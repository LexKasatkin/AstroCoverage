import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
from astropy.time import Time
import astropy.units as u
import sep
import subprocess
import re
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy_healpix import HEALPix
import warnings

warnings.filterwarnings("ignore", category=FITSFixedWarning)

NSIDE = 128
healpix_instance = HEALPix(nside=NSIDE, order='nested', frame='icrs')


class FitsAnalyzer:
    def __init__(self, astap_path=None, cfg_global=None):
        """
        astap_path: путь к Astap для plate solving
        cfg_global: словарь с глобальной конфигурацией (latitude, longitude, elevation)
        """
        self.astap_path = astap_path
        self.cfg_global = cfg_global or {}

    # ============================
    # Основной метод анализа файла
    # ============================
    def process_file(self, fits_path, skip_db=False):
        """
        Анализ одного FITS файла. Возвращает словарь record.
        skip_db: если True и запись в БД уже есть, пропускаем файл
        """
        json_path = os.path.splitext(fits_path)[0] + ".json"
        if skip_db and os.path.exists(json_path):
            print(f"Skipping {fits_path} (JSON exists and record in DB)")
            return None

        try:
            with fits.open(fits_path, ignore_missing_end=True) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                if data is None:
                    return None

                # Поддержка 3D массивов
                if data.ndim == 3:
                    data = data[0] if data.shape[0] <= 4 else data[:, :, 0]
                data = data.astype(float)

                # Frame type
                if self.detect_frame_type(header, fits_path) != "LIGHT":
                    return None

                # Pixel scale и FWHM
                pixel_scale = self.compute_pixel_scale(header, data.shape)
                fwhm_px, fwhm_arcsec = self.compute_fwhm(data, pixel_scale)

                # RA / DEC
                ra, dec = self.get_radec(header)

                # Polygon и WCS
                polygon, wcs_source, plate_solved = self.get_polygon(fits_path, header, data.shape)

                # ASTAP анализ (HFD, STARS)
                hfd, stars = self.run_astap_analysis(fits_path)

                # BBox & Healpix
                min_ra, max_ra, min_dec, max_dec = self.compute_bbox(polygon) if polygon else (None, None, None, None)
                healpix = self.compute_healpix(ra, dec)

                # FOV
                fov_width = header.get("NAXIS1", 0) * pixel_scale / 3600 if pixel_scale else None
                fov_height = header.get("NAXIS2", 0) * pixel_scale / 3600 if pixel_scale else None

                # Observer & date
                observer = self.get_observer_location(header)
                date_loc = header.get("DATE-LOC") or self.get_date_loc(header)

                # Altitude / Azimuth / Airmass
                altitude, azimuth, airmass = self.compute_altaz(header, ra, dec, observer)

                # Moon & Sun
                moon_info, sun_info = self.compute_solar_moon(header)

                # Сбор record
                record = {
                    "file_path": os.path.abspath(fits_path),
                    "file_name": os.path.basename(fits_path),
                    "ra": ra, "dec": dec, "healpix": healpix,
                    "min_ra": min_ra, "max_ra": max_ra, "min_dec": min_dec, "max_dec": max_dec,
                    "fov_width": fov_width, "fov_height": fov_height, "pixel_scale": pixel_scale,
                    "hfd": hfd, "fwhm_px": fwhm_px, "fwhm_arcsec": fwhm_arcsec, "stars": stars,
                    "airmass": airmass, "altitude": altitude, "azimuth": azimuth,
                    "exptime": header.get("EXPTIME"), "gain": header.get("GAIN"),
                    "offset": header.get("OFFSET"), "ccd_temp": header.get("CCD-TEMP"),
                    "camera": header.get("CAMERAID") or header.get("INSTRUME"),
                    "telescope": header.get("TELESCOP"), "filter": header.get("FILTER"),
                    "plate_solved": plate_solved, "wcs_source": wcs_source,
                    "date_obs": header.get("DATE-OBS"), "date_loc": date_loc,
                    "latitude": observer.lat.deg, "longitude": observer.lon.deg,
                    "elevation": observer.height.value,
                    "polygon": polygon,
                    "moon_alt": moon_info["alt"], "moon_az": moon_info["az"],
                    "sun_alt": sun_info["alt"], "sun_az": sun_info["az"]
                }

            # Запись JSON
            with open(json_path, "w", encoding="utf-8") as fjson:
                json.dump(record, fjson, indent=4)

            return record

        except Exception as e:
            print("ERROR:", fits_path, e)
            return None

    # ============================
    # Вспомогательные методы
    # ============================

    def detect_frame_type(self, header, filename):
        imagetyp = header.get("IMAGETYP") or header.get("FRAME") or header.get("TYPE")
        if imagetyp:
            t = imagetyp.strip().upper()
            if "LIGHT" in t: return "LIGHT"
            if "DARK" in t: return "DARK"
            if "FLAT" in t: return "FLAT"
            if "BIAS" in t or "OFFSET" in t: return "BIAS"
        name = filename.lower()
        if "dark" in name: return "DARK"
        if "flat" in name: return "FLAT"
        if "bias" in name or "offset" in name: return "BIAS"
        return "LIGHT"

    def compute_fwhm(self, data, pixel_scale=None):
        try:
            img = np.array(data, dtype=np.float32)
            bkg = sep.Background(img)
            data_sub = img - bkg.back()
            objects = sep.extract(data_sub, thresh=8.0*bkg.globalrms)
            if len(objects) == 0: return None, None
            a, b = objects['a'], objects['b']
            valid = (a>0) & (b>0)
            if not np.any(valid): return None, None
            sigmas = 0.5*(a[valid]+b[valid])
            fwhm_px = float(np.median(sigmas)*2.355)
            fwhm_arcsec = fwhm_px*pixel_scale if pixel_scale else None
            return fwhm_px, fwhm_arcsec
        except:
            return None, None

    def compute_pixel_scale(self, header, image_shape=None, polygon=None):
        try:
            wcs = WCS(header)
            scales_deg = proj_plane_pixel_scales(wcs)
            scale_arcsec = np.mean(np.abs(scales_deg)) * 3600
            if 0 < scale_arcsec < 180: return scale_arcsec
        except:
            pass

        if image_shape:
            h, w = image_shape
            if polygon:
                ras = [p[0] for p in polygon]
                decs = [p[1] for p in polygon]
                ra_span = max(ras)-min(ras)
                dec_span = max(decs)-min(decs)
            else:
                ra_span = dec_span = 1.0
            return (ra_span*3600/w + dec_span*3600/h)/2
        return None

    def get_radec(self, header):
        ra = dec = None
        for k1, k2 in [("RA", "DEC"), ("OBJCTRA", "OBJCTDEC"), ("CRVAL1", "CRVAL2")]:
            if k1 in header and k2 in header:
                try:
                    ra, dec = float(header[k1]), float(header[k2])
                    break
                except:
                    pass
        return ra, dec

    def get_polygon(self, fits_path, header, image_shape):
        polygon = None
        wcs_header = None
        wcs_source = "NONE"
        plate_solved = False

        wcs_path = os.path.splitext(fits_path)[0] + ".wcs"
        polygon, wcs_header = self.read_polygon_from_wcs_file(wcs_path, image_shape)

        if polygon:
            wcs_source = "WCS_FILE"
            plate_solved = True
        elif self.astap_path and self.solve_to_wcs(fits_path):
            polygon, wcs_header = self.read_polygon_from_wcs_file(wcs_path, image_shape)
            if polygon:
                wcs_source = "ASTAP"
                plate_solved = True
        else:
            polygon = self.compute_polygon_safe(header, image_shape)
            wcs_source = "FITS"

        return polygon, wcs_source, plate_solved

    def read_wcs_header(self, wcs_path):
        if not os.path.exists(wcs_path): return None
        try:
            with open(wcs_path, "r", encoding="utf-8") as f:
                return fits.Header.fromstring(f.read(), sep="\n")
        except:
            return None

    def read_polygon_from_wcs_file(self, wcs_path, image_shape):
        header = self.read_wcs_header(wcs_path)
        if header is None: return None, None
        try:
            wcs = WCS(header)
            h, w = image_shape
            pix = [(0, 0), (w, 0), (w, h), (0, h)]
            world = wcs.all_pix2world(pix, 0)
            polygon = [[float(x), float(y)] for x, y in world]
            return polygon, header
        except:
            return None, None

    def compute_polygon_safe(self, header, image_shape):
        try:
            wcs = WCS(header)
            h, w = image_shape
            pix = [(0, 0), (w, 0), (w, h), (0, h)]
            world = wcs.all_pix2world(pix, 0)
            polygon = [[float(x), float(y)] for x, y in world]
            return polygon
        except:
            return None

    def compute_bbox(self, polygon):
        ras = [p[0] for p in polygon]
        decs = [p[1] for p in polygon]
        min_dec, max_dec = min(decs), max(decs)
        ras_sorted = sorted(ras)
        span_normal = ras_sorted[-1] - ras_sorted[0]
        span_wrap = (ras_sorted[0]+360) - ras_sorted[-1]
        if span_wrap < span_normal:
            min_ra = ras_sorted[-1]
            max_ra = ras_sorted[0]+360
        else:
            min_ra, max_ra = ras_sorted[0], ras_sorted[-1]
        return min_ra, max_ra, min_dec, max_dec

    def compute_healpix(self, ra, dec):
        if ra is None or dec is None: return None
        return int(healpix_instance.skycoord_to_healpix(SkyCoord(ra=ra*u.deg, dec=dec*u.deg)))

    def get_observer_location(self, header):
        lat = header.get("SITELAT") or self.cfg_global.get("latitude", 0.0)
        lon = header.get("SITELONG") or self.cfg_global.get("longitude", 0.0)
        ele = header.get("SITEALT") or self.cfg_global.get("elevation", 0.0)
        return EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=ele*u.m)

    def get_date_loc(self, header):
        if "DATE-OBS" not in header: return None
        observer = self.get_observer_location(header)
        t_obs = Time(header["DATE-OBS"])
        offset = observer.lon.deg/360*24*u.hour
        t_local = t_obs + offset
        return t_local.iso

    def compute_altaz(self, header, ra, dec, observer):
        if ra is not None and dec is not None and "DATE-OBS" in header:
            t_obs = Time(header["DATE-OBS"])
            altaz = SkyCoord(ra=ra*u.deg, dec=dec*u.deg).transform_to(
                AltAz(obstime=t_obs, location=observer)
            )
            altitude = altaz.alt.deg
            azimuth = altaz.az.deg
            airmass = 1/np.cos(np.radians(90-altitude)) if altitude>0 else None
        else:
            altitude = header.get("CENTALT")
            azimuth = header.get("CENTAZ")
            airmass = header.get("AIRMASS")
        return altitude, azimuth, airmass

    def compute_solar_moon(self, header):
        if "DATE-OBS" not in header: return {"alt":None,"az":None}, {"alt":None,"az":None}
        t_obs = Time(header["DATE-OBS"])
        observer = self.get_observer_location(header)
        altaz_frame = AltAz(obstime=t_obs, location=observer)
        moon_coord = get_body('moon', t_obs, location=observer).transform_to(altaz_frame)
        sun_coord = get_body('sun', t_obs, location=observer).transform_to(altaz_frame)
        return {"alt": moon_coord.alt.deg, "az": moon_coord.az.deg}, {"alt": sun_coord.alt.deg, "az": sun_coord.az.deg}

    def solve_to_wcs(self, fits_path):
        if not self.astap_path or not os.path.exists(self.astap_path) or not os.path.exists(fits_path):
            return False
        try:
            subprocess.run([self.astap_path, "-f", fits_path, "-o", fits_path],
                           capture_output=True, text=True, check=True, timeout=60)
            return True
        except:
            return False

    def run_astap_analysis(self, fits_path):
        if not self.astap_path:
            return None, None
        try:
            result = subprocess.run([self.astap_path, "-f", fits_path, "-analyse"],
                                    capture_output=True, text=True, timeout=15)
            hfd = re.search(r"HFD_MEDIAN\s*=\s*([\d\.]+)", result.stdout)
            stars = re.search(r"STARS\s*=\s*(\d+)", result.stdout)
            return float(hfd.group(1)) if hfd else None, int(stars.group(1)) if stars else None
        except:
            return None, None
    