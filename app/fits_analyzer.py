import os
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
from photutils.detection import DAOStarFinder
from photutils.psf import fit_fwhm

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
                fwhm_px, fwhm_arcsec, snr_median, sky_snr = self.compute_fwhm(data, pixel_scale)

                # sky brightness / sky backgroud / sky rms
                sky_brightness, sky_background, sky_rms = self.compute_sky_brightness(
                    data,
                    pixel_scale=pixel_scale,
                    gain=header.get("GAIN"),
                    exposure=header.get("EXPTIME")
                )
                bortle_float, bortle_class = self.estimate_bortle(sky_brightness)
                
                # RA / DEC
                ra, dec = self.get_radec(header)

                # Polygon и WCS
                polygon, wcs_source, plate_solved = self.get_polygon(fits_path, header, data.shape)

                # ASTAP анализ (HFD, STARS)
                hfd, stars = self.run_astap_analysis(fits_path)
                hfd_arcsec = hfd * pixel_scale if hfd and pixel_scale else None

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
                moon_phase= moon_info["phase"]

                # Сбор record
                record = {
                    "file_path": os.path.abspath(fits_path),
                    "file_name": os.path.basename(fits_path),
                    "ra": ra, "dec": dec, "healpix": healpix,
                    "min_ra": min_ra, "max_ra": max_ra, "min_dec": min_dec, "max_dec": max_dec,
                    "fov_width": fov_width, "fov_height": fov_height, "pixel_scale": pixel_scale,
                    "hfd": hfd, "hfd_arcsec": hfd_arcsec, "snr_median": snr_median, "sky_snr": sky_snr,
                    "sky_background": sky_background, "sky_rms": sky_rms, "sky_brightness": sky_brightness,
                    "bortle": bortle_class, "bortle_float": bortle_float,
                    "fwhm_px": fwhm_px, "fwhm_arcsec": fwhm_arcsec, "stars": stars,
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
                    "sun_alt": sun_info["alt"], "sun_az": sun_info["az"],
                    "moon_phase": moon_phase
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

    def compute_fwhm_photutils(self, data, pixel_scale=None):
        try:
            data = np.asarray(data, dtype=np.float32)

            # Фон
            bkg = sep.Background(data)
            data_sub = data - bkg.back()

            # Поиск звезд
            finder = DAOStarFinder(
                threshold=4.0 * bkg.globalrms,
                fwhm=4.0,
                sharplo=0.2,
                sharphi=1.0,
                roundlo=-1.0,
                roundhi=1.0
            )

            stars = finder(data_sub)
            if stars is None or len(stars) < 5:
                return None, None

            # Координаты звезд
            xy = list(zip(stars["xcentroid"], stars["ycentroid"]))

            # PSF fit
            fwhm_values = fit_fwhm(
                data_sub,
                xypos=xy,
                fwhm=4.0,
                fit_shape=(9, 9)
            )

            # Очистка мусора
            fwhm_values = np.array(fwhm_values)
            fwhm_values = fwhm_values[np.isfinite(fwhm_values)]
            fwhm_values = fwhm_values[(fwhm_values > 1.0) & (fwhm_values < 15.0)]

            if len(fwhm_values) == 0:
                return None, None

            fwhm_px = float(np.median(fwhm_values))
            fwhm_arcsec = fwhm_px * pixel_scale if pixel_scale else None

            return fwhm_px, fwhm_arcsec

        except Exception as e:
            print("FWHM error:", e)
            return None, None

    def compute_fwhm(self, data, pixel_scale=None, snr_threshold=5.0):
        """
        Расчет FWHM и апертурного SNR, близкого к PixInsight:
        - фильтрация по эллиптичности
        - исключение краёв кадра
        - защита от шумов и пересветов
        - медианное значение (устойчиво к выбросам)
        - фильтрация по SNR
        """

        try:
            img = np.asarray(data, dtype=np.float32)

            # Фон
            bkg = sep.Background(img)
            data_sub = img - bkg.back()
            rms = bkg.globalrms

            # Детекция объектов
            thresh = 3.0 * rms
            objects = sep.extract(data_sub, thresh=thresh)

            if len(objects) == 0:
                return None, None, None

            h, w = img.shape

            # Параметры объектов
            a = objects['a']   # полуоси
            b = objects['b']
            x = objects['x']
            y = objects['y']
            flux = objects['flux']

            # Эллиптичность
            ellipticity = 1.0 - (b / a)

            # Начальная фильтрация
            mask = (
                (a > 1.2) & (a < 8.0) & 
                (ellipticity < 0.4) &
                (x > 0.1 * w) & (x < 0.9 * w) &
                (y > 0.1 * h) & (y < 0.9 * h)
            )

            if not np.any(mask):
                return None, None, None

            # Апертурный SNR с учетом FWHM
            sigmas = 0.5 * (a[mask] + b[mask])
            fwhm_px_est = sigmas * 2.355
            # радиус апертуры ≈ FWHM / 2
            r_ap = fwhm_px_est / 2
            npix = np.pi * r_ap**2

            snr_aperture = flux[mask] / (np.sqrt(npix) * rms)
            snr_mask = snr_aperture > snr_threshold

            if not np.any(snr_mask):
                return None, None, None

            # Финальная маска
            final_mask = np.zeros_like(mask, dtype=bool)
            final_mask[np.where(mask)[0][snr_mask]] = True

            # Медианный FWHM
            fwhm_px = float(np.median(sigmas[snr_mask]) * 2.355)
            fwhm_arcsec = fwhm_px * pixel_scale if pixel_scale else None

            # Медианный SNR
            snr_median = float(np.median(snr_aperture[snr_mask]))

            sky_snr = np.median(data) / rms

            return fwhm_px, fwhm_arcsec, snr_median, sky_snr

        except Exception as e:
            print("FWHM computation error:", e)
            return None, None, None


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
    
    # ==========================================================
    # SKY BRIGHTNESS
    # ==========================================================
    def compute_sky_brightness(self, data, pixel_scale, gain, exposure, zeropoint=21.5):
        try:
            bkg = sep.Background(data)
            sky_adu = np.median(bkg.back())
            sky_rms = bkg.globalrms

            if pixel_scale is None or gain is None or exposure is None:
                return None, sky_adu, sky_rms

            sky_e = sky_adu * gain
            sky_flux = sky_e / (exposure * pixel_scale ** 2)

            if sky_flux <= 0:
                return None, sky_adu, sky_rms

            sky_mag = zeropoint - 2.5 * np.log10(sky_flux)

            return sky_mag, sky_adu, sky_rms

        except Exception as e:
            print("Sky brightness error:", e)
            return None, None, None
        
    def estimate_bortle(self, sky_brightness):
        """
        Estimate Bortle class from sky brightness (mag/arcsec^2)
        Returns:
            bortle_float, bortle_int
        """
        if sky_brightness is None:
            return None, None

        table = [
            (22.0, 1),
            (21.7, 2),
            (21.3, 3),
            (20.8, 4),
            (20.3, 5),
            (19.8, 6),
            (19.1, 7),
            (18.5, 8),
            (17.5, 9),
        ]

        # brighter sky → higher bortle
        for i in range(len(table) - 1):
            m1, b1 = table[i]
            m2, b2 = table[i + 1]

            if m1 >= sky_brightness >= m2:
                # linear interpolation
                t = (m1 - sky_brightness) / (m1 - m2)
                bortle = b1 + t * (b2 - b1)
                return round(bortle, 2), int(round(bortle))

        if sky_brightness > 22.0:
            return 1.0, 1
        if sky_brightness < 17.5:
            return 9.0, 9

        return None, None


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
        """
        Возвращает координаты Луны и Солнца в данный момент наблюдения
        и фазу Луны (0-1: 0 - новая, 0.5 - полная)
        """
        if "DATE-OBS" not in header:
            return {"alt": None, "az": None, "phase": None}, {"alt": None, "az": None}

        t_obs = Time(header["DATE-OBS"])
        observer = self.get_observer_location(header)
        altaz_frame = AltAz(obstime=t_obs, location=observer)

        # Луна и Солнце в небесных координатах для расчета фазы
        moon_icrs = get_body('moon', t_obs, location=observer)
        sun_icrs = get_body('sun', t_obs, location=observer)

        # Луна и Солнце в горизонте для координат alt/az
        moon_altaz = moon_icrs.transform_to(altaz_frame)
        sun_altaz = sun_icrs.transform_to(altaz_frame)

        # Фаза Луны (0-1)
        elongation = moon_icrs.separation(sun_icrs)  # угол Луна-Солнце
        moon_phase = (1 + np.cos(elongation.rad)) / 2  # 0 - новолуние, 1 - полнолуние

        moon_info = {"alt": moon_altaz.alt.deg, "az": moon_altaz.az.deg, "phase": moon_phase}
        sun_info = {"alt": sun_altaz.alt.deg, "az": sun_altaz.az.deg}

        return moon_info, sun_info

    def solve_to_wcs(self, fits_path):
        if not self.astap_path or not os.path.exists(self.astap_path) or not os.path.exists(fits_path):
            return False
        try:
            subprocess.run([self.astap_path, "-f", fits_path, "-o", fits_path],
                            capture_output=True, text=True, check=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
            return True
        except:
            return False

    def run_astap_analysis(self, fits_path):
        if not self.astap_path:
            return None, None
        try:
            result = subprocess.run([self.astap_path, "-f", fits_path, "-analyse"],
                                    capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            hfd = re.search(r"HFD_MEDIAN\s*=\s*([\d\.]+)", result.stdout)
            stars = re.search(r"STARS\s*=\s*(\d+)", result.stdout)
            return float(hfd.group(1)) if hfd else None, int(stars.group(1)) if stars else None
        except:
            return None, None