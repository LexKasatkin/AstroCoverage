import os
import sys
import json
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
from astropy.time import Time
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy_healpix import HEALPix
import sep
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from fits_db import FitsDatabase

warnings.filterwarnings("ignore", category=FITSFixedWarning)

NSIDE = 128
healpix_instance = HEALPix(nside=NSIDE, order='nested', frame='icrs')
cfg_global = {}

# ============================
# FRAME TYPE
# ============================
def detect_frame_type(header, filename):
    imagetyp = header.get("IMAGETYP") or header.get("FRAME") or header.get("TYPE")

    if imagetyp:
        t = imagetyp.strip().upper()
        if "LIGHT" in t:
            return "LIGHT"
        if "DARK" in t:
            return "DARK"
        if "FLAT" in t:
            return "FLAT"
        if "BIAS" in t or "OFFSET" in t:
            return "BIAS"

    name = filename.lower()
    if "dark" in name:
        return "DARK"
    if "flat" in name:
        return "FLAT"
    if "bias" in name or "offset" in name:
        return "BIAS"

    return "LIGHT"

# ============================
# Utils
# ============================
def compute_fwhm_sep(image, pixel_scale=None):
    try:
        img = np.array(image, dtype=np.float32)
        bkg = sep.Background(img)
        data_sub = img - bkg.back()  # вызываем метод

        objects = sep.extract(data_sub, thresh=8.0 * bkg.globalrms)
        if len(objects) == 0:
            return None, None

        a = objects['a']
        b = objects['b']

        valid = (a > 0) & (b > 0)
        if not np.any(valid):
            return None, None

        sigmas = 0.5 * (a[valid] + b[valid])
        fwhm_px = float(np.median(sigmas) * 2.355)
        fwhm_arcsec = fwhm_px * pixel_scale if pixel_scale is not None else None

        return fwhm_px, fwhm_arcsec

    except Exception as e:
        print("Ошибка при вычислении FWHM:", e)
        return None, None


def compute_pixel_scale(header, image_shape=None, polygon=None):
    try:
        wcs = WCS(header)
        scales_deg = proj_plane_pixel_scales(wcs)
        scale_arcsec = np.mean(np.abs(scales_deg)) * 3600
        if 0 < scale_arcsec < 180:
            return scale_arcsec
    except:
        pass

    if image_shape:
        h, w = image_shape
        if polygon:
            ras = [p[0] for p in polygon]
            decs = [p[1] for p in polygon]
            ra_span = max(ras) - min(ras)
            dec_span = max(decs) - min(decs)
        else:
            ra_span = dec_span = 1.0
        return (ra_span * 3600 / w + dec_span * 3600 / h) / 2
    return None


def read_wcs_header(wcs_path):
    if not os.path.exists(wcs_path):
        return None
    try:
        with open(wcs_path, "r", encoding="utf-8") as f:
            return fits.Header.fromstring(f.read(), sep="\n")
    except:
        return None


def read_polygon_from_wcs_file(wcs_path, image_shape):
    header = read_wcs_header(wcs_path)
    if header is None:
        return None, None
    try:
        wcs = WCS(header)
        h, w = image_shape
        pix = [(0, 0), (w, 0), (w, h), (0, h)]
        world = wcs.all_pix2world(pix, 0)
        polygon = [[float(x), float(y)] for x, y in world]
        return polygon, header
    except:
        return None, None


def compute_polygon_safe(header, image_shape):
    try:
        wcs = WCS(header)
        h, w = image_shape
        pix = [(0, 0), (w, 0), (w, h), (0, h)]
        world = wcs.all_pix2world(pix, 0)
        polygon = [[float(x), float(y)] for x, y in world]
        return polygon
    except:
        return None


def compute_bbox(polygon):
    ras = [p[0] for p in polygon]
    decs = [p[1] for p in polygon]
    min_dec, max_dec = min(decs), max(decs)
    ras_sorted = sorted(ras)
    span_normal = ras_sorted[-1] - ras_sorted[0]
    span_wrap = (ras_sorted[0] + 360) - ras_sorted[-1]
    if span_wrap < span_normal:
        min_ra = ras_sorted[-1]
        max_ra = ras_sorted[0] + 360
    else:
        min_ra, max_ra = ras_sorted[0], ras_sorted[-1]
    return min_ra, max_ra, min_dec, max_dec


def compute_healpix(ra, dec):
    if ra is None or dec is None:
        return None
    return int(healpix_instance.skycoord_to_healpix(SkyCoord(ra=ra*u.deg, dec=dec*u.deg)))


def get_observer_location_from_header(header):
    lat = header.get("SITELAT") or cfg_global.get("latitude", 0.0)
    lon = header.get("SITELONG") or cfg_global.get("longitude", 0.0)
    ele = header.get("SITEALT") or cfg_global.get("elevation", 0.0)
    return EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=ele*u.m)


def get_date_loc(header):
    if "DATE-OBS" not in header:
        return None
    observer = get_observer_location_from_header(header)
    t_obs = Time(header["DATE-OBS"])
    offset = observer.lon.deg / 360 * 24 * u.hour
    t_local = t_obs + offset
    return t_local.iso


def compute_solar_moon(header):
    if "DATE-OBS" not in header:
        return None, None
    t_obs = Time(header["DATE-OBS"])
    observer = get_observer_location_from_header(header)
    altaz_frame = AltAz(obstime=t_obs, location=observer)
    moon_coord = get_body('moon', t_obs, location=observer).transform_to(altaz_frame)
    sun_coord = get_body('sun', t_obs, location=observer).transform_to(altaz_frame)
    return {"alt": moon_coord.alt.deg, "az": moon_coord.az.deg}, {"alt": sun_coord.alt.deg, "az": sun_coord.az.deg}


def solve_to_wcs(astap_path, fits_path):
    if not astap_path or not os.path.exists(astap_path) or not os.path.exists(fits_path):
        return False
    try:
        subprocess.run([astap_path, "-f", fits_path, "-o", fits_path],
                       capture_output=True, text=True, check=True, timeout=60)
        return True
    except:
        return False


def run_astap_analysis(astap_path, fits_path):
    if not astap_path:
        return None, None
    try:
        result = subprocess.run([astap_path, "-f", fits_path, "-analyse"],
                                capture_output=True, text=True, timeout=15)
        hfd = re.search(r"HFD_MEDIAN\s*=\s*([\d\.]+)", result.stdout)
        stars = re.search(r"STARS\s*=\s*(\d+)", result.stdout)
        return float(hfd.group(1)) if hfd else None, int(stars.group(1)) if stars else None
    except:
        return None, None

# ============================
# FITS processing
# ============================
def process_single_fits(args):
    fits_path, astap_path, skip_db = args

    # Проверка: есть JSON и запись в БД
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

            if data.ndim == 3:
                data = data[0] if data.shape[0] <= 4 else data[:, :, 0]

            data = data.astype(float)

            # ============================
            # FRAME TYPE CHECK
            # ============================
            frame_type = detect_frame_type(header, os.path.basename(fits_path))
            if frame_type != "LIGHT":
                return None

            # ============================
            # Compute pixel scale & FWHM
            # ============================
            pixel_scale = compute_pixel_scale(header, data.shape)
            fwhm_px, fwhm_arcsec = compute_fwhm_sep(data, pixel_scale)

            # ============================
            # RA / DEC
            # ============================
            ra = dec = None
            for k1, k2 in [("RA", "DEC"), ("OBJCTRA", "OBJCTDEC"), ("CRVAL1", "CRVAL2")]:
                if k1 in header and k2 in header:
                    try:
                        ra, dec = float(header[k1]), float(header[k2])
                        break
                    except:
                        pass

            # ============================
            # WCS / polygon
            # ============================
            polygon = None
            wcs_header = None
            wcs_source = "NONE"
            plate_solved = False

            wcs_path = os.path.splitext(fits_path)[0] + ".wcs"
            polygon, wcs_header = read_polygon_from_wcs_file(wcs_path, data.shape)

            if polygon:
                wcs_source = "WCS_FILE"
                plate_solved = True
            elif astap_path and solve_to_wcs(astap_path, fits_path):
                polygon, wcs_header = read_polygon_from_wcs_file(wcs_path, data.shape)
                if polygon:
                    wcs_source = "ASTAP"
                    plate_solved = True
            else:
                polygon = compute_polygon_safe(header, data.shape)
                wcs_source = "FITS"

            hfd, stars_count = run_astap_analysis(astap_path, fits_path)

            # ============================
            # BBox & Healpix
            # ============================
            min_ra, max_ra, min_dec, max_dec = compute_bbox(polygon) if polygon else (None, None, None, None)
            healpix = compute_healpix(ra, dec)

            # ============================
            # FOV
            # ============================
            fov_width = fov_height = None
            if pixel_scale and "NAXIS1" in header and "NAXIS2" in header:
                fov_width = header["NAXIS1"] * pixel_scale / 3600
                fov_height = header["NAXIS2"] * pixel_scale / 3600

            # ============================
            # Observer & time
            # ============================
            observer = get_observer_location_from_header(header)
            date_loc = header.get("DATE-LOC") or get_date_loc(header)

            if ra is not None and dec is not None and "DATE-OBS" in header:
                t_obs = Time(header["DATE-OBS"])
                altaz = SkyCoord(ra=ra*u.deg, dec=dec*u.deg).transform_to(
                    AltAz(obstime=t_obs, location=observer)
                )
                altitude = altaz.alt.deg
                azimuth = altaz.az.deg
                airmass = 1 / np.cos(np.radians(90 - altitude)) if altitude > 0 else None
            else:
                altitude = header.get("CENTALT")
                azimuth = header.get("CENTAZ")
                airmass = header.get("AIRMASS")

            moon_info, sun_info = compute_solar_moon(header)

            # ============================
            # Record
            # ============================
            record = {
                "file_path": os.path.abspath(fits_path),
                "file_name": os.path.basename(fits_path),
                "ra": ra, "dec": dec, "healpix": healpix,
                "min_ra": min_ra, "max_ra": max_ra, "min_dec": min_dec, "max_dec": max_dec,
                "fov_width": fov_width, "fov_height": fov_height, "pixel_scale": pixel_scale,
                "hfd": hfd, "fwhm_px": fwhm_px, "fwhm_arcsec": fwhm_arcsec, "stars": stars_count,
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
# DATABASE GENERATION WITH CHECK
# ============================
def generate_database(config_path, db_path):
    global cfg_global
    with open(config_path) as f:
        cfg_global = json.load(f)

    astap_path = cfg_global.get("astap_path")
    db = FitsDatabase(db_path)

    fits_files = []
    for scope in cfg_global.get("telescopes", []):
        fits_dir = scope.get("fits_dir")
        if not fits_dir or not os.path.exists(fits_dir):
            continue
        for dp, dn, fn in os.walk(fits_dir):
            for f in fn:
                if f.lower().endswith((".fits", ".fit", ".fts")):
                    fits_files.append(os.path.join(dp, f))

    files_to_process = []
    skip_db_flags = {}
    for f in fits_files:
        json_path = os.path.splitext(f)[0] + ".json"
        in_db = db.check_record_exists(os.path.abspath(f))
        in_json = os.path.exists(json_path)

        if in_db and in_json:
            print(f"Skipping {f} (already in DB and JSON exists)")
            continue

        files_to_process.append(f)
        skip_db_flags[f] = in_db

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_fits, (f, astap_path, skip_db_flags[f])): f
            for f in files_to_process
        }

        for fut in as_completed(futures):
            record = fut.result()
            if not record:
                continue

            record_id = db.insert_record(record)
            if record_id:
                print(f"Record successfully inserted: {record['file_name']} (ID={record_id})")
            else:
                print(f"WARNING: Failed to insert record: {record['file_name']}")

    db.close()
    print("DATABASE READY:", db_path)

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
    CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
    DB_PATH = os.path.join(BASE_DIR, "data.sqlite")
    generate_database(CONFIG_PATH, DB_PATH)
