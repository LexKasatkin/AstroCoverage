import os
import sys
import json
import subprocess
import sqlite3
import re
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

warnings.filterwarnings("ignore", category=FITSFixedWarning)

NSIDE = 128
healpix_instance = HEALPix(nside=NSIDE, order='nested', frame='icrs')
cfg_global = {}

# ============================
# Utils
# ============================
def compute_fwhm_arcsec(fwhm_px, pixel_scale):
    if fwhm_px is None or pixel_scale is None:
        return None
    return fwhm_px * pixel_scale

def compute_fwhm_sep(image, pixel_scale=None):
    try:
        img = np.array(image, dtype=np.float32)
        bkg = sep.Background(img)
        data_sub = img - bkg
        objects = sep.extract(data_sub, thresh=3.0*bkg.globalrms)
        if len(objects) == 0:
            return None, None
        a = objects['a']
        b = objects['b']
        sigmas = 0.5 * (a + b)
        fwhm_px = float(np.median(sigmas) * 2.355)
        fwhm_arcsec = fwhm_px * pixel_scale if pixel_scale else None
        return fwhm_px, fwhm_arcsec
    except Exception as e:
        print("FWHM SEP error:", e)
        return None, None

# ============================
# WCS helpers
# ============================
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

def compute_polygon_from_header(header, shape):
    return compute_polygon_safe(header, shape)

def compute_bbox(polygon):
    ras = [p[0] for p in polygon]
    decs = [p[1] for p in polygon]
    min_dec = min(decs)
    max_dec = max(decs)
    ras_sorted = sorted(ras)
    span_normal = ras_sorted[-1] - ras_sorted[0]
    span_wrap = (ras_sorted[0] + 360) - ras_sorted[-1]
    if span_wrap < span_normal:
        min_ra = ras_sorted[-1]
        max_ra = ras_sorted[0] + 360
    else:
        min_ra = ras_sorted[0]
        max_ra = ras_sorted[-1]
    return min_ra, max_ra, min_dec, max_dec

def compute_healpix(ra, dec):
    if ra is None or dec is None:
        return None
    return int(healpix_instance.skycoord_to_healpix(SkyCoord(ra=ra*u.deg, dec=dec*u.deg)))

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
        return (ra_span*3600/w + dec_span*3600/h)/2
    return None

# ============================
# ASTAP
# ============================
def solve_to_wcs(astap_path, fits_path):
    if not astap_path or not os.path.exists(astap_path) or not os.path.exists(fits_path):
        return False
    try:
        subprocess.run([astap_path, "-f", fits_path, "-o", fits_path, "-update"],
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
# Observer helpers
# ============================
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
    return (
        {"alt": moon_coord.alt.deg, "az": moon_coord.az.deg},
        {"alt": sun_coord.alt.deg, "az": sun_coord.az.deg}
    )

# ============================
# MAIN
# ============================
def process_fits(astap_path, fits_path, conn, cur):
    json_path = fits_path + ".json"
    cur.execute("SELECT 1 FROM fits_data WHERE file_path = ?", (os.path.abspath(fits_path),))
    if cur.fetchone():
        return
    try:
        with fits.open(fits_path, ignore_missing_end=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            if data is None:
                return
            if data.ndim == 3:
                data = data[0] if data.shape[0] <= 4 else data[:, :, 0]
            data = data.astype(float)

            ra = dec = None
            for k1, k2 in [("RA", "DEC"), ("OBJCTRA", "OBJCTDEC"), ("CRVAL1", "CRVAL2")]:
                if k1 in header and k2 in header:
                    try:
                        ra, dec = float(header[k1]), float(header[k2])
                        break
                    except:
                        pass

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
                polygon = compute_polygon_from_header(header, data.shape)
                wcs_header = header
                wcs_source = "FITS"

            hfd, stars_count = run_astap_analysis(astap_path, fits_path)

            pixel_scale = compute_pixel_scale(header, data.shape, polygon)
            fwhm_px, fwhm_arcsec = compute_fwhm_sep(data, pixel_scale)

            min_ra, max_ra, min_dec, max_dec = compute_bbox(polygon) if polygon else (None, None, None, None)
            healpix = compute_healpix(ra, dec)

            fov_width = fov_height = None
            if pixel_scale and "NAXIS1" in header and "NAXIS2" in header:
                fov_width = header["NAXIS1"] * pixel_scale / 3600
                fov_height = header["NAXIS2"] * pixel_scale / 3600

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

            record = {
                "file_name": os.path.basename(fits_path),
                "file_path": os.path.abspath(fits_path),
                "ra": ra,
                "dec": dec,
                "healpix": healpix,
                "min_ra": min_ra,
                "max_ra": max_ra,
                "min_dec": min_dec,
                "max_dec": max_dec,
                "fov_width": fov_width,
                "fov_height": fov_height,
                "pixel_scale": pixel_scale,
                "hfd": hfd,
                "fwhm_px": fwhm_px,
                "fwhm_arcsec": fwhm_arcsec,
                "stars": stars_count,
                "airmass": airmass,
                "altitude": altitude,
                "azimuth": azimuth,
                "exptime": header.get("EXPTIME"),
                "gain": header.get("GAIN"),
                "offset": header.get("OFFSET"),
                "ccd_temp": header.get("CCD-TEMP"),
                "camera": header.get("CAMERAID") or header.get("INSTRUME"),
                "telescope": header.get("TELESCOP"),
                "filter": header.get("FILTER"),
                "plate_solved": plate_solved,
                "wcs_source": wcs_source,
                "date_obs": header.get("DATE-OBS"),
                "date_loc": date_loc,
                "latitude": observer.lat.deg,
                "longitude": observer.lon.deg,
                "elevation": observer.height.value,
                "polygon": polygon,
                "moon_alt": moon_info["alt"] if moon_info else None,
                "moon_az": moon_info["az"] if moon_info else None,
                "sun_alt": sun_info["alt"] if sun_info else None,
                "sun_az": sun_info["az"] if sun_info else None
            }

            with open(json_path, "w") as f:
                json.dump(record, f, indent=2)

            cur.execute("""
                INSERT OR REPLACE INTO fits_data (
                    file_name, file_path,
                    ra, dec, healpix,
                    min_ra, max_ra, min_dec, max_dec,
                    fov_width, fov_height, pixel_scale,
                    hfd, fwhm_px, fwhm_arcsec, stars,
                    airmass, altitude, azimuth,
                    exptime, gain, offset, ccd_temp,
                    camera, telescope, filter,
                    plate_solved, wcs_source,
                    date_obs, date_loc,
                    latitude, longitude, elevation,
                    polygon,
                    moon_alt, moon_az,
                    sun_alt, sun_az
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record["file_name"], record["file_path"],
                record["ra"], record["dec"], record["healpix"],
                record["min_ra"], record["max_ra"], record["min_dec"], record["max_dec"],
                record["fov_width"], record["fov_height"], record["pixel_scale"],
                record["hfd"], record["fwhm_px"], record["fwhm_arcsec"], record["stars"],
                record["airmass"], record["altitude"], record["azimuth"],
                record["exptime"], record["gain"], record["offset"], record["ccd_temp"],
                record["camera"], record["telescope"], record["filter"],
                int(record["plate_solved"]), record["wcs_source"],
                record["date_obs"], record["date_loc"],
                record["latitude"], record["longitude"], record["elevation"],
                json.dumps(record["polygon"]),
                record["moon_alt"], record["moon_az"],
                record["sun_alt"], record["sun_az"]
            ))

            conn.commit()

    except Exception as e:
        print("ERROR:", fits_path, e)

# ============================
# DB INIT
# ============================
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fits_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_path TEXT UNIQUE,
            ra REAL,
            dec REAL,
            healpix INTEGER,
            min_ra REAL,
            max_ra REAL,
            min_dec REAL,
            max_dec REAL,
            fov_width REAL,
            fov_height REAL,
            pixel_scale REAL,
            hfd REAL,
            fwhm_px REAL,
            fwhm_arcsec REAL,
            stars INTEGER,
            airmass REAL,
            altitude REAL,
            azimuth REAL,
            exptime REAL,
            gain REAL,
            offset REAL,
            ccd_temp REAL,
            camera TEXT,
            telescope TEXT,
            filter TEXT,
            plate_solved INTEGER,
            wcs_source TEXT,
            date_obs TEXT,
            date_loc TEXT,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            polygon TEXT,
            moon_alt REAL,
            moon_az REAL,
            sun_alt REAL,
            sun_az REAL
        )
    """)
    conn.commit()
    return conn, cur

# ============================
# ENTRY
# ============================
def generate_database(config_path, db_path):
    global cfg_global
    with open(config_path) as f:
        cfg_global = json.load(f)

    astap_path = cfg_global.get("astap_path")
    conn, cur = init_db(db_path)

    for scope in cfg_global.get("telescopes", []):
        fits_dir = scope.get("fits_dir")
        if not fits_dir or not os.path.exists(fits_dir):
            continue

        for date_folder in sorted(os.listdir(fits_dir)):
            date_path = os.path.join(fits_dir, date_folder)
            if not os.path.isdir(date_path):
                continue

            for name in sorted(os.listdir(date_path)):
                if name.lower().endswith((".fits", ".fit", ".fts")):
                    process_fits(astap_path, os.path.join(date_path, name), conn, cur)

    conn.close()
    print("DATABASE READY:", db_path)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
    CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
    DB_PATH = os.path.join(BASE_DIR, "data.sqlite")
    generate_database(CONFIG_PATH, DB_PATH)
