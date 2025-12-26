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
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from scipy.optimize import curve_fit

# ==================================================
# CONFIG
# ==================================================

warnings.filterwarnings("ignore", category=FITSFixedWarning)

# ==================================================
# FWHM
# ==================================================

def gaussian_2d(coords, amp, x0, y0, sx, sy, offset):
    x, y = coords
    return offset + amp * np.exp(
        -(((x - x0) ** 2) / (2 * sx ** 2) + ((y - y0) ** 2) / (2 * sy ** 2))
    ).ravel()


def measure_fwhm(data, max_stars=50):
    try:
        sigma_bkg = mad_std(data)
        finder = DAOStarFinder(fwhm=3.0, threshold=5 * sigma_bkg)
        stars = finder(data)
        if stars is None:
            return None

        fwhm_list = []
        for s in stars[:max_stars]:
            x, y = int(s["xcentroid"]), int(s["ycentroid"])
            r = 7
            cut = data[y-r:y+r+1, x-r:x+r+1]
            if cut.shape != (2*r+1, 2*r+1):
                continue

            yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
            p0 = (cut.max(), r, r, 2.0, 2.0, np.median(cut))

            try:
                popt, _ = curve_fit(gaussian_2d, (xx, yy), cut.ravel(), p0=p0, maxfev=2000)
                sx, sy = popt[3], popt[4]
                if sx > 0 and sy > 0:
                    fwhm_list.append(2.355 * np.sqrt(sx * sy))
            except:
                continue

        return np.median(fwhm_list) if fwhm_list else None
    except:
        return None


# ==================================================
# WCS helpers
# ==================================================

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
        return [[float(x), float(y)] for x, y in world], header
    except:
        return None, None


def compute_polygon_from_header(header, shape):
    try:
        wcs = WCS(header)
        h, w = shape
        pix = [(0, 0), (w, 0), (w, h), (0, h)]
        world = wcs.all_pix2world(pix, 0)
        return [[float(x), float(y)] for x, y in world]
    except:
        return None


def extract_wcs_fields(header):
    keys = [
        "CTYPE1", "CTYPE2",
        "CRVAL1", "CRVAL2",
        "CRPIX1", "CRPIX2",
        "CDELT1", "CDELT2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2"
    ]
    return {k: header.get(k) for k in keys if k in header}


# ==================================================
# ASTAP
# ==================================================

def solve_to_wcs(astap_path, fits_path):
    if not os.path.exists(astap_path) or not os.path.exists(fits_path):
        return False
    try:
        subprocess.run(
            [astap_path, "-f", fits_path, "-o", fits_path, "-update"],
            capture_output=True, text=True, check=True
        )
        return True
    except:
        return False


def run_astap_analysis(astap_path, fits_path):
    try:
        result = subprocess.run(
            [astap_path, "-f", fits_path, "-analyse"],
            capture_output=True, text=True, timeout=15
        )
        hfd = re.search(r"HFD_MEDIAN\s*=\s*([\d\.]+)", result.stdout)
        stars = re.search(r"STARS\s*=\s*(\d+)", result.stdout)
        return (
            float(hfd.group(1)) if hfd else None,
            int(stars.group(1)) if stars else None
        )
    except:
        return None, None


# ==================================================
# MAIN PROCESS
# ==================================================

def process_fits(astap_path, fits_path, conn, cur):
    json_path = fits_path + ".json"

    if os.path.exists(json_path):
        print("SKIP:", fits_path)
        return

    print("PROCESS:", fits_path)

    try:
        with fits.open(fits_path, ignore_missing_end=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header

            if data is None:
                return

            if data.ndim == 3:
                data = data[0] if data.shape[0] <= 4 else data[:, :, 0]

            data = data.astype(float)

            # RA / DEC
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

            if polygon is None and astap_path:
                if solve_to_wcs(astap_path, fits_path):
                    polygon, wcs_header = read_polygon_from_wcs_file(wcs_path, data.shape)
                    if polygon:
                        wcs_source = "ASTAP"
                        plate_solved = True

            if polygon is None:
                polygon = compute_polygon_from_header(header, data.shape)
                wcs_header = header
                wcs_source = "FITS"

            hfd, stars_count = run_astap_analysis(astap_path, fits_path)

            fwhm_px = measure_fwhm(data)
            fwhm_arcsec = None
            try:
                if fwhm_px:
                    w = WCS(header)
                    px = np.sqrt((w.pixel_scale_matrix ** 2).sum(axis=0))
                    fwhm_arcsec = float(fwhm_px * px.mean() * 3600)
            except:
                pass

            wcs_fields = extract_wcs_fields(wcs_header) if wcs_header else {}

            record = {
                "file_name": os.path.basename(fits_path),
                "ra": ra,
                "dec": dec,
                "polygon": polygon,
                "hfd": hfd,
                "stars": stars_count,
                "fwhm_px": fwhm_px,
                "fwhm_arcsec": fwhm_arcsec,
                "wcs_fields": wcs_fields,
                "wcs_source": wcs_source,
                "plate_solved": plate_solved,
                "exptime": header.get("EXPTIME"),
                "date_obs": header.get("DATE-OBS"),
                "date_loc": header.get("DATE-LOC"),
                "instrument": header.get("INSTRUME"),
                "camera": header.get("CAMERAID"),
                "telescope": header.get("TELESCOP"),
                "ccd_temp": header.get("CCD-TEMP"),
                "gain": header.get("GAIN"),
                "offset": header.get("OFFSET"),
                "header": {k: str(v) for k, v in header.items()}
            }

            # JSON
            with open(json_path, "w") as f:
                json.dump(record, f, indent=2)

            # DB
            cur.execute("""
                INSERT OR REPLACE INTO fits_data (
                    file_name, ra, dec, polygon,
                    hfd, stars,
                    fwhm_px, fwhm_arcsec,
                    wcs_fields, wcs_source, plate_solved,
                    exptime, date_obs, date_loc,
                    instrument, camera, telescope,
                    ccd_temp, gain, offset,
                    header
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record["file_name"],
                record["ra"], record["dec"],
                json.dumps(record["polygon"]),
                record["hfd"], record["stars"],
                record["fwhm_px"], record["fwhm_arcsec"],
                json.dumps(record["wcs_fields"]),
                record["wcs_source"], int(record["plate_solved"]),
                record["exptime"], record["date_obs"], record["date_loc"],
                record["instrument"], record["camera"], record["telescope"],
                record["ccd_temp"], record["gain"], record["offset"],
                json.dumps(record["header"])
            ))

            conn.commit()
            print("OK:", fits_path)

    except Exception as e:
        print("ERROR:", fits_path, e)


# ==================================================
# DATABASE
# ==================================================

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    # conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fits_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT UNIQUE,
            ra REAL,
            dec REAL,
            polygon TEXT,
            hfd REAL,
            stars INTEGER,
            fwhm_px REAL,
            fwhm_arcsec REAL,
            wcs_fields TEXT,
            wcs_source TEXT,
            plate_solved INTEGER,
            exptime REAL,
            date_obs TEXT,
            date_loc TEXT,
            instrument TEXT,
            camera TEXT,
            telescope TEXT,
            ccd_temp REAL,
            gain REAL,
            offset REAL,
            header TEXT
        )
    """)
    conn.commit()
    return conn, cur


# ==================================================
# ENTRY
# ==================================================

def generate_database(config_path, db_path):
    with open(config_path) as f:
        cfg = json.load(f)

    astap_path = cfg.get("astap_path")
    conn, cur = init_db(db_path)

    for scope in cfg.get("telescopes", []):
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
