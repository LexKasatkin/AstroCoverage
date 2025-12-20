import os
import sys
import json
import subprocess
import math
from astropy.io import fits
from astropy.wcs import WCS

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_JSON_PATH = os.path.join(BASE_DIR, "data.json")

# -------------------- ASTAP solver --------------------

def solve_with_astap(astap_path: str, fits_path: str, timeout_sec: int = 10):
    if not os.path.exists(astap_path) or not os.path.exists(fits_path):
        return None
    try:
        subprocess.run([astap_path, "-f", fits_path, "-o", fits_path],
                       capture_output=True, text=True, timeout=timeout_sec, check=True)
    except Exception as e:
        print("ASTAP error:", e)
        return None

    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            wcs = WCS(hdr)
            h, w = hdul[0].data.shape
            pix = [(0,0),(w,0),(w,h),(0,h)]
            world = wcs.all_pix2world(pix, 0)
            polygon = [[float(ra), float(dec)] for ra, dec in world]
            for ra, dec in polygon:
                if not (-360 <= ra <= 360 and -90 <= dec <= 90):
                    return None
            return polygon
    except Exception as e:
        print("Error reading WCS from ASTAP FITS:", e)
        return None

# -------------------- helpers --------------------

def read_ra_dec(header):
    for ra_key, dec_key in [("RA","DEC"),("OBJCTRA","OBJCTDEC"),("CRVAL1","CRVAL2")]:
        if ra_key in header and dec_key in header:
            try:
                return float(header[ra_key]), float(header[dec_key])
            except:
                pass
    return None, None

def polygon_from_center_fov(ra_center, dec_center, fov_w, fov_h, rotation=0.0):
    """Возвращает 4 вершины кадра с учетом поворота"""
    dx = fov_w/2
    dy = fov_h/2
    theta = math.radians(rotation)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    corners = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    polygon = []
    for x, y in corners:
        xr = x * cos_t - y * sin_t
        yr = x * sin_t + y * cos_t
        ra = ra_center + xr / math.cos(math.radians(dec_center))
        dec = dec_center + yr
        polygon.append([ra, dec])
    return polygon

def fov_from_nina(header):
    """Вычисляем FOV для NINA из размеров кадра и пикселя"""
    try:
        naxis1 = float(header.get("NAXIS1"))
        naxis2 = float(header.get("NAXIS2"))
        xpix = float(header.get("XPIXSZ")) / 1000  # um -> mm
        ypix = float(header.get("YPIXSZ")) / 1000
        foc_len = float(header.get("FOCALLEN"))  # мм
        fov_w = math.degrees(naxis1 * xpix / foc_len)
        fov_h = math.degrees(naxis2 * ypix / foc_len)
        rotation = float(header.get("OBJCTROT", 0.0))
        return fov_w, fov_h, rotation
    except:
        return None, None, 0.0

# -------------------- main generator --------------------

def generate():
    if not os.path.exists(CONFIG_PATH):
        print("Config not found:", CONFIG_PATH)
        return

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    astap_path = cfg.get("astap_path")
    result = {}

    for scope in cfg.get("telescopes", []):
        sid = scope["id"]
        fits_dir = scope["fits_dir"]

        if not os.path.exists(fits_dir):
            continue

        for date_folder in sorted(os.listdir(fits_dir)):
            date_path = os.path.join(fits_dir, date_folder)
            if not os.path.isdir(date_path):
                continue

            for name in sorted(os.listdir(date_path)):
                if not name.lower().endswith((".fits", ".fit", ".fts")):
                    continue

                path = os.path.join(date_path, name)
                polygon = None

                try:
                    with fits.open(path, memmap=False) as hdul:
                        hdu = hdul[0]
                        header = hdu.header
                        ra, dec = read_ra_dec(header)

                        # 1. Если есть WCS
                        if "CTYPE1" in header and "CTYPE2" in header:
                            wcs = WCS(header)
                            h, w = hdu.data.shape
                            pix = [(0,0),(w,0),(w,h),(0,h)]
                            world = wcs.all_pix2world(pix,0)
                            polygon = [[float(ra), float(dec)] for ra, dec in world]

                        # 2. Если RA/DEC + FOV в заголовке
                        elif ra is not None and dec is not None and ("FOV_W" in header and "FOV_H" in header):
                            fov_w = float(header["FOV_W"])
                            fov_h = float(header["FOV_H"])
                            rotation = float(header.get("CROTA2", 0.0))
                            polygon = polygon_from_center_fov(ra, dec, fov_w, fov_h, rotation)

                        # 3. NINA: вычисляем FOV из размера пикселя и фокального расстояния
                        elif ra is not None and dec is not None:
                            fov_w, fov_h, rotation = fov_from_nina(header)
                            if fov_w is not None:
                                polygon = polygon_from_center_fov(ra, dec, fov_w, fov_h, rotation)

                        # 4. Если ничего не сработало — пробуем ASTAP
                        elif astap_path:
                            polygon = solve_with_astap(astap_path, path, timeout_sec=10)

                except Exception as e:
                    print("FITS error:", path, e)
                    continue

                if not polygon:
                    print("Skipping:", path)
                    continue

                result.setdefault(sid, {}).setdefault(date_folder, []).append({
                    "file": name,
                    "ra": float(ra) if ra else None,
                    "dec": float(dec) if dec else None,
                    "polygon": polygon
                })

    os.makedirs(os.path.dirname(DATA_JSON_PATH), exist_ok=True)
    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Data saved to", DATA_JSON_PATH)

# -------------------- entry --------------------

if __name__ == "__main__":
    generate()
