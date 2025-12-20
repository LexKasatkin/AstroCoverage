import os
import sys
import json
from astropy.io import fits
from astropy.wcs import WCS

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_JSON_PATH = os.path.join(BASE_DIR, "data.json")

def generate():
    if not os.path.exists(CONFIG_PATH):
        print("Config not found:", CONFIG_PATH)
        return

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    result = {}

    for scope in cfg.get("telescopes", []):
        sid = scope["id"]
        fits_dir = scope["fits_dir"]

        if not os.path.exists(fits_dir):
            print("FITS dir not found:", fits_dir)
            continue

        for date_folder in sorted(os.listdir(fits_dir)):
            date_path = os.path.join(fits_dir, date_folder)
            if not os.path.isdir(date_path):
                continue

            for name in sorted(os.listdir(date_path)):
                if not name.lower().endswith((".fits", ".fit", ".fts")):
                    continue

                path = os.path.join(date_path, name)
                try:
                    with fits.open(path, memmap=False) as hdul:
                        hdu = hdul[0]
                        if "CTYPE1" not in hdu.header:
                            continue

                        wcs = WCS(hdu.header)
                        h, w = hdu.data.shape

                        corners = [
                            wcs.pixel_to_world(0, 0),
                            wcs.pixel_to_world(w, 0),
                            wcs.pixel_to_world(w, h),
                            wcs.pixel_to_world(0, h)
                        ]

                        polygon = [[c.ra.deg, c.dec.deg] for c in corners]

                        date = date_folder
                        result.setdefault(sid, {}).setdefault(date, []).append({
                            "file": name,
                            "polygon": polygon
                        })

                except Exception as e:
                    print("FITS error:", path, e)

    os.makedirs(os.path.dirname(DATA_JSON_PATH), exist_ok=True)
    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Data saved to", DATA_JSON_PATH)

if __name__ == "__main__":
    generate()
