import os, sys, json
from astropy.io import fits
from astropy.wcs import WCS

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_JSON_PATH = os.path.join(BASE_DIR, "data.json")

def generate():
    if not os.path.exists(CONFIG_PATH):
        return

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    result = {}

    for scope in cfg.get("telescopes", []):
        sid = scope["id"]
        fits_dir = scope["fits_dir"]

        if not os.path.exists(fits_dir):
            continue

        for root, _, files in os.walk(fits_dir):
            for name in files:
                if not name.lower().endswith((".fits", ".fit", ".fts")):
                    continue

                path = os.path.join(root, name)
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

                        date_obs = hdu.header.get("DATE-OBS", "")
                        date = date_obs[:10] if date_obs else "unknown"
                        time = date_obs[11:19] if len(date_obs) >= 19 else ""

                        result.setdefault(sid, {}).setdefault(date, []).append({
                            "file": name,
                            "datetime": f"{date} {time}".strip(),
                            "polygon": polygon
                        })

                except Exception as e:
                    print("FITS error:", path, e)

    os.makedirs(os.path.dirname(DATA_JSON_PATH), exist_ok=True)
    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
