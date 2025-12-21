import os
import sys
import json
import subprocess
from astropy.io import fits
from astropy.wcs import WCS

# -------------------- paths --------------------

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_JSON_PATH = os.path.join(BASE_DIR, "data.json")

# -------------------- helpers --------------------

def read_ra_dec(header):
    """Попытка получить RA/DEC из header"""
    for ra_key, dec_key in [
        ("RA", "DEC"),
        ("OBJCTRA", "OBJCTDEC"),
        ("CRVAL1", "CRVAL2"),
    ]:
        if ra_key in header and dec_key in header:
            try:
                return float(header[ra_key]), float(header[dec_key])
            except Exception:
                pass
    return None, None


def solve_to_wcs(
    astap_path: str,
    input_fits: str,
    output_fits: str,
    timeout_sec: int = 10,
) -> bool:
    """
    Запускает ASTAP:
    - пишет solved FITS
    - рядом создаёт solved.wcs.wcs
    """
    if not os.path.exists(astap_path) or not os.path.exists(input_fits):
        return False

    os.makedirs(os.path.dirname(output_fits), exist_ok=True)

    cmd = [
        astap_path,
        "-f", input_fits,
        "-o", output_fits,
    ]

    print("Running ASTAP:", " ".join(cmd))

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
        return True

    except subprocess.CalledProcessError as e:
        print("ASTAP error:", e.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("ASTAP timeout")
        return False


def read_polygon_from_wcs_file(wcs_path: str, image_shape):
    """
    Читает polygon из ASTAP *.wcs.wcs
    """
    if not os.path.exists(wcs_path):
        return None

    try:
        with open(wcs_path, encoding="utf-8") as f:
            text = f.read()

        header = fits.Header.fromstring(text, sep="\n")

        if "CTYPE1" not in header or "CTYPE2" not in header:
            return None

        wcs = WCS(header)

        h, w = image_shape
        pix = [
            (0, 0),
            (w, 0),
            (w, h),
            (0, h),
        ]

        world = wcs.all_pix2world(pix, 0)
        return [[float(ra), float(dec)] for ra, dec in world]

    except Exception as e:
        print("WCS read error:", e)
        return None

# -------------------- main --------------------

def generate():
    global generation_status

    generation_status.update({
        "running": True,
        "stage": "init",
        "current": "",
        "done": 0,
        "total": 0,
        "error": None,
    })

    try:
        if not os.path.exists(CONFIG_PATH):
            raise RuntimeError("Config not found")

        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)

        astap_path = cfg.get("astap_path")
        result = {}

        # ===== STAGE 1: count total files =====
        generation_status["stage"] = "counting"

        total = 0
        for scope in cfg.get("telescopes", []):
            fits_dir = scope["fits_dir"]
            if not os.path.exists(fits_dir):
                continue

            for date_folder in os.listdir(fits_dir):
                date_path = os.path.join(fits_dir, date_folder)
                if not os.path.isdir(date_path):
                    continue

                for name in os.listdir(date_path):
                    if name.lower().endswith((".fits", ".fit", ".fts")):
                        total += 1

        generation_status["total"] = total

        # ===== STAGE 2: processing =====
        generation_status["stage"] = "processing"

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

                    generation_status["current"] = (
                        f"{scope['name']} / {date_folder} / {name}"
                    )

                    input_fits = os.path.join(date_path, name)
                    solved_fits = input_fits + ".solved.fits"
                    wcs_file = input_fits + ".solved.wcs"

                    ra = dec = None
                    polygon = None

                    try:
                        with fits.open(input_fits) as hdul:
                            if hdul[0].data is None:
                                raise RuntimeError("FITS has no image data")

                            data = hdul[0].data
                            header = hdul[0].header
                            ra, dec = read_ra_dec(header)

                            if "CTYPE1" in header and "CTYPE2" in header:
                                wcs = WCS(header)
                                h, w = data.shape
                                pix = [(0, 0), (w, 0), (w, h), (0, h)]
                                world = wcs.all_pix2world(pix, 0)
                                polygon = [[float(r), float(d)] for r, d in world]
                            else:
                                if not os.path.exists(wcs_file):
                                    ok = solve_to_wcs(
                                        astap_path,
                                        input_fits,
                                        solved_fits,
                                    )
                                    if not ok:
                                        raise RuntimeError("ASTAP failed")

                                polygon = read_polygon_from_wcs_file(
                                    wcs_file,
                                    data.shape
                                )

                    except Exception as e:
                        polygon = {"status": "failed", "reason": str(e)}

                    result.setdefault(sid, {}).setdefault(date_folder, []).append({
                        "file": name,
                        "ra": float(ra) if ra is not None else None,
                        "dec": float(dec) if dec is not None else None,
                        "polygon": polygon,
                    })

                    generation_status["done"] += 1

        # ===== SAVE RESULT =====
        generation_status["stage"] = "saving"

        with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        generation_status["error"] = str(e)

    finally:
        generation_status["running"] = False
        generation_status["stage"] = "done"


# ===== GENERATION STATUS =====

generation_status = {
    "running": False,
    "stage": "",
    "current": "",
    "done": 0,
    "total": 0,
    "error": None,
}

if __name__ == "__main__":
    generate()
