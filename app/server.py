import os
import sys
import json
import threading
import sqlite3
from flask import Flask, send_from_directory, request, jsonify

from coverage_generator import generate, generation_status
from watcher import start_watch

# ======================
# PATHS
# ======================

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
WEB_DIR = os.path.join(BASE_DIR, "coverage-web")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DB_PATH = os.path.join(BASE_DIR, "data.sqlite")  # <-- общая БД с fits_analyzer

app = Flask(__name__)

# ======================
# DATABASE
# ======================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ======================
# CONFIG
# ======================

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"telescopes": [], "astap_path": ""}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ======================
# UTILS
# ======================

def parse_bool(v):
    if v is None:
        return None
    return v.lower() in ("1", "true", "yes", "y")


# ======================
# STATIC FILES
# ======================

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(WEB_DIR, path)


# ======================
# DATA API
# ======================

@app.route("/data.json")
def data_json():
    try:
        conn = get_db()
        cur = conn.cursor()

        args = request.args
        where = []
        params = []

        if "plate_solved" in args:
            v = parse_bool(args.get("plate_solved"))
            if v is not None:
                where.append("plate_solved = ?")
                params.append(1 if v else 0)

        if "has_wcs" in args:
            v = parse_bool(args.get("has_wcs"))
            where.append("wcs_fields IS NOT NULL" if v else "wcs_fields IS NULL")

        if "telescope" in args:
            where.append("telescope = ?")
            params.append(args["telescope"])

        if "camera" in args:
            where.append("camera = ?")
            params.append(args["camera"])

        if "min_hfd" in args:
            where.append("hfd >= ?")
            params.append(float(args["min_hfd"]))

        if "max_hfd" in args:
            where.append("hfd <= ?")
            params.append(float(args["max_hfd"]))

        if "date_from" in args:
            where.append("date_obs >= ?")
            params.append(args["date_from"])

        if "date_to" in args:
            where.append("date_obs <= ?")
            params.append(args["date_to"])

        sql = "SELECT * FROM fits_data"
        if where:
            sql += " WHERE " + " AND ".join(where)

        limit = int(args.get("limit", 1000))
        offset = int(args.get("offset", 0))
        sql += f" LIMIT {limit} OFFSET {offset}"

        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        result = {}

        for r in rows:
            d = dict(r)

            # распарсить json поля
            for key in ("polygon", "wcs_fields", "header"):
                if d.get(key):
                    try:
                        d[key] = json.loads(d[key])
                    except:
                        pass

            telescope = d.get("telescope") or "unknown"
            date_raw = d.get("date_obs")
            if not date_raw:
                continue

            # приводим к YYYY-MM-DD
            date = date_raw[:10]

            if not date:
                continue

            if telescope not in result:
                result[telescope] = {}

            if date not in result[telescope]:
                result[telescope][date] = []

            result[telescope][date].append(d)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================
# CONFIG API
# ======================

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify(load_config())


@app.route("/config/astap", methods=["POST"])
def set_astap():
    cfg = load_config()
    cfg["astap_path"] = request.json.get("path", "")
    save_config(cfg)
    return jsonify({"status": "ok"})


@app.route("/config/telescope", methods=["POST"])
def add_telescope():
    cfg = load_config()
    d = request.json
    cfg["telescopes"].append({
        "id": d["name"].lower().replace(" ", "_"),
        "name": d["name"],
        "fits_dir": d["path"]
    })
    save_config(cfg)
    generate()
    return jsonify({"status": "ok"})


@app.route("/config/telescope/<tid>", methods=["PUT"])
def update_telescope(tid):
    cfg = load_config()
    for t in cfg["telescopes"]:
        if t["id"] == tid:
            t["name"] = request.json["name"]
            t["fits_dir"] = request.json["path"]
    save_config(cfg)
    generate()
    return jsonify({"status": "ok"})


@app.route("/config/telescope/<tid>", methods=["DELETE"])
def delete_telescope(tid):
    cfg = load_config()
    cfg["telescopes"] = [t for t in cfg["telescopes"] if t["id"] != tid]
    save_config(cfg)
    generate()
    return jsonify({"status": "ok"})


# ======================
# REGEN
# ======================

@app.route("/regen_data", methods=["POST"])
def regen_data():
    try:
        generate()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/generation_status")
def generation_status_api():
    return jsonify(generation_status)


# ======================
# RUN
# ======================

def run_server():
    threading.Thread(
        target=start_watch,
        args=(lambda: print("coverage updated"),),
        daemon=True
    ).start()

    app.run(port=8000)


if __name__ == "__main__":
    run_server()

