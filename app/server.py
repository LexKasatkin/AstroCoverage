import os
import sys
import json
import threading
from flask import Flask, send_from_directory, request, jsonify
from coverage_generator import generate
from watcher import start_watch

# Базовая директория
BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
WEB_DIR = os.path.join(BASE_DIR, "coverage-web")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_JSON_PATH = os.path.join(BASE_DIR, "data.json")

app = Flask(__name__)

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
# STATIC FILES
# ======================

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(WEB_DIR, path)

@app.route("/data.json")
def data_json():
    return send_from_directory(BASE_DIR, "data.json")

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
    tid = d["name"].lower().replace(" ", "_")
    cfg["telescopes"].append({
        "id": tid,
        "name": d["name"],
        "fits_dir": d["path"]
    })
    save_config(cfg)
    generate()
    return jsonify({"status": "ok"})

@app.route("/config/telescope/<tid>", methods=["PUT"])
def update_telescope(tid):
    cfg = load_config()
    d = request.json
    for t in cfg["telescopes"]:
        if t["id"] == tid:
            t["name"] = d["name"]
            t["fits_dir"] = d["path"]
            break
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
# DATA.JSON REGEN
# ======================

@app.route("/regen_data", methods=["POST"])
def regen_data():
    try:
        if os.path.exists(DATA_JSON_PATH):
            with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
                f.write("{}")
        generate()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ======================
# RUN SERVER
# ======================

def run_server():
    threading.Thread(target=start_watch, args=(lambda: print("coverage updated"),), daemon=True).start()
    app.run(port=8000)

if __name__ == "__main__":
    run_server()
