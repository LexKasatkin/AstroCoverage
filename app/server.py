import os
import json
from flask import Flask, request, jsonify
from fits_db import FitsDatabase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data.sqlite")

app = Flask(__name__)

# ------------------ Проверка одного файла ------------------
@app.route("/api/skip_db")
def skip_db():
    file_path = request.args.get("file_path")
    if not file_path:
        return jsonify({"skip_db": False})

    file_path = os.path.normcase(os.path.abspath(file_path))
    db = FitsDatabase(DB_PATH)
    try:
        exists = db.check_record_exists(file_path)
    finally:
        db.close()

    return jsonify({"skip_db": exists})

# ------------------ Пакетная проверка ------------------
@app.route("/api/skip_db_batch", methods=["POST"])
def skip_db_batch():
    files = request.json.get("files", [])
    result = {}
    db = FitsDatabase(DB_PATH)
    try:
        for f in files:
            path = os.path.normcase(os.path.abspath(f))
            result[f] = db.check_record_exists(path)
    finally:
        db.close()
    return jsonify(result)

# ------------------ Вставка новой записи ------------------
@app.route("/api/insert", methods=["POST"])
def insert_record():
    data = request.json
    if not data:
        return jsonify({"error": "empty payload"}), 400

    # Нормализуем путь перед вставкой
    data["file_path"] = os.path.normcase(os.path.abspath(data["file_path"]))

    db = FitsDatabase(DB_PATH)
    try:
        db.insert_many([data])
    except Exception as e:
        print(f"[DB INSERT ERROR] {data.get('file_path')}: {e}")
        return jsonify({"error": "insert failed"}), 500
    finally:
        db.close()

    return jsonify({"status": "inserted"})

# ------------------ Запуск ------------------
if __name__ == "__main__":
    app.run(port=8000, threaded=True)
