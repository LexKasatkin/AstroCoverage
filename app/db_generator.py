import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from fits_db import FitsDatabase
from fits_analyzer import FitsAnalyzer

# ============================
# Обёртка для ProcessPoolExecutor
# ============================
def process_single_fits(args):
    fits_path, astap_path, cfg_global, skip_db = args
    analyzer = FitsAnalyzer(astap_path, cfg_global)
    return analyzer.process_file(fits_path, skip_db)

# ============================
# Генерация БД из FITS файлов
# ============================
def generate_database(config_path, db_path):
    # Загрузка конфигурации
    with open(config_path) as f:
        cfg_global = json.load(f)

    astap_path = cfg_global.get("astap_path")
    db = FitsDatabase(db_path)

    # Сбор всех FITS файлов
    fits_files = []
    for scope in cfg_global.get("telescopes", []):
        fits_dir = scope.get("fits_dir")
        if not fits_dir or not os.path.exists(fits_dir):
            continue
        for dp, dn, fn in os.walk(fits_dir):
            for f in fn:
                if f.lower().endswith((".fits", ".fit", ".fts")):
                    fits_files.append(os.path.join(dp, f))

    # Определяем, какие файлы нужно обработать
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

    # Параллельная обработка
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_fits, (f, astap_path, cfg_global, skip_db_flags[f])): f
            for f in files_to_process
        }

        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                record = fut.result()
            except Exception as e:
                print(f"ERROR processing {futures[fut]}:", e)
                continue
            if not record:
                continue

            record_id = db.insert_record(record)
            if record_id:
                print(f"[{i}/{total}] Record inserted: {record['file_name']} (ID={record_id})")
            else:
                print(f"[{i}/{total}] WARNING: Failed to insert record: {record['file_name']}")

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
