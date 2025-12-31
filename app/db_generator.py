import os
import sys
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from fits_db import FitsDatabase
from fits_analyzer import FitsAnalyzer
import psutil

# ============================
# Безопасная обработка одного файла
# ============================
def process_single_fits_safe(args):
    fits_path, astap_path, cfg_global, skip_db = args
    try:
        analyzer = FitsAnalyzer(astap_path, cfg_global)
        return analyzer.process_file(fits_path, skip_db)
    except Exception as e:
        print(f"\nERROR processing file:\n{fits_path}")
        print(str(e))
        traceback.print_exc()
        return None

# ============================
# Подбор количества потоков
# ============================
def suggest_max_workers(mem_per_worker_gb=1.5):
    available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
    max_by_mem = max(1, int(available_mem_gb // mem_per_worker_gb))
    max_by_cpu = os.cpu_count() or 2
    return min(max_by_mem, max_by_cpu, 8)  # ограничим разумно

# ============================
# Основная логика
# ============================
def generate_database(config_path, db_path, scan_dirs):
    start_time = time.perf_counter()

    # ----------------------------
    # Загружаем конфиг или создаем новый
    # ----------------------------
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating new empty config.json")
        cfg_global = {
            "astap_path": "C:\\Program Files\\astap\\astap_cli.exe",
            "telescopes": []
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_global, f, indent=4)
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_global = json.load(f)

    astap_path = cfg_global.get("astap_path")
    db = FitsDatabase(db_path)

    # ---------------------------------
    # Сбор FITS файлов
    # ---------------------------------
    fits_files = []
    for root in scan_dirs:
        for dp, _, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith((".fits", ".fit", ".fts")):
                    fits_files.append(os.path.join(dp, f))

    print(f"\nFound {len(fits_files)} FITS files")

    # ---------------------------------
    # Фильтрация
    # ---------------------------------
    files_to_process = []
    skip_db_flags = {}

    for f in fits_files:
        json_path = os.path.splitext(f)[0] + ".json"
        in_db = db.check_record_exists(os.path.abspath(f))
        in_json = os.path.exists(json_path)

        if in_db and in_json:
            continue

        files_to_process.append(f)
        skip_db_flags[f] = in_db

    print(f"To process: {len(files_to_process)} files")

    # ---------------------------------
    # Параллельная обработка
    # ---------------------------------
    max_workers = suggest_max_workers()
    print(f"Using {max_workers} threads")

    processed = 0
    inserted = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_fits_safe,
                (f, astap_path, cfg_global, skip_db_flags[f])
            ): f
            for f in files_to_process
        }

        for i, fut in enumerate(as_completed(futures), 1):
            fpath = futures[fut]
            result = fut.result()

            processed += 1

            if not result:
                print(f"[{i}/{len(files_to_process)}] Failed: {fpath}")
                continue

            record_id = db.insert_record(result)
            if record_id:
                inserted += 1
                print(f"[{i}/{len(files_to_process)}] Inserted: {result['file_name']}")
            else:
                print(f"[{i}/{len(files_to_process)}] DB insert failed")

    db.close()

    elapsed = time.perf_counter() - start_time

    print("\n==============================")
    print("DATABASE GENERATION DONE")
    print("==============================")
    print(f"Total FITS found : {len(fits_files)}")
    print(f"Processed        : {processed}")
    print(f"Inserted         : {inserted}")
    print(f"Time elapsed     : {elapsed:.2f} sec")
    if processed:
        print(f"Avg per file     : {elapsed / processed:.2f} sec")
    print("==============================\n")

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    # Определяем базовую директорию для exe или скрипта
    if getattr(sys, "frozen", False):
        BASE_DIR = os.path.dirname(sys.executable)
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
    DB_PATH = os.path.join(BASE_DIR, "data.sqlite")

    scan_dirs = sys.argv[1:]
    if not scan_dirs:
        print("No directories provided.")
        print("Usage:")
        print("  db_generator.exe D:\\FITS1 D:\\FITS2")
        sys.exit(1)

    generate_database(CONFIG_PATH, DB_PATH, scan_dirs)
