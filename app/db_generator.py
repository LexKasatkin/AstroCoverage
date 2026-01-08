import os
import sys
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import requests

from fits_analyzer import FitsAnalyzer

session = requests.Session()

# ------------------ Пакетная проверка ------------------
def check_skip_db_batch(api_host, api_port, files):
    try:
        url = f"http://{api_host}:{api_port}/api/skip_db_batch"
        r = session.post(url, json={"files": files}, timeout=60)
        if r.status_code == 200:
            return r.json()  # словарь file_path -> True/False
    except Exception as e:
        print(f"[SKIP_DB_BATCH ERROR] {e}")
    return {f: False for f in files}

# ------------------ Обработка одного FITS ------------------
def process_single_fits_safe(args):
    fits_path, astap_path, cfg_global = args
    try:
        analyzer = FitsAnalyzer(astap_path, cfg_global)
        return analyzer.process_file(fits_path)
    except Exception as e:
        print(f"\nERROR processing:\n{fits_path}")
        print(e)
        traceback.print_exc()
        return None

# ------------------ Отправка результата ------------------
def send_to_server(api_url, data):
    try:
        r = session.post(api_url, json=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"[SEND ERROR] {data.get('file_name')}: {e}")
        return False

# ------------------ Подбор числа потоков ------------------
def suggest_max_workers(mem_per_worker_gb=1.5):
    mem = psutil.virtual_memory().available / (1024 ** 3)
    return min(os.cpu_count() or 2, max(1, int(mem // mem_per_worker_gb)), 8)

# ------------------ Главная функция ------------------
def generate_database(api_host, api_port, scan_dirs):
    start = time.perf_counter()

    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    astap_path = cfg["astap_path"]
    api_url = f"http://{api_host}:{api_port}/api/insert"

    # ------------------ Собираем все FITS-файлы ------------------
    fits_files = []
    for root in scan_dirs:
        for dp, _, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith((".fits", ".fit", ".fts")):
                    fits_files.append(os.path.join(dp, f))

    # ------------------ Пакетная проверка ------------------
    print("Checking which files already exist in DB...")
    files_status = check_skip_db_batch(api_host, api_port, fits_files)
    new_fits = [f for f in fits_files if not files_status.get(f, False)]
    skipped_files = [f for f in fits_files if files_status.get(f, False)]

    print(f"Found {len(new_fits)} new files to process")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files already in DB:")
        for f in skipped_files:
            print("  ", f)

    # ------------------ Потоки для обработки ------------------
    workers = suggest_max_workers()
    print(f"Using {workers} threads")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_single_fits_safe, (f, astap_path, cfg)): f
            for f in new_fits
        }

        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result and send_to_server(api_url, result):
                print(f"[{i}/{len(futures)}] Sent: {result['file_name']}")

    print(f"Done in {time.perf_counter() - start:.1f}s")


# ------------------ Точка входа ------------------
if __name__ == "__main__":
    scan_dirs = sys.argv[1:]
    if not scan_dirs:
        print("Usage: db_generator.exe D:\\FITS1 D:\\FITS2")
        sys.exit(1)

    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    generate_database(cfg["api"]["host"], cfg["api"]["port"], scan_dirs)
