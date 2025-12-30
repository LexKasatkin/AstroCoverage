import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fits_db import FitsDatabase
from fits_analyzer import FitsAnalyzer
import psutil
import argparse

def process_single_fits(args):
    fits_path, astap_path, cfg_global, skip_db = args
    analyzer = FitsAnalyzer(astap_path, cfg_global)
    return analyzer.process_file(fits_path, skip_db)

def suggest_max_workers(mem_per_proc_gb=2):
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    max_by_mem = max(1, int(available_mem_gb // mem_per_proc_gb))
    max_by_cpu = os.cpu_count() or 2
    return min(max_by_mem, max_by_cpu)

def generate_database(config_path, db_path, scan_dirs=None):
    start_time = time.perf_counter()
    with open(config_path) as f:
        cfg_global = json.load(f)
    astap_path = cfg_global.get("astap_path")
    db = FitsDatabase(db_path)

    fits_files = []
    if scan_dirs:
        for fits_dir in scan_dirs:
            if not os.path.exists(fits_dir):
                print(f"Directory does not exist: {fits_dir}")
                continue
            for dp, dn, fn in os.walk(fits_dir):
                for f in fn:
                    if f.lower().endswith((".fits", ".fit", ".fts")):
                        fits_files.append(os.path.join(dp, f))
    else:
        for scope in cfg_global.get("telescopes", []):
            fits_dir = scope.get("fits_dir")
            if not fits_dir or not os.path.exists(fits_dir):
                continue
            for dp, dn, fn in os.walk(fits_dir):
                for f in fn:
                    if f.lower().endswith((".fits", ".fit", ".fts")):
                        fits_files.append(os.path.join(dp, f))

    print(f"Found {len(fits_files)} FITS files")

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

    total = len(files_to_process)
    print(f"Processing {total} files...")
    processed = 0
    inserted = 0
    max_workers = suggest_max_workers()
    print(f"Using max_workers={max_workers} for parallel processing")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_fits, (f, astap_path, cfg_global, skip_db_flags[f])): f
                   for f in files_to_process}

        for i, fut in enumerate(as_completed(futures), 1):
            fpath = futures[fut]
            try:
                record = fut.result()
            except Exception as e:
                print(f"ERROR processing {fpath}: {e}")
                continue
            processed += 1
            if not record:
                continue
            record_id = db.insert_record(record)
            if record_id:
                inserted += 1
                print(f"[{i}/{total}] Inserted: {record['file_name']} (ID={record_id})")
            else:
                print(f"[{i}/{total}] WARNING: Failed to insert {record['file_name']}")

    db.close()
    elapsed = time.perf_counter() - start_time
    print("\n==============================")
    print("DATABASE GENERATION FINISHED")
    print("==============================")
    print(f"Total FITS found     : {len(fits_files)}")
    print(f"Processed            : {processed}")
    print(f"Inserted into DB     : {inserted}")
    print(f"Elapsed time         : {elapsed:.2f} sec")
    if processed > 0:
        print(f"Average per file     : {elapsed / processed:.2f} sec")
    print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FITS database")
    parser.add_argument(
        "dirs",
        nargs="*",
        help="Directories to scan for FITS files (overrides config.json telescopes)"
    )
    args = parser.parse_args()

    # Папка, где лежит exe или скрипт
    BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
    CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
    DB_PATH = os.path.join(BASE_DIR, "data.sqlite")

    generate_database(CONFIG_PATH, DB_PATH, scan_dirs=args.dirs)
