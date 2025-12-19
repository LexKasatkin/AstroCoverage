import os
import sys
import time
import threading
import json           # <- вот этого не хватало
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from coverage_generator import generate


BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.abspath(".")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

class FitsHandler(FileSystemEventHandler):
    def __init__(self, cb):
        self.lock = False
        self.cb = cb

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith((".fits", ".fit", ".fts")):
            return
        if self.lock:
            return
        self.lock = True
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        time.sleep(2)  # ждём, пока файл допишется
        print("New FITS detected → regenerating coverage")
        generate()
        if self.cb:
            self.cb()
        self.lock = False

def load_telescopes():
    if not os.path.exists(CONFIG_PATH):
        return []
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("telescopes", [])

def start_watch(cb=None):
    handler = FitsHandler(cb)
    observer = Observer()
    for t in load_telescopes():
        path = t.get("fits_dir")
        if path and os.path.exists(path):
            observer.schedule(handler, path, recursive=True)
            print("Watching:", path)
        else:
            print("Invalid path:", path)
    observer.start()
    observer.join()
