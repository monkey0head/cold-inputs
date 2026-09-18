"""Set up raw data for the Beauty2014, Sports2014, and Toys2014 datasets.

For each dataset:
  1. Downloads the Amazon 2014 ratings CSV from SNAP and adds the header
     expected by the preprocessing step.
  2. Downloads the Amazon 2014 item metadata from SNAP, parses it, and saves
     it as a CSV.

Usage (from the project root, with SEQ_REC_DATA_PATH set):
    python runs/get_data/setup_amazon2014.py

The script is idempotent: it skips any file that already exists.
"""

import ast
import gzip
import json
import logging
import os

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEQ_REC_DATA_PATH = os.environ["SEQ_REC_DATA_PATH"]

SNAP_BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"

DATASETS = {
    "Beauty2014": {
        "ratings_url": f"{SNAP_BASE_URL}/ratings_Beauty.csv",
        "meta_url": f"{SNAP_BASE_URL}/meta_Beauty.json.gz",
    },
    "Sports2014": {
        "ratings_url": f"{SNAP_BASE_URL}/ratings_Sports_and_Outdoors.csv",
        "meta_url": f"{SNAP_BASE_URL}/meta_Sports_and_Outdoors.json.gz",
    },
    "Toys2014": {
        "ratings_url": f"{SNAP_BASE_URL}/ratings_Toys_and_Games.csv",
        "meta_url": f"{SNAP_BASE_URL}/meta_Toys_and_Games.json.gz",
    },
}

META_COLUMNS = ["item_id", "title", "description", "categories", "brand", "price"]
# SNAP ratings CSVs have no header; this is their column order.
RATINGS_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def _parse_meta_line(line: str) -> dict | None:
    """Parse one line from the Amazon 2014 metadata JSON-gz file.

    The file uses Python-literal syntax (not valid JSON) for some lines, so we
    try json.loads first, then ast.literal_eval as fallback.
    """
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(line)
    except Exception:
        return None


def download_file(url: str, dest: str) -> None:
    log.info("Downloading %s -> %s", url, dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    log.info("Download complete: %s", dest)


def build_metadata_csv(gz_path: str, out_csv: str) -> None:
    log.info("Parsing metadata from %s", gz_path)
    records = []
    skipped = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = _parse_meta_line(line)
            if rec is None:
                skipped += 1
                continue
            desc = rec.get("description", "")
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc)
            records.append(
                {
                    "item_id": rec.get("asin", ""),
                    "title": rec.get("title", ""),
                    "description": desc,
                    "categories": str(rec.get("categories", "")),
                    "brand": rec.get("brand", ""),
                    "price": rec.get("price", ""),
                }
            )
    log.info("Parsed %d metadata records (%d lines skipped)", len(records), skipped)
    df = pd.DataFrame(records, columns=META_COLUMNS)
    df.to_csv(out_csv, index=False)
    log.info("Saved metadata CSV: %s  (%d rows)", out_csv, len(df))


def setup_dataset(name: str, cfg: dict) -> None:
    raw_dir = os.path.join(SEQ_REC_DATA_PATH, "raw", name)
    os.makedirs(raw_dir, exist_ok=True)

    # --- interactions CSV ---
    dst_csv = os.path.join(raw_dir, f"{name}.csv")
    if os.path.exists(dst_csv):
        log.info("[%s] interactions CSV already exists, skipping.", name)
    else:
        ratings_raw = os.path.join(raw_dir, f"{name}_ratings_raw.csv")
        if not os.path.exists(ratings_raw):
            download_file(cfg["ratings_url"], ratings_raw)
        else:
            log.info("[%s] ratings CSV already downloaded, skipping.", name)
        log.info("[%s] Adding header to SNAP ratings CSV -> %s", name, dst_csv)
        ratings = pd.read_csv(ratings_raw, header=None, names=RATINGS_COLUMNS)
        ratings.to_csv(dst_csv, index=False)

    # --- metadata ---
    dst_meta_csv = os.path.join(raw_dir, f"{name}_meta.csv")
    if os.path.exists(dst_meta_csv):
        log.info("[%s] metadata CSV already exists, skipping download.", name)
        return

    gz_path = os.path.join(raw_dir, f"{name}_meta_raw.json.gz")
    if not os.path.exists(gz_path):
        download_file(cfg["meta_url"], gz_path)
    else:
        log.info("[%s] metadata gz already downloaded, skipping.", name)

    build_metadata_csv(gz_path, dst_meta_csv)


def main() -> None:
    for name, cfg in DATASETS.items():
        log.info("=== Setting up %s ===", name)
        setup_dataset(name, cfg)
    log.info("All done.")


if __name__ == "__main__":
    main()
