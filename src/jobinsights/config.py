"""Filesystem layout. Everything large lives under DATA_DIR, which is gitignored."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("JOBINSIGHTS_DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

RAW_POSTINGS = RAW_DIR / "postings.csv"
CLEAN_POSTINGS = PROCESSED_DIR / "postings_clean.parquet"

KAGGLE_DATASET = "arshkon/linkedin-job-postings"
KAGGLE_FILE = "postings.csv"
CACHE_LLM = Path("cache") / "llm"
