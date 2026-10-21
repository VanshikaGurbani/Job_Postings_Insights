"""Download and clean the raw postings table."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

from . import config

# Columns the pipeline uses. Missing ones are tolerated (the HF mirror lacks a few).
KEEP_COLS = [
    "job_id",
    "company_name",
    "title",
    "description",
    "location",
    "min_salary",
    "med_salary",
    "max_salary",
    "pay_period",
    "currency",
    "remote_allowed",
    "formatted_work_type",
    "formatted_experience_level",
    "skills_desc",
    "listed_time",
]
MIN_DESCRIPTION_CHARS = 50


def download(source: str, dest: Path = config.RAW_POSTINGS, from_file: Path | None = None) -> Path:
    """Place the raw postings CSV at `dest`.

    source:
      kaggle - full dataset (123,849 rows). Uses the user's own Kaggle credentials
               (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
      hf     - public Hugging Face subset (33,246 rows), no login.
      local  - copy a CSV you already downloaded (`from_file`).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source == "local":
        if from_file is None or not from_file.is_file():
            raise FileNotFoundError("--file must point to an existing postings CSV")
        shutil.copyfile(from_file, dest)
    elif source == "kaggle":
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise SystemExit("Install the extra first: pip install -e '.[kaggle]'") from e
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(config.KAGGLE_DATASET, config.KAGGLE_FILE, path=str(dest.parent))
        zipped = dest.parent / (config.KAGGLE_FILE + ".zip")
        if zipped.exists():
            shutil.unpack_archive(zipped, dest.parent)
            zipped.unlink()
    elif source == "hf":
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(config.HF_DATASET, config.HF_FILE, repo_type="dataset")
        shutil.copyfile(cached, dest)
    else:
        raise ValueError(f"unknown source {source!r}")
    if not dest.is_file():
        raise FileNotFoundError(f"download finished but {dest} does not exist")
    return dest


_WS = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Collapse whitespace. Non-strings (NaN) become an empty string."""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", text).strip()


def clean_postings(raw: pd.DataFrame) -> pd.DataFrame:
    """Select columns, normalise text, drop duplicates and near-empty descriptions."""
    df = raw[[c for c in KEEP_COLS if c in raw.columns]].copy()
    for col in ("title", "description", "location", "company_name", "skills_desc"):
        if col in df.columns:
            df[col] = df[col].map(clean_text)
    df = df.drop_duplicates(subset="job_id", keep="first")
    df = df[df["description"].str.len() >= MIN_DESCRIPTION_CHARS]
    return df.reset_index(drop=True)


def clean(src: Path = config.RAW_POSTINGS, dest: Path = config.CLEAN_POSTINGS) -> pd.DataFrame:
    raw = pd.read_csv(src)
    df = clean_postings(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"raw rows: {len(raw):,}  clean rows: {len(df):,}  -> {dest}")
    return df


def load_clean(path: Path = config.CLEAN_POSTINGS) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run `jobinsights download` then `jobinsights clean`.")
    return pd.read_parquet(path)
