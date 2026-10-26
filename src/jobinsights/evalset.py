"""The hand-labelled evaluation set: which postings, and how gold labels are stored.

Only job ids and labels are committed (data/labels/), never posting text.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

from . import config

IDS_PATH = config.LABELS_DIR / "eval_ids.json"
GOLD_PATH = config.LABELS_DIR / "gold.jsonl"

TECH_TITLE = re.compile(
    r"\b(?:data|analyst|analytics|scientist|machine learning|ml|ai|software|developer|"
    r"engineer|bi|statistician|quantitative)\b",
    re.IGNORECASE,
)
REMOTE_KEYWORD = re.compile(r"remote|work from home|hybrid", re.IGNORECASE)

# (stratum name, size). The strata are NOT population-proportional; see the README.
STRATA = (("tech_title", 50), ("remote_keyword", 40), ("random_rest", 60))


def make_eval_ids(df: pd.DataFrame, seed: int = 42, path: Path = IDS_PATH) -> dict[int, str]:
    """Draw the stratified sample once and store it. Refuses to overwrite existing labels' ids."""
    if path.is_file():
        raise SystemExit(f"{path} exists; delete it deliberately if you want a new sample")
    is_tech = df["title"].str.contains(TECH_TITLE)
    has_kw = df["description"].str.contains(REMOTE_KEYWORD)
    pools = {
        "tech_title": df[is_tech],
        "remote_keyword": df[~is_tech & has_kw],
        "random_rest": df[~is_tech & ~has_kw],
    }
    chosen: dict[int, str] = {}
    for name, size in STRATA:
        pick = pools[name].sample(size, random_state=seed)
        chosen.update({int(j): name for j in pick["job_id"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": seed, "strata": {str(k): v for k, v in chosen.items()}}, indent=1),
        encoding="utf-8",
    )
    print(f"{len(chosen)} postings -> {path}")
    return chosen


def load_ids(path: Path = IDS_PATH) -> dict[int, str]:
    if not path.is_file():
        raise SystemExit("No eval set yet. Run `jobinsights make-eval-set`.")
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8"))["strata"].items()}


def load_gold(path: Path = GOLD_PATH) -> dict[int, dict]:
    """Latest label per job id (the file is append-only so edits are history)."""
    gold: dict[int, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                gold[int(rec["job_id"])] = rec
    return gold


def save_gold(rec: dict, path: Path = GOLD_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def labeling_order(ids: dict[int, str], seed: int = 0) -> list[int]:
    """Fixed shuffled order shared by the LLM run and the labelling app, so that labelling only
    part of the set still gives a random (roughly stratified) subset."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order
