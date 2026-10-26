"""The labelled evaluation set: which postings, and how gold labels are stored.

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
STRATA = (("tech_title", 12), ("remote_keyword", 12), ("random_rest", 16))


def make_eval_ids(df: pd.DataFrame, seed: int = 42, path: Path = IDS_PATH) -> dict[int, str]:
    """Draw the stratified sample once and store it. Refuses to overwrite existing labels' ids."""
    if path.is_file():
        raise SystemExit(f"{path} exists; delete it deliberately if you want a new sample")
    df = df.drop_duplicates(subset=["company_name", "title"])  # reposted ads would repeat
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


FLAGGED = "visa_flagged"


def add_visa_flagged(rules: pd.DataFrame, n_each: int = 10, seed: int = 1, path: Path = IDS_PATH):
    """Append postings the visa rule flags (friendly / restricted) to the sample.

    Random postings almost never mention work authorisation (about 5%), so a random sample
    cannot test the rule. These extras measure the precision of its flags; recall is not
    estimable from them and the report says so.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if FLAGGED in data["strata"].values():
        return
    friendly = rules[rules["visa_rules"] == "friendly"].sample(n_each, random_state=seed)
    restricted = rules[rules["visa_rules"].isin(["restricted", "mixed"])].sample(
        n_each, random_state=seed
    )
    for j in list(friendly["job_id"]) + list(restricted["job_id"]):
        data["strata"].setdefault(str(int(j)), FLAGGED)
    data["visa_flagged_seed"] = seed
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"added {2 * n_each} visa-flagged postings to {path}")


def labeling_order(ids: dict[int, str], seed: int = 0) -> list[int]:
    """Fixed order shared by the LLM run and the labelling app, so that labelling only part of
    the set still gives a random subset. Every third posting is a visa-flagged one."""
    main = sorted(j for j, s in ids.items() if s != FLAGGED)
    flagged = sorted(j for j, s in ids.items() if s == FLAGGED)
    rng = random.Random(seed)
    rng.shuffle(main)
    rng.shuffle(flagged)
    order: list[int] = []
    while main or flagged:
        for _ in range(2):
            if main:
                order.append(main.pop())
        if flagged:
            order.append(flagged.pop())
    return order
