"""Structured extraction with a local open-weights instruct model (free, CPU).

The model is prompted for one JSON object per posting. Results are cached on disk by
(model, job_id, prompt version) so a re-run only pays for postings not yet processed.
Outputs that do not parse are counted and reported rather than silently dropped.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from . import config
from .salary import ANNUAL_FACTOR
from .taxonomy import normalize_skill

PROMPT_VERSION = "v2"
CACHE_DIR = Path("cache") / "llm"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_TOKENS = 3000

SYSTEM = """Extract fields from the job posting. Output ONLY a JSON object with keys remote, salary, skills.
remote: "remote" if the job can be done from home, "hybrid" if part office and part remote, "onsite" if the text says it must be done in person at a workplace, otherwise "unknown".
salary: {"present": bool, "min": number|null, "max": number|null, "period": "hourly"|"weekly"|"monthly"|"yearly"|null}. Use ONLY base pay amounts that literally appear in the posting text (not bonuses or commission); if the text states no pay, present is false and min, max and period are null.
skills: list of skills, tools, technologies and credentials the candidate must have, copied from the posting."""

EXAMPLE_IN = (
    "Title: Cashier\n\nPosting:\nWe are hiring a cashier for our store. Handle cash and POS "
    "systems. Must be able to lift 25 lbs. Apply today."
)
EXAMPLE_OUT = (
    '{"remote": "unknown", "salary": {"present": false, "min": null, "max": null, '
    '"period": null}, "skills": ["Cash Handling", "POS Systems"]}'
)

_PERIODS = {
    "hour": "hourly", "hourly": "hourly", "hr": "hourly",
    "week": "weekly", "weekly": "weekly",
    "month": "monthly", "monthly": "monthly",
    "year": "yearly", "yearly": "yearly", "annual": "yearly", "annually": "yearly",
}  # fmt: skip

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def load_model(name: str = DEFAULT_MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    return tok, model


def _cache_path(model: str, job_id: int) -> Path:
    return CACHE_DIR / model.replace("/", "__") / PROMPT_VERSION / f"{job_id}.json"


def parse_output(text: str) -> dict | None:
    """Parse and validate the model's JSON. None if unusable."""
    m = _JSON.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    remote = obj.get("remote")
    sal = obj.get("salary") if isinstance(obj.get("salary"), dict) else {}
    skills = obj.get("skills") if isinstance(obj.get("skills"), list) else []
    if remote not in ("remote", "hybrid", "onsite", "unknown"):
        return None
    lo, hi = _num(sal.get("min")), _num(sal.get("max"))
    period = _PERIODS.get(str(sal.get("period")).strip().lower())
    return {
        "remote": remote,
        "salary": {"present": bool(sal.get("present")), "min": lo, "max": hi, "period": period},
        "skills": [s for s in skills if isinstance(s, str)],
    }


def _num(x: object) -> float | None:
    """Numbers, or strings like "$40" / "85,000". Anything else is None."""
    if isinstance(x, int | float) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"\d[\d,]*\.?\d*", x)
        return float(m.group(0).replace(",", "")) if m else None
    return None


def to_pay(salary: dict) -> tuple[float | None, float | None]:
    """Annualise an LLM salary object. (None, None) if absent or the period is missing."""
    if not salary.get("present") or salary.get("min") is None:
        return None, None
    period = salary.get("period")
    if period not in ANNUAL_FACTOR:
        return None, None
    hi = salary["max"] if salary.get("max") is not None else salary["min"]
    return salary["min"] * ANNUAL_FACTOR[period], hi * ANNUAL_FACTOR[period]


def clean_skills(raw: list[str]) -> list[str]:
    """Canonical taxonomy names only; unknown strings are dropped."""
    seen: dict[str, None] = {}
    for s in raw:
        canon = normalize_skill(s)
        if canon:
            seen[canon] = None
    return list(seen)


def generate(tok, model, title: str, description: str) -> tuple[str, int, int]:
    import torch

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE_IN},
        {"role": "assistant", "content": EXAMPLE_OUT},
        {"role": "user", "content": f"Title: {title}\n\nPosting:\n{description}"},
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    if ids.shape[1] > MAX_PROMPT_TOKENS:  # keep the system prompt, trim the posting's tail
        keep_tail = ids[:, -200:]
        ids = torch.cat([ids[:, : MAX_PROMPT_TOKENS - 200], keep_tail], dim=1)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=300, do_sample=False)
    new = out[0, ids.shape[1] :]
    return tok.decode(new, skip_special_tokens=True), int(ids.shape[1]), int(len(new))


def run(
    df: pd.DataFrame, model_name: str = DEFAULT_MODEL, dest: Path | None = None
) -> pd.DataFrame:
    tok, model = load_model(model_name)
    rows, failed, t0 = [], 0, time.time()
    for i, rec in enumerate(df.itertuples(index=False), 1):
        path = _cache_path(model_name, rec.job_id)
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
        else:
            text, n_in, n_out = generate(tok, model, rec.title, rec.description)
            cached = {"job_id": rec.job_id, "raw": text, "in": n_in, "out": n_out}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cached), encoding="utf-8")
            print(f"[{i}/{len(df)}] {time.time() - t0:.0f}s elapsed", flush=True)
        parsed = parse_output(cached["raw"])
        ok = parsed is not None
        failed += not ok
        parsed = parsed or {
            "remote": "unknown",
            "salary": {"present": False, "min": None, "max": None, "period": None},
            "skills": [],
        }
        lo, hi = to_pay(parsed["salary"])
        rows.append(
            {
                "job_id": rec.job_id,
                "remote_llm": parsed["remote"],
                "skills_llm": clean_skills(parsed["skills"]),
                "skills_llm_raw": parsed["skills"],
                "salary_llm_low": lo,
                "salary_llm_high": hi,
                "llm_parse_ok": ok,
            }
        )
    out_df = pd.DataFrame(rows)
    print(f"{len(out_df)} postings, {failed} unparseable outputs")
    tag = model_name.split("/")[-1]
    dest = dest or config.PROCESSED_DIR / f"extracted_llm_{tag}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dest, index=False)
    return out_df
