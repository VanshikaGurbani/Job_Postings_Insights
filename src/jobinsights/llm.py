"""Structured extraction with the Claude API (skills, salary, work arrangement).

Calls are cached on disk by (model, job_id, prompt version) so nothing is paid for twice,
and a hard dollar cap stops the run before it overspends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config
from .salary import ANNUAL_FACTOR
from .taxonomy import normalize_skill

PROMPT_VERSION = "v1"
CACHE_DIR = Path("cache") / "llm"

# $ per million tokens (input, output). Anthropic list prices as of 2026-10; check before a big run.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

SYSTEM = """You extract structured fields from a US job posting.

remote: how the posting says the work is done. "remote" = the employee works from home or \
anywhere (fully remote); "hybrid" = a mix of office and remote; "onsite" = the posting says the \
work is in person / on site / at a named workplace and does not offer remote; "unknown" = the \
text does not say. Do not infer from the job type alone.

salary: only the BASE pay range stated in the posting text, copied as written. Ignore bonuses, \
commission, OTE, sign-on pay, benefits, stipends, and company revenue. If the text gives a single \
number use it for both min and max. period is the unit the text states (hourly, weekly, monthly, \
yearly); if no unit is stated use null. If there is no base pay in the text, present=false and the \
numbers are null. Do not use any structured metadata, only the text.

skills: the concrete skills, tools, technologies, methods, credentials and competencies the \
posting asks the candidate to have or use. Short canonical names (e.g. "Python", "SQL", \
"Excel", "Project Management", "Customer Service", "Communication"). Do not list the employer's \
products, benefits, or generic words like "experience". No duplicates."""

SCHEMA = {
    "type": "object",
    "properties": {
        "remote": {"type": "string", "enum": ["remote", "hybrid", "onsite", "unknown"]},
        "salary": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "min": {"type": ["number", "null"]},
                "max": {"type": ["number", "null"]},
                "period": {
                    "type": ["string", "null"],
                    "enum": ["hourly", "weekly", "monthly", "yearly", None],
                },
            },
            "required": ["present", "min", "max", "period"],
            "additionalProperties": False,
        },
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["remote", "salary", "skills"],
    "additionalProperties": False,
}


def estimate_cost(df: pd.DataFrame, model: str, out_tokens: int = 250) -> float:
    """Rough dollar estimate: ~4 characters per token plus the fixed prompt."""
    p_in, p_out = PRICES[model]
    in_tok = (df["description"].str.len().sum() + df["title"].str.len().sum()) / 4
    in_tok += len(df) * (len(SYSTEM) / 4 + 60)
    return (in_tok * p_in + len(df) * out_tokens * p_out) / 1e6


def _cache_path(model: str, job_id: int) -> Path:
    return CACHE_DIR / model / PROMPT_VERSION / f"{job_id}.json"


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
    """Canonical taxonomy names only; unknown strings are dropped (counted by the caller)."""
    seen: dict[str, None] = {}
    for s in raw:
        canon = normalize_skill(s)
        if canon:
            seen[canon] = None
    return list(seen)


def extract_one(client, model: str, job_id: int, title: str, description: str) -> dict:
    path = _cache_path(model, job_id)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Title: {title}\n\nPosting:\n{description}"}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"job {job_id}: stop_reason={resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    result = {
        "job_id": job_id,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "output": json.loads(text),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def run(
    df: pd.DataFrame, model: str, max_usd: float, dest: Path | None = None, client=None
) -> pd.DataFrame:
    """Extract for every row of `df`, stopping if the running cost would pass `max_usd`."""
    if model not in PRICES:
        raise SystemExit(f"unknown model {model!r}; known: {', '.join(PRICES)}")
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    p_in, p_out = PRICES[model]
    spent, rows = 0.0, []
    for rec in df.itertuples(index=False):
        cached = _cache_path(model, rec.job_id).is_file()
        res = extract_one(client, model, rec.job_id, rec.title, rec.description)
        if not cached:
            spent += (res["input_tokens"] * p_in + res["output_tokens"] * p_out) / 1e6
        out = res["output"]
        lo, hi = to_pay(out["salary"])
        rows.append(
            {
                "job_id": rec.job_id,
                "remote_llm": out["remote"],
                "skills_llm": clean_skills(out["skills"]),
                "skills_llm_raw": out["skills"],
                "salary_llm_low": lo,
                "salary_llm_high": hi,
                "llm_in_tokens": res["input_tokens"],
                "llm_out_tokens": res["output_tokens"],
            }
        )
        if spent > max_usd:
            print(f"stopping: spent ${spent:.3f} > cap ${max_usd:.2f} after {len(rows)} postings")
            break
    out_df = pd.DataFrame(rows)
    print(f"{len(out_df)} postings, ${spent:.4f} spent this run (cached calls are free)")
    dest = dest or config.PROCESSED_DIR / f"extracted_llm_{model}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dest, index=False)
    return out_df
