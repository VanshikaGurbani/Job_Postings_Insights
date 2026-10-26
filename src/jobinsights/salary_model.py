"""Salary regression on postings that disclose pay, with honest out-of-fold error metrics.

Only postings that state pay can be modelled (about a third of the data), and they are not a
random sample, so the numbers describe pay-transparent postings, not all postings. Pay is never
imputed for postings that do not state it. Folds are grouped by normalised title so near-duplicate
postings (the same role posted many times) cannot leak between train and test.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

from . import config
from .roles import role_family, state_of
from .taxonomy import SKILL_NAMES

OUT_DIR = config.DATA_DIR.parent / "outputs"
N_FOLDS = 5
MIN_ANNUAL, MAX_ANNUAL = 15_000.0, 600_000.0


def build_frame() -> pd.DataFrame:
    clean = pd.read_parquet(config.CLEAN_POSTINGS)
    rules = pd.read_parquet(config.PROCESSED_DIR / "extracted_rules.parquet")
    df = clean.merge(rules, on="job_id")
    df["role"] = df["title"].map(role_family)
    df["state"] = df["location"].map(state_of).fillna("NA")
    df["annual_mid"] = (df["salary_annual_low"] + df["salary_annual_high"]) / 2
    df["experience"] = df["formatted_experience_level"].fillna("Unknown")
    return df


def coverage(df: pd.DataFrame) -> dict:
    """Share of postings that state pay, overall and by group (shows who is missing)."""
    has = df["annual_mid"].notna()
    out = {"overall": round(float(has.mean()), 3), "n_postings": int(len(df))}
    for col in ("role", "experience"):
        out[f"by_{col}"] = has.groupby(df[col]).mean().round(3).to_dict()
    out["by_remote"] = has.groupby(df["remote_rules"]).mean().round(3).to_dict()
    return out


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z ]", "", t.lower()).strip()


def features(train: pd.DataFrame, test: pd.DataFrame):
    """Fit encoders on train only; return sparse train/test matrices."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=3000, stop_words="english")
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
    cat_cols = ["role", "experience", "state", "remote_rules", "formatted_work_type"]

    def skills(frame: pd.DataFrame) -> sparse.csr_matrix:
        idx = {s: i for i, s in enumerate(SKILL_NAMES)}
        rows, cols = [], []
        for r, lst in enumerate(frame["skills_rules"]):
            for s in lst:
                rows.append(r)
                cols.append(idx[s])
        return sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(frame), len(SKILL_NAMES))
        )

    cats = lambda f: f[cat_cols].fillna("NA").astype(str)  # noqa: E731
    xtr = sparse.hstack(
        [tfidf.fit_transform(train["title"]), ohe.fit_transform(cats(train)), skills(train)]
    ).tocsr()
    xte = sparse.hstack(
        [tfidf.transform(test["title"]), ohe.transform(cats(test)), skills(test)]
    ).tocsr()
    return xtr, xte


def _metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y, p = np.exp(y_true_log), np.exp(y_pred_log)
    ss_res = np.sum((y_true_log - y_pred_log) ** 2)
    ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
    return {
        "mae_usd": round(float(np.mean(np.abs(y - p))), 0),
        "median_ae_usd": round(float(np.median(np.abs(y - p))), 0),
        "mape": round(float(np.mean(np.abs(y - p) / y)), 3),
        "r2_log": round(float(1 - ss_res / ss_tot), 3),
    }


def run() -> dict:
    df = build_frame()
    cov = coverage(df)
    d = df[df["annual_mid"].between(MIN_ANNUAL, MAX_ANNUAL)].reset_index(drop=True)
    y = np.log(d["annual_mid"].to_numpy())
    groups = d["title"].map(_norm_title)
    oof = {name: np.zeros(len(d)) for name in ("median", "role_exp_median", "ridge", "gbm")}
    for tr, te in GroupKFold(N_FOLDS).split(d, y, groups):
        train, test = d.iloc[tr], d.iloc[te]
        oof["median"][te] = DummyRegressor(strategy="median").fit(train, y[tr]).predict(test)
        key = ["role", "experience"]
        med = pd.Series(y[tr], index=train.index).groupby([train[k] for k in key]).median()
        fallback = float(np.median(y[tr]))
        oof["role_exp_median"][te] = [
            med.get((r, e), fallback) for r, e in zip(test["role"], test["experience"], strict=True)
        ]
        xtr, xte = features(train, test)
        oof["ridge"][te] = Ridge(alpha=3.0).fit(xtr, y[tr]).predict(xte)
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
        oof["gbm"][te] = gbm.fit(xtr.toarray(), y[tr]).predict(xte.toarray())
    results = {name: _metrics(y, p) for name, p in oof.items()}
    ridge_by_role = {
        role: _metrics(y[m], oof["ridge"][m]) | {"n": int(m.sum())}
        for role in d["role"].unique()
        if (m := (d["role"] == role).to_numpy()).sum() >= 50
    }
    summary = {
        "n_modelled": int(len(d)),
        "n_postings": int(len(df)),
        "salary_sources": d["salary_source"].value_counts().to_dict(),
        "cv": f"{N_FOLDS}-fold GroupKFold grouped by normalised title, out-of-fold predictions",
        "target": "log annual mid-point (USD); hourly x2080, weekly x52, monthly x12",
        "models": results,
        "ridge_by_role": ridge_by_role,
        "coverage": cov,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "salary_model.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary
