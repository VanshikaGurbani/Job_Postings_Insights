"""Cluster postings by their skill sets and evaluate the clustering honestly.

Each posting is a binary vector over the skill taxonomy. For each k we report the mean
silhouette (cosine) and the stability of the partition across random seeds (mean pairwise
adjusted Rand index), so k is chosen from measurements rather than hard-coded.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import config
from .roles import role_family
from .taxonomy import FAMILY, SKILL_NAMES

OUT_DIR = config.DATA_DIR.parent / "outputs"
K_RANGE = range(4, 21, 2)
N_SEEDS = 5
MIN_SKILLS = 2  # postings with fewer hard skills carry too little signal to cluster
SVD_DIMS = 15
# Soft skills (communication, teamwork...) appear in most postings of every kind, so they only
# blur the clusters; the representation uses the other families.
HARD = [i for i, s in enumerate(SKILL_NAMES) if FAMILY[s] != "Soft skills"]


def skill_matrix(skill_lists: pd.Series) -> np.ndarray:
    idx = {s: i for i, s in enumerate(SKILL_NAMES)}
    X = np.zeros((len(skill_lists), len(SKILL_NAMES)), dtype=np.float32)
    for r, skills in enumerate(skill_lists):
        for s in skills:
            X[r, idx[s]] = 1.0
    return X


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def sweep(X: np.ndarray, seed: int = 0, sil_sample: int = 5000) -> pd.DataFrame:
    """Silhouette and seed-stability for each k. X rows are L2-normalised (cosine geometry)."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), min(sil_sample, len(X)), replace=False)
    rows = []
    for k in K_RANGE:
        labelings = [KMeans(k, n_init=3, random_state=s).fit_predict(X) for s in range(N_SEEDS)]
        ari = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labelings, 2)]
        sil = silhouette_score(X[sample], labelings[0][sample])
        rows.append({"k": k, "silhouette": sil, "stability_ari": float(np.mean(ari))})
    return pd.DataFrame(rows)


def choose_k(table: pd.DataFrame) -> int:
    """Highest silhouette among partitions that are stable (ARI >= 0.7); else the most stable."""
    stable = table[table["stability_ari"] >= 0.7]
    pool = stable if len(stable) else table.sort_values("stability_ari", ascending=False).head(1)
    return int(pool.sort_values("silhouette", ascending=False).iloc[0]["k"])


def name_clusters(
    X: np.ndarray, labels: np.ndarray, titles: pd.Series, top_n: int = 4
) -> dict[int, dict]:
    """Name each cluster by the skills with the highest lift (share inside / share overall)."""
    overall = X.mean(axis=0)
    out = {}
    for c in sorted(set(labels)):
        mask = labels == c
        share = X[mask].mean(axis=0)
        lift = np.where(overall > 0.01, share / np.maximum(overall, 1e-9), 0)
        # require the skill to be common inside the cluster, otherwise rare skills dominate
        score = np.where(share >= 0.25, lift * share, 0)
        top = [SKILL_NAMES[i] for i in np.argsort(-score)[:top_n] if score[i] > 0]
        roles = titles[mask].map(role_family).value_counts(normalize=True)
        out[int(c)] = {
            "name": " / ".join(top[:3]) or "mixed",
            "size": int(mask.sum()),
            "top_skills": top,
            "top_role_family": roles.index[0],
            "top_role_family_share": round(float(roles.iloc[0]), 3),
            "tech_title_share": round(float(1 - roles.get("Other", 0.0)), 3),
        }
    return out


def represent(skill_lists: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows kept, binary hard-skill matrix, embedding); IDF-weighted, SVD-reduced, L2-normed."""
    binary = skill_matrix(skill_lists)[:, HARD]
    keep = binary.sum(axis=1) >= MIN_SKILLS
    binary = binary[keep]
    idf = np.log(len(binary) / (1 + binary.sum(axis=0))) + 1
    z = TruncatedSVD(SVD_DIMS, random_state=0).fit_transform(_l2(binary * idf))
    return keep, binary, _l2(z)


def run() -> dict:
    clean = pd.read_parquet(config.CLEAN_POSTINGS)
    rules = pd.read_parquet(config.PROCESSED_DIR / "extracted_rules.parquet")
    df = clean.merge(rules[["job_id", "skills_rules"]], on="job_id")
    # Baseline = what the original notebook did: raw binary skills, k fixed at 15.
    base = _l2(skill_matrix(df["skills_rules"]))
    base = base[base.any(axis=1)]
    base_lab = KMeans(15, n_init=3, random_state=0).fit_predict(base)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(base), 5000, replace=False)
    baseline = {
        "k": 15,
        "silhouette": round(float(silhouette_score(base[pick], base_lab[pick])), 3),
    }
    keep, binary, Z = represent(df["skills_rules"])
    df = df[keep].reset_index(drop=True)
    table = sweep(Z)
    k = choose_k(table)
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    full = np.zeros((len(df), len(SKILL_NAMES)), dtype=np.float32)
    full[:, HARD] = binary
    names = name_clusters(full, labels, df["title"])
    xy = TruncatedSVD(2, random_state=0).fit_transform(Z)
    pd.DataFrame(
        {"job_id": df["job_id"], "cluster": labels, "x": xy[:, 0], "y": xy[:, 1]}
    ).to_parquet(config.PROCESSED_DIR / "clusters.parquet", index=False)
    summary = {
        "n_clustered": int(len(df)),
        "n_total": int(len(clean)),
        "min_hard_skills": MIN_SKILLS,
        "representation": f"hard-skill binary x IDF -> SVD({SVD_DIMS}) -> L2",
        "baseline_original_method": baseline,
        "chosen_k": k,
        "sweep": table.round(4).to_dict(orient="records"),
        "clusters": names,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clusters.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("baseline (original method):", baseline)
    print(table.round(3).to_string(index=False))
    print(f"chosen k={k}")
    for c, info in names.items():
        print(
            c,
            info["size"],
            info["name"],
            "|",
            info["top_role_family"],
            info["top_role_family_share"],
        )
    return summary
