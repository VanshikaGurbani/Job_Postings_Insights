"""Static figures for the README, drawn from outputs/insights.json (needs the `plots` extra)."""

from __future__ import annotations

from pathlib import Path

FIG_DIR = Path("docs") / "figures"
BLUE, ORANGE, TEAL, GREY, DARK = "#2a6f97", "#e9a03b", "#2a9d8f", "#c9ced6", "#264653"


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Install the extra first: pip install -e '.[plots]'") from e
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8a8f98",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def skill_premium(res: dict, path: Path) -> None:
    plt = _plt()
    p = res["premiums"]
    rows = p["skills"]
    show = rows[:9] + rows[-5:]
    show = show[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, r in enumerate(show):
        lo, hi = r["ci95"]
        sig = lo > 0 or hi < 0
        color = (TEAL if r["premium_pct"] > 0 else ORANGE) if sig else GREY
        ax.plot([lo, hi], [i, i], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(r["premium_pct"], i, "o", color=color, ms=6)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_yticks(range(len(show)), [f"{r['skill']}  (n={r['n']})" for r in show])
    ax.set_xlabel("Adjusted pay difference when the posting lists the skill (%)")
    ax.set_title(
        f"Pay associated with each skill ({p['n_postings']:,} tech postings that state pay)",
        loc="left",
    )
    ax.text(
        0,
        -0.17,
        "Adjusted for role, experience level and work arrangement. Bars: 95% interval, resampling "
        "companies.\nColoured = interval excludes zero. Associations, not causal effects.",
        transform=ax.transAxes,
        fontsize=8,
        color="#6b7280",
        va="top",
    )
    fig.savefig(path)
    plt.close(fig)


def role_skills(res: dict, path: Path, role: str = "Data Analyst", top: int = 12) -> None:
    plt = _plt()
    rows = res["demand"][role][:top][::-1]
    n_entry, n_senior = rows[0]["n_entry"], rows[0]["n_senior"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ys = range(len(rows))
    h = 0.38
    ax.barh([y + h / 2 for y in ys], [100 * r["entry"] for r in rows], h, color=TEAL,
            label=f"Entry / associate / internship (n={n_entry})")  # fmt: skip
    ax.barh([y - h / 2 for y in ys], [100 * r["senior"] for r in rows], h, color=BLUE,
            label=f"Mid-senior and above (n={n_senior})")  # fmt: skip
    ax.set_yticks(list(ys), [r["skill"] for r in rows])
    ax.set_xlabel("% of postings that list the skill")
    ax.set_title(f"Technical skills in {role} postings, by seniority")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def sponsor_share(res: dict, path: Path) -> None:
    plt = _plt()
    rows = res["sponsors_by_role"][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ys = range(len(rows))
    h = 0.38
    ax.barh([y + h / 2 for y in ys], [100 * r["shortlist"] for r in rows], h, color=TEAL,
            label="Not restricted, and sponsorship text or company H-1B history")  # fmt: skip
    ax.barh([y - h / 2 for y in ys], [100 * r["restricted"] for r in rows], h, color=ORANGE,
            label="Posting says no sponsorship / citizens only")  # fmt: skip
    ax.set_yticks(list(ys), [f"{r['role']} ({r['postings']:,})" for r in rows])
    ax.set_xlabel("% of postings")
    ax.set_title("How many postings an F-1 / OPT candidate can reasonably pursue")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.savefig(path)
    plt.close(fig)


def top_employers(res: dict, path: Path, top: int = 12) -> None:
    plt = _plt()
    rows = res["top_employers"][:top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(range(len(rows)), [r["openings"] for r in rows], color=BLUE)
    for i, r in enumerate(rows):
        ax.text(
            r["openings"] + 0.6,
            i,
            f"{r['h1b_approvals']:,} H-1B approvals FY21-23",
            va="center",
            fontsize=8,
            color="#6b7280",
        )
    ax.set_yticks(range(len(rows)), [r["company_name"] for r in rows])
    ax.set_xlabel("Open data / AI postings that do not rule out sponsorship")
    ax.set_title("Employers with H-1B history hiring the most data and AI roles")
    ax.set_xlim(0, max(r["openings"] for r in rows) * 1.55)
    fig.savefig(path)
    plt.close(fig)


def learning_path(res: dict, path: Path) -> None:
    plt = _plt()
    paths = res["learning_path"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True)
    for ax, (role, lp) in zip(axes.ravel(), paths.items(), strict=False):
        xs = [0] + list(range(1, len(lp["steps"]) + 1))
        ys = [100 * lp["ready_before"]] + [100 * s["ready_share"] for s in lp["steps"]]
        ax.plot(xs, ys, "-o", color=TEAL, ms=4)
        order = "\n".join(f"{i}. {s['skill']}" for i, s in enumerate(lp["steps"], 1))
        ax.text(0.03, 0.97, order, transform=ax.transAxes, va="top", fontsize=7.5, color="#374151")
        ax.set_title(f"{role} (n={lp['n_postings']})")
        ax.set_xlabel("technical skills learned, in this order")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of postings where you\nhave >= half the listed skills")
    fig.suptitle("Which skills to learn first (entry-level postings)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_all(res: dict, out_dir: Path = FIG_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "skill_premium.png": skill_premium,
        "data_analyst_skills.png": role_skills,
        "sponsor_share.png": sponsor_share,
        "top_employers.png": top_employers,
        "learning_path.png": learning_path,
    }
    paths = []
    for name, fn in jobs.items():
        fn(res, out_dir / name)
        paths.append(out_dir / name)
    return paths
