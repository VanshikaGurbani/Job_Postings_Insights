"""Command line entry point: `jobinsights <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config, data, evalset, extract


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jobinsights", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="fetch the raw postings CSV into data/raw/")
    d.add_argument("--source", choices=["kaggle", "hf", "local"], default="kaggle")
    d.add_argument("--file", type=Path, help="existing CSV to copy (with --source local)")
    d.add_argument("--dest", type=Path, default=config.RAW_POSTINGS)

    c = sub.add_parser("clean", help="clean the raw CSV into a parquet file")
    c.add_argument("--src", type=Path, default=config.RAW_POSTINGS)
    c.add_argument("--dest", type=Path, default=config.CLEAN_POSTINGS)

    e = sub.add_parser("extract-rules", help="run the regex baseline (skills, remote, salary)")
    e.add_argument("--workers", type=int, default=None)

    sub.add_parser("make-eval-set", help="draw the stratified 150-posting evaluation sample")
    sub.add_parser("label", help="open the hand-labelling app (needs the dashboard extra)")

    m = sub.add_parser("llm-extract", help="Claude structured extraction on the eval set")
    m.add_argument("--model", default="claude-haiku-5-5")
    m.add_argument("--max-usd", type=float, required=True, help="hard spend cap for this run")
    m.add_argument("--all-rows", action="store_true", help="every clean posting, not just eval set")
    m.add_argument("--yes", action="store_true", help="confirm that real API credits are spent")

    sub.add_parser("evaluate", help="precision/recall/F1 of each extractor vs gold labels")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        path = data.download(args.source, args.dest, args.file)
        print(f"raw postings at {path}")
    elif args.command == "clean":
        data.clean(args.src, args.dest)
    elif args.command == "extract-rules":
        extract.extract(workers=args.workers)
    elif args.command == "make-eval-set":
        evalset.make_eval_ids(data.load_clean())
    elif args.command == "label":
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).with_name("label_app.py")),
            ],
            check=True,
        )
    elif args.command == "llm-extract":
        from . import llm

        df = data.load_clean()
        if not args.all_rows:
            df = df[df["job_id"].isin(evalset.load_ids())]
        est = llm.estimate_cost(df, args.model)
        print(
            f"{len(df)} postings, model {args.model}, "
            f"estimated cost ${est:.2f}, cap ${args.max_usd}"
        )
        if not args.yes:
            raise SystemExit("Dry run only. Re-run with --yes to spend credits.")
        llm.run(df, args.model, args.max_usd)
    elif args.command == "evaluate":
        from . import evaluate

        evaluate.evaluate()


if __name__ == "__main__":
    main()
