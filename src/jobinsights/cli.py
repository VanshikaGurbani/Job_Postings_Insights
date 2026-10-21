"""Command line entry point: `jobinsights <command>`."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config, data


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
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        path = data.download(args.source, args.dest, args.file)
        print(f"raw postings at {path}")
    elif args.command == "clean":
        data.clean(args.src, args.dest)


if __name__ == "__main__":
    main()
