#!/usr/bin/env python3
"""Check or sync manuscript figures using SHA-256 hashes.

Run ``python dev/sync_figures.py`` to check, or add ``--sync`` to copy missing
and changed files from dev/figures to paper/Figures. Paper-only files are kept.
Paths default to this checkout regardless of the current working directory.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Check figures, optionally copy differences, and return a shell exit code."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check hashes without copying (default)")
    mode.add_argument("--sync", action="store_true", help="copy missing or changed figures, then verify hashes")
    parser.add_argument("--source", type=Path, default=root / "dev" / "figures", help="source figure directory")
    parser.add_argument("--destination", type=Path, default=root / "paper" / "Figures", help="paper figure directory")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    destination = args.destination.resolve()

    try:
        if not source.is_dir():
            parser.error(f"source directory does not exist: {source}")
        if not destination.is_dir():
            parser.error(f"destination directory does not exist: {destination}")
        if source == destination or source in destination.parents or destination in source.parents:
            parser.error("source and destination directories must not overlap")
        files = sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
        if not files:
            parser.error(f"no source figures found in {source}")

        differences = 0
        for relative in files:
            original = source / relative
            target = destination / relative
            # Do not follow destination links outside the selected directory.
            if not target.resolve().is_relative_to(destination):
                parser.error(f"destination file resolves outside the destination directory: {target}")
            original_hash = sha256(original)
            if target.is_file() and sha256(target) == original_hash:
                continue
            differences += 1
            status = "CHANGED" if target.exists() else "MISSING"
            if args.sync:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                if sha256(target) != original_hash:
                    parser.exit(2, f"Hash verification failed after copying: {target}\n")
                status = "COPIED"
            print(f"{status}: {relative.as_posix()}")

        if args.sync:
            print(f"Synced {differences} file(s); all {len(files)} source figures match by SHA-256.")
        elif differences:
            print(f"{differences} of {len(files)} source figures differ. Run with --sync to copy them.")
            return 1
        else:
            print(f"All {len(files)} source figures match by SHA-256.")
        return 0
    except OSError as exc:
        parser.exit(2, f"Figure check/sync failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
