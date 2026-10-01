#!/usr/bin/env python3
"""Safely update the package version and changelog.

Examples
--------
    python dev/bump_version.py --patch
    python dev/bump_version.py --minor --dry-run
    python dev/bump_version.py 1.2.0rc1
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Literal

from packaging.version import InvalidVersion, Version

_VERSION_PATTERN = re.compile(r'(__version__\s*=\s*")([^"]+)(")')
_CITATION_VERSION_PATTERN = re.compile(r"^(version:\s*)(\S+)\s*$", re.MULTILINE)
_UNRELEASED_PATTERN = re.compile(r"^(?P<heading>## \[Unreleased\][^\n]*\n)(?P<body>.*?)(?=^## \[|\Z)",
                                 re.MULTILINE | re.DOTALL, )


def read_current_version(meta_file: Path) -> Version:
    """Read and validate ``__version__`` from the package metadata file."""
    content = meta_file.read_text(encoding="utf-8")
    match = _VERSION_PATTERN.search(content)
    if match is None:
        raise ValueError(f"Could not find __version__ in {meta_file}")
    try:
        return Version(match.group(2))
    except InvalidVersion as exc:
        raise ValueError(f"Current version {match.group(2)!r} is invalid") from exc


def increment_version(current: Version, part: Literal["major", "minor", "patch"]) -> Version:
    """Return the next final release for the requested semantic component."""
    major, minor, patch = (*current.release, 0, 0)[:3]
    if part == "major":
        return Version(f"{major + 1}.0.0")
    if part == "minor":
        return Version(f"{major}.{minor + 1}.0")
    return Version(f"{major}.{minor}.{patch + 1}")


def parse_requested_version(raw: str) -> Version:
    """Parse an explicit PEP 440 version."""
    try:
        return Version(raw)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid version: {raw!r}") from exc


def prepare_updates(meta_content: str, changelog_content: str, new_version: Version, *,
                    release_date: date | None = None, ) -> tuple[str, str]:
    """Prepare both updated files in memory without modifying the filesystem."""
    match = _VERSION_PATTERN.search(meta_content)
    if match is None:
        raise ValueError("Could not find __version__ in metadata content")

    unreleased = _UNRELEASED_PATTERN.search(changelog_content)
    if unreleased is None:
        raise ValueError("Could not find an [Unreleased] section in CHANGELOG.md")
    body = unreleased.group("body").strip()
    if not body:
        raise ValueError("The [Unreleased] changelog section is empty")

    updated_meta = _VERSION_PATTERN.sub(rf"\g<1>{new_version}\g<3>", meta_content, count=1)
    day = release_date or date.today()
    replacement = f"## [Unreleased]\n\n## [{new_version}] - {day.isoformat()}\n\n{body}\n\n"
    updated_changelog = (
            changelog_content[: unreleased.start()] + replacement + changelog_content[unreleased.end():].lstrip("\n"))
    return updated_meta, updated_changelog


def prepare_citation_update(citation_content: str, new_version: Version) -> str:
    """Return CITATION.cff content with its top-level ``version`` set to ``new_version``."""
    if _CITATION_VERSION_PATTERN.search(citation_content) is None:
        raise ValueError("Could not find a top-level 'version:' entry in CITATION.cff")
    return _CITATION_VERSION_PATTERN.sub(rf"\g<1>{new_version}", citation_content, count=1)


def _write_temp(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False, )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def atomic_update(files: dict[Path, str]) -> None:
    """Replace several UTF-8 text files and restore originals on failure."""
    originals = {path: path.read_bytes() for path in files}
    temporary = {path: _write_temp(path, content) for path, content in files.items()}
    replaced: list[Path] = []
    try:
        for path, temp_path in temporary.items():
            os.replace(temp_path, path)
            replaced.append(path)
    except Exception:
        for path in replaced:
            restore = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
            restore_path = Path(restore.name)
            with restore:
                restore.write(originals[path])
                restore.flush()
                os.fsync(restore.fileno())
            os.replace(restore_path, path)
        raise
    finally:
        for temp_path in temporary.values():
            temp_path.unlink(missing_ok=True)


def update_project(meta_file: Path, changelog_file: Path, new_version: Version, *, dry_run: bool = False,
                   release_date: date | None = None, citation_file: Path | None = None, ) -> tuple[str, str]:
    """Validate and update the project files (and CITATION.cff, if given), optionally without writing."""
    current = read_current_version(meta_file)
    if new_version <= current:
        raise ValueError(f"New version {new_version} must be greater than current version {current}")
    updated_meta, updated_changelog = prepare_updates(meta_file.read_text(encoding="utf-8"),
                                                      changelog_file.read_text(encoding="utf-8"), new_version,
                                                      release_date=release_date, )
    files = {meta_file: updated_meta, changelog_file: updated_changelog}
    if citation_file is not None:
        files[citation_file] = prepare_citation_update(citation_file.read_text(encoding="utf-8"), new_version)
    if not dry_run:
        atomic_update(files)
    return updated_meta, updated_changelog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("version", nargs="?", help="Explicit PEP 440 version, for example 1.2.0rc1")
    target.add_argument("--major", action="store_true", help="Increment the major version")
    target.add_argument("--minor", action="store_true", help="Increment the minor version")
    target.add_argument("--patch", action="store_true", help="Increment the patch version")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print changes without writing files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    meta_file = project_root / "src" / "geodesiq" / "_meta.py"
    changelog_file = project_root / "CHANGELOG.md"
    citation_file = project_root / "CITATION.cff"
    current = read_current_version(meta_file)

    if args.version is not None:
        new_version = parse_requested_version(args.version)
    else:
        part: Literal["major", "minor", "patch"]
        if args.major:
            part = "major"
        elif args.minor:
            part = "minor"
        else:
            part = "patch"
        new_version = increment_version(current, part)

    try:
        update_project(meta_file, changelog_file, new_version, dry_run=args.dry_run,
                       citation_file=citation_file if citation_file.exists() else None, )
    except ValueError as exc:
        build_parser().error(str(exc))

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} geodesiq from {current} to {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
