import importlib.util
from datetime import date
from pathlib import Path

import pytest
from packaging.version import Version

SCRIPT = Path(__file__).parents[1] / "dev" / "bump_version.py"
SPEC = importlib.util.spec_from_file_location("bump_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bump_version = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bump_version)


def project_files(tmp_path: Path, unreleased: str = "### Fixed\n\n- Corrected a defect.\n") -> tuple[Path, Path]:
    meta = tmp_path / "_meta.py"
    changelog = tmp_path / "CHANGELOG.md"
    meta.write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    changelog.write_text(f"# Changelog\n\n## [Unreleased]\n\n{unreleased}\n## [1.2.3] - 2026-01-01\n\n- Previous.\n",
                         encoding="utf-8", )
    return meta, changelog


def test_increment_version() -> None:
    current = Version("1.2.3")
    assert bump_version.increment_version(current, "patch") == Version("1.2.4")
    assert bump_version.increment_version(current, "minor") == Version("1.3.0")
    assert bump_version.increment_version(current, "major") == Version("2.0.0")
    assert bump_version.increment_version(Version("1.2"), "patch") == Version("1.2.1")


def test_explicit_prerelease_and_atomic_update(tmp_path: Path) -> None:
    meta, changelog = project_files(tmp_path)
    bump_version.update_project(meta, changelog, Version("1.3.0rc1"), release_date=date(2026, 7, 10))
    assert '__version__ = "1.3.0rc1"' in meta.read_text(encoding="utf-8")
    updated = changelog.read_text(encoding="utf-8")
    assert "## [Unreleased]\n\n## [1.3.0rc1] - 2026-07-10" in updated
    assert "- Corrected a defect." in updated


def test_rejects_non_increasing_version(tmp_path: Path) -> None:
    meta, changelog = project_files(tmp_path)
    with pytest.raises(ValueError, match="must be greater"):
        bump_version.update_project(meta, changelog, Version("1.2.3"))


def test_rejects_empty_unreleased_section(tmp_path: Path) -> None:
    meta, changelog = project_files(tmp_path, unreleased="")
    with pytest.raises(ValueError, match="empty"):
        bump_version.update_project(meta, changelog, Version("1.2.4"))


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    meta, changelog = project_files(tmp_path)
    before = (meta.read_text(encoding="utf-8"), changelog.read_text(encoding="utf-8"))
    updated = bump_version.update_project(meta, changelog, Version("1.2.4"), dry_run=True)
    assert updated != before
    assert (meta.read_text(encoding="utf-8"), changelog.read_text(encoding="utf-8")) == before


def test_citation_version_is_updated_atomically(tmp_path: Path) -> None:
    meta, changelog = project_files(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text('cff-version: 1.2.0\ntitle: "x"\nversion: 1.2.3\nlicense: MIT\n', encoding="utf-8")

    bump_version.update_project(meta, changelog, Version("1.3.0"), citation_file=citation)

    assert citation.read_text(encoding="utf-8") == 'cff-version: 1.2.0\ntitle: "x"\nversion: 1.3.0\nlicense: MIT\n'


def test_citation_without_version_is_rejected_before_writing(tmp_path: Path) -> None:
    meta, changelog = project_files(tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text("cff-version: 1.2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        bump_version.update_project(meta, changelog, Version("1.3.0"), citation_file=citation)

    assert '__version__ = "1.2.3"' in meta.read_text(encoding="utf-8")
