"""Prepare the manuscript in ``paper/`` for an arXiv submission and pack it into a ready-to-upload zip.

Steps:

1. Copy the LaTeX sources (``main.tex``, class, bibliography style and database, ``Extras/``, ``Figures/``) into a
   temporary work folder and strip all comments from every ``.tex`` file (verbatim-like environments such as
   ``minted`` are left untouched; a comment after code keeps its ``%`` so that no line-end space is introduced).
2. Compile there with ``latexmk -pdf -shell-escape`` (needed by ``minted``) and read the recorder file (``main.fls``)
   to find the files that the document really uses. Unused figures, old figure versions and templates are dropped.
3. Copy only the used files, the generated ``main.bbl`` and the ``_minted`` cache into a staging folder, and switch
   ``minted`` to ``frozencache``: arXiv compiles without ``-shell-escape``, so the highlighted code comes from the cache.
   The local ``minted.sty``, ``fvextra.sty`` and ``latex2pydata.sty`` are shipped too, because the cache is only valid
   for the minted version that wrote it and arXiv's TeX Live has an older one. ``\linenumbers`` is removed, so the
   arXiv version has no line numbers.
4. Check that the staged sources compile with plain ``pdflatex`` (no shell escape, no BibTeX), without errors or
   undefined references, and with the same number of pages as the full build.
5. Write the zip.

Usage (from the repository root)::

    python dev/prepare_arxiv.py                      # writes arxiv/geodesiq_arxiv.zip
    python dev/prepare_arxiv.py -o some/where.zip --keep   # keep the work folders for inspection

Needs ``latexmk`` and ``pdflatex`` (with minted v3 and Pygments) on the PATH. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN = "main.tex"
SOURCES = ["main.tex", "SciPost.cls", "SciPost_bibstyle.bst", "references.bib", "Extras", "Figures"]

# Environments whose content is copied verbatim (a % there is code, not a comment).
VERBATIM_ENVS = {"verbatim", "verbatim*", "minted", "lstlisting", "comment", "Verbatim"}
# Packages shipped with the submission so that arXiv uses the versions that wrote the minted cache.
VENDORED_PACKAGES = ["minted.sty", "fvextra.sty", "latex2pydata.sty"]
# Auxiliary files that are never part of the submission (main.bbl is added explicitly).
AUX_SUFFIXES = {".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".blg", ".listing", ".synctex.gz"}


def find_comment(line: str) -> int:
    """Return the index of the first unescaped ``%`` in ``line``, or ``-1``."""
    for match in re.finditer("%", line):
        i = match.start()
        backslashes = len(line[:i]) - len(line[:i].rstrip("\\"))
        if backslashes % 2 == 0:
            return i
    return -1


def strip_comments(text: str) -> str:
    """Remove LaTeX comments, keeping verbatim-like environments and the content after ``\\end{document}`` out."""
    envs = set(VERBATIM_ENVS)
    envs.update(re.findall(r"\\(?:newtcblisting|DeclareTCBListing|newminted\w*)\{?\\?(\w+)\}?", text))
    begin = re.compile(r"\\begin\{(" + "|".join(re.escape(e) for e in envs) + r")\}")

    out: list[str] = []
    in_env: str | None = None
    for line in text.splitlines():
        if in_env is not None:
            out.append(line)
            if f"\\end{{{in_env}}}" in line:
                in_env = None
            continue
        i = find_comment(line)
        code = line if i < 0 else line[:i]
        if (m := begin.search(code)) and f"\\end{{{m.group(1)}}}" not in code[m.end() :]:
            # Keep the rest of the line as it is: the options of the environment may follow.
            in_env = m.group(1)
            out.append(line if i < 0 else code + "%")
            continue
        if i < 0:
            out.append(line)
        elif code.strip():
            out.append(code + "%")  # the % still suppresses the line-end space
        # a comment-only line is dropped
        if "\\end{document}" in code:
            break

    cleaned = "\n".join(out) + "\n"
    return re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse the blank lines left by removed comment blocks


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        print(result.stdout[-4000:], result.stderr[-2000:], sep="\n")
        sys.exit(f"Command failed in {cwd}: {' '.join(cmd)}")


def page_count(log: Path) -> int:
    match = re.search(r"Output written on .*?\((\d+) pages?", log.read_text(errors="replace"), re.S)
    if not match:
        sys.exit(f"No output pages found in {log}")
    return int(match.group(1))


def check_log(log: Path) -> None:
    text = log.read_text(errors="replace")
    problems = [
        line
        for line in text.splitlines()
        if line.startswith("!") or "undefined" in line.lower() and ("Reference" in line or "Citation" in line)
    ]
    if problems:
        sys.exit(f"Problems in {log}:\n" + "\n".join(problems[:20]))


def used_files(work: Path) -> set[Path]:
    """Files inside ``work`` that pdflatex read, from the recorder file ``main.fls``."""
    used: set[Path] = set()
    for line in (work / "main.fls").read_text(errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        path = Path(line[6:].strip())
        path = (work / path) if not path.is_absolute() else path
        try:
            rel = path.resolve().relative_to(work.resolve())
        except ValueError:
            continue  # a file of the TeX distribution
        if rel.parts[0] == "_minted" or rel.suffix in AUX_SUFFIXES | {".minted"} or not path.is_file():
            continue  # the minted cache is copied as a whole; temporary files are gone
        used.add(rel)
    return used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "arxiv" / "geodesiq_arxiv.zip")
    parser.add_argument("--keep", action="store_true", help="keep the work and staging folders")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="arxiv-"))
    work, stage, check = tmp / "work", tmp / "stage", tmp / "check"
    try:
        # 1. copy and strip comments
        print(f"Copying sources from {PAPER} to {work}")
        work.mkdir()
        for name in SOURCES:
            src = PAPER / name
            if src.is_dir():
                shutil.copytree(src, work / name, ignore=shutil.ignore_patterns("build*", "_minted*"))
            else:
                shutil.copy2(src, work / name)
        for tex in work.rglob("*.tex"):
            tex.write_text(strip_comments(tex.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")

        # 2. full build with shell escape
        print("Full build (latexmk, shell escape)")
        run(["latexmk", "-pdf", "-shell-escape", "-recorder", "-interaction=nonstopmode", "-halt-on-error", MAIN], work)
        check_log(work / "main.log")
        pages = page_count(work / "main.log")

        # 3. stage the used files
        files = sorted(used_files(work))
        print(f"Staging {len(files)} used files, main.bbl and the minted cache")
        stage.mkdir()
        for rel in files:
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work / rel, stage / rel)
        shutil.copy2(work / "main.bbl", stage / "main.bbl")
        shutil.copytree(work / "_minted", stage / "_minted")
        main_tex = (stage / MAIN).read_text(encoding="utf-8")
        main_tex, n = re.subn(r"\\usepackage(\[[^\]]*\])?\{minted\}", _frozen, main_tex, count=1)
        if n != 1:
            sys.exit("Could not find \\usepackage{minted} in main.tex")
        # The arXiv version has no line numbers (they are only for the journal submission).
        main_tex, n = re.subn(r"^[ \t]*\\linenumbers[ \t]*%?\n", "", main_tex, flags=re.M)
        if n == 0:
            print("  note: no \\linenumbers found in main.tex")
        (stage / MAIN).write_text(main_tex, encoding="utf-8", newline="\n")
        # The cache only matches the minted version that wrote it, and arXiv's TeX Live is older: ship the local
        # minted, fvextra and latex2pydata (they require each other's latest versions), which TeX loads before the
        # system copies.
        for package in VENDORED_PACKAGES:
            result = subprocess.run(["kpsewhich", package], capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout.strip():
                sys.exit(f"kpsewhich could not find {package}")
            shutil.copy2(result.stdout.strip(), stage / package)

        # 4. check as arXiv would compile it: pdflatex only, no shell escape, no BibTeX
        print("Check build (pdflatex x3, no shell escape)")
        shutil.copytree(stage, check)
        for _ in range(3):
            run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", MAIN], check)
        check_log(check / "main.log")
        if (check_pages := page_count(check / "main.log")) != pages:
            sys.exit(f"Page count differs: {check_pages} (check build) vs {pages} (full build)")

        # 5. zip
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())
        shutil.copy2(check / "main.pdf", args.output.with_suffix(".pdf"))
        dropped = sorted(
            p.relative_to(work).as_posix()
            for p in work.rglob("*")
            if p.is_file() and p.parts[len(work.parts)] in ("Extras", "Figures") and p.relative_to(work) not in files
        )
        print(f"\nDone: {args.output} ({args.output.stat().st_size / 1e6:.1f} MB, {pages} pages)")
        print(f"Preview of the check build: {args.output.with_suffix('.pdf')}")
        print("Included:", ", ".join(p.as_posix() for p in files))
        print("Dropped (unused):", ", ".join(dropped) or "none")
    finally:
        if args.keep:
            print(f"Work folders kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def _frozen(match: re.Match[str]) -> str:
    options = match.group(1)
    options = f"[{options[1:-1]},frozencache]" if options else "[frozencache]"
    return rf"\usepackage{options}{{minted}}"


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
