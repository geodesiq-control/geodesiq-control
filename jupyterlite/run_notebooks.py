"""
Run notebooks of the built JupyterLite site in a headless browser and fail if any cell does not execute cleanly.

    python jupyterlite/run_notebooks.py smoke_test.ipynb landau_zener.ipynb
    python jupyterlite/run_notebooks.py --browser msedge --screenshots shots smoke_test.ipynb

The site in jupyterlite/_output (or --site) is served on a free local port. For each notebook, the stored outputs are
cleared and all cells are run through the JupyterLab menus, as a user would. A notebook passes when every code cell
has an execution count and no output contains a Python traceback. Needs `playwright` and a browser
(`playwright install chromium`, or an installed Edge/Chrome through --browser msedge/chrome).
"""

import argparse
import functools
import http.server
import re
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

PROMPTS = ".jp-Notebook .jp-CodeCell .jp-InputArea-prompt"


def serve(site: Path) -> http.server.ThreadingHTTPServer:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def click_menu(page: Page, menu: str, item: str) -> None:
    page.locator("#jp-MainMenu .lm-MenuBar-itemLabel", has_text=re.compile(f"^{menu}$")).click()
    page.locator(".lm-Menu-item", has=page.locator(".lm-Menu-itemLabel", has_text=re.compile(f"^{item}$"))).click()


def prompts(page: Page) -> list[str]:
    # textContent, not innerText: cells outside the viewport are not rendered and have no innerText.
    return page.eval_on_selector_all(PROMPTS, "els => els.map(e => e.textContent.trim())")


def run_notebook(page: Page, url: str, timeout: float) -> tuple[bool, str]:
    t0 = time.time()
    page.goto(url)
    page.wait_for_selector(".jp-NotebookPanel .jp-Cell", timeout=180_000)
    # The kernel is ready when the status bar reports it idle; the first start downloads the environment. A notebook
    # whose kernelspec matches no kernel of the site asks for one first: take the proposed one.
    deadline = time.time() + 600
    while "Idle" not in page.inner_text("#jp-main-statusbar"):
        select = page.locator(".jp-Dialog .jp-mod-accept", has_text="Select")
        if select.count():
            select.click()
        if time.time() > deadline:
            return False, "the kernel did not start within 600 s"
        time.sleep(1)
    t_ready = time.time() - t0

    page.locator(".jp-NotebookPanel .jp-Cell").first.click()
    click_menu(page, "Edit", "Clear Outputs of All Cells")
    page.wait_for_function(f"() => [...document.querySelectorAll('{PROMPTS}')].every(e => !/\\d/.test(e.textContent))")
    click_menu(page, "Run", "Run All Cells")

    deadline = time.time() + timeout
    page.wait_for_function(f"() => /[\\d*]/.test(document.querySelector('{PROMPTS}').textContent)", timeout=60_000)
    while any("*" in p for p in prompts(page)):
        if time.time() > deadline:
            return False, f"timeout after {timeout:.0f} s"
        time.sleep(2)
    t_run = time.time() - t0 - t_ready

    counts = prompts(page)
    not_run = [i for i, p in enumerate(counts) if not re.search(r"\d", p)]
    outputs = page.eval_on_selector_all(".jp-OutputArea-output", "els => els.map(e => e.textContent)")
    # dict.fromkeys drops duplicates: an error output contains a nested element with the same class.
    errors = [o for o in dict.fromkeys(outputs) if "Traceback" in o or re.search(r"^\w+Error: ", o, re.MULTILINE)]
    timing = f"kernel ready {t_ready:.0f} s, run all {t_run:.0f} s"
    if errors or not_run:
        details = "\n".join(e.strip()[-3000:] for e in errors)
        return False, f"{timing}; {len(counts) - len(not_run)}/{len(counts)} cells ran\n{details}"
    return True, f"{timing}; {len(counts)}/{len(counts)} cells ran"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="+", help="notebook paths inside the site, e.g. landau_zener.ipynb")
    parser.add_argument("--site", type=Path, default=Path(__file__).parent / "_output", help="built site directory")
    parser.add_argument("--browser", default="chromium", help="chromium (Playwright's), msedge or chrome")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds allowed for running each notebook")
    parser.add_argument("--screenshots", type=Path, help="directory for a screenshot of each notebook")
    args = parser.parse_args()

    server = serve(args.site)
    base = f"http://127.0.0.1:{server.server_address[1]}/lab/index.html?path="
    failed = []
    with sync_playwright() as p:
        channel = None if args.browser == "chromium" else args.browser
        browser = p.chromium.launch(channel=channel, headless=True)
        for notebook in args.notebooks:
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            try:
                ok, message = run_notebook(page, base + notebook, args.timeout)
            except Exception as exc:  # A missing element or a timeout of the page itself.
                ok, message = False, f"{type(exc).__name__}: {exc}"
            if args.screenshots:
                args.screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=args.screenshots / f"{Path(notebook).stem}.png")
            print(f"{'PASS' if ok else 'FAIL'} {notebook}: {message}", flush=True)
            if not ok:
                failed.append(notebook)
            page.close()
        browser.close()
    server.shutdown()

    if failed:
        print(f"{len(failed)} notebook(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
