"""Capture the app's key states as PNGs (TRIA-13 card screenshots, demo and paper figures).

Drives the Microsoft Edge already installed on Windows through Playwright (no browser download).
Needs the app running (`task app`). Output: results/ui/*.png
Run: uv run --with playwright python -m scripts.ui_screens [--base http://localhost:3000] [--skip-bench]
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "results" / "ui"
LONG = 240_000  # a full screen + benchmark on a laptop GPU (much slower on battery)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(args.base)
        page.wait_for_selector(".dot.ok", timeout=LONG)  # model loaded
        page.screenshot(path=OUT / "1-first-run.png")

        page.get_by_role("button", name="Check eligibility").click()  # empty input: the error summary
        page.wait_for_selector(".alert[role=alert]")
        page.screenshot(path=OUT / "2-validation-error.png")

        page.locator(".start .btn").click()  # load the first validation example
        page.get_by_role("button", name="Check eligibility").click()
        page.wait_for_selector(".outcome .verdict", timeout=LONG)
        rows = page.locator(".crf-main:not([disabled])")
        rows.nth(1).click()  # select a row: its evidence lights up in the note
        page.wait_for_timeout(600)
        page.screenshot(path=OUT / "3-screen-results.png", full_page=True)

        if not args.skip_bench:
            page.get_by_role("link", name="Benchmarks").click()
            page.get_by_role("button", name="Compare the three methods").click()
            page.wait_for_selector("table.data", timeout=LONG)
            page.screenshot(path=OUT / "4-bench-decoders.png", full_page=True)
            page.get_by_role("tab", name="Memory").click()
            page.get_by_role("button", name="Compare attention methods").click()
            page.locator("div:not([hidden]) > .bench table.data").wait_for(timeout=LONG)
            page.screenshot(path=OUT / "5-bench-attention.png", full_page=True)

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        mobile.goto(args.base)
        mobile.wait_for_selector(".dot.ok", timeout=LONG)
        mobile.screenshot(path=OUT / "6-mobile-first-run.png", full_page=True)
        mobile.locator(".start .btn").tap()
        mobile.get_by_role("button", name="Check eligibility").tap()
        mobile.wait_for_selector(".outcome .verdict", timeout=LONG)
        mobile.locator(".crf-main:not([disabled])").nth(1).tap()
        mobile.wait_for_timeout(600)
        mobile.screenshot(path=OUT / "7-mobile-results.png", full_page=True)
        browser.close()
    review = OUT.parent.parent / ".impeccable" / "review"  # the design review reads these two names
    review.mkdir(parents=True, exist_ok=True)
    for src, dst in (("3-screen-results.png", "desktop.png"), ("7-mobile-results.png", "mobile.png")):
        (review / dst).write_bytes((OUT / src).read_bytes())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
