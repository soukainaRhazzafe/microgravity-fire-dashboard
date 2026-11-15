import argparse
import csv
import json
import re
import time
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
import asyncio
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


DEFAULT_URL = (
    "https://psi.nasa.gov/physci/repo/search"
    "?q=&data_source=psi&data_type=investigation"
)

FIELDS = [
    "title",
    "objective",
    "approach",
    "hypothesis",
    "research_area",
    "research_impacts",
    "hardware",
    "investigation_id",
    "source_url",
    "scraped_at",
    "scrape_error",
  
]

ALIASES = {
    "objective": ["Objectives", "Objective"],
    "approach": ["Approach"],
    "hypothesis": ["Hypothesis"],
    "research_impacts": ["Research Impacts/Earth Benefits"],
    "research_area": ["Research Area"],
    "hardware": ["Hardware"],
}

# Material icon names can appear as text beside section headings.
ICON_WORDS = {
    "info",
    "pallet",
    "table_view",
    "image",
    "sim_card_download",
    "menu_book",
}


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize(text):
    words = [
        word for word in clean(text).split()
        if word not in ICON_WORDS
    ]
    return " ".join(words).strip().rstrip(":").casefold()


def extract_field(soup, aliases):
    wanted = {normalize(alias) for alias in aliases}

    # First handle structured label/value tables.
    for row in soup.select("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) >= 2:
            if normalize(cells[0].get_text(" ", strip=True)) in wanted:
                return clean(
                    " ".join(cell.get_text(" ", strip=True) for cell in cells[1:])
                )

    # Then handle headings, with strong/b labels as a fallback.
    tags = soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b"]
    )

    for tag in tags:
        if normalize(tag.get_text(" ", strip=True)) not in wanted:
            continue

        lines = soup.get_text("\n", strip=True).splitlines()
        lines = [clean(line) for line in lines if clean(line)]

        heading_names = {
            normalize(node.get_text(" ", strip=True))
            for node in tags
        }

        for index, line in enumerate(lines):
            if normalize(line) not in wanted:
                continue

            content = []
            for following in lines[index + 1:]:
                key = normalize(following)
                if not key:
                    continue
                if key in heading_names:
                    break
                content.append(following)

            # Hardware tables use "Identifier" as their column heading.
            if "hardware" in wanted:
                content = [
                    value for value in content
                    if normalize(value) != "identifier"
                ]

            value = clean(" ".join(content))
            if value.casefold().startswith("no associated"):
                return ""
            if value:
                return value

    return ""


async def investigation_links(page):
    links = await page.locator("a[href]").evaluate_all(
        "(anchors) => anchors.map(a => a.href)"
    )
    pattern = re.compile(
        r"https://psi\.nasa\.gov/physci/repo/data/"
        r"investigations/PSI-\d+",
        re.IGNORECASE,
    )
    found = set()
    for link in links:
        match = pattern.match(urljoin(page.url, link))
        if match:
            found.add(match.group(0))
    return sorted(found)


async def collect_search_results(page, url):
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)

    await page.wait_for_selector(
        'a[href*="/data/investigations/PSI-"]',
        timeout=45000,
    )

    found = set()
    seen_pages = set()

    while True:
        current = await investigation_links(page)
        signature = tuple(current)

        if signature in seen_pages:
            break

        seen_pages.add(signature)
        found.update(current)
        print(f"Collected {len(found)} unique investigation links")

        next_button = page.get_by_role(
            "button", name=re.compile(r"next page", re.IGNORECASE)
        )

        if await next_button.count() == 0:
            break

        button = next_button.first
        if not await button.is_visible() or not await button.is_enabled():
            break

        await button.click()
        await page.wait_for_timeout(2000)

    return sorted(found)


async def scrape_investigation(page, url):
    record = {field: "" for field in FIELDS}
    record["source_url"] = url
    record["investigation_id"] = url.rstrip("/").split("/")[-1]
    record["scraped_at"] = datetime.now(timezone.utc).isoformat()

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_function(
            """() => [...document.querySelectorAll('h1')]
                .some(h => /PSI-\\d+/i.test(h.textContent))""",
            timeout=45000,
        )
        await page.wait_for_timeout(1500)

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        main = soup.find("main") or soup

        for unwanted in main.select("script, style, nav, footer"):
            unwanted.decompose()

        title = next(
            (
                heading for heading in main.find_all("h1")
                if re.search(r"PSI-\d+", heading.get_text())
            ),
            None,
        )

        record["title"] = (
            clean(title.get_text(" ", strip=True)) if title else ""
        )

        for field, aliases in ALIASES.items():
            record[field] = extract_field(main, aliases)

        if not record["title"] or not record["objective"]:
            record["scrape_error"] = (
                "Missing title or objective: review page structure/loading."
            )

    except Exception as error:
        record["scrape_error"] = str(error)

    return record


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    # Changed default parameter value to True for headless browser inside Google Colab
    parser.add_argument("--headless", action="store_false", default=True)
    args, unknown = parser.parse_known_args()

    output = Path("output")
    output.mkdir(exist_ok=True)
    records = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=args.headless)
        context = await browser.new_context()
        search_page = await context.new_page()

        try:
            urls = await collect_search_results(search_page, args.url)
        except Exception:
            await search_page.screenshot(
                path=str(output / "psi_search_debug.png"),
                full_page=True,
            )
            await browser.close()
            raise RuntimeError(
                "Could not read investigation links. "
                "Check the browser and output/psi_search_debug.png."
            )

        if not urls:
            await browser.close()
            raise RuntimeError(
                "No investigation links found."
            )

        page = await context.new_page()

        for index, url in enumerate(urls, start=1):
            print(f"[{index}/{len(urls)}] {url}")
            record = await scrape_investigation(page, url)
            records.append(record)

            if record["scrape_error"]:
                print("  WARNING:", record["scrape_error"])

            await asyncio.sleep(1)

        await browser.close()

    with (output / "investigations.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} investigations.")
    print("CSV: output/psi_investigations.csv")



if __name__ == "__main__":
    asyncio.run(main_async())