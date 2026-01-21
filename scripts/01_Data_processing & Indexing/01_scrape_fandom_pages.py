#!/usr/bin/env python3
"""
01_scrape_fandom_pages.py

Download raw HTML pages for a Fandom wiki and also extract plain-text
article content for each page.

Pipeline:

1) 00_generate_urls.py  --> data/raw/url_lists/<domain>_urls.txt
2) 01_scrape_fandom_pages.py       (this script)
    -> downloads each URL into:
         data/raw/fandom_html/<domain>/*.html
         data/raw/fandom_html/<domain>/*.txt

UPDATED SAVING PATTERN:
- Save files using the Fandom article/page ID (e.g., 12345.html, 12345.txt)
- If article ID cannot be extracted, fall back to URL-safe filename
  (so scraping doesn't fail completely).

Config:
  - Reads base_url from configs/scraping.yaml
"""

import sys
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlsplit, unquote

import requests
from bs4 import BeautifulSoup  # used for plain-text extraction


# ---------------------- CONFIG LOADING ----------------------


def load_scraping_config():
    """
    Load configs/scraping.yaml relative to project root.

    Needs at least:
      base_url: "https://money-heist.fandom.com"
    """
    try:
        import yaml
    except ImportError:
        print(
            "ERROR: PyYAML is not installed.\n"
            "Install it with: pip install pyyaml"
        )
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "scraping.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "base_url" not in cfg:
        print("ERROR: 'base_url' must be defined in configs/scraping.yaml")
        sys.exit(1)

    return cfg, project_root

def create_logger(project_root: Path, script_name: str = "01_scrape_fandom_pages"):
    log_dir = project_root / "data" / "logs" / "scraping"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{timestamp}_{script_name}.log"
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logger initialized for {script_name}")
    logger.info(f"Log file: {log_path}")
    return logger, log_path


# ---------------------- HELPERS -----------------------------


def safe_filename_from_url(url: str) -> str:
    """
    Turn a wiki URL into a safe relative filename, e.g.:

      https://money-heist.fandom.com/wiki/Professor
        -> money-heist_fandom_com_wiki_professor.html
    """
    parts = urlsplit(url)
    host = parts.netloc.replace(".", "_")
    path = unquote(parts.path).strip("/").replace("/", "_").replace(" ", "_")

    base = f"{host}_{path}" if path else host
    base = base[:180]  # avoid insane filesystem length
    # keep alnum, dot, dash, underscore
    base = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
    if not base:
        base = "page"

    return base + ".html"


def read_url_list(path: Path):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def extract_plain_text(html: str) -> str:
    """
    Extract readable plain text from a Fandom HTML page.
    Removes scripts, styles, etc., focusing on article content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, and other non-text elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Fandom articles commonly use:
    # <div class="mw-parser-output"> ... </div>
    content = soup.find("div", class_="mw-parser-output")

    if content:
        text = content.get_text(separator="\n", strip=True)
    else:
        # Fallback: extract all text
        text = soup.get_text(separator="\n", strip=True)

    # Cleanup multiple blank lines
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join([line for line in lines if line])

    return cleaned


def fetch_html(session: requests.Session, url: str, timeout: int = 20) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_article_id(html: str):
    """
    Try to extract the Fandom/MediaWiki page/article ID from the HTML.

    Returns:
      - string of digits (e.g., "12345") if found
      - None otherwise
    """
    # Fast regex attempts first (avoid full soup when possible)
    # Common: "wgArticleId":12345 or wgArticleId = 12345
    m = re.search(r'wgArticleId"\s*:\s*(\d+)', html)
    if m:
        return m.group(1)
    m = re.search(r"\bwgArticleId\s*=\s*(\d+)", html)
    if m:
        return m.group(1)

    # Meta tag: <meta property="mw:pageId" content="12345">
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", {"property": "mw:pageId"})
    if meta and meta.get("content") and str(meta["content"]).isdigit():
        return str(meta["content"])

    # Sometimes pageId appears elsewhere
    meta2 = soup.find("meta", {"name": "pageId"})
    if meta2 and meta2.get("content") and str(meta2["content"]).isdigit():
        return str(meta2["content"])

    return None


# ---------------------- MAIN SCRAPER ------------------------


def scrape_all(cfg, project_root: Path, logger: logging.Logger):
    base_url = cfg["base_url"].rstrip("/")
    domain = urlparse(base_url).netloc.split(".")[0]  # e.g. "money-heist"

    url_list_path = project_root / "data" / "raw" / "url_lists" / f"{domain}_urls.txt"
    output_dir = project_root / "data" / "raw" / "fandom_html" / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    if not url_list_path.exists():
        logger.error(f"URL list file not found: {url_list_path}")
        sys.exit(1)

    urls = read_url_list(url_list_path)
    logger.info(f"Loaded {len(urls)} URLs from {url_list_path}")
    logger.info(f"Output directory: {output_dir}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "FandomHTMLScraper/1.0 (+research; contact=you@example.com)"
        }
    )

    total = len(urls)
    success = 0
    skipped = 0
    failed = 0
    id_collisions = 0

    for i, url in enumerate(urls, start=1):
        logger.info(f"[{i}/{total}] Fetching → {url}")
        ok = False

        for attempt in range(1, 4):  # up to 3 attempts
            try:
                html = fetch_html(session, url)

                article_id = extract_article_id(html)
                if article_id:
                    fname = f"{article_id}.html"
                else:
                    # fallback (keeps scraping going)
                    fname = safe_filename_from_url(url)
                    logger.warning(f"    Could not extract article ID; using URL filename: {fname}")

                out_path = output_dir / fname

                if out_path.exists():
                    logger.info(f"    SKIP (exists) → {out_path.name}")
                    skipped += 1
                    ok = True
                    break

                # -------- Save HTML --------
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(html)
                logger.info(f"    Saved HTML → {out_path}")

                # -------- Extract & Save Plain Text --------
                plain_text = extract_plain_text(html)
                txt_path = out_path.with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(plain_text)
                logger.info(f"    Saved TEXT → {txt_path}")

                success += 1
                ok = True
                break

            except Exception as e:
                logger.warning(
                    f"    Attempt {attempt}/3 failed for {url}: {e}", exc_info=False
                )
                time.sleep(1.0 * attempt)

        if not ok:
            logger.error(f"    FAILED: {url}")
            failed += 1

        # polite delay between requests
        time.sleep(0.5)

    logger.info("=== Scrape summary ===")
    logger.info(f"Total URLs:    {total}")
    logger.info(f"Downloaded:    {success}")
    logger.info(f"Skipped (had): {skipped}")
    logger.info(f"Failed:        {failed}")
    if id_collisions:
        logger.info(f"ID collisions: {id_collisions} (saved with fallback names)")


# ---------------------- ENTRYPOINT --------------------------


if __name__ == "__main__":
    cfg, project_root = load_scraping_config()
    logger, log_path = create_logger(project_root)

    start = time.time()
    logger.info("=== Starting 01_scrape_fandom_pages ===")

    try:
        scrape_all(cfg, project_root, logger)
    except Exception as e:
        logger.error(f"Unhandled error in 01_scrape_fandom_pages: {e}", exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 01_scrape_fandom_pages in {elapsed:.2f} seconds")
    logger.info(f"Log file: {log_path}")

    print(f"📝 Log saved to: {log_path}")