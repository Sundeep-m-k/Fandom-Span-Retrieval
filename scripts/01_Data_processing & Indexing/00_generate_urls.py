#!/usr/bin/env python3
"""
00_generate_urls.py

Generate a clean list of article URLs for a Fandom wiki.

Aligned with the new system design:

- Reads settings from configs/scraping.yaml
- Uses:
    * Special:AllPages (if start_url is set)
    * Category pages (category_urls)
- Writes output to:
    data/raw/url_lists/<domain>_urls.txt

Example configs/scraping.yaml:

    base_url: "https://money-heist.fandom.com"
    start_url: "https://money-heist.fandom.com/wiki/Special:AllPages"
    category_urls:
      - "https://money-heist.fandom.com/wiki/Category:Characters"
      - "https://money-heist.fandom.com/wiki/Category:Episodes"
      - "https://money-heist.fandom.com/wiki/Category:Locations"

"""
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit, unquote, urlparse
import requests
from bs4 import BeautifulSoup
# Global logger (configured in main)
logger = logging.getLogger("00_generate_urls")


# ---------------------- CONFIG LOADING ----------------------


def load_scraping_config():
    """
    Load configs/scraping.yaml relative to project root.

    Expected keys:
      - base_url (required)
      - start_url (optional)
      - category_urls (optional list)
    """
    try:
        import yaml
    except ImportError:
        print(
            "ERROR: PyYAML is not installed.\n"
            "Install it with: pip install pyyaml"
        )
        sys.exit(1)

    # Project root = parent of "scripts" directory
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


# ---------------------- LOGGING SETUP -----------------------
def create_logger(project_root: Path, script_name: str = "00_generate_urls"):
    """
    Create a logger that logs to both console and a file under:
        data/logs/scraping/<timestamp>_00_generate_urls.log
    """
    log_dir = project_root / "data" / "logs" / "scraping"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{timestamp}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers if re-run in same process (e.g., notebook)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Console handler
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


def normalize_url(url: str) -> str:
    """
    Normalize wiki URLs:
    - decode %XX
    - replace spaces with underscores
    - drop query + fragment
    """
    parts = urlsplit(url)
    path = unquote(parts.path).replace(" ", "_")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def filter_article(url: str) -> bool:
    """
    Keep only real /wiki/ article pages (exclude File:, Help:, etc.).
    """
    if "/wiki/" not in url:
        return False

    page = url.split("/wiki/", 1)[1]

    skip_prefixes = (
        "File:",
        "User:",
        "User_blog:",
        "Template:",
        "Help:",
        "Special:",
        "Forum:",
        "Message_Wall:",
        "Category:",  # drop category pages from “content”
        "Talk:",
        "MediaWiki:",
        "Module:",
    )
    return not any(page.startswith(p) for p in skip_prefixes)


# ---------------------- 1) SPECIAL:ALLPAGES -----------------


def scrape_allpages(base_url: str, start_url: str, session: requests.Session, delay: float = 0.5):
    """
    Crawl Special:AllPages and collect /wiki/ links.
    """
    url = start_url
    results = []
    seen = set()

    while url:
        logger.info(f"[AllPages] Fetching {url}")
        r = session.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.select(".mw-allpages-chunk li > a, .mw-allpages-group li > a"):
            href = a.get("href")
            if not href or not href.startswith("/wiki/"):
                continue
            full = normalize_url(urljoin(base_url, href))
            if full not in seen:
                seen.add(full)
                results.append(full)

        # Detect "next page"
        next_url = None

        # 1) <link rel="next">
        head_next = soup.find("link", rel=lambda v: v and "next" in v.lower() if v else False)
        if head_next and head_next.get("href"):
            next_url = head_next.get("href")

        # 2) anchor.mw-nextlink
        if not next_url:
            a_next = soup.select_one("a.mw-nextlink")
            if a_next and a_next.get("href"):
                next_url = a_next.get("href")

        # 3) pager link inside .mw-allpages-nav
        if not next_url:
            for a in soup.select(".mw-allpages-nav a[href]"):
                if a.get_text(strip=True).lower().startswith("next page"):
                    next_url = a.get("href")
                    break

        url = urljoin(base_url, next_url) if next_url else None
        time.sleep(delay)

    logger.info(f"[AllPages] Collected {len(results)} raw URLs")
    return results


# ---------------------- 2) CATEGORY SCRAPING ----------------


def scrape_category(base_url: str, category_url: str, session: requests.Session):
    """
    Extract article URLs from a single Fandom category page.
    (No pagination handling to keep it simple for now.)
    """
    logger.info(f"[Category] Fetching {category_url}")
    r = session.get(category_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    urls = []
    selectors = [
        "a.category-page__member-link",  # newer layout
        ".category-page__members a",     # older layout
        ".category-page__content a",     # fallback
    ]

    for sel in selectors:
        for a in soup.select(sel):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                full = normalize_url(base_url.rstrip("/") + href)
            else:
                full = normalize_url(href)
            urls.append(full)

    logger.info(f"[Category] Found {len(urls)} URLs in {category_url}")
    return urls


def scrape_all_categories(base_url: str, category_urls, session: requests.Session):
    if not category_urls:
        logger.info("No category_urls in scraping.yaml — skipping category scrape.")
        return []

    collected = []
    for cu in category_urls:
        try:
            collected.extend(scrape_category(base_url, cu, session))
        except Exception as e:
            logger.warning(f"Error in category {cu}: {e}", exc_info=True)

    logger.info(f"[Category] Total raw category URLs: {len(collected)}")
    return collected


# ---------------------- MAIN PIPELINE -----------------------


def get_all_urls(cfg, project_root: Path):
    base_url: str = cfg["base_url"].rstrip("/")
    start_url: str | None = cfg.get("start_url")
    category_urls = cfg.get("category_urls", []) or []

    logger.info(f"Base URL: {base_url}")
    logger.info(f"Start URL (AllPages): {start_url}")
    logger.info(f"Category URLs: {category_urls}")

    session = requests.Session()
    session.headers.update({"User-Agent": "FandomURLFetcher/1.0"})

    combined = []

    # 1) AllPages
    if start_url:
        combined.extend(scrape_allpages(base_url, start_url, session))
    else:
        logger.info("start_url not set in scraping.yaml — skipping AllPages.")

    # 2) Categories
    combined.extend(scrape_all_categories(base_url, category_urls, session))

    # Deduplicate + filter
    combined = list(set(combined))
    logger.info(f"Combined (before filtering): {len(combined)}")

    filtered = [u for u in combined if filter_article(u)]
    logger.info(f"After filtering to real article pages: {len(filtered)}")

    # Show a few samples
    for u in filtered[:5]:
        logger.info(f"Sample URL: {u}")

    # Output path: data/raw/url_lists/<domain>_urls.txt
    domain = urlparse(base_url).netloc.split(".")[0]  # e.g. "money-heist"
    url_list_dir = project_root / "data" / "raw" / "url_lists"
    url_list_dir.mkdir(parents=True, exist_ok=True)
    output_path = url_list_dir / f"{domain}_urls.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(filtered)))

    logger.info(f"Saved URL list to: {output_path}")
    return output_path


if __name__ == "__main__":
    cfg, project_root = load_scraping_config()
    logger, log_path = create_logger(project_root)

    start_time = time.time()
    logger.info("=== Starting 00_generate_urls ===")

    try:
        output_path = get_all_urls(cfg, project_root)
    except Exception as e:
        logger.error(f"Unhandled error in 00_generate_urls: {e}", exc_info=True)
        raise

    elapsed = time.time() - start_time
    logger.info(f"Finished 00_generate_urls in {elapsed:.2f} seconds")
    logger.info(f"URL list: {output_path}")
    logger.info(f"Log file: {log_path}")

    print(f"Saved URL list to: {output_path}")
    print(f"Log saved to: {log_path}")