#!/usr/bin/env python3
"""
02_parse_html_to_sections.py

Parse raw Fandom HTML pages into clean text sections + internal link annotations.

Input:
    data/raw/fandom_html/<domain>/*.html

Output (UPDATED: under data/interim/<domain>/):
    1) data/interim/<domain>/sections_parsed_<domain>.jsonl
    2) data/interim/<domain>/sections_parsed_<domain>_by_page/<article_id|page_name|stem>.jsonl

Key note:
- Link offsets must match the saved section text. To keep it simple and reliable,
  we build text + offsets from the same buffer (no get_text()+find()).
"""

import sys
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin, unquote

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


# ---------------------- CONFIG LOADING ----------------------


def load_scraping_config():
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is not installed. Install it with: pip install pyyaml")
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


# ---------------------- LOGGING -----------------------------


def create_logger(project_root: Path, script_name: str = "02_parse_html_to_sections"):
    log_dir = project_root / "data" / "logs" / "preprocessing"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{ts}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()

    fmt = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized for {script_name}")
    logger.info(f"Log file: {log_path}")
    return logger, log_path


# ---------------------- PAGE METADATA HELPERS ----------------

WG_ARTICLE_ID_RE = re.compile(r'"wgArticleId"\s*:\s*(\d+)')
WG_PAGENAME_RE = re.compile(r'"wgPageName"\s*:\s*"([^"]+)"')
WG_TITLE_RE = re.compile(r'"wgTitle"\s*:\s*"([^"]+)"')


def get_article_id(soup: BeautifulSoup):
    for script in soup.find_all("script"):
        if not script.string:
            continue
        m = WG_ARTICLE_ID_RE.search(script.string)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def get_page_name(soup: BeautifulSoup):
    for script in soup.find_all("script"):
        if not script.string:
            continue
        m = WG_PAGENAME_RE.search(script.string)
        if m:
            return m.group(1)
    return None


def get_wg_title(soup: BeautifulSoup):
    for script in soup.find_all("script"):
        if not script.string:
            continue
        m = WG_TITLE_RE.search(script.string)
        if m:
            return m.group(1)
    return None


def get_title_from_html(soup: BeautifulSoup):
    h1 = soup.select_one("h1.page-header__title")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    h1 = soup.select_one("h1#firstHeading") or soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return None


def get_main_content_div(soup: BeautifulSoup):
    for sel in [
        "#mw-content-text .mw-parser-output",
        "div.mw-parser-output",
        "div.page-content div.mw-parser-output",
        "#mw-content-text",
        "div.page-content",
        "main",
        "article",
    ]:
        node = soup.select_one(sel)
        if node:
            return node
    return soup.body or soup


# ---------------------- URL & LINK HELPERS ------------------

SKIP_NAMESPACES = (
    "File:",
    "User:",
    "User_blog:",
    "Template:",
    "Help:",
    "Special:",
    "Forum:",
    "Message_Wall:",
    "Category:",
    "Talk:",
    "MediaWiki:",
    "Module:",
)


def classify_link(href: str, base_url: str):
    if not href:
        return "other", None, None

    href = href.strip()
    base_url = base_url.rstrip("/")

    if href.startswith("#"):
        return "other", None, None

    if href.startswith("http://") or href.startswith("https://"):
        if href.startswith(base_url + "/wiki/"):
            path = href[len(base_url):]
            if path.startswith("/wiki/"):
                page = unquote(path[len("/wiki/"):]).replace(" ", "_")
                if any(page.startswith(ns) for ns in SKIP_NAMESPACES):
                    ns = page.split(":", 1)[0] + ":"
                    if ns.lower() == "file:":
                        return "file", None, href
                    if ns.lower() == "category:":
                        return "category", None, href
                    return "other", None, href
                return "internal", page, href
            return "other", None, href
        return "external", None, href

    if href.startswith("/wiki/"):
        page = unquote(href[len("/wiki/"):]).replace(" ", "_")
        if any(page.startswith(ns) for ns in SKIP_NAMESPACES):
            ns = page.split(":", 1)[0] + ":"
            if ns.lower() == "file:":
                return "file", None, urljoin(base_url, href)
            if ns.lower() == "category:":
                return "category", None, urljoin(base_url, href)
            return "other", None, urljoin(base_url, href)
        resolved = urljoin(base_url, href)
        return "internal", page, resolved

    resolved = urljoin(base_url, href)
    return "other", None, resolved


# ---------------------- SECTION PARSING ----------------------

SKIP_SECTION_TITLES = {
    "references",
    "external links",
    "see also",
    "navigation",
    "contents",
}

SKIP_CONTAINER_CLASSES = {
    "toc",
    "portable-infobox",
    "infobox",
    "navbox",
}


def is_in_ignored_container(el):
    for parent in el.parents:
        if not getattr(parent, "attrs", None):
            continue

        if parent.get("id", "") == "toc":
            return True

        classes = parent.get("class", [])
        if any(cls in SKIP_CONTAINER_CLASSES for cls in classes):
            return True

    return False


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _append(buf: str, s: str) -> str:
    s = _norm_ws(s)
    if not s:
        return buf
    if not buf:
        return s
    if not buf.endswith((" ", "\n")):
        return buf + " " + s
    return buf + s


def extract_text_and_links_from_block(el, base_offset: int, base_url: str, page_map: dict):
    text = ""
    links = []

    for node in el.descendants:
        if isinstance(node, NavigableString):
            if node.parent and isinstance(node.parent, Tag) and node.parent.name == "a":
                continue
            text = _append(text, str(node))
            continue

        if isinstance(node, Tag) and node.name == "a":
            anchor = node.get_text(" ", strip=True)
            if not anchor:
                continue

            href = node.get("href")
            link_type, target_page_name, resolved_url = classify_link(href, base_url)

            start = len(text)
            text = _append(text, anchor)
            end = len(text)

            target_article_id = None
            if link_type == "internal" and target_page_name:
                target_article_id = page_map.get(target_page_name)

            links.append({
                "anchor_text": anchor,
                "start": base_offset + start,
                "end": base_offset + end,
                "link_type": link_type,
                "target_page_name": target_page_name,
                "target_article_id": target_article_id,
                "resolved_url": resolved_url,
            })

    return text, links


def iter_sections_with_links(content_div: BeautifulSoup,
                             page_title: str | None,
                             base_url: str,
                             page_map: dict):
    current_section = "Introduction"
    current_blocks = []
    current_links = []
    current_offset = 0

    def flush():
        return "\n".join(current_blocks).strip()

    for el in content_div.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if is_in_ignored_container(el):
            continue

        if el.name in ("h2", "h3"):
            text = flush()
            if text and not current_section.startswith("__SKIP__"):
                yield current_section, text, list(current_links)

            current_blocks.clear()
            current_links.clear()
            current_offset = 0

            sec_title = el.get_text(" ", strip=True) or "Untitled"
            sec_title = sec_title.replace("[edit]", "").strip()
            sec_title_lower = sec_title.lower()

            if sec_title_lower in SKIP_SECTION_TITLES:
                current_section = f"__SKIP__:{sec_title}"
                continue

            if page_title and sec_title_lower == page_title.lower():
                current_section = f"__SKIP__:{sec_title}"
                continue

            current_section = sec_title
            continue

        if current_section.startswith("__SKIP__"):
            continue

        if el.name == "p":
            block_text, block_links = extract_text_and_links_from_block(
                el, base_offset=current_offset, base_url=base_url, page_map=page_map
            )
            if block_text:
                current_blocks.append(block_text)
                current_links.extend(block_links)
                current_offset += len(block_text) + 1  # +1 for '\n'

        elif el.name in ("ul", "ol"):
            for li in el.find_all("li", recursive=False):
                block_text, block_links = extract_text_and_links_from_block(
                    li, base_offset=current_offset, base_url=base_url, page_map=page_map
                )
                if block_text:
                    current_blocks.append(block_text)
                    current_links.extend(block_links)
                    current_offset += len(block_text) + 1  # +1 for '\n'

    text = flush()
    if text and not current_section.startswith("__SKIP__"):
        yield current_section, text, list(current_links)


def make_wiki_url(base_url: str, page_name: str | None):
    if not page_name:
        return None
    return base_url.rstrip("/") + "/wiki/" + page_name


# ---------------------- PAGE MAP BUILD ----------------


def build_page_map(html_dir: Path, logger: logging.Logger):
    page_map = {}
    html_files = sorted(html_dir.glob("*.html"))
    logger.info(f"Building page_map from {len(html_files)} HTML files")

    for i, path in enumerate(html_files, start=1):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        except Exception as e:
            logger.warning(f"[page_map {i}/{len(html_files)}] Failed to read {path}: {e}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        page_name = get_page_name(soup)
        article_id = get_article_id(soup)

        if page_name is not None and article_id is not None:
            page_map[page_name] = article_id

        if i % 200 == 0:
            logger.info(f"[page_map] Processed {i}/{len(html_files)} files")

    logger.info(f"[page_map] Built mapping for {len(page_map)} pages")
    return page_map


# ---------------------- CORE -----------------------------


def parse_all_html(cfg, project_root: Path, logger: logging.Logger):
    base_url = cfg["base_url"].rstrip("/")
    domain = urlparse(base_url).netloc.split(".")[0]

    html_dir = project_root / "data" / "raw" / "fandom_html" / domain

    # UPDATED: write into data/interim/<domain>/
    interim_root = project_root / "data" / "interim" / domain
    interim_root.mkdir(parents=True, exist_ok=True)

    out_path = interim_root / f"sections_parsed_{domain}.jsonl"
    per_page_dir = interim_root / f"sections_parsed_{domain}_by_page"
    per_page_dir.mkdir(parents=True, exist_ok=True)

    if not html_dir.exists():
        logger.error(f"HTML directory not found: {html_dir}")
        sys.exit(1)

    html_files = sorted(list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm")))
    if not html_files:
        logger.error(f"No .html/.htm files found in: {html_dir}")
        sys.exit(1)

    logger.info(f"Base URL: {base_url}")
    logger.info(f"HTML dir: {html_dir}")
    logger.info(f"Master output: {out_path}")
    logger.info(f"Per-page output dir: {per_page_dir}")
    logger.info(f"Found {len(html_files)} HTML files to process")

    page_map = build_page_map(html_dir, logger)

    pages_seen = set()
    pages_processed = 0
    pages_skipped = 0
    sections_written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, path in enumerate(html_files, start=1):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
            except Exception as e:
                logger.warning(f"[{i}/{len(html_files)}] Failed to read {path}: {e}")
                pages_skipped += 1
                continue

            soup = BeautifulSoup(html, "html.parser")

            page_name = get_page_name(soup)
            article_id = get_article_id(soup)
            wg_title = get_wg_title(soup)
            html_title = get_title_from_html(soup)

            if wg_title:
                display_title = wg_title
            elif html_title:
                display_title = html_title
            elif page_name:
                display_title = page_name.replace("_", " ")
            else:
                display_title = None

            content_div = get_main_content_div(soup)
            if not content_div:
                logger.warning(f"[{i}/{len(html_files)}] No content div for {path}")
                pages_skipped += 1
                continue

            url = make_wiki_url(base_url, page_name)

            if article_id is not None:
                page_key = str(article_id)
            elif page_name:
                page_key = page_name
            else:
                page_key = path.stem

            safe_page_key = page_key.replace("/", "_")
            page_out_path = per_page_dir / f"{safe_page_key}.jsonl"

            pages_seen.add(article_id or page_name or str(path))

            sec_count_for_page = 0
            with open(page_out_path, "w", encoding="utf-8") as pfout:
                for section_title, section_text, links in iter_sections_with_links(
                    content_div,
                    display_title,
                    base_url=base_url,
                    page_map=page_map,
                ):
                    txt_clean = (section_text or "").strip()
                    if not txt_clean:
                        continue

                    if txt_clean.lower() in {"to be added", "to be added.", "tba"}:
                        continue

                    rec = {
                        "article_id": article_id,
                        "page_name": page_name,
                        "title": display_title,
                        "section": section_title,
                        "text": txt_clean,
                        "url": url,
                        "source_path": str(path.relative_to(project_root)),
                        "links": links,
                    }

                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    fout.write(line)
                    pfout.write(line)

                    sections_written += 1
                    sec_count_for_page += 1

            if sec_count_for_page == 0:
                pages_skipped += 1
                logger.info(f"[{i}/{len(html_files)}] No non-empty sections for {article_id or page_name or path.name}")
                try:
                    if page_out_path.exists():
                        page_out_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete empty per-page file {page_out_path}: {e}")
            else:
                pages_processed += 1
                logger.info(
                    f"[{i}/{len(html_files)}] Parsed {article_id or page_name or path.name} "
                    f"({sec_count_for_page} sections) -> {page_out_path}"
                )

    logger.info("=== Parse summary ===")
    logger.info(f"Pages seen (unique):             {len(pages_seen)}")
    logger.info(f"Pages processed (with sections): {pages_processed}")
    logger.info(f"Pages skipped:                   {pages_skipped}")
    logger.info(f"Total sections written:          {sections_written}")
    logger.info(f"Master JSONL:                    {out_path}")
    logger.info(f"Per-page JSONLs dir:             {per_page_dir}")

    return out_path, per_page_dir


# ---------------------- ENTRYPOINT --------------------------


if __name__ == "__main__":
    cfg, project_root = load_scraping_config()
    logger, log_path = create_logger(project_root)

    start = time.time()
    logger.info("=== Starting 02_parse_html_to_sections ===")

    try:
        out_path, per_page_dir = parse_all_html(cfg, project_root, logger)
    except Exception as e:
        logger.error(f"Unhandled error in 02_parse_html_to_sections: {e}", exc_info=True)
        raise

    elapsed = time.time() - start
    logger.info(f"Finished 02_parse_html_to_sections in {elapsed:.2f} seconds")
    logger.info(f"Master sections file: {out_path}")
    logger.info(f"Per-page sections dir: {per_page_dir}")
    logger.info(f"Log file: {log_path}")

    print(f"Master sections written to: {out_path}")
    print(f"Per-page sections written under: {per_page_dir}")
    print(f"Log saved to: {log_path}")