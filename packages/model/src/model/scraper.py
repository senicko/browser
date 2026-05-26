import csv
import datetime
import json
import logging
import os
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .tokenizer import StemTokenizer

import joblib
import requests
from schema import Document

from .paths import ARTIFACTS_DIR, MODEL_PACKAGE_ROOT

API_BASE_URL = "https://en.wikipedia.org/w/api.php"

MIN_CHUNK_TOKENS = 20

_END_SECTION_RE = re.compile(
    r"^=+\s*(References|Notes|Citations|External links|See also|Further reading|Bibliography)\s*=+\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_HEADER_LINE_RE = re.compile(r"^=+\s*[^=\n]+?\s*=+\s*$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _clean_article_text(text: str) -> str:
    """
    Removes wikitext-style headers and truncates the article at common
    end-matter sections (References, External links, See also, ...).
    """

    match = _END_SECTION_RE.search(text)
    if match:
        text = text[: match.start()]

    text = _HEADER_LINE_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text).strip()
    return text


def fetch_wikipedia_articles(
    titles: list[str],
    logger: logging.Logger,
    response_cache_file: str | Path | None = None,
) -> list[dict[str, str]]:
    """
    Fetches wikipedia articles.
    """

    cache = None
    raw_articles: list[dict[str, str]] = []

    if response_cache_file:
        os.makedirs(os.path.dirname(response_cache_file), exist_ok=True)
        cache = open(response_cache_file, "w", encoding="utf-8")

    if cache:
        cache.write("[\n")

    first_entry = True

    for title in titles:
        logger.debug(f"fetching: {title}")

        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "redirects": 1,
        }

        while True:
            response = requests.get(
                API_BASE_URL,
                params=params,
                headers={
                    "User-Agent": "CultureArticlesDownloader/0.0 (class-step-dipping@duck.com)"
                },
            )

            if response.status_code == 200:
                body = response.json()

                if "errors" in body:
                    logger.error(f"errors: {body['errors']}")
                    break

                if "warnings" in body:
                    logger.warning(f"warnings: {body['warnings']}")

                pages = body.get("query", {}).get("pages", {})

                for page_id, page_data in pages.items():
                    if page_id != "-1":
                        raw_articles.append(
                            {
                                "title": page_data.get("title"),
                                "text": page_data.get("extract", ""),
                            }
                        )

                        if cache:
                            if not first_entry:
                                cache.write(",\n")
                            json.dump(raw_articles[-1], cache)

                        first_entry = False

                # Wikipedia API may return continue for pagination.
                # It shouldn't happen when downloading articles one by one though.
                if "continue" in body:
                    print("pagination limit reached")
                    params.update(body["continue"])
                else:
                    break
            elif response.status_code in (429, 502, 503, 504):
                logger.debug(f"failed with status {response.status_code}")
                raw_cooldown = response.headers.get("Retry-After", "10")

                try:
                    # Handle standard cooldown
                    cooldown = float(raw_cooldown)
                except ValueError:
                    # Handle date-string cooldown
                    try:
                        retry_date = parsedate_to_datetime(raw_cooldown)
                        now_utc = datetime.datetime.now(datetime.timezone.utc)

                        time_delta = retry_date - now_utc
                        cooldown = time_delta.total_seconds()

                        if cooldown < 0:
                            cooldown = 10
                    except (TypeError, ValueError):
                        cooldown = 10

                logger.debug(f"rate limited: cooldown={cooldown:.2f}")
                time.sleep(cooldown)
            else:
                logger.error(f"unrecoverable status {response.status_code}")
                break

    if cache:
        cache.write("\n]")
        cache.close()

    return raw_articles


def load_titles(titles_csv_file: str | Path) -> list[str]:
    """
    Loads corpus titles as a list of strings.
    """

    titles = []

    with open(titles_csv_file, newline="", encoding="utf-8") as articles_csv:
        articles_reader = csv.reader(articles_csv, delimiter=",")
        next(articles_reader, None)

        for row in articles_reader:
            titles.append(row[1])

    return titles


def build_corpus( 
    logger: logging.Logger,
    titles_csv: str | Path = "culture_petscan.csv",
    corpus_save_to: str | Path = ARTIFACTS_DIR / "corpus.pkl",
    raw_articles_cache: str | Path = ARTIFACTS_DIR / "culture_articles_cache.json",
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> list[Document]:
    """
    Builds a new corpus from cached articles. Allows to download articles 
    in case they aren't available locally.
    """

    if raw_articles_cache.exists():
        with open(raw_articles_cache, "r", encoding="utf-8") as f:
            raw_articles = json.load(f)
    else:
        input("You don't have any cached wikipedia articles. Do you want to start downloading the corpus?")
        titles = load_titles(MODEL_PACKAGE_ROOT / titles_csv) 
        raw_articles = fetch_wikipedia_articles(
            titles=titles,
            response_cache_file=raw_articles_cache,
            logger=logger,
        )

    tokenizer = StemTokenizer()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=tokenizer.token_count,
        is_separator_regex=False
    )

    documents: list[Document] = []
    truncated_articles = 0
    dropped_chunks = 0

    for i, article in enumerate(raw_articles):
        title = article.get("title", "")
        raw_text = article.get("text", "")
        text = _clean_article_text(raw_text)

        if len(text) < len(raw_text):
            truncated_articles += 1

        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            if tokenizer.token_count(chunk) < min_chunk_tokens:
                dropped_chunks += 1
                continue

            documents.append(
                Document(
                    title=title,
                    text=chunk
                )
            )

        if i % 100 == 0:
            logger.debug(f"processed {i} articles")

    logger.debug(
        f"cleaned {truncated_articles} articles; dropped {dropped_chunks} "
        f"chunks below {min_chunk_tokens} tokens"
    )

    corpus_save_to.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(documents, corpus_save_to)

    return documents


def load_corpus(
    corpus_path: str | Path = ARTIFACTS_DIR / "corpus.pkl",
    raw_articles_cache: str | Path = ARTIFACTS_DIR / "culture_articles_cache.json",
) -> list[Document]:
    """
    Loads corpus as a list of Article object instances.
    """

    corpus_path = Path(corpus_path)
    raw_articles_cache = Path(raw_articles_cache)

    if corpus_path.exists():
        return joblib.load(corpus_path)

    return None