"""Scrape ~500 recent news articles: RSS feeds give the links, trafilatura pulls the article body.

Steps: read every feed in FEEDS -> interleave links source by source (so no outlet dominates)
-> download pages with a small thread pool -> keep articles with >= MIN_WORDS words
-> drop duplicate URLs and titles -> number them A0001... -> save to config.ARTICLES_CSV.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import pandas as pd
import requests
import trafilatura

from . import config

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT = 20  # seconds per request
DELAY = 0.5  # pause after each page download (per worker), to be polite
WORKERS = 6
MIN_WORDS = 150
MAX_PER_SOURCE = 50  # keeps one big feed from dominating the corpus
COLUMNS = ["article_id", "source", "url", "title", "date", "text"]

# Several sections per outlet: world, politics, business, tech, science and sport.
FEEDS = {
    "BBC": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://feeds.bbci.co.uk/sport/rss.xml",
    ],
    "The Guardian": [
        "https://www.theguardian.com/world/rss",
        "https://www.theguardian.com/us-news/rss",
        "https://www.theguardian.com/politics/rss",
        "https://www.theguardian.com/uk/business/rss",
        "https://www.theguardian.com/uk/technology/rss",
        "https://www.theguardian.com/science/rss",
        "https://www.theguardian.com/environment/rss",
        "https://www.theguardian.com/uk/sport/rss",
    ],
    "Al Jazeera": ["https://www.aljazeera.com/xml/rss/all.xml"],
    "NPR": [
        "https://feeds.npr.org/1001/rss.xml",
        "https://feeds.npr.org/1006/rss.xml",
        "https://feeds.npr.org/1007/rss.xml",
        "https://feeds.npr.org/1014/rss.xml",
        "https://feeds.npr.org/1019/rss.xml",
    ],
    "CBS News": [
        "https://www.cbsnews.com/latest/rss/world",
        "https://www.cbsnews.com/latest/rss/politics",
        "https://www.cbsnews.com/latest/rss/moneywatch",
        "https://www.cbsnews.com/latest/rss/technology",
        "https://www.cbsnews.com/latest/rss/science",
    ],
    "ABC News": [
        "https://abcnews.go.com/abcnews/internationalheadlines",
        "https://abcnews.go.com/abcnews/politicsheadlines",
        "https://abcnews.go.com/abcnews/technologyheadlines",
        "https://abcnews.go.com/abcnews/topstories",
    ],
    "DW": ["https://rss.dw.com/rdf/rss-en-all"],
    "The Independent": [
        "https://www.independent.co.uk/news/world/rss",
        "https://www.independent.co.uk/sport/rss",
    ],
    "Euronews": ["https://www.euronews.com/rss"],
    "Fox News": [
        "https://moxie.foxnews.com/google-publisher/world.xml",
        "https://moxie.foxnews.com/google-publisher/politics.xml",
        "https://moxie.foxnews.com/google-publisher/sports.xml",
    ],
    "PBS NewsHour": ["https://www.pbs.org/newshour/feeds/rss/headlines"],
    "CNBC": ["https://www.cnbc.com/id/100003114/device/rss/rss.html"],
    "TechCrunch": ["https://techcrunch.com/feed/"],
    "Ars Technica": ["https://feeds.arstechnica.com/arstechnica/index"],
    "ScienceDaily": ["https://www.sciencedaily.com/rss/all.xml"],
    "Science News": ["https://www.sciencenews.org/feed"],
    "Space.com": ["https://www.space.com/feeds/all"],
}

# Links that are not text articles (videos, live blogs, galleries, podcasts).
SKIP_URL = re.compile(
    r"/(videos?|live|live-updates|liveblog|audio|av|gallery|galleries|podcasts?|"
    r"pictures|in-pictures|sounds)/",
    re.I,
)


def _get(url):
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)


TRACKING = re.compile(r"^(utm_|at_|traffic_source|maca|cmpid|ito|ref)", re.I)


def _norm_url(url):
    """Drop tracking tags (e.g. ?at_medium=RSS) and the fragment; keep real ids like ?id=123."""
    p = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(p.query) if not TRACKING.match(k)])
    return urlunsplit(("https", p.netloc.lower(), p.path.rstrip("/"), query, ""))


def _norm_title(title):
    return re.sub(r"[^a-z0-9 ]", "", str(title).lower()).strip()


def _feed_date(entry):
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    return time.strftime("%Y-%m-%d", t) if t else ""


# Site boilerplate. Lines matching DROP_LINE are removed;
# everything from a CUT_FROM line on is removed.
DROP_LINE = re.compile(
    r"^(- )?(Published|Updated|Share|Date:|Source:|Summary:|Share:|___+|Join today"
    r"|Already a member\?.*"
    r"|Become an Independent member.*|Want to bookmark your favourite articles.*"
    r"|.*\b(sign up (for|to)|subscribe to|our newsletter|we use cookies|cookie (policy"
    r"|settings)|create an account)\b.*"
    r"|Accuracy and availability of transcripts.*|Copyright ©.*|\(SOUNDBITE OF [^)]*\)"
    r"|Leave your feedback.*|Share Copy URL.*|Your support helps make our show possible.*"
    r"|This article was originally (written|published) in \w+\.?|Edited by:.*|Related:.*"
    r"|Read more:.*"
    r"|Do you have an opinion on the issues raised in this article\?.*|What you need to know"
    r"|You must confirm your public display name.*|Please logout and then login again.*"
    r"|Want great stories delivered.*|- (Stream|TV): .*"
    r"|Good morning\. You're reading the .* newsletter.*"
    r"|This newsletter was edited by.*|A version of this article first appeared in .*newsletter.*"
    r"|This article contains content provided by .*)$",
    re.I,
)
CUT_FROM = re.compile(
    r"^(Story Source:|Journal Reference:|Cite This Page:|Related Stories?:?)$", re.I
)
AFTER_LABEL = re.compile(r"^- (Date|Source):$")  # ScienceDaily: the next line is the label's value
# PBS pages come out as one block with share/donate text glued to the story.
INLINE = [
    re.compile(r"^By — .*?Share Copy URL \S+\s*|Leave your feedback\s*"),
    re.compile(r"^Email Facebook Twitter .*? [AP]M E[SD]T\s*"),
    re.compile(r"\s*A free press is a cornerstone of a healthy democracy\..*$", re.S),
]
# Author bio at the end of the page, e.g. "Jane Doe is a staff writer for Space.com ..."
BIO = re.compile(
    r"^[A-Z][\w.'’ ()/,-]{2,60}? (is|was|joined|served as)\b.{0,80}\b(writer|editor|journalist|"
    r"reporter|correspondent|contributor|Space\.com)\b",
    re.I,
)
LONG_PROMO = (
    "Accuracy",
    "Copyright",
    "Become",
    "This article contains",
)  # promo lines that can be long
LEAD_IN = re.compile(r"^Updated on:.*?(EDT|EST|GMT|UTC)\s*/?\s*(CBS News|CBS/AP|CBS)?\s*", re.I)


def clean_text(text, title=""):
    """Remove site boilerplate, collapse spaces, keep one newline between paragraphs."""
    lines, skip_next = [], False
    text = str(text)
    for pattern in INLINE:
        text = pattern.sub("", text)
    for p in text.split("\n"):
        p = LEAD_IN.sub("", re.sub(r"\s+", " ", p).strip())
        if skip_next or not p:
            skip_next = False
            continue
        if CUT_FROM.match(p):
            break
        skip_next = bool(AFTER_LABEL.match(p))
        # long paragraphs that mention "newsletter" etc. are real text;
        # only drop short promo lines
        if DROP_LINE.match(p) and (len(p) < 200 or p.startswith(LONG_PROMO)):
            continue
        lines.append(p)
    while lines and BIO.match(lines[-1]):
        lines.pop()
    if lines and title and _norm_title(lines[0]) == _norm_title(title):
        lines = lines[1:]  # many sites repeat the headline as the first line
    if lines:
        lines[0] = lines[0].removeprefix("- ")  # ScienceDaily summary bullet
    return "\n".join(dict.fromkeys(lines))  # also drops exact repeated paragraphs


def collect_links():
    """Read all feeds. Returns candidates (source, url, title, date), interleaved by source."""

    def read(source, url):
        try:
            entries = feedparser.parse(_get(url).content).entries
        except Exception as e:  # one dead feed should not stop the scrape
            print(f"  feed failed: {url} ({e.__class__.__name__})")
            return []
        return [
            {
                "source": source,
                "url": e.link,
                "title": clean_text(e.get("title", "")),
                "date": _feed_date(e),
            }
            for e in entries
            if e.get("link")
        ]

    jobs = [(s, u) for s, urls in FEEDS.items() for u in urls]
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda j: read(*j), jobs))

    per_source = {s: [] for s in FEEDS}
    for (source, _), entries in zip(jobs, results):
        per_source[source].append(entries)
    # Within a source, alternate sections; then alternate sources.
    by_source = [
        [e for grp in zip_longest(*feeds) for e in grp if e] for feeds in per_source.values()
    ]
    ordered = [e for grp in zip_longest(*by_source) for e in grp if e]

    seen_url, seen_title, out = set(), set(), []
    for c in ordered:
        key, tkey = _norm_url(c["url"]), _norm_title(c["title"])
        if key in seen_url or tkey in seen_title or SKIP_URL.search(urlsplit(c["url"]).path):
            continue
        seen_url.add(key)
        seen_title.add(tkey)
        out.append(c)
    return out


def _extract(html_bytes, url, precision):
    """trafilatura body + metadata as a dict ({} if nothing found).

    Bytes let it detect the encoding.
    """
    raw = trafilatura.extract(
        html_bytes,
        url=url,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        include_tables=False,
        favor_precision=precision,
    )
    return json.loads(raw) if raw else {}


def fetch_article(cand):
    """Download one page and extract its body. Returns an article dict, or None if unusable."""
    try:
        resp = _get(cand["url"])
        time.sleep(DELAY)
        if resp.status_code != 200:
            return None
        meta = _extract(resp.content, resp.url, precision=True)
        if len((meta.get("text") or "").split()) < MIN_WORDS:  # precision mode misses some layouts
            meta = _extract(resp.content, resp.url, precision=False)
    except Exception:
        return None
    text = clean_text(meta.get("text") or "", cand["title"])
    if len(text.split()) < MIN_WORDS:
        return None
    date = cand["date"] or (meta.get("date") or "")[:10]
    return {
        "source": cand["source"],
        "url": _norm_url(cand["url"]),
        "title": cand["title"] or clean_text(meta.get("title") or ""),
        "date": date,
        "text": text,
    }


def scrape_news(n=config.N_ARTICLES, batch=60):
    """Scrape about n articles (stops at n good ones) and save them to ARTICLES_CSV."""
    cands = collect_links()
    print(f"{len(cands)} candidate links from {sum(map(len, FEEDS.values()))} feeds")
    rows, seen_url, seen_title, seen_text = [], set(), set(), set()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i in range(0, len(cands), batch):
            for art in ex.map(fetch_article, cands[i : i + batch]):
                if art is None:
                    continue
                keys = (_norm_url(art["url"]), _norm_title(art["title"]), hash(art["text"]))
                if keys[0] in seen_url or keys[1] in seen_title or keys[2] in seen_text:
                    continue  # redirects / syndicated copies of the same story
                if sum(r["source"] == art["source"] for r in rows) >= MAX_PER_SOURCE:
                    continue
                seen_url.add(keys[0])
                seen_title.add(keys[1])
                seen_text.add(keys[2])
                rows.append(art)
            print(f"  checked {min(i + batch, len(cands))}/{len(cands)} links, kept {len(rows)}")
            if len(rows) >= n:
                break
    df = pd.DataFrame(rows[:n])
    df.insert(0, "article_id", [f"A{i + 1:04d}" for i in range(len(df))])
    df = df[COLUMNS]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ARTICLES_CSV, index=False)
    print(f"saved {len(df)} articles to {config.ARTICLES_CSV}")
    return df


def load_articles():
    """Read the saved corpus (scrapes it first if the CSV does not exist)."""
    if not config.ARTICLES_CSV.exists():
        return scrape_news()
    return pd.read_csv(config.ARTICLES_CSV, dtype=str).fillna("")


def corpus_summary(df):
    """Articles per source with word-length stats, plus an 'All' row."""
    d = df.assign(words=df["text"].str.split().str.len())
    agg = dict(
        articles=("article_id", "count"),
        mean_words=("words", "mean"),
        median_words=("words", "median"),
        min_words=("words", "min"),
        max_words=("words", "max"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    table = d.groupby("source").agg(**agg).sort_values("articles", ascending=False)
    table.loc["All"] = d.assign(source="All").groupby("source").agg(**agg).iloc[0]
    return table.round(0).astype(
        {
            "articles": int,
            "mean_words": int,
            "median_words": int,
            "min_words": int,
            "max_words": int,
        }
    )
