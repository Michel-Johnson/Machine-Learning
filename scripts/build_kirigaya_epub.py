#!/usr/bin/env python3
"""Archive public posts from https://kirigaya.cn into EPUB files with images embedded.

Only posts the site returns without authentication are included. Private posts
(API code 4003, or isPrivate) are skipped. WebP images are transcoded to PNG or
JPEG so typical EPUB readers can display them; other images are embedded as
published when they are already a reasonable size.

Dependencies:
    python3 -m pip install --user ebooklib markdown beautifulsoup4 pillow
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from ebooklib import epub
from PIL import Image, ImageDraw, ImageFont

ORIGIN = "https://kirigaya.cn"
LIST_URL = ORIGIN + "/api/blog/range-query-blog-by-pageId?pageId={page}"
ARTICLE_URL = ORIGIN + "/api/blog/fetch-blog-by-seq?seq={seq}"
ARTICLE_PAGE = ORIGIN + "/blog/article?seq={seq}"
USER_AGENT = (
    "KirigayaOfflineArchive/1.0 "
    "(personal EPUB of public posts; respects private posts)"
)
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
SINGLE_FILE_LIMIT = 62 * 1024 * 1024
YEAR_FILE_LIMIT = 55 * 1024 * 1024

CSS = """
body {
  margin: 1.1em 1.15em 2em;
  line-height: 1.75;
  font-family: "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", serif;
  color: #1f2422;
}
h1, h2, h3, h4 { line-height: 1.35; font-family: "PingFang SC", "Noto Sans CJK SC", sans-serif; }
h1 { font-size: 1.55em; margin-bottom: 0.3em; }
a { color: #1d6a68; }
img { max-width: 100%; height: auto; }
p.meta, p.volume-note { color: #6b6560; font-size: 0.92em; }
p.cover { text-align: center; }
blockquote { margin: 1em 0; padding: 0.2em 0 0.2em 0.9em; border-left: 3px solid #55a2a0; color: #333; }
pre, code { font-family: "Sarasa Mono SC", "Source Code Pro", monospace; }
pre {
  background: #f6f3ee;
  padding: 0.75em 0.9em;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.45;
}
code { background: #f3efe8; }
pre code { background: transparent; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #d9d3c7; padding: 0.35em 0.5em; vertical-align: top; }
.math-block pre, .math-inline { font-family: "Sarasa Mono SC", monospace; }
.math-block pre { background: #f4f1ea; }
.missing-image { color: #8a3b32; font-size: 0.9em; }
ul, ol { padding-left: 1.3em; }
"""

FENCE_RE = re.compile(r"(```[\s\S]*?```)|(~~~[\s\S]*?~~~)")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
DISPLAY_MATH_RE = re.compile(r"\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]")
INLINE_MATH_RE = re.compile(
    r"(?<!\\)(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\\)\$(?!\$)|\\\((.+?)\\\)"
)
SEQ_LINK_RE = re.compile(
    r"(?:https?://(?:www\.)?kirigaya\.cn)?(?:/#)?/blog/article\?seq=(\d+)"
)
IMAGE_EXT_RE = re.compile(
    r"\.(?:png|jpe?g|gif|webp|svg|bmp|avif)(?:$|\?)", re.IGNORECASE
)
ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


class RateLimiter:
    def __init__(self, per_second: float) -> None:
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            time.sleep(delay)


LIMITER = RateLimiter(8)


def fetch_bytes(url: str, timeout: float = 40, retries: int = 4) -> tuple[bytes, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": ORIGIN + "/",
        "Accept": "*/*",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        LIMITER.wait()
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
                content_type = response.headers.get("Content-Type", "")
                if len(data) > MAX_IMAGE_BYTES:
                    raise RuntimeError(f"response larger than {MAX_IMAGE_BYTES} bytes")
                return data, content_type
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_error}")


def fetch_json(url: str) -> dict:
    data, _ = fetch_bytes(url, timeout=40, retries=4)
    return json.loads(data.decode("utf-8"))


def decode_article_text(text: str) -> str:
    raw = base64.b64decode(text)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    else:
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
    return raw.decode("utf-8")


def parse_time(value: str) -> datetime:
    match = re.search(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2}))?",
        value or "",
    )
    if not match:
        return datetime.min
    year, month, day, hour, minute = match.groups()
    return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0))


def collection_name(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return ""


def list_public_posts() -> tuple[list[dict], int]:
    posts: list[dict] = []
    seen: set[int] = set()
    private_count = 0
    page = 1
    while page <= 300:
        payload = fetch_json(LIST_URL.format(page=page))
        if payload.get("code") != 200:
            break
        rows = payload.get("data") or []
        real = [row for row in rows if row]
        if not real:
            break
        for row in real:
            seq = row.get("seq")
            if seq is None or seq in seen:
                continue
            seen.add(seq)
            if row.get("isPrivate") is True:
                private_count += 1
                continue
            posts.append(
                {
                    "seq": int(seq),
                    "name": row.get("name") or f"未命名 {seq}",
                    "adjustTime": row.get("adjustTime") or "",
                    "tags": row.get("tags") or [],
                    "imgUrl": row.get("imgUrl") or "",
                    "viewCount": row.get("viewCount") or 0,
                }
            )
        print(f"listed page {page}: {len(real)} posts", flush=True)
        if len(real) < 9:
            break
        page += 1
    posts.sort(key=lambda item: (parse_time(item["adjustTime"]), item["seq"]), reverse=True)
    return posts, private_count


def load_or_fetch_article(post: dict, cache_dir: Path) -> dict | None:
    path = cache_dir / "articles" / f"{post['seq']}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("markdown") is not None:
                return cached
        except json.JSONDecodeError:
            pass

    payload = fetch_json(ARTICLE_URL.format(seq=post["seq"]))
    code = payload.get("code")
    if code == 4003:
        return None
    if code != 200 or not payload.get("data"):
        raise RuntimeError(payload.get("msg") or f"code {code}")
    data = payload["data"]
    if data.get("isPrivate") is True:
        return None
    text = data.get("text") or ""
    markdown_text = decode_article_text(text) if text else ""
    article = {
        "seq": int(data.get("seq", post["seq"])),
        "name": data.get("name") or post["name"],
        "adjustTime": data.get("adjustTime") or post["adjustTime"],
        "tags": data.get("tags") or post.get("tags") or [],
        "imgUrl": data.get("imgUrl") or post.get("imgUrl") or "",
        "viewCount": data.get("viewCount") or post.get("viewCount") or 0,
        "collection": collection_name(data.get("collection")),
        "markdown": markdown_text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
    return article


def fetch_articles(posts: list[dict], cache_dir: Path, workers: int) -> tuple[list[dict], list[str]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    articles: dict[int, dict] = {}
    failures: list[str] = []
    print(f"fetching {len(posts)} public articles", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(load_or_fetch_article, post, cache_dir): post for post in posts
        }
        done = 0
        for future in as_completed(future_map):
            post = future_map[future]
            done += 1
            try:
                article = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"seq {post['seq']} {post['name']}: {exc}")
                print(f"FAIL {post['seq']}: {exc}", flush=True)
                continue
            if article is None:
                failures.append(f"seq {post['seq']} {post['name']}: 非公开，已跳过")
                continue
            articles[article["seq"]] = article
            if done % 25 == 0 or done == len(posts):
                print(f"articles {done}/{len(posts)}", flush=True)

    # One more sequential pass for failures that were transport errors.
    retry_posts = []
    still_failed = []
    for line in failures:
        match = re.match(r"seq (\d+) ", line)
        if not match:
            still_failed.append(line)
            continue
        seq = int(match.group(1))
        if seq in articles or "非公开" in line:
            if "非公开" in line:
                still_failed.append(line)
            continue
        retry_posts.append(next(post for post in posts if post["seq"] == seq))
    failures = still_failed
    for post in retry_posts:
        try:
            article = load_or_fetch_article(post, cache_dir)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"seq {post['seq']} {post['name']}: {exc}")
            continue
        if article:
            articles[article["seq"]] = article
    ordered = [articles[post["seq"]] for post in posts if post["seq"] in articles]
    return ordered, failures


def _math_block(latex: str) -> str:
    return (
        '\n\n<div class="math-block"><pre>'
        + html.escape(latex.strip())
        + "</pre></div>\n\n"
    )


def _math_inline(latex: str) -> str:
    return '<span class="math-inline">' + html.escape(latex.strip()) + "</span>"


def transform_markdown(source: str) -> str:
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    parts: list[str] = []
    last = 0
    for match in FENCE_RE.finditer(source):
        if match.start() > last:
            parts.append(_transform_prose(source[last : match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_transform_prose(source[last:]))
    return "".join(parts)


def _transform_prose(chunk: str) -> str:
    inlines: list[str] = []

    def hold_inline(match: re.Match[str]) -> str:
        inlines.append(match.group(0))
        return f"INLINECODEHOLDER{len(inlines) - 1}END"

    chunk = INLINE_CODE_RE.sub(hold_inline, chunk)

    def display(match: re.Match[str]) -> str:
        latex = match.group(1) if match.group(1) is not None else match.group(2)
        return _math_block(latex or "")

    def inline(match: re.Match[str]) -> str:
        latex = match.group(1) if match.group(1) is not None else match.group(2)
        return _math_inline(latex or "")

    chunk = DISPLAY_MATH_RE.sub(display, chunk)
    chunk = INLINE_MATH_RE.sub(inline, chunk)
    for index, code in enumerate(inlines):
        chunk = chunk.replace(f"INLINECODEHOLDER{index}END", code)
    return chunk


def markdown_to_soup(source: str) -> BeautifulSoup:
    transformed = transform_markdown(source)
    rendered = markdown.markdown(
        transformed,
        extensions=["extra", "sane_lists"],
        output_format="html",
    )
    rendered = ILLEGAL_XML_RE.sub("", rendered)
    return BeautifulSoup(f"<div id='root'>{rendered}</div>", "html.parser")


def normalize_url(src: str | None) -> str | None:
    if not src:
        return None
    src = src.strip().strip("\"'")
    if not src or src.startswith("#") or src.lower().startswith("javascript:"):
        return None
    if src.startswith("data:"):
        return src
    if src.startswith("//"):
        src = "https:" + src
    elif src.startswith("/"):
        src = ORIGIN + src
    elif not src.startswith(("http://", "https://")):
        src = urllib.parse.urljoin(ORIGIN + "/", src)
    parsed = urllib.parse.urlparse(src)
    if parsed.scheme not in ("http", "https"):
        return None
    return src


def is_image_href(href: str) -> bool:
    path = urllib.parse.urlparse(href).path
    return bool(IMAGE_EXT_RE.search(path))


def prepare_soup(article: dict, available_seqs: set[int]) -> BeautifulSoup:
    soup = markdown_to_soup(article.get("markdown") or "")
    root = soup.find(id="root")
    assert root is not None
    for tag in root.find_all(["script", "iframe", "object", "embed", "form", "style"]):
        tag.decompose()
    for tag in root.find_all(True):
        for attr in list(tag.attrs):
            if attr.lower().startswith("on"):
                del tag.attrs[attr]

    for anchor in list(root.find_all("a")):
        href = normalize_url(anchor.get("href"))
        if not href or href.startswith("data:"):
            continue
        match = SEQ_LINK_RE.search(href)
        if match and int(match.group(1)) in available_seqs:
            anchor["href"] = f"p{match.group(1)}.xhtml"
            continue
        if anchor.find("img") is None and is_image_href(href):
            text = anchor.get_text(strip=True)
            if text in {"", "图片", "image", "img", "封面"} or text == href or is_image_href(text):
                image = soup.new_tag("img", src=href, alt=text or "图片")
                anchor.replace_with(image)
                continue
        anchor["href"] = href

    for image in root.find_all("img"):
        src = image.get("data-src") or image.get("data-original") or image.get("src")
        normalized = normalize_url(src)
        if normalized:
            image["src"] = normalized
        if image.has_attr("srcset"):
            del image["srcset"]
    return soup


def iter_image_urls(soup: BeautifulSoup, cover_url: str | None) -> list[str]:
    urls: list[str] = []
    root = soup.find(id="root")
    for image in root.find_all("img") if root else []:
        src = image.get("src") or ""
        if src:
            urls.append(src)
    if cover_url and cover_url not in urls:
        urls.insert(0, cover_url)
    return urls


def sniff_kind(data: bytes, content_type: str, url: str) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    head = data[:200].lstrip().lower()
    if head.startswith(b"<svg") or b"<svg" in head:
        return "svg"
    if data[4:8] == b"ftyp":
        return "avif"
    lowered = (content_type or "").split(";")[0].lower()
    for label, kind in (
        ("png", "png"),
        ("jpeg", "jpeg"),
        ("jpg", "jpeg"),
        ("gif", "gif"),
        ("webp", "webp"),
        ("svg", "svg"),
    ):
        if label in lowered:
            return kind
    path = urllib.parse.urlparse(url).path.lower()
    for ext, kind in (
        (".png", "png"),
        (".jpg", "jpeg"),
        (".jpeg", "jpeg"),
        (".gif", "gif"),
        (".webp", "webp"),
        (".svg", "svg"),
    ):
        if path.endswith(ext):
            return kind
    return "unknown"


def uses_alpha(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA"):
        extrema = image.getextrema()
        alpha = extrema[-1]
        return alpha[0] < 255
    return False


def looks_like_diagram(image: Image.Image) -> bool:
    sample = image.convert("RGB").resize((64, 64), Image.Resampling.BOX)
    colors = sample.getcolors(64 * 64)
    if not colors:
        return False
    return len(colors) <= 1100


def resize_max(image: Image.Image, edge: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= edge:
        return image
    scale = edge / longest
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def save_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def save_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def encode_image(data: bytes, kind: str, profile: str) -> tuple[bytes, str, str] | None:
    """Return bytes, extension, media type. None if the payload is not an image."""
    max_edge = 2400 if profile == "faithful" else 1600
    jpeg_q = 92 if profile == "faithful" else 82
    passthrough = 900_000 if profile == "faithful" else 0

    if kind == "svg":
        if b"<svg" not in data[:500].lower() and b"<svg" not in data[:500]:
            return None
        if len(data) > 2_000_000:
            return None
        return data, "svg", "image/svg+xml"

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        if kind in {"jpeg", "png", "gif"} and len(data) < passthrough:
            media = {"jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif"}[kind]
            ext = "jpg" if kind == "jpeg" else kind
            return data, ext, media
        return None

    if getattr(image, "is_animated", False) and kind == "gif":
        return data, "gif", "image/gif"

    if (
        profile == "faithful"
        and kind in {"jpeg", "png", "gif"}
        and len(data) <= passthrough
        and max(image.size) <= max_edge
    ):
        media = {"jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif"}[kind]
        ext = "jpg" if kind == "jpeg" else kind
        return data, ext, media

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        if uses_alpha(rgba):
            png = save_png(resize_max(rgba, max_edge))
            return png, "png", "image/png"
        image = rgba.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    image = resize_max(image, max_edge)
    png = save_png(image)
    jpeg = save_jpeg(image, jpeg_q)
    if looks_like_diagram(image) and len(png) <= (2_500_000 if profile == "faithful" else 1_200_000):
        return png, "png", "image/png"
    if len(jpeg) <= len(png):
        return jpeg, "jpg", "image/jpeg"
    return png, "png", "image/png"


def image_cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return cache_dir / "images" / digest


def load_cached_image(cache_dir: Path, url: str) -> tuple[bytes, str] | None:
    path = image_cache_path(cache_dir, url)
    meta_path = path.with_suffix(".meta.json")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return path.read_bytes(), meta.get("content_type", "")
    return None


def store_cached_image(cache_dir: Path, url: str, data: bytes, content_type: str) -> None:
    path = image_cache_path(cache_dir, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_suffix(".meta.json").write_text(
        json.dumps({"url": url, "content_type": content_type}, ensure_ascii=False),
        encoding="utf-8",
    )


def download_image(cache_dir: Path, url: str) -> tuple[bytes, str]:
    cached = load_cached_image(cache_dir, url)
    if cached:
        return cached
    if url.startswith("data:"):
        header, payload = url.split(",", 1)
        if ";base64" in header:
            data = base64.b64decode(payload)
        else:
            data = urllib.parse.unquote_to_bytes(payload)
        content_type = header[5:].split(";")[0] if header.startswith("data:") else ""
    else:
        data, content_type = fetch_bytes(url, timeout=40, retries=3)
    if not data:
        raise RuntimeError("empty image")
    store_cached_image(cache_dir, url, data, content_type)
    return data, content_type


def download_all_images(
    urls: list[str], cache_dir: Path, workers: int
) -> tuple[dict[str, tuple[bytes, str]], dict[str, str]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique = []
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)
    blobs: dict[str, tuple[bytes, str]] = {}
    errors: dict[str, str] = {}
    print(f"downloading {len(unique)} images", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(download_image, cache_dir, url): url for url in unique}
        done = 0
        for future in as_completed(future_map):
            url = future_map[future]
            done += 1
            try:
                blobs[url] = future.result()
            except Exception as exc:  # noqa: BLE001
                errors[url] = str(exc)
            if done % 40 == 0 or done == len(unique):
                print(f"images {done}/{len(unique)} ok={len(blobs)} fail={len(errors)}", flush=True)
    return blobs, errors


def encode_all(
    blobs: dict[str, tuple[bytes, str]], profile: str
) -> dict[str, dict]:
    encoded: dict[str, dict] = {}
    for url, (data, content_type) in blobs.items():
        kind = sniff_kind(data, content_type, url)
        result = encode_image(data, kind, profile)
        if result is None:
            continue
        body, ext, media = result
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        encoded[url] = {
            "bytes": body,
            "ext": ext,
            "media": media,
            "href": f"images/{digest}.{ext}",
            "kind": kind,
        }
    return encoded


def apply_images(soup: BeautifulSoup, article: dict, encoded: dict[str, dict], errors: dict[str, str]) -> str:
    root = soup.find(id="root")
    assert root is not None
    cover = normalize_url(article.get("imgUrl"))
    body_srcs = [img.get("src") for img in root.find_all("img")]

    for image in list(root.find_all("img")):
        src = image.get("src") or ""
        if src in encoded:
            image["src"] = encoded[src]["href"]
            if not image.get("alt"):
                image["alt"] = "图片"
            continue
        alt = image.get("alt") or "图片"
        note = soup.new_tag("p")
        note["class"] = "missing-image"
        reason = errors.get(src, "无法嵌入")
        note.append(f"[图片未能嵌入：{alt}] {reason}")
        if src.startswith("http"):
            link = soup.new_tag("a", href=src)
            link.string = "原始地址"
            note.append(" ")
            note.append(link)
        image.replace_with(note)

    if cover and cover not in body_srcs:
        if cover in encoded:
            cover_p = soup.new_tag("p")
            cover_p["class"] = "cover"
            cover_img = soup.new_tag("img")
            cover_img["src"] = encoded[cover]["href"]
            cover_img["alt"] = "封面"
            cover_p.append(cover_img)
            root.insert(0, cover_p)
        elif cover in errors:
            note = soup.new_tag("p")
            note["class"] = "missing-image"
            note.append(f"[封面未能嵌入] {errors[cover]}")
            root.insert(0, note)

    inner = root.decode_contents()
    return ILLEGAL_XML_RE.sub("", inner)


def chapter_html(article: dict, body_html: str) -> str:
    title = article["name"]
    tags = "、".join(article.get("tags") or [])
    when = article.get("adjustTime") or ""
    collection = article.get("collection") or ""
    source = ARTICLE_PAGE.format(seq=article["seq"])
    meta_bits = [html.escape(when)]
    if tags:
        meta_bits.append(html.escape(tags))
    if collection:
        meta_bits.append("合集：" + html.escape(collection))
    meta = " · ".join(bit for bit in meta_bits if bit)
    return (
        "<html><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='meta'>{meta}<br/>原文：<a href='{html.escape(source)}'>{html.escape(source)}</a></p>"
        f"{body_html}"
        "</body></html>"
    )


def make_cover(title: str, subtitle: str, footer: str) -> bytes:
    image = Image.new("RGB", (1400, 1960), "#faf8f2")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1400, 28), fill="#428381")
    draw.rectangle((0, 1932, 1400, 1960), fill="#428381")
    font_big = ImageFont.truetype(FONT_PATH, 92)
    font_mid = ImageFont.truetype(FONT_PATH, 48)
    font_small = ImageFont.truetype(FONT_PATH, 32)
    draw.text((120, 720), title, font=font_big, fill="#1f2a28")
    draw.text((120, 860), subtitle, font=font_mid, fill="#3d4a46")
    draw.text((120, 1680), footer, font=font_small, fill="#6b6560")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def preface_html(volume_label: str, articles: list[dict], private_count: int, failures: list[str], image_note: str, missing_images: int) -> str:
    items = []
    current_year = None
    for article in articles:
        year = str(parse_time(article["adjustTime"]).year) if parse_time(article["adjustTime"]) != datetime.min else "日期未知"
        if year != current_year:
            if current_year is not None:
                items.append("</ul>")
            items.append(f"<h2>{html.escape(year)}</h2><ul>")
            current_year = year
        items.append(
            f"<li><a href='p{article['seq']}.xhtml'>{html.escape(article['adjustTime'])} {html.escape(article['name'])}</a></li>"
        )
    if current_year is not None:
        items.append("</ul>")
    fail_html = ""
    if failures:
        rows = "".join(f"<li>{html.escape(line)}</li>" for line in failures[:80])
        fail_html = f"<h2>未能收录</h2><ul>{rows}</ul>"
    return (
        "<html><body>"
        "<h1>关于这本离线书</h1>"
        f"<p>{html.escape(volume_label)}</p>"
        "<p>正文来自锦恢的公开博客「汇尘轩」（<a href='https://kirigaya.cn/home'>https://kirigaya.cn/home</a>）。"
        "文章与图片的著作权归原作者锦恢（LSTM-Kirigaya）所有。这里只收录站点未登录即可阅读的文章，"
        f"私密或无权限文章已跳过（列表中约 {private_count} 篇）。抓取日期：2026-10-01。</p>"
        f"<p>{html.escape(image_note)} 未能嵌入的图片：{missing_images} 处。</p>"
        "<p>数学公式保留 LaTeX 原文，方便对照。文内指向本站其他公开文章的链接会转到对应章节。</p>"
        + "".join(items)
        + fail_html
        + "</body></html>"
    )


def build_book(
    articles: list[dict],
    bodies: dict[int, str],
    encoded: dict[str, dict],
    *,
    title: str,
    subtitle: str,
    footer: str,
    identifier: str,
    volume_label: str,
    private_count: int,
    failures: list[str],
    image_note: str,
    missing_images: int,
) -> epub.EpubBook:
    book = epub.EpubBook()
    book.set_identifier(identifier)
    book.set_title(title)
    book.set_language("zh")
    book.add_author("锦恢")
    book.add_metadata("DC", "source", ORIGIN + "/home")
    book.add_metadata(
        "DC",
        "rights",
        "文章与图片版权归原作者锦恢（LSTM-Kirigaya）所有。本文件只收录 kirigaya.cn 上公开可见的内容，供离线阅读。",
    )
    book.add_metadata("DC", "date", "2026-10-01")
    book.add_metadata("DC", "description", "汇尘轩公开文章离线合集，图片已嵌入。")
    book.set_cover("cover.jpg", make_cover("汇尘轩", subtitle, footer), create_page=True)

    style = epub.EpubItem(
        uid="style-main",
        file_name="style/main.css",
        media_type="text/css",
        content=CSS,
    )
    book.add_item(style)

    used_hrefs = set()
    for article in articles:
        body = bodies[article["seq"]]
        for match in re.findall(r"""src=['"](images/[^'"]+)['"]""", body):
            used_hrefs.add(match)
    href_to_item = {}
    for info in encoded.values():
        if info["href"] not in used_hrefs or info["href"] in href_to_item:
            continue
        item = epub.EpubImage()
        item.file_name = info["href"]
        item.media_type = info["media"]
        item.content = info["bytes"]
        book.add_item(item)
        href_to_item[info["href"]] = item

    preface = epub.EpubHtml(
        uid="preface",
        title="关于这本离线书",
        file_name="preface.xhtml",
        lang="zh",
    )
    preface.content = preface_html(
        volume_label, articles, private_count, failures, image_note, missing_images
    )
    preface.add_item(style)
    book.add_item(preface)

    chapters = []
    for article in articles:
        chapter = epub.EpubHtml(
            uid=f"post-{article['seq']}",
            title=article["name"],
            file_name=f"p{article['seq']}.xhtml",
            lang="zh",
        )
        chapter.content = chapter_html(article, bodies[article["seq"]])
        chapter.add_item(style)
        book.add_item(chapter)
        chapters.append(chapter)

    by_year: dict[str, list] = defaultdict(list)
    for article, chapter in zip(articles, chapters):
        moment = parse_time(article["adjustTime"])
        year = str(moment.year) if moment != datetime.min else "日期未知"
        by_year[year].append(chapter)
    toc = [preface]
    for year in sorted(by_year, reverse=True):
        toc.append((epub.Section(year), tuple(by_year[year])))
    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", preface, *chapters]
    return book


def write_book(path: Path, book: epub.EpubBook) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book, {})


def article_image_bytes(article_urls: list[str], encoded: dict[str, dict], already: set[str]) -> int:
    total = 0
    for url in article_urls:
        info = encoded.get(url)
        if not info or info["href"] in already:
            continue
        already.add(info["href"])
        total += len(info["bytes"])
    return total


def split_articles(
    articles: list[dict], urls_by_seq: dict[int, list[str]], encoded: dict[str, dict], limit: int
) -> list[list[dict]]:
    groups: list[list[dict]] = []
    by_year: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        moment = parse_time(article["adjustTime"])
        year = str(moment.year) if moment != datetime.min else "unknown"
        by_year[year].append(article)

    def pack(chunk: list[dict]) -> list[list[dict]]:
        volumes: list[list[dict]] = []
        current: list[dict] = []
        seen: set[str] = set()
        size = 0
        for article in chunk:
            extra = article_image_bytes(urls_by_seq.get(article["seq"], []), encoded, set(seen))
            extra += len(article.get("markdown") or "") + 2000
            if current and size + extra > limit:
                volumes.append(current)
                current = []
                seen = set()
                size = 0
                extra = article_image_bytes(urls_by_seq.get(article["seq"], []), encoded, seen)
                extra += len(article.get("markdown") or "") + 2000
            current.append(article)
            seen.update(
                encoded[url]["href"]
                for url in urls_by_seq.get(article["seq"], [])
                if url in encoded
            )
            size += extra
        if current:
            volumes.append(current)
        return volumes

    for year in sorted(by_year, reverse=True):
        groups.extend(pack(by_year[year]))
    return groups


def verify_epub(path: Path) -> dict:
    report = {"path": str(path), "chapters": 0, "images": 0, "remote_imgs": 0, "broken_refs": 0, "unreadable": 0}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        image_names = [name for name in names if name.startswith("images/")]
        report["images"] = len(image_names)
        for name in image_names:
            data = archive.read(name)
            if name.endswith(".svg"):
                if b"<svg" not in data[:800].lower() and b"<svg" not in data[:800]:
                    report["unreadable"] += 1
                continue
            try:
                image = Image.open(io.BytesIO(data))
                image.verify()
            except Exception:
                report["unreadable"] += 1
        for name in names:
            if not name.endswith(".xhtml") or name in {"nav.xhtml", "cover.xhtml"}:
                continue
            if name == "preface.xhtml":
                continue
            report["chapters"] += 1
            text = archive.read(name).decode("utf-8", errors="replace")
            for src in re.findall(r"""<img[^>]+src=['"]([^'"]+)['"]""", text):
                if src.startswith(("http://", "https://")):
                    report["remote_imgs"] += 1
                elif src not in names and not src.startswith("data:"):
                    report["broken_refs"] += 1
    report["bytes"] = path.stat().st_size
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an EPUB of public kirigaya.cn posts")
    parser.add_argument("--out", type=Path, default=Path("epub"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/kirigaya_cache"))
    parser.add_argument("--limit", type=int, default=0, help="Only the newest N public posts; 0 means all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    posts, private_count = list_public_posts()
    if args.limit:
        posts = posts[: args.limit]
    articles, failures = fetch_articles(posts, args.cache, args.workers)
    available = {article["seq"] for article in articles}
    print(f"ready {len(articles)} articles, private skipped about {private_count}", flush=True)

    soups = {}
    all_urls: list[str] = []
    urls_by_seq: dict[int, list[str]] = {}
    for article in articles:
        soup = prepare_soup(article, available)
        soups[article["seq"]] = soup
        cover = normalize_url(article.get("imgUrl"))
        urls = iter_image_urls(soup, cover)
        urls_by_seq[article["seq"]] = urls
        all_urls.extend(urls)

    blobs, image_errors = download_all_images(all_urls, args.cache, max(args.workers, 6))
    profile = "faithful"
    encoded = encode_all(blobs, profile)
    raw_total = sum(len(info["bytes"]) for info in encoded.values())
    print(f"encoded images {len(encoded)} bytes {raw_total} errors {len(image_errors)}", flush=True)
    if raw_total > SINGLE_FILE_LIMIT:
        profile = "compact"
        encoded = encode_all(blobs, profile)
        raw_total = sum(len(info["bytes"]) for info in encoded.values())
        print(f"re-encoded compact bytes {raw_total}", flush=True)

    bodies = {}
    missing_images = 0
    for article in articles:
        body = apply_images(soups[article["seq"]], article, encoded, image_errors)
        bodies[article["seq"]] = body
        missing_images += body.count("missing-image")

    image_note = (
        "正文图片和封面已嵌入本书。WebP 已转为 PNG 或 JPEG，方便阅读器显示，画面仍是原文图片。"
        + ("" if profile == "faithful" else "文件较大时另做了压缩。")
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("kirigaya-blog*.epub"):
        old.unlink()

    if raw_total <= SINGLE_FILE_LIMIT:
        groups = [articles]
    else:
        groups = split_articles(articles, urls_by_seq, encoded, YEAR_FILE_LIMIT)

    reports = []
    total = len(groups)
    for index, group in enumerate(groups, start=1):
        if total == 1:
            filename = "kirigaya-blog.epub"
            title = "汇尘轩：锦恢的博客"
            subtitle = f"公开文章 {len(group)} 篇"
            footer = "来源 kirigaya.cn"
            volume_label = f"这是公开文章的完整离线合集，共 {len(group)} 篇。"
            identifier = "kirigaya-cn-public-blog-2026-10-01"
        else:
            moment_values = [parse_time(item["adjustTime"]) for item in group]
            years = sorted({item.year for item in moment_values if item != datetime.min})
            year_label = str(years[0]) if len(years) == 1 else "多年度"
            filename = f"kirigaya-blog-{index:02d}-{year_label}.epub"
            title = f"汇尘轩：锦恢的博客（{index}/{total}）"
            subtitle = f"{year_label} · {len(group)} 篇"
            footer = f"第 {index} 卷 / 共 {total} 卷"
            volume_label = (
                f"这是公开文章离线合集的第 {index} 卷，共 {total} 卷，本卷 {len(group)} 篇。"
                "拆成多卷是为了让每个文件都能带上原图。"
            )
            identifier = f"kirigaya-cn-public-blog-2026-10-01-{index}"
        group_failures = failures if index == 1 else []
        group_private = private_count if index == 1 else 0
        book = build_book(
            group,
            bodies,
            encoded,
            title=title,
            subtitle=subtitle,
            footer=footer,
            identifier=identifier,
            volume_label=volume_label,
            private_count=group_private,
            failures=group_failures,
            image_note=image_note,
            missing_images=missing_images if index == 1 else 0,
        )
        path = args.out / filename
        write_book(path, book)
        report = verify_epub(path)
        reports.append(report)
        print("WROTE", report, flush=True)

    print(json.dumps({"profile": profile, "articles": len(articles), "reports": reports}, ensure_ascii=False, indent=2))
    image_total = sum(report["images"] for report in reports)
    bad = [
        report
        for report in reports
        if report["broken_refs"] or report["remote_imgs"] or report["unreadable"] or report["chapters"] == 0
    ]
    if all_urls and image_total == 0:
        bad.append({"error": "no embedded images"})
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
