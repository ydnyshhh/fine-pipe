from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import httpx
import tldextract
import trafilatura
from datasketch import MinHash, MinHashLSH
from readability import Document as ReadabilityDocument
from rich.console import Console

from typing import Optional, Tuple

from fine_pipe.settings import Settings
from fine_pipe.schema import Document, DatasetRow, RunStats, RunSummary, sha1_text

console = Console()


_WHITESPACE_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"[.!?]\s+")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


# A tiny English stopword set
_EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "to", "of", "in", "on", "for", "with", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those",
    "as", "not", "we", "you", "they", "i", "he", "she", "them",
    "his", "her", "their", "our", "us", "your",
}

# PII regex patterns
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
# Loose phone pattern
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _now() -> int:
    return int(time.time())


def _normalize_text_for_hash(text: str) -> str:
    """Normalization used for exact dedup/minhash consistency."""
    text = text.strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def _safe_json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# URL filtering

def _canonicalize_url(url: str, drop_query_prefixes: list[str]) -> Optional[str]:
    """
    Normalize URLs:
      - strip fragments (#...)
      - drop tracking query params (utm_*, fbclid, etc.)
    """
    
    try:
        u = url.strip()
        if not u:
            return None

        parsed = urlparse(u)
        if not parsed.scheme:
            # If user pasted "example.com/..." treat as https
            parsed = urlparse("https://" + u)

        # remove fragment
        parsed = parsed._replace(fragment="")

        # filter query params
        if parsed.query:
            q = []
            for k, v in parse_qsl(parsed.query, keep_blank_values=True):
                k_low = k.lower()
                # drop exact keys or prefix matches
                if any(k_low == p or k_low.startswith(p) for p in drop_query_prefixes):
                    continue
                q.append((k, v))
            new_query = urlencode(q, doseq=True)
            parsed = parsed._replace(query=new_query)

        # normalize netloc casing
        parsed = parsed._replace(netloc=parsed.netloc.lower())

        return urlunparse(parsed)
    except Exception:
        return None
    

def _url_domain(url: str) -> str:
    """Get registrable domain for allow/deny rules and stats."""
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    
def _should_drop_by_extension(url: str, drop_extensions: list[str]) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in drop_extensions)


def _filter_urls(urls: list[str], settings: Settings) -> list[str]:
    cfg = settings.url_filter
    if not cfg.enabled:
        return urls

    out: list[str] = []
    seen: set[str] = set()

    for raw in urls:
        canon = _canonicalize_url(raw, cfg.drop_query_prefixes)
        if not canon:
            continue

        if _should_drop_by_extension(canon, cfg.drop_extensions):
            continue

        dom = _url_domain(canon)
        if cfg.allow_domains:
            # allowlist mode
            if dom not in {d.lower() for d in cfg.allow_domains}:
                continue
        if cfg.deny_domains:
            if dom in {d.lower() for d in cfg.deny_domains}:
                continue

        if canon not in seen:
            seen.add(canon)
            out.append(canon)

    return out


# Fetch (async)


async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    settings: Settings,
) -> Document:
    doc = Document(url=url)
    cfg = settings.fetch
    doc.meta["domain"] = _url_domain(url)

    async with sem:
        # simple retry loop (you can swap to tenacity later)
        last_err: Optional[str] = None
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = await client.get(url, follow_redirects=True)
                doc.fetched_at = _now()
                doc.status_code = resp.status_code
                doc.final_url = str(resp.url)
                ct = resp.headers.get("content-type", "")
                doc.content_type = ct

                # status checks
                if resp.status_code >= 400:
                    doc.drop("fetch", "http_error", f"HTTP {resp.status_code}")
                    return doc

                # content type allowlist
                ok_ct = any(a in ct.lower() for a in cfg.accept_content_types)
                if not ok_ct:
                    doc.drop("fetch", "bad_content_type", f"content-type={ct}")
                    return doc

                # decode to text (httpx handles charset; fallback safe)
                doc.html = resp.text
                doc.add_event("fetch", "pass", "fetched", status=resp.status_code, content_type=ct)
                return doc

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                last_err = f"{type(e).__name__}: {e}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

            # backoff
            await asyncio.sleep(min(2.0 * (attempt + 1), 8.0))

        # exhausted retries
        doc.fetch_error = last_err
        doc.drop("fetch", "fetch_failed", last_err or "unknown fetch failure")
        return doc
    
    
async def _fetch_all(urls: list[str], settings: Settings) -> list[Document]:
    cfg = settings.fetch
    sem = asyncio.Semaphore(cfg.max_concurrency)

    headers = {"user-agent": cfg.user_agent}
    timeout = httpx.Timeout(timeout=cfg.timeout_s, connect=cfg.connect_timeout_s)

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        tasks = [_fetch_one(client, sem, u, settings) for u in urls]
        return await asyncio.gather(*tasks)


# Extraction


def _extract_text(doc: Document, settings: Settings) -> Document:
    cfg = settings.extract
    if doc.dropped:
        return doc
    if not cfg.enabled:
        return doc

    if not doc.html:
        doc.drop("extract", "no_html", "No HTML to extract")
        return doc

    html = doc.html

    # Primary extractor
    try:
        if cfg.primary == "trafilatura":
            text = trafilatura.extract(
                html,
                include_tables=cfg.include_tables,
                include_comments=cfg.include_comments,
                favor_recall=True,
            )
            title = trafilatura.extract_metadata(html).title if trafilatura.extract_metadata(html) else None
        else:
            rd = ReadabilityDocument(html)
            title = rd.short_title()
            text = rd.summary(html_partial=True)

            # Readability returns HTML; attempt a simple strip via trafilatura
            text = trafilatura.extract(text) or ""

        doc.title = (title or "").strip() if title else None
        doc.text = (text or "").strip() if text else None
        doc.extracted_at = _now()
    except Exception as e:
        doc.extract_error = f"{type(e).__name__}: {e}"
        doc.text = None

    # Fallback extractor if too short / failed
    if not doc.text or len(doc.text) < cfg.min_extracted_chars:
        if cfg.fallback != "none":
            try:
                if cfg.fallback == "readability":
                    rd = ReadabilityDocument(html)
                    title2 = rd.short_title()
                    html2 = rd.summary(html_partial=True)
                    text2 = trafilatura.extract(html2) or ""
                else:
                    text2 = trafilatura.extract(html) or ""
                    title2 = trafilatura.extract_metadata(html).title if trafilatura.extract_metadata(html) else None

                text2 = (text2 or "").strip()
                if len(text2) >= cfg.min_extracted_chars:
                    doc.text = text2
                    if title2 and not doc.title:
                        doc.title = title2.strip()
            except Exception as e:
                doc.extract_error = doc.extract_error or f"{type(e).__name__}: {e}"

    if not doc.text or len(doc.text) < cfg.min_extracted_chars:
        doc.drop("extract", "extraction_failed", f"Extracted chars < {cfg.min_extracted_chars}")
        return doc

    # light cleanup
    doc.text = _WHITESPACE_RE.sub(" ", doc.text).strip()
    doc.add_event("extract", "pass", "extracted", chars=len(doc.text))
    return doc

# Language detection/filtering

from typing import Optional

def _detect_lang_fast_langdetect(text: str) -> tuple[Optional[str], Optional[float]]:
    """
    fast_langdetect.detect(text) returns:
      [{"lang": "en", "score": 0.95}]
    We return (lang, score).
    """
    try:
        import fast_langdetect as f  # type: ignore
    except Exception:
        return None, None

    sample = text[:4000].replace("\n", " ").strip()
    if not sample:
        return None, None

    try:
        out = f.detect(sample)

        # Expected: list[dict]
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                lang = first.get("lang") or first.get("language")
                conf = first.get("score") or first.get("prob") or first.get("confidence")
                if lang is None:
                    return None, None
                return str(lang), float(conf) if conf is not None else 1.0

        # Fallbacks (just in case API changes)
        if isinstance(out, dict):
            lang = out.get("lang") or out.get("language")
            conf = out.get("score") or out.get("prob") or out.get("confidence")
            return (str(lang), float(conf)) if lang and conf is not None else (None, None)

        if isinstance(out, str) and out:
            return out, 1.0

        return None, None
    except Exception:
        return None, None




_FASTTEXT_MODEL = None

def _detect_lang_fasttext(text: str) -> tuple[Optional[str], Optional[float]]:
    """
    Optional fastText language ID support.
    Requires FASTTEXT_LID_MODEL env var pointing to lid.176.bin or lid.176.ftz.
    """
    global _FASTTEXT_MODEL
    try:
        import fasttext  # type: ignore
    except Exception:
        return None, None

    model_path = os.environ.get("FASTTEXT_LID_MODEL")
    if not model_path or not Path(model_path).exists():
        return None, None

    if _FASTTEXT_MODEL is None:
        _FASTTEXT_MODEL = fasttext.load_model(model_path)

    # fastText expects a single line; truncate long text for speed
    sample = text[:4000].replace("\n", " ")
    labels, probs = _FASTTEXT_MODEL.predict(sample, k=1)
    if not labels:
        return None, None
    lang = labels[0].replace("__label__", "")
    conf = float(probs[0]) if probs else None
    return lang, conf


def _language_filter(doc: Document, settings: Settings, stats: RunStats) -> Document:
    cfg = settings.language
    if doc.dropped:
        return doc
    if not cfg.enabled:
        return doc

    if not doc.text:
        doc.drop("language", "no_text", "No text for language detection")
        stats.bump_drop("language", "no_text")
        return doc

    if len(doc.text) < cfg.min_chars:
        doc.drop("language", "too_short", f"chars < {cfg.min_chars}")
        stats.bump_drop("language", "too_short")
        return doc

    if cfg.detector == "fast_langdetect":
        lang, conf = _detect_lang_fast_langdetect(doc.text)
    elif cfg.detector == "fasttext":
        lang, conf = _detect_lang_fasttext(doc.text)
    else:
        doc.drop("language", "bad_detector", f"Unknown language.detector='{cfg.detector}'")
        stats.bump_drop("language", "bad_detector")
        return doc

    doc.lang = lang
    doc.lang_conf = conf

    if not lang or conf is None:
        doc.drop("language", "lang_unknown", "Language detection failed")
        stats.bump_drop("language", "lang_unknown")
        return doc

    if conf < cfg.min_confidence:
        doc.drop("language", "low_conf", f"conf={conf:.3f} < {cfg.min_confidence}")
        stats.bump_drop("language", "low_conf")
        return doc

    if cfg.target_langs and lang not in cfg.target_langs:
        doc.drop("language", "lang_mismatch", f"lang={lang} not in {cfg.target_langs}")
        stats.bump_drop("language", "lang_mismatch")
        return doc

    stats.bump_lang(lang)
    doc.add_event("language", "pass", "language ok", lang=lang, conf=conf)
    return doc


# Gopher-style quality filter

def _gopher_quality(doc: Document, settings: Settings, stats: RunStats) -> Document:
    cfg = settings.gopher
    if doc.dropped:
        return doc
    if not cfg.enabled:
        doc.quality_score = 1.0
        return doc

    text = doc.text or ""
    words = _WORD_RE.findall(text)
    n_words = len(words)

    if n_words < cfg.min_words:
        doc.drop("gopher", "too_few_words", f"words={n_words} < {cfg.min_words}")
        stats.bump_drop("gopher", "too_few_words")
        return doc
    if n_words > cfg.max_words:
        doc.drop("gopher", "too_many_words", f"words={n_words} > {cfg.max_words}")
        stats.bump_drop("gopher", "too_many_words")
        return doc

    # ratios
    total_chars = max(len(text), 1)
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    uppers = sum(ch.isupper() for ch in text)
    symbols = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)

    digit_ratio = digits / total_chars
    symbol_ratio = symbols / total_chars
    upper_ratio = (uppers / max(letters, 1))

    if symbol_ratio > cfg.max_symbol_ratio:
        doc.drop("gopher", "symbol_ratio", f"{symbol_ratio:.3f} > {cfg.max_symbol_ratio}")
        stats.bump_drop("gopher", "symbol_ratio")
        return doc
    if digit_ratio > cfg.max_digit_ratio:
        doc.drop("gopher", "digit_ratio", f"{digit_ratio:.3f} > {cfg.max_digit_ratio}")
        stats.bump_drop("gopher", "digit_ratio")
        return doc
    if upper_ratio > cfg.max_upper_ratio:
        doc.drop("gopher", "upper_ratio", f"{upper_ratio:.3f} > {cfg.max_upper_ratio}")
        stats.bump_drop("gopher", "upper_ratio")
        return doc

    # repetition heuristic: repeated lines fraction
    lines = [ln.strip().lower() for ln in (doc.text or "").splitlines() if ln.strip()]
    rep_frac = 0.0
    if len(lines) >= 5:
        uniq = len(set(lines))
        rep_frac = 1.0 - (uniq / len(lines))

    if rep_frac > cfg.repetition_threshold:
        doc.drop("gopher", "repetition", f"rep_frac={rep_frac:.3f} > {cfg.repetition_threshold}")
        stats.bump_drop("gopher", "repetition")
        return doc

    # simple quality score: prefer longer but not extreme; penalize ratios a bit
    score = 1.0
    score -= min(symbol_ratio / (cfg.max_symbol_ratio + 1e-9), 1.0) * 0.2
    score -= min(digit_ratio / (cfg.max_digit_ratio + 1e-9), 1.0) * 0.2
    score -= min(rep_frac / (cfg.repetition_threshold + 1e-9), 1.0) * 0.2
    score += min(n_words / 1000.0, 1.0) * 0.2

    doc.quality_score = float(max(0.0, min(1.5, score)))
    doc.meta.update({
        "word_count": n_words,
        "digit_ratio": digit_ratio,
        "symbol_ratio": symbol_ratio,
        "upper_ratio": upper_ratio,
        "rep_frac": rep_frac,
    })
    doc.add_event("gopher", "pass", "quality ok", quality_score=doc.quality_score)
    return doc

# C4-style filters

def _c4_filter(doc: Document, settings: Settings, stats: RunStats) -> Document:
    cfg = settings.c4
    if doc.dropped:
        return doc
    if not cfg.enabled:
        return doc

    text = (doc.text or "").strip()
    low = text.lower()

    # blacklist phrases
    for phrase in cfg.blacklist_phrases:
        if phrase.lower() in low:
            doc.drop("c4", "blacklist_phrase", f"matched phrase: {phrase}")
            stats.bump_drop("c4", "blacklist_phrase")
            return doc

    # min sentences
    # heuristic: split by punctuation; requires a few sentence breaks
    sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sents) < cfg.min_sentences:
        doc.drop("c4", "too_few_sentences", f"sentences={len(sents)} < {cfg.min_sentences}")
        stats.bump_drop("c4", "too_few_sentences")
        return doc

    # stopword fraction (works best for English; for multilingual you’ll gate by language)
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if words:
        sw = sum(1 for w in words if w in _EN_STOPWORDS)
        frac = sw / len(words)
        doc.meta["stopword_fraction"] = frac
        if frac < cfg.min_stopword_fraction:
            doc.drop("c4", "low_stopword_fraction", f"{frac:.4f} < {cfg.min_stopword_fraction}")
            stats.bump_drop("c4", "low_stopword_fraction")
            return doc

    doc.add_event("c4", "pass", "c4 ok")
    return doc

# Dedup: exact + MinHash


def _choose_better(a: Document, b: Document, keep_policy: str) -> Document:
    """
    Decide which doc to keep when duplicates are found.
    - highest_quality: keep higher quality_score; tie -> longer text
    - longest: keep longer text
    """
    a_score = a.quality_score or 0.0
    b_score = b.quality_score or 0.0
    a_len = len(a.text or "")
    b_len = len(b.text or "")

    if keep_policy == "longest":
        return a if a_len >= b_len else b

    # highest_quality
    if a_score != b_score:
        return a if a_score > b_score else b
    return a if a_len >= b_len else b


def _exact_dedup(docs: list[Document], settings: Settings, stats: RunStats) -> list[Document]:
    cfg = settings.dedup
    if not cfg.enabled or not cfg.exact_dedup:
        return docs

    seen: dict[str, Document] = {}
    out: list[Document] = []

    for doc in docs:
        if doc.dropped:
            out.append(doc)
            continue

        norm = _normalize_text_for_hash(doc.text or "")
        h = sha1_text(norm)
        doc.exact_hash = h
        doc.doc_id = h  # stable id

        if h not in seen:
            seen[h] = doc
            out.append(doc)
        else:
            # duplicate: keep best, drop the other
            best = _choose_better(seen[h], doc, cfg.keep_policy)
            other = doc if best is seen[h] else seen[h]
            seen[h] = best

            other.drop("dedup_exact", "duplicate", "Exact duplicate text")
            stats.bump_drop("dedup_exact", "duplicate")

    return out


def _minhash_signature(text: str, shingle_size: int, num_perm: int) -> MinHash:
    """
    Build a MinHash signature from word shingles.
    """
    mh = MinHash(num_perm=num_perm)
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < shingle_size:
        # fallback: hash individual words
        for w in words:
            mh.update(w.encode("utf-8"))
        return mh

    for i in range(len(words) - shingle_size + 1):
        sh = " ".join(words[i:i + shingle_size])
        mh.update(sh.encode("utf-8"))
    return mh


def _minhash_dedup(docs: list[Document], settings: Settings, stats: RunStats) -> list[Document]:
    cfg = settings.dedup
    if not cfg.enabled or not cfg.minhash:
        return docs

    # only consider non-dropped docs
    keepers = [d for d in docs if not d.dropped and d.text]
    if len(keepers) <= 1:
        return docs

    lsh = MinHashLSH(threshold=cfg.jaccard_threshold, num_perm=cfg.num_perm)
    sigs: dict[str, MinHash] = {}

    # insert signatures
    for d in keepers:
        doc_id = d.doc_id or d.ensure_doc_id()
        mh = _minhash_signature(d.text or "", cfg.shingle_size, cfg.num_perm)
        sigs[doc_id] = mh
        lsh.insert(doc_id, mh)

    # cluster by union-find-ish approach
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = [d.doc_id or d.ensure_doc_id() for d in keepers]
    for doc_id in ids:
        for cand in lsh.query(sigs[doc_id]):
            if cand != doc_id:
                union(doc_id, cand)

    # group by root
    groups: dict[str, list[Document]] = {}
    id_to_doc = {d.doc_id or d.ensure_doc_id(): d for d in keepers}

    for doc_id in ids:
        root = find(doc_id)
        groups.setdefault(root, []).append(id_to_doc[doc_id])

    # pick representative per group
    for root, group_docs in groups.items():
        if len(group_docs) <= 1:
            continue

        # pick best
        best = group_docs[0]
        for d in group_docs[1:]:
            best = _choose_better(best, d, cfg.keep_policy)

        cluster_id = root
        best.minhash_cluster_id = cluster_id

        for d in group_docs:
            if d is best:
                continue
            d.minhash_cluster_id = cluster_id
            d.drop("dedup_minhash", "near_duplicate", f"Near-duplicate cluster={cluster_id}")
            stats.bump_drop("dedup_minhash", "near_duplicate")

    return docs

# PII detect + redact/drop

def _apply_pii(doc: Document, settings: Settings, stats: RunStats) -> Document:
    cfg = settings.pii
    if doc.dropped:
        return doc
    if not cfg.enabled:
        return doc

    text = doc.text or ""
    matches = 0
    pii_types: list[str] = []

    def _count_and_maybe_redact(pattern: re.Pattern, label: str, token: str) -> None:
        nonlocal text, matches, pii_types
        found = list(pattern.finditer(text))
        if not found:
            return
        matches += len(found)
        pii_types.append(label)
        if cfg.mode == "redact":
            text = pattern.sub(token, text)

    if cfg.redact_email:
        _count_and_maybe_redact(_EMAIL_RE, "email", "[EMAIL]")
    if cfg.redact_phone:
        _count_and_maybe_redact(_PHONE_RE, "phone", "[PHONE]")
    if cfg.redact_ip:
        _count_and_maybe_redact(_IPV4_RE, "ip", "[IP]")

    doc.pii_matches = matches
    doc.pii_types = sorted(set(pii_types))

    if cfg.mode == "drop" and matches >= cfg.drop_if_pii_matches_gte:
        doc.drop("pii", "pii_drop", f"PII matches={matches} >= {cfg.drop_if_pii_matches_gte}")
        stats.bump_drop("pii", "pii_drop")
        return doc

    if cfg.mode == "redact" and matches > 0:
        doc.text = text
        doc.pii_redacted = True
        doc.add_event("pii", "warn", "pii redacted", matches=matches, types=doc.pii_types)
    else:
        doc.add_event("pii", "pass", "pii ok", matches=matches)

    return doc

# Output: build rows + write shards

def _to_dataset_row(doc: Document) -> DatasetRow:
    dom = doc.meta.get("domain", "") or _url_domain(doc.url)
    fetched_at = doc.fetched_at or _now()

    filters_triggered: list[str] = []
    for r in getattr(doc, "drop_reasons", []) or []:
        stage = getattr(r, "stage", None)
        code = getattr(r, "code", None)
        if stage and code:
            filters_triggered.append(f"{stage}:{code}")

    return DatasetRow(
        doc_id=doc.doc_id or doc.ensure_doc_id(),
        url=doc.final_url or doc.url,
        domain=dom,
        fetched_at=fetched_at,
        title=doc.title or "",
        text=doc.text or "",
        lang=doc.lang or "",
        lang_conf=float(doc.lang_conf or 0.0),
        quality_score=float(doc.quality_score or 0.0),
        dedup_cluster_id=doc.minhash_cluster_id or "",
        pii_redacted=bool(doc.pii_redacted),
        pii_types=list(doc.pii_types),
        filters_triggered=filters_triggered,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_shards(
    out_dir: Path,
    rows: list[DatasetRow],
    fmt: str,
    shard_size: int,
) -> int:
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    shard_idx = 0

    buf: list[dict[str, Any]] = []
    for row in rows:
        buf.append(row.to_dict())
        if len(buf) >= shard_size:
            shard_path = shard_dir / f"shard_{shard_idx:05d}.{ 'parquet' if fmt=='parquet' else 'jsonl' }"
            if fmt == "parquet":
                _write_parquet(shard_path, buf)
            else:
                _write_jsonl(shard_path, buf)
            n += 1
            shard_idx += 1
            buf = []

    if buf:
        shard_path = shard_dir / f"shard_{shard_idx:05d}.{ 'parquet' if fmt=='parquet' else 'jsonl' }"
        if fmt == "parquet":
            _write_parquet(shard_path, buf)
        else:
            _write_jsonl(shard_path, buf)
        n += 1

    return n

# Pipeline class

class Pipeline:
    """
    Orchestrates all stages.

    Design:
    - Each stage is a pure-ish function operating on Document
    - We keep dropped docs for stats/debugging, but only write kept docs
    - Settings are validated by settings.py (Pydantic)
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(
        self,
        urls: list[str],
        out_dir: Path,
        export_format: Optional[str] = None,
    ) -> RunSummary:
        """
        Main entrypoint called by CLI/UI.

        Writes:
          - out_dir/shards/shard_*.parquet|jsonl
          - out_dir/stats.json

        Returns:
          RunSummary(total, kept, dropped, shards_written, dataset_dir, stats_path)
        """
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stats = RunStats()
        stats.total_urls = len(urls)

        # ---- URL filter stage (operates on list) ----
        filtered_urls = _filter_urls(urls, self.settings)

        # ---- Fetch (async) ----
        docs = asyncio.run(_fetch_all(filtered_urls, self.settings))
        stats.fetched = sum(1 for d in docs if (d.status_code is not None) and not d.dropped)

        # ---- Extract ----
        for i, d in enumerate(docs):
            docs[i] = _extract_text(d, self.settings)
        stats.extracted = sum(1 for d in docs if (d.text is not None) and not d.dropped)

        # ---- Language ----
        for i, d in enumerate(docs):
            docs[i] = _language_filter(d, self.settings, stats)

        # ---- Gopher quality ----
        for i, d in enumerate(docs):
            docs[i] = _gopher_quality(d, self.settings, stats)

        # ---- Exact dedup ----
        docs = _exact_dedup(docs, self.settings, stats)

        # ---- MinHash dedup ----
        docs = _minhash_dedup(docs, self.settings, stats)

        # ---- C4 filter ----
        for i, d in enumerate(docs):
            docs[i] = _c4_filter(d, self.settings, stats)

        # ---- PII ----
        for i, d in enumerate(docs):
            docs[i] = _apply_pii(d, self.settings, stats)

        # ---- Collect kept docs ----
        kept_docs = [d for d in docs if not d.dropped and (d.text is not None)]
        dropped_docs = [d for d in docs if d.dropped]

        stats.kept = len(kept_docs)
        stats.dropped = len(dropped_docs)
        stats.finish()

        # ---- Write dataset ----
        fmt = (export_format or self.settings.output.format).lower()
        shard_size = self.settings.output.shard_size

        rows = [_to_dataset_row(d) for d in kept_docs]
        shards_written = _write_shards(out_dir, rows, fmt=fmt, shard_size=shard_size)

        # ---- Write stats.json ----
        stats_path = out_dir / "stats.json"
        if self.settings.output.write_stats:
            # Provide drop counts + top-level counts + language counts
            _safe_json_write(
                stats_path,
                {
                    "started_at": stats.started_at,
                    "finished_at": stats.finished_at,
                    "total_urls": stats.total_urls,
                    "fetched": stats.fetched,
                    "extracted": stats.extracted,
                    "kept": stats.kept,
                    "dropped": stats.dropped,
                    "drop_counts": stats.drop_counts,
                    "lang_counts": stats.lang_counts,
                    "output_format": fmt,
                    "shards_written": shards_written,
                },
            )

        return RunSummary(
            total=len(docs),
            kept=len(kept_docs),
            dropped=len(dropped_docs),
            shards_written=shards_written,
            dataset_dir=out_dir,
            stats_path=stats_path if self.settings.output.write_stats else None,
        )
        
        