from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
import time
import hashlib



DropStage = Literal[
    "url_filter",
    "fetch",
    "extract",
    "language",
    "gopher",
    "dedup_exact",
    "dedup_minhash",
    "c4",
    "pii",
    "write",
]

DocStatus = Literal["ok", "dropped", "error"]


def sha1_text(text: str) -> str:
    """Stable hash used for doc IDs and exact dedup."""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def now_unix() -> int:
    """Unix timestamp (seconds)."""
    return int(time.time())



# Core pipeline objects



@dataclass
class StageEvent:
    """
    A lightweight log entry that records what happened at a stage.

    Useful for:
    - debugging why docs were dropped
    - showing a per-doc audit trail in Gradio
    """
    stage: str
    action: Literal["pass", "drop", "warn", "info", "error"]
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: int = field(default_factory=now_unix)



@dataclass
class DropReason:
    """
    Structured drop reason:
    - stage: which stage decided to drop
    - code: short machine-readable code (good for stats aggregation)
    - detail: human readable message
    """
    stage: DropStage
    code: str
    detail: str
    
    
@dataclass
class Document:
    """
    The central object that flows through the pipeline.

    Philosophy:
    - stages mutate this object (fill html/text/lang/score/meta)
    - stages may drop it, but never delete it
    - drop reasons and stage logs are first-class for observability
    """
    url: str
    
    
    # Fetch output
    fetched_at: Optional[int] = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    final_url: Optional[str] = None  # after redirects
    html: Optional[str] = None
    fetch_error: Optional[str] = None

    # Extraction output
    title: Optional[str] = None
    text: Optional[str] = None
    extracted_at: Optional[int] = None
    extract_error: Optional[str] = None

    # Language
    lang: Optional[str] = None
    lang_conf: Optional[float] = None

    # Quality
    quality_score: Optional[float] = None
    
    # Dedup
    doc_id: Optional[str] = None             # stable ID (usually sha1(normalized_text))
    exact_hash: Optional[str] = None         # exact dedup hash (can equal doc_id)
    minhash_cluster_id: Optional[str] = None # assigned cluster ID for near-dup groups

    # PII
    pii_redacted: bool = False
    pii_types: list[str] = field(default_factory=list)
    pii_matches: int = 0

    # State
    status: DocStatus = "ok"
    dropped: bool = False
    drop_reasons: list[DropReason] = field(default_factory=list)

    # Observability
    events: list[StageEvent] = field(default_factory=list)

    # Arbitrary metadata for extensions (domain, word counts, ratios, etc.)
    meta: dict[str, Any] = field(default_factory=dict)
    
    def add_event(
        self,
        stage: str,
        action: Literal["pass", "drop", "warn", "info", "error"],
        message: str,
        **data: Any,
    ) -> None:
        self.events.append(StageEvent(stage=stage, action=action, message=message, data=data))

    def drop(self, stage: DropStage, code: str, detail: str) -> None:
        """
        Mark document as dropped.
        keep it around for stats, debugging, UI audit trails.
        """
        self.dropped = True
        self.status = "dropped"
        self.drop_reasons.append(DropReason(stage=stage, code=code, detail=detail))
        self.add_event(stage=stage, action="drop", message=detail, code=code)

    def mark_error(self, stage: DropStage, detail: str, code: str = "error") -> None:
        """
        Record an error without necessarily dropping.
        """
        self.status = "error"
        self.add_event(stage=stage, action="error", message=detail, code=code)
        
    
    def ensure_doc_id(self) -> str:
        """
        Set doc_id if missing.
        Convention: doc_id is derived from the cleaned/normalized text.
        """
        if self.doc_id:
            return self.doc_id
        if not self.text:
            # fallback: url-based id (not ideal, but prevents crashes)
            self.doc_id = sha1_text(self.url)
            return self.doc_id
        self.doc_id = sha1_text(self.text)
        return self.doc_id
    
    def to_preview_row(self) -> dict[str, Any]:
        """
        Small row for Gradio preview tables.
        """
        text_len = len(self.text) if self.text else 0
        reasons = "; ".join([f"{r.stage}:{r.code}" for r in self.drop_reasons]) if self.drop_reasons else ""
        return {
            "url": self.url,
            "final_url": self.final_url or "",
            "status": self.status,
            "lang": self.lang or "",
            "quality_score": self.quality_score if self.quality_score is not None else "",
            "text_len": text_len,
            "dropped": self.dropped,
            "reasons": reasons,
        }
        

# Dataset row

@dataclass
class DatasetRow:
    """
    The normalized dataset record.

    stable and pretraining-friendly:
    - always store url + text
    - keep metadata fields that help filtering/sampling later
    """
    doc_id: str
    url: str
    domain: str
    fetched_at: int
    title: str
    text: str
    lang: str
    lang_conf: float
    quality_score: float
    dedup_cluster_id: str
    pii_redacted: bool
    pii_types: list[str]
    filters_triggered: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "domain": self.domain,
            "fetched_at": self.fetched_at,
            "title": self.title,
            "text": self.text,
            "lang": self.lang,
            "lang_conf": self.lang_conf,
            "quality_score": self.quality_score,
            "dedup_cluster_id": self.dedup_cluster_id,
            "pii_redacted": self.pii_redacted,
            "pii_types": self.pii_types,
            "filters_triggered": self.filters_triggered,
        }
        
            

# Run outputs 

@dataclass
class RunStats:
    """
    Aggregated stats across the run.
    Store counts per stage and reason code.
    """
    started_at: int = field(default_factory=now_unix)
    finished_at: Optional[int] = None

    total_urls: int = 0
    fetched: int = 0
    extracted: int = 0
    kept: int = 0
    dropped: int = 0

    # Drop reason counters: {"stage:code": count}
    drop_counts: dict[str, int] = field(default_factory=dict)

    # Language distribution
    lang_counts: dict[str, int] = field(default_factory=dict)

    def bump_drop(self, stage: str, code: str) -> None:
        key = f"{stage}:{code}"
        self.drop_counts[key] = self.drop_counts.get(key, 0) + 1

    def bump_lang(self, lang: str) -> None:
        self.lang_counts[lang] = self.lang_counts.get(lang, 0) + 1

    def finish(self) -> None:
        self.finished_at = now_unix()


@dataclass
class RunSummary:
    """
    What Pipeline.run() returns to CLI/UI.

    Keep it small but useful:
    - counts
    - where outputs are written
    - where to find stats
    """
    total: int
    kept: int
    dropped: int
    shards_written: int
    dataset_dir: Path
    stats_path: Optional[Path] = None
    summary_path: Optional[Path] = None
