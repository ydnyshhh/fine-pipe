from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict


# Stage Config Models

class URLFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    enabled: bool = True
    drop_extensions: list[str] = Field(
        default_factory=lambda: [
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
            ".mp4", ".mov", ".avi",
            ".zip", ".tar", ".gz", ".7z",
            ".pdf",
        ]    
        
    )
    drop_query_prefixes: list[str] = Field(
        default_factory=lambda: ["utm_", "fbclid", "gclid", "ref", "session"]
    )
    deny_domains: list[str] = Field(default_factory=list)
    allow_domains: list[str] = Field(default_factory=list) # if non-empty -> allowlist mode
    
    
class FetchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    enabled: bool = True
    timeout_s: float = 20.0
    connect_timeout_s: float = 10.0
    max_retries: int = 3
    max_concurrency: int = 50
    user_agent: str = "fine-pipe/0.1 (research)"
    accept_content_types: list[str] = Field(default_factory=lambda: ["text/html", "text/plain"])
    # render JS-heavy pages if extraction fails (later)
    render_fallback: bool = False

class ExtractConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    primary: Literal["trafilatura", "readability"] = "trafilatura"
    fallback: Literal["none", "readability", "trafilatura"] = "readability"
    include_tables: bool = False
    include_comments: bool = False
    min_extracted_chars: int = 200  # if less treat as extraction failure


class LanguageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    detector: Literal["fast_langdetect", "fasttext", "pycld3"] = "fast_langdetect"
    target_langs: list[str] = Field(default_factory=lambda: ["en"])
    min_confidence: float = 0.80
    min_chars: int = 200
    allow_mixed: bool = False  # if False drop mixed-language docs


class GopherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_words: int = 80
    max_words: int = 20000
    max_symbol_ratio: float = 0.20
    max_digit_ratio: float = 0.25
    max_upper_ratio: float = 0.35
    repetition_threshold: float = 0.35  # simple repeated-line or repeated-ngram proxy
    

class DedupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    exact_dedup: bool = True

    # MinHash params
    minhash: bool = True
    shingle_size: int = 5
    num_perm: int = 128
    jaccard_threshold: float = 0.85

    # Strategy for choosing representative doc in cluster
    keep_policy: Literal["highest_quality", "longest"] = "highest_quality"


class C4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_sentences: int = 3
    min_stopword_fraction: float = 0.02 
    blacklist_phrases: list[str] = Field(
        default_factory=lambda: [
            "enable javascript",
            "cookie policy",
            "accept cookies",
            "privacy policy",
            "terms of service",
            "all rights reserved",
            "sign up",
            "subscribe",
            "404 not found",
            "page not found",
        ]
    )


class PIIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: Literal["redact", "drop"] = "redact"  # redact spans or drop whole doc
    redact_email: bool = True
    redact_phone: bool = True
    redact_ip: bool = True
    # If mode=drop drop when this many PII matches are found
    drop_if_pii_matches_gte: int = 3


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["parquet", "jsonl"] = "parquet"
    shard_size: int = 50000  # docs per shard
    write_stats: bool = True
    
# Top-level Settings model

class Settings(BaseModel):
    """
    One object passed through the pipeline.
    Every stage reads its section from here.
    """
    model_config = ConfigDict(extra="forbid")

    url_filter: URLFilterConfig = Field(default_factory=URLFilterConfig)
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    gopher: GopherConfig = Field(default_factory=GopherConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    c4: C4Config = Field(default_factory=C4Config)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

# YAML Loader

def load_settings(path: Path) -> Settings:
    """
    Load YAML -> validate -> return Settings.

    This is the single entrypoint used by:
      - CLI: fine-pipe run / fine-pipe ui
      - Gradio UI: build_app(settings)
      - tests
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Settings.model_validate(raw)
    except ValidationError as e:
        # Raise a clean error with context
        raise ValueError(f"Invalid config YAML: {path}\n{e}") from e   
             