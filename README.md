# fine-pipe

Turn a list of web URLs into a **clean, structured pretraining dataset** (Parquet / JSONL) using a practical, FineWeb-style filtering pipeline:

- URL filtering (extensions, domain allow/deny, tracking params)
- Fetch + retries (HTTP)
- Text extraction (trafilatura / readability)
- Language filtering
- Gopher-like quality filtering (heuristics)
- MinHash / exact dedup
- C4-like filters (boilerplate / spam patterns)
- PII detection + redaction (or drop)
- Output shards + run stats + dataset preview in UI (Gradio)

> Goal: paste URLs → get dataset rows like `{doc_id, url, title, text, lang, quality_score, ...}` suitable for pretraining.

---

## Features

- **CLI-first** for reproducible runs (`fine-pipe run`)
- **Gradio UI** for interactive runs + dataset preview + downloads (`fine-pipe ui`)
- **Config-driven**: pipeline behavior controlled by YAML (`configs/default.yaml`)
- **Output shards**: Parquet (default) or JSONL
- **Stats + summaries**: `stats.json`, `summary.json`

---

## Repository layout

```text
fine-pipe/
  configs/
    default.yaml
  src/
    fine_pipe/
      __init__.py
      cli.py
      pipeline.py
      schema.py
      settings.py
      ui/
        __init__.py
        gradio_app.py
  data/
    out/        # default CLI outputs
    out_ui/     # default UI outputs
  pyproject.toml
  README.md
```
## Install (recommended: uv)

### 1) Install uv

**Windows (PowerShell):**

```
winget install --id Astral.uv -e
```
**macOS/Linux:**

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**verify:**
```
uv --version
```
### 2) Clone the repo

```
git clone https://github.com/<your-username>/fine-pipe.git
cd fine-pipe
```
### 3) Create the environment + install deps

```
uv sync
```
This reads ```pyproject.toml```, creates ```.venv/```, and installs everything.

---

## Quickstart (CLI)

### 1) Create a URL list

**Create ```urls.txt``` in repo root:**

```txt
https://en.wikipedia.org/wiki/Transformer_(deep_learning)
https://arxiv.org/abs/1706.03762
```
### 2) Validate config

```
uv run fine-pipe validate-config --config configs/default.yaml
```
### 3) Run pipeline

```
uv run fine-pipe run --config configs/default.yaml --urls urls.txt --out data/out
```
After it finishes, check:

```txt
data/out/
  shards/
    shard_00000.parquet
  stats.json
  summary.json
```
## Quickstart (UI)

### Launch Gradio UI:

```
uv run fine-pipe ui --config configs/default.yaml
```
In the UI you can:

- paste URLs or upload urls.txt
- run the pipeline
- preview actual dataset rows (url/title/text/lang/etc.)
- click a row to view full text
- download shards (.parquet), dataset.jsonl, and stats files

UI outputs are written to:

```
data/out_ui/<timestamp>/
```

## Configuration

The pipeline is configured via YAML.

Default:

```
configs/default.yaml
```
**Key sections:**

- url_filter – drop extensions, allow/deny domains, remove tracking params
- fetch – timeouts, retries, concurrency
- extract – trafilatura/readability settings
- language – detector + target languages
- gopher – heuristic quality filters
- dedup – exact + MinHash config
- c4 – boilerplate patterns
- pii – detect + redact/drop
- output – parquet/jsonl, shard sizes










