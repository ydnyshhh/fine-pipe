from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from fine_pipe.settings import Settings, load_settings
from fine_pipe.pipeline import Pipeline


# -----------------------------
# Helpers
# -----------------------------

def _parse_urls(text: str) -> list[str]:
    urls: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    return urls


def _read_urls_file(file_obj: Any) -> list[str]:
    if file_obj is None:
        return []
    fp = getattr(file_obj, "name", None)
    if not fp:
        return []
    p = Path(fp)
    if not p.exists():
        return []
    return _parse_urls(p.read_text(encoding="utf-8", errors="ignore"))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _safe_json(obj: Any) -> Any:
    """Make objects JSON-serializable for Gradio."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if hasattr(obj, "model_dump"):  # pydantic
        return _safe_json(obj.model_dump())
    if hasattr(obj, "__dict__"):
        try:
            return _safe_json(vars(obj))
        except Exception:
            return str(obj)
    return str(obj)


def _find_shards(out_dir: Path) -> list[Path]:
    shards_dir = out_dir / "shards"
    if not shards_dir.exists():
        return []
    return sorted(shards_dir.glob("*.parquet"))


def _read_rows_from_first_shard(out_dir: Path, limit: int = 200) -> list[dict[str, Any]]:
    shard_files = _find_shards(out_dir)
    if not shard_files:
        return []
    import pyarrow.parquet as pq

    t = pq.read_table(str(shard_files[0]))
    rows = t.to_pylist()
    return rows[:limit]


def _preview_table(rows: list[dict[str, Any]], limit: int = 50) -> tuple[list[str], list[list[Any]]]:
    """
    Convert rows -> table for UI.
    """
    headers = ["url", "title", "lang", "lang_conf", "quality_score", "text_preview"]
    table: list[list[Any]] = []

    for r in rows[:limit]:
        txt = (r.get("text") or "")
        preview = (txt[:220] + "…") if len(txt) > 220 else txt
        table.append([
            r.get("url", ""),
            r.get("title", ""),
            r.get("lang", ""),
            float(r.get("lang_conf") or 0.0),
            float(r.get("quality_score") or 0.0),
            preview,
        ])

    return headers, table


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path


def _write_selected_record(path: Path, row: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _call_pipeline_run(pipe: Pipeline, *, urls: list[str], out_dir: Path) -> Any:
    """
    Call Pipeline.run robustly even if signature changes.
    """
    sig = inspect.signature(pipe.run)
    kwargs: dict[str, Any] = {}

    # common names
    if "urls" in sig.parameters:
        kwargs["urls"] = urls
    if "out_dir" in sig.parameters:
        kwargs["out_dir"] = out_dir

    # If your pipeline supports other params later, add here.
    return pipe.run(**kwargs)


# -----------------------------
# Main UI
# -----------------------------

def build_app(
    settings: Optional[Settings] = None,
    default_cfg_path: str = "configs/default.yaml",
    default_out_base: str = "data/out_ui",
) -> gr.Blocks:
    with gr.Blocks(title="fine-pipe") as demo:
        gr.Markdown(
            """
# fine-pipe
Paste URLs → run the pipeline → preview the **actual pretraining dataset rows** → download shards/JSONL.

**Tip:** If you see `kept=0`, it means filters dropped everything (e.g., `language:low_conf`).
"""
        )

        # UI state: full dataset rows + current run output dir
        rows_state = gr.State([])        # list[dict]
        run_out_state = gr.State("")     # str

        with gr.Row():
            cfg_path = gr.Textbox(
                label="Config YAML path",
                value=default_cfg_path,
                placeholder="configs/default.yaml",
            )
            out_base = gr.Textbox(
                label="Output base directory",
                value=default_out_base,
                placeholder="data/out_ui",
            )

        with gr.Row():
            urls_text = gr.Textbox(
                label="Paste URLs (one per line)",
                lines=10,
                placeholder="https://...\nhttps://...\n",
            )
            urls_file = gr.File(
                label="Or upload urls.txt",
                file_types=[".txt"],
            )

        with gr.Row():
            max_urls = gr.Number(label="Max URLs (0 = all)", value=0, precision=0)
            preview_limit = gr.Number(label="Preview rows (from first shard)", value=50, precision=0)
            run_btn = gr.Button("Run pipeline", variant="primary")

        with gr.Accordion("Outputs", open=True):
            run_log = gr.Textbox(label="Run log", lines=8)

            with gr.Row():
                stats_json = gr.JSON(label="stats.json (parsed)")
                summary_json = gr.JSON(label="summary.json (parsed)")

            gr.Markdown("## Dataset preview (kept docs)")

            # Dataframe for preview table
            preview_df = gr.Dataframe(
                label="Preview table (click a row to view full text)",
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                selected_meta = gr.JSON(label="Selected record (metadata)")
                selected_text = gr.Textbox(label="Selected record (full text)", lines=14)

            with gr.Row():
                download_selected = gr.File(label="Download selected record (JSON)")
                download_all = gr.File(label="Download outputs (shards + dataset.jsonl + stats)", file_count="multiple")

        def _run(
            cfg_path_str: str,
            out_base_str: str,
            pasted: str,
            uploaded: Any,
            max_urls_val: float,
            preview_limit_val: float,
        ):
            t0 = time.time()

            # 1) collect urls
            urls = _parse_urls(pasted)
            urls.extend(_read_urls_file(uploaded))
            urls = _dedupe_preserve_order(urls)

            if not urls:
                return (
                    "No URLs provided.",
                    {},
                    {},
                    [],      # preview table
                    [],      # rows_state
                    "",      # run_out_state
                    None,    # selected download
                    [],      # all downloads
                    {},      # selected meta
                    "",      # selected text
                )

            if max_urls_val and int(max_urls_val) > 0:
                urls = urls[: int(max_urls_val)]

            # 2) load settings
            cfg_path_p = Path(cfg_path_str).expanduser().resolve()
            if settings is None:
                settings_obj = load_settings(cfg_path_p)
            else:
                settings_obj = settings

            # 3) choose run output dir (timestamped)
            out_base_p = Path(out_base_str).expanduser().resolve()
            run_out = out_base_p / time.strftime("%Y%m%d_%H%M%S")
            run_out.mkdir(parents=True, exist_ok=True)

            # 4) run pipeline
            log_lines: list[str] = []
            log_lines.append(f"Config: {cfg_path_p}")
            log_lines.append(f"Out: {run_out}")
            log_lines.append(f"URLs: {len(urls)}")
            log_lines.append("")

            try:
                pipe = Pipeline(settings=settings_obj)
                summary = _call_pipeline_run(pipe, urls=urls, out_dir=run_out)

                # 5) read stats/summary written by pipeline (if present)
                stats_path = run_out / "stats.json"
                summary_path = run_out / "summary.json"

                stats_obj = {}
                if stats_path.exists():
                    stats_obj = json.loads(stats_path.read_text(encoding="utf-8"))

                summary_obj = _safe_json(summary)
                if summary_path.exists():
                    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))

                # 6) dataset preview rows
                rows = _read_rows_from_first_shard(run_out, limit=max(50, int(preview_limit_val or 50)))
                _, table = _preview_table(rows, limit=int(preview_limit_val or 50))

                # 7) downloads: shards + dataset.jsonl + stats + summary
                files: list[str] = []
                if stats_path.exists():
                    files.append(str(stats_path))
                if summary_path.exists():
                    files.append(str(summary_path))

                shard_files = _find_shards(run_out)
                for f in shard_files:
                    files.append(str(f))

                # Also create dataset.jsonl for convenience (from preview rows only).
                if rows:
                    jsonl_path = _write_jsonl(run_out / "dataset.jsonl", rows)
                    files.append(str(jsonl_path))

                dt = time.time() - t0
                log_lines.append(f"Done in {dt:.2f}s")
                log_lines.append(f"kept preview rows: {len(rows)}")
                if not shard_files:
                    log_lines.append("No shards found. Probably kept=0 (all docs dropped).")

                # reset selection outputs
                return (
                    "\n".join(log_lines),
                    stats_obj,
                    summary_obj,
                    table,
                    rows,
                    str(run_out),
                    None,
                    files,
                    {},
                    "",
                )

            except Exception as e:
                log_lines.append("ERROR:")
                log_lines.append(str(e))
                return (
                    "\n".join(log_lines),
                    {},
                    {},
                    [],
                    [],
                    str(run_out),
                    None,
                    [],
                    {},
                    "",
                )

        def _on_row_select(evt: gr.SelectData, rows: list[dict[str, Any]], run_out_str: str):
            """
            When user clicks a row in preview_df, show full record and allow download.
            """
            if not rows:
                return {}, "", None

            # evt.index is usually (row, col)
            idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
            if idx < 0 or idx >= len(rows):
                return {}, "", None

            r = rows[idx]
            meta = {k: v for k, v in r.items() if k != "text"}
            full_text = r.get("text", "") or ""

            run_out = Path(run_out_str) if run_out_str else Path(".")
            sel_path = _write_selected_record(run_out / "selected_record.json", r)

            return _safe_json(meta), full_text, str(sel_path)

        # Wire run
        run_btn.click(
            _run,
            inputs=[cfg_path, out_base, urls_text, urls_file, max_urls, preview_limit],
            outputs=[
                run_log,
                stats_json,
                summary_json,
                preview_df,
                rows_state,
                run_out_state,
                download_selected,
                download_all,
                selected_meta,
                selected_text,
            ],
        )

        # Wire selection
        preview_df.select(
            _on_row_select,
            inputs=[rows_state, run_out_state],
            outputs=[selected_meta, selected_text, download_selected],
        )

    return demo
