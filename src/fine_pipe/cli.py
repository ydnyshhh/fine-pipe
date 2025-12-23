from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from typing import Any

import typer
from rich.console import Console
from rich.traceback import install as install_rich_traceback


app = typer.Typer(
    name = "fine_pipe",
    help = (
        "Paste URLs -> Get cleaned dataset (Parquet/JSONL)"
        "with language filtering, Gopher/C4 filters, MinHash dedup, and PII removal."  
    ),
    add_completion=False,
)

console = Console()


def _read_urls(
    urls_file: Optional[Path] = None,
    urls_text: Optional[str] = None,
) -> list[str]:
    """
    Read URLs from one of three sources (priority order):
    1) --text "<url1>\\n<url2>..."
    2) --urls /path/to/urls.txt
    3) stdin (when piped): 'cat urls.txt | fine-pipe run...'
    
    Returns a raw list (un-normalized). URL normalization happens in the pipeline.
    
    """
    
    if urls_text:
        urls = [line.strip() for line in urls_text.splitlines() if line.strip()]
        return urls
    
    if urls_file is not None:
        if not urls_file.exists():
            raise typer.BadParameter(f"URL file does not exist: {urls_file}")
        data = urls_file.read_text(encoding="utf-8", errors="ignore")
        urls = [line.strip() for line in data.splitlines() if line.strip()]
        return urls
    
    # If stdin is piped, sys.stdin.isatty() == False
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        urls = [line.strip() for line in data.splitlines() if line.strip()]
        return urls
    
    raise typer.BadParameter(
        "No URLs provided. Use --urls urls.txt, --text, or pipe via stdin."   
    )            
    
def _ensure_out_dir(out_dir: Path) -> Path:
    """
    Create output directory if needed and return a resolved absolute path.
    """    
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents = True, exist_ok=True)
    return out_dir


def _print_run_banner(config: Path, out_dir: Path, n_urls: int, dry_run: bool) -> None:
    console.print("[bold]url2fineweb[/bold] CLI")
    console.print(f"[bold]Config:[/bold] {config}")
    console.print(f"[bold]Out dir:[/bold] {out_dir}")
    console.print(f"[bold]Input URLs:[/bold] {n_urls}")
    if dry_run:
        console.print("[yellow]Dry-run enabled: pipeline will NOT fetch/process pages.[/yellow]")

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

# Commands

@app.command()
def run(
    out: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Output directory. Will create shards/ and stats files inside it.",
    ),
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",        
    ),
    urls: Optional[Path] = typer.Option(
        None,
        "--urls",
        help="Path to newline-separated URLs file.",
    ),
    text: Optional[str] = typer.Option(
        None,
        "--text",
        help="Newline-separated URLs passed as a single string (useful in UI/CI).",
    ),
    max_urls: Optional[int] = typer.Option(
        None,
        "--max-urls",
        min=1,
        help="Process only the first N URLs (after reading; before pipeline filtering).",
    ),
    export_format: str = typer.Option(
        "parquet",
        "--format",
        "-f",
        help="Dataset export format.",
        case_sensitive=False,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry_run",
        help="Validate  config + URL input and exit without running the pipeline.",
    ),
    save_input_urls: bool = typer.Option(
        True,
        "--save-input-urls/--no-save-input-urls",
        help="Save the raw input URL list to out_dir/input_urls.txt for reproducibility.",
    ),
    json_summary: bool = typer.Option(
        False,
        "--json",
        help="Print machine-readable JSON summary to stdout (useful for scripts/CI).",
    ),
):
    """
    Run the full pipeline:

    URLs -> URL filtering -> fetch -> text extraction -> language filtering ->
    Gopher filtering -> MinHash dedup -> C4 filters -> PII detect/remove ->
    write dataset shards + stats.

    This command is used both for local runs and for headless dataset generation.
    """
    out_dir = _ensure_out_dir(out)
    input_urls = _read_urls(urls_file=urls, urls_text=text)

    if max_urls is not None:
        input_urls = input_urls[:max_urls]

    _print_run_banner(config=config, out_dir=out_dir, n_urls=len(input_urls), dry_run=dry_run)

    # Store raw input URL list for reproducibility
    if save_input_urls:
        (out_dir / "input_urls.txt").write_text(
            "\n".join(input_urls) + "\n", encoding="utf-8"
        )
    
    
    # Load settings from YAML
    
    from fine_pipe.settings import load_settings

    settings = load_settings(config)

    # Validate export format early
    export_format = export_format.lower().strip()
    if export_format not in {"parquet", "jsonl"}:
        raise typer.BadParameter("--format must be one of: parquet, jsonl")

    if dry_run:
        console.print("[green]Dry-run OK.[/green] Config loaded and URLs parsed.")
        raise typer.Exit(code=0)
    
    # Run pipeline

    from fine_pipe.pipeline import Pipeline

    pipeline = Pipeline(settings=settings)
    
    # Expected behavior: pipeline runs all stages and writes outputs into out_dir
    # and returns a RunSummary-like object with counts and output paths.
    summary = pipeline.run(
        urls=input_urls,
        out_dir=out_dir,
        export_format=export_format,
    )
    
    console.print("[bold green]Done.[/bold green]")
    try:
        # If summary is a dataclass
        d = asdict(summary)  # type: ignore[arg-type]
    except Exception:
        # fallback: try using __dict__
        d = getattr(summary, "__dict__", {"summary": str(summary)})

    # Always write a summary.json to disk
    _write_json(out_dir / "summary.json", d)

    console.print(f"kept={d.get('kept')} dropped={d.get('dropped')} total={d.get('total')}")
    console.print(f"dataset_dir={d.get('dataset_dir', str(out_dir))}")
    if d.get("stats_path"):
        console.print(f"stats={d['stats_path']}")

    
    if json_summary:
        # Print to stdout in JSON
        sys.stdout.write(json.dumps(d, ensure_ascii=False) + "\n")
    


@app.command()
def ui(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind the Gradio server.",
    ),
    port: int = typer.Option(
        7860,
        "--port",
        help="Port for the Gradio server.",
    ),
    share: bool = typer.Option(
        False,
        "--share",
        help="Create a public Gradio link (useful for demos).",
    ),
):
    """
    Launch the Gradio UI.

    The UI should be a thin wrapper that calls the same Pipeline used by `run`.
    """
    from fine_pipe.settings import load_settings
    settings = load_settings(config)
    
    
    # Import Gradio app builder only when needed
    from fine_pipe.ui.gradio_app import build_app

    demo = build_app(settings=settings)
    console.print(f"[bold]Launching Gradio UI[/bold] at http://{host}:{port}")
    demo.launch(server_name=host, server_port=port, share=share)

@app.command("validate-config")
def validate_config(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",
    )
):
    """
    Validate the YAML config and exit.

    Useful for CI and for quickly checking that your config schema still matches.
    """
    from fine_pipe.settings import load_settings
    _ = load_settings(config)
    console.print(f"[green]Config OK:[/green] {config}")
    
@app.command("version")
def version():
    """
    Print the package version.
    """
    try:
        from importlib.metadata import version as pkg_version
        console.print(pkg_version("fine-pipe"))
    except Exception:
        console.print("0.0.0 (dev)")


if __name__ == "__main__":
    app()                