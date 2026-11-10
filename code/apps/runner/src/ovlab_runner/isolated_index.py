"""Self-contained, offline browser for isolated export artifacts."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from urllib.parse import quote


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _href(path: str) -> str:
    return quote(path, safe="/")


def _label(path: str) -> str:
    return path.replace("__", " · ").replace("-", " ").replace("_", " ")


def _table(path: Path) -> str:
    # Keep the index small, while linking every complete CSV alongside its preview.
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            headers = next(reader, [])
            if not headers:
                return '<p class="muted">This table has no recorded rows.</p>'
            rows = []
            for _, row in zip(range(8), reader):
                rows.append(row)
    except (OSError, UnicodeError, csv.Error):
        return '<p class="muted">Preview unavailable; open the complete CSV file.</p>'
    def cell(value):
        value = str(value)
        return _escape(value if len(value) <= 240 else value[:237] + "…")
    head = "".join(f"<th scope=\"col\">{cell(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell(item)}</td>" for item in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div><p class="muted">Preview: up to 8 rows; open the CSV for all records.</p>'


def render_isolated_index(root: Path, *, category: str, metadata: dict[str, object]) -> str:
    """Render artifact links and previews without network requests or JavaScript.

    Call before computing the publication inventory so the page itself is
    covered by its checksum. The generated page remains usable through file://.
    """
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path not in {root / "metadata.json", root / "index.html"})
    images = [path for path in files if path.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg", ".webp"}]
    pdfs = [path for path in files if path.suffix.lower() == ".pdf"]
    tables = [path for path in files if path.suffix.lower() == ".csv"]
    pdf_set = set(pdfs)
    image_paths = {str(path.relative_to(root).with_suffix("")) for path in images}
    source = metadata.get("source", {})
    source = source if isinstance(source, dict) else {}
    run = source.get("run") or metadata.get("name") or "Unknown"
    experiment = metadata.get("experiment", {})
    experiment = experiment if isinstance(experiment, dict) else {}
    checkpoint = metadata.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    model = metadata.get("model", {})
    model = model if isinstance(model, dict) else {}
    summary_path = root / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                training_model = summary.get("model")
                if isinstance(training_model, dict):
                    model = training_model
        except (OSError, UnicodeError, ValueError):
            pass
    cards = [("Run", run), ("Experiment", experiment.get("name") or experiment.get("id")),
             ("Model", model.get("name") or model.get("profile_id")),
             ("Checkpoint", checkpoint.get("checkpoint_id") or checkpoint.get("repository")),
             ("Episodes", len(source.get("episodes", [])) if isinstance(source.get("episodes"), list) else None),
             ("Scope", source.get("scope") or category)]
    cards_html = "".join(f'<div class="card"><span>{_escape(label)}</span><strong>{_escape(value if value is not None else "Not recorded")}</strong></div>' for label, value in cards)
    def rel(path):
        return path.relative_to(root).as_posix()
    def figure(path):
        relative = rel(path)
        pdf = path.with_suffix(".pdf")
        pdf_link = f' <a href="{_href(rel(pdf))}">PDF</a>' if pdf in pdf_set else ""
        return (f'<figure><a href="{_href(relative)}"><img src="{_href(relative)}" '
                f'alt="{_escape(_label(path.stem))}" loading="lazy"></a><figcaption>'
                f'<strong>{_escape(_label(path.stem))}</strong><span>{_escape(rel(path.parent))}</span>'
                f'<a href="{_href(relative)}">Open image</a>{pdf_link}</figcaption></figure>')
    overview_images = [p for p in images if "episodes" not in p.relative_to(root).parts]
    episode_images = [p for p in images if "episodes" in p.relative_to(root).parts]
    def gallery(paths):
        return '<div class="gallery">' + ''.join(figure(p) for p in paths) + '</div>' if paths else '<p class="muted">No figures were recorded for this section.</p>'
    table_html = []
    for path in tables:
        relative = rel(path)
        preview = _table(path) if "overview" in path.relative_to(root).parts or relative == "metrics.csv" else ""
        table_html.append(f'<article class="table-card"><h3>{_escape(_label(path.stem))}</h3><p><a href="{_href(relative)}">Open complete CSV</a> <span class="muted">{_escape(relative)}</span></p>{preview}</article>')
    unpaired_pdfs = [path for path in pdfs if str(path.relative_to(root).with_suffix("")) not in image_paths]
    visual_and_tables = set(images) | pdf_set | set(tables)
    other = [path for path in files if path not in visual_and_tables]
    inventory = [*unpaired_pdfs, *other]
    inventory_html = ''.join(f'<li><a href="{_href(rel(path))}">{_escape(rel(path))}</a> <span class="muted">{path.stat().st_size:,} bytes</span></li>' for path in inventory)
    omitted = metadata.get("omitted_figures", [])
    omitted_html = ""
    if isinstance(omitted, list) and omitted:
        omitted_html = '<section><h2>Unavailable figures</h2><p>These figures could not be produced from the recorded data.</p><ul>' + ''.join(f'<li>{_escape(item)}</li>' for item in omitted) + '</ul></section>'
    metadata_html = _escape(json.dumps({key: value for key, value in metadata.items() if key != "files"}, ensure_ascii=False, indent=2, sort_keys=True))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Isolated export · {_escape(run)}</title><style>
:root{{color-scheme:light;--ink:#17304a;--muted:#536b80;--border:#cddce8;--blue:#1559a3;--bg:#f3f7fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}a{{color:var(--blue)}}a:focus-visible{{outline:3px solid #f0a11b;outline-offset:3px}}header{{background:#102e4c;color:#fff;padding:2.3rem max(1.25rem,calc((100% - 1280px)/2))}}header p{{margin:.3rem 0;color:#d4e4f3}}h1{{margin:.15rem 0;overflow-wrap:anywhere}}h2{{margin-top:0}}main{{max-width:1320px;margin:auto;padding:1.5rem 1.25rem 3rem}}nav{{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0 1.5rem}}section{{background:white;border:1px solid var(--border);border-radius:12px;padding:1.3rem;margin:1rem 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.7rem}}.card{{background:#f3f7fb;border:1px solid var(--border);border-radius:8px;padding:.8rem;min-width:0}}.card span,.card strong{{display:block;overflow-wrap:anywhere}}.card span,.muted,figcaption span{{color:var(--muted);font-size:.88rem}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,370px),1fr));gap:1rem}}figure{{margin:0;border:1px solid var(--border);border-radius:8px;overflow:hidden}}figure img{{display:block;width:100%;height:auto;background:#fff}}figcaption{{padding:.8rem;overflow-wrap:anywhere}}figcaption strong,figcaption span{{display:block}}.table-card{{border-top:1px solid var(--border);padding:1rem 0}}.table-card h3{{margin:0}}.table-wrap{{overflow:auto;max-height:30rem}}table{{border-collapse:collapse;font-size:.84rem}}th,td{{border:1px solid var(--border);padding:.4rem;text-align:left;vertical-align:top;white-space:nowrap}}th{{background:#eef4fa;position:sticky;top:0}}li{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.82rem;background:#f3f7fb;padding:1rem;border-radius:8px}}
</style></head><body><header><p>OVLAB / { _escape(category) } / isolated export</p><h1>Run evidence</h1><p>Recorded data, figures and downloadable tables from one run.</p></header>
<main><nav aria-label="Sections"><a href="#overview">Overview</a><a href="#figures">Figures</a><a href="#episodes">Episode figures</a><a href="#tables">Tables</a><a href="#files">Files and metadata</a></nav>
<section id="overview"><h2>Overview</h2><div class="cards">{cards_html}</div></section>
<section id="figures"><h2>Overview figures</h2>{gallery(overview_images)}</section>
<section id="episodes"><h2>Episode figures</h2>{gallery(episode_images)}</section>
<section id="tables"><h2>Data tables</h2>{''.join(table_html) or '<p class="muted">No CSV tables were recorded.</p>'}</section>
{omitted_html}<section id="files"><h2>Files and metadata</h2><p><a href="metadata.json">Open complete export metadata and checksums</a></p><ul>{inventory_html}</ul><details><summary>Recorded export metadata</summary><pre>{metadata_html}</pre></details></section>
</main></body></html>'''
