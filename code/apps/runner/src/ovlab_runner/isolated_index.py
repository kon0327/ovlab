"""Self-contained, offline browser for isolated export artifacts."""
from __future__ import annotations

import csv
import html
import json
from itertools import islice
from pathlib import Path
from urllib.parse import quote

from .comparison_index import COMPARISON_CSS, COMPARISON_SCRIPT, render_quick_comparison


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
            preview = list(islice(reader, 9))
            rows, has_more = preview[:8], len(preview) > 8
    except (OSError, UnicodeError, csv.Error):
        return '<p class="muted">Preview unavailable; open the complete CSV file.</p>'
    def cell(value):
        value = str(value)
        return _escape(value) if len(value) <= 240 else f'<span title="{_escape(value)}">{_escape(value[:237])}…</span>'
    if not rows:
        return '<p class="muted">This table has column definitions but no recorded rows.</p>'
    head = "".join(f"<th scope=\"col\">{cell(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell(item)}</td>" for item in row) + "</tr>" for row in rows)
    note = 'First 8 rows; open the CSV for all records.' if has_more else f'{len(rows)} recorded {"row" if len(rows) == 1 else "rows"}; all rows shown.'
    return f'<div class="table-wrap" tabindex="0" role="region" aria-label="Preview of {_escape(path.name)}"><table><caption>{note}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_isolated_index(root: Path, *, category: str, metadata: dict[str, object], comparison_summary=None) -> str:
    """Render artifact links and previews without network requests.

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
    grouped = metadata.get("export_type") == "grouped"
    run = source.get("run") or source.get("group") or metadata.get("name") or "Unknown"
    kind_label = "Grouped comparison" if grouped else "Isolated export"
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
    if grouped:
        comparison = metadata.get("comparison", {})
        cards = [("Group", run), ("Runs", len(source.get("runs", metadata.get("source_integrity", {})))),
                 ("Comparison", str((comparison_summary or comparison).get("status", "unavailable")).replace("_", " ")),
                 ("Shared tasks", len((comparison_summary or {}).get("task_ids", [])))]
    cards_html = "".join(f'<div class="card"><span>{_escape(label)}</span><strong>{_escape(value if value is not None else "Not recorded")}</strong></div>' for label, value in cards)
    def rel(path):
        return path.relative_to(root).as_posix()
    def figure(path):
        relative = rel(path)
        pdf = path.with_suffix(".pdf")
        title = _label(path.stem.rsplit('__', 1)[-1]).capitalize()
        context = path.stem.rsplit('__', 1)[0] if '__' in path.stem else 'Group comparison' if grouped else 'Run overview'
        pdf_link = f' <a href="{_href(rel(pdf))}">Open PDF</a>' if pdf in pdf_set else ""
        return (f'<figure><a href="{_href(relative)}" data-preview="{_escape(title)}" aria-label="Enlarge {_escape(title)}"><img src="{_href(relative)}" '
                f'alt="{_escape(title)} · {_escape(context)}" loading="lazy" decoding="async"></a><figcaption>'
                f'<strong>{_escape(title)}</strong><span class="figure-context">{_escape(context)}</span>'
                f'<div class="figure-actions"><a href="{_href(relative)}">Open image</a>{pdf_link}</div></figcaption></figure>')
    overview_images = [p for p in images if "episodes" not in p.relative_to(root).parts]
    episode_images = [p for p in images if "episodes" in p.relative_to(root).parts]
    def gallery(paths):
        return '<div class="gallery">' + ''.join(figure(p) for p in paths) + '</div>' if paths else '<p class="muted">No figures were recorded for this section.</p>'
    table_html = []
    for path in tables:
        relative = rel(path)
        preview = _table(path) if grouped or "overview" in path.relative_to(root).parts or relative == "metrics.csv" else ""
        table_html.append(f'<article class="table-card"><h3>{_escape(_label(path.stem))}</h3><p><a href="{_href(relative)}">Open complete CSV</a> <span class="muted">{_escape(relative)}</span></p>{preview}</article>')
    unpaired_pdfs = [path for path in pdfs if str(path.relative_to(root).with_suffix("")) not in image_paths]
    visual_and_tables = set(images) | pdf_set | set(tables)
    other = [path for path in files if path not in visual_and_tables]
    inventory = [*unpaired_pdfs, *other]
    inventory_html = ''.join(f'<li><a href="{_href(rel(path))}">{_escape(rel(path))}</a> <span class="muted">{path.stat().st_size:,} bytes</span></li>' for path in inventory)
    omitted = metadata.get("omitted_figures", [])
    omitted_html = ""
    if isinstance(omitted, list) and omitted:
        omitted_html = '<section class="recording-gaps"><h2>Unavailable figures</h2><p>These figures could not be produced from the recorded data. The available measurements below remain accessible.</p><ul>' + ''.join(f'<li>{_escape(item)}</li>' for item in omitted) + '</ul></section>'
    metadata_html = _escape(json.dumps({key: value for key, value in metadata.items() if key != "files"}, ensure_ascii=False, indent=2, sort_keys=True))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{kind_label} · {_escape(run)}</title><style>
:root{{color-scheme:light;--ink:#17304a;--muted:#536b80;--border:#cddce8;--blue:#1559a3;--bg:#f3f7fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}a{{color:var(--blue)}}a:focus-visible{{outline:3px solid #f0a11b;outline-offset:3px}}header{{background:#102e4c;color:#fff;padding:2.3rem max(1.25rem,calc((100% - 1280px)/2))}}header p{{margin:.3rem 0;color:#d4e4f3}}h1{{margin:.15rem 0;overflow-wrap:anywhere}}h2{{margin-top:0}}main{{max-width:1320px;margin:auto;padding:1.5rem 1.25rem 3rem}}nav{{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0 1.5rem}}section{{background:white;border:1px solid var(--border);border-radius:12px;padding:1.3rem;margin:1rem 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.7rem}}.card{{background:#f3f7fb;border:1px solid var(--border);border-radius:8px;padding:.8rem;min-width:0}}.card span,.card strong{{display:block;overflow-wrap:anywhere}}.card span,.muted,figcaption span{{color:var(--muted);font-size:.88rem}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,370px),1fr));gap:1rem}}figure{{margin:0;border:1px solid var(--border);border-radius:8px;overflow:hidden}}figure img{{display:block;width:100%;height:auto;background:#fff}}figcaption{{padding:.8rem;overflow-wrap:anywhere}}figcaption strong,figcaption span{{display:block}}.table-card{{border-top:1px solid var(--border);padding:1rem 0}}.table-card h3{{margin:0}}.table-wrap{{overflow:auto;max-height:30rem}}table{{border-collapse:collapse;font-size:.84rem}}th,td{{border:1px solid var(--border);padding:.4rem;text-align:left;vertical-align:top;white-space:nowrap}}th{{background:#eef4fa;position:sticky;top:0}}li{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.82rem;background:#f3f7fb;padding:1rem;border-radius:8px}}
{COMPARISON_CSS}
{EXPORT_POLISH_CSS}
</style></head><body><a class="skip-link" href="#overview">Skip to export overview</a><header><p>OVLAB / {_escape({'train': 'Training', 'evaluate': 'Evaluation'}.get(category, category))} / {kind_label}</p><h1>{'Group comparison' if grouped else 'Run evidence'}</h1><p>{_escape(run)}</p></header>
<main id="export-content"><nav aria-label="Sections"><a href="#overview">Overview</a><a href="#figures">Figures ({len(overview_images)})</a>{f'<a href="#episodes">Episode figures ({len(episode_images)})</a>' if episode_images else ''}<a href="#tables">Tables ({len(tables)})</a><a href="#files">Files and metadata</a></nav>
<section id="overview"><h2>Overview</h2><div class="cards">{cards_html}</div></section>
{render_quick_comparison(comparison_summary, scope_label=str(run), heading_level=2) if comparison_summary else ''}
{omitted_html}
<section id="figures"><h2>Overview figures</h2>{gallery(overview_images)}</section>
{f'<section id="episodes"><h2>Episode figures</h2>{gallery(episode_images)}</section>' if episode_images else ''}
<section id="tables"><h2>Data tables</h2>{''.join(table_html) or '<p class="muted">No CSV tables were recorded.</p>'}</section>
<section id="files"><h2>Files and metadata</h2><p><a href="metadata.json">Open complete export metadata and checksums</a></p><ul>{inventory_html}</ul><details><summary>Recorded export metadata</summary><pre>{metadata_html}</pre></details></section>
<footer>Offline export · Figures and complete tables remain available as individual files · <a href="#overview">Back to overview</a></footer></main>
<dialog id="figure-preview" aria-labelledby="figure-preview-title"><div class="preview-toolbar"><h2 id="figure-preview-title">Figure preview</h2><button type="button" data-close-preview>Close preview</button></div><img alt=""><p><a data-original-figure href="#figures">Open original image</a></p></dialog>
{COMPARISON_SCRIPT if comparison_summary else ''}{FIGURE_PREVIEW_SCRIPT}</body></html>'''


EXPORT_POLISH_CSS = """
html{scroll-behavior:smooth}section{scroll-margin-top:1rem}header p{overflow-wrap:anywhere}button,select{font:inherit}main{min-width:0}.skip-link{position:fixed;top:-100px;left:1rem;z-index:20;background:white;padding:.75rem 1rem}.skip-link:focus{top:1rem}
nav a,.figure-actions a{min-height:44px;display:inline-flex;align-items:center}nav a{padding:.2rem .45rem}.figure-actions{display:flex;gap:1rem;flex-wrap:wrap}.figure-context{font: .73rem/1.6 ui-monospace,monospace;overflow-wrap:anywhere;margin:.3rem 0}
.table-wrap{border:1px solid var(--border);border-radius:7px;margin-top:.7rem;overscroll-behavior-inline:contain}.table-wrap table{width:100%}.table-wrap caption{text-align:left;padding:.6rem;background:#f0f5fa;color:#415b74;font-size:.78rem}.table-wrap th,.table-wrap td{white-space:normal;min-width:125px;max-width:42ch;overflow-wrap:anywhere}.table-wrap td{font-variant-numeric:tabular-nums}.table-wrap th{z-index:1}
figure>a{display:block}figure>a:hover{outline:2px solid #447bae;outline-offset:-2px}figcaption strong{font-size:1rem}.recording-gaps{border-left:3px solid #b57c28;background:#fff9ed}.recording-gaps h2{font-size:1.1rem}.recording-gaps li{font-size:.84rem;margin:.4rem 0}footer{font-size:.8rem;color:#465e76;margin:1.5rem 0}
dialog{width:min(1200px,96vw);max-height:94vh;border:1px solid #a9bfd3;border-radius:12px;padding:1rem;color:#17304a;background:white;overflow:auto}dialog::backdrop{background:rgb(12 27 43 / .7)}dialog .preview-toolbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;position:sticky;top:-1rem;background:white;padding:.3rem 0 .7rem;z-index:1}dialog h2{font-size:1.1rem;margin:0}dialog button{min-height:44px;padding:.5rem .8rem;background:#eaf2fc;border:1px solid #8ba6bf;border-radius:7px;cursor:pointer}dialog img{display:block;max-width:100%;height:auto;margin:auto}dialog a{min-height:44px;display:inline-flex;align-items:center}
.quick-comparison h2{font-size:1.2rem;margin:0 0 .5rem}:focus-visible{outline:3px solid #a64e00;outline-offset:3px}
@media(max-width:640px){header{padding:1.4rem 1rem}main{padding:1rem .65rem}section{padding:1rem .8rem}.cards{grid-template-columns:minmax(0,1fr)}.gallery{grid-template-columns:minmax(0,1fr)}dialog{padding:.7rem}nav{gap:.3rem}.table-wrap{max-height:400px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{.skip-link,nav,dialog{display:none}section{break-inside:auto}figure{break-inside:avoid}.table-wrap{max-height:none;overflow:visible}.table-wrap th{position:static}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

FIGURE_PREVIEW_SCRIPT = """<script>
(()=>{const dialog=document.getElementById('figure-preview');if(!dialog||typeof dialog.showModal!=='function')return;
 const image=dialog.querySelector('img'),title=dialog.querySelector('h2'),original=dialog.querySelector('[data-original-figure]');let trigger=null;
 document.querySelectorAll('a[data-preview]').forEach(link=>link.addEventListener('click',event=>{
 if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();trigger=link;image.src=link.href;image.alt=link.querySelector('img').alt;title.textContent=link.dataset.preview;original.href=link.href;dialog.showModal();}));
 dialog.querySelector('[data-close-preview]').addEventListener('click',()=>dialog.close());
 dialog.addEventListener('close',()=>{image.removeAttribute('src');if(trigger)trigger.focus({preventScroll:true});});})();
</script>"""
