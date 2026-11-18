"""Lightweight experiment navigation and scoped comparison previews."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote

from ovlab_runner.comparison_index import COMPARISON_CSS, COMPARISON_SCRIPT, render_quick_comparison
from ovlab_runner.report_views import publish_report_view, report_page


def _escape(value):
    return html.escape(str(value), quote=True)


def _read(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _inside(root, value):
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_relative_to(root) else None


def report_locations(plan, result, root):
    """Resolve current artifacts, never trusting stale paths outside this export."""
    locations = {}
    for node in plan['nodes']:
        path = node['path']
        for phase, record in result.get('nodes', {}).get(path, {}).items():
            if phase not in ('train', 'evaluate') or not record.get('run_id'):
                continue
            run_id = record['run_id']
            base = root / phase / path
            profile = 'system-performance' if phase == 'train' else node.get('configuration', {}).get(phase, {}).get('document', {}).get('reporting', {}).get('profile', 'libero-task-default')
            profile_root = base / 'interactive' / run_id / profile
            build = _read(profile_root / 'latest.json').get('build_id')
            report = None
            if isinstance(build, str) and Path(build).name == build and build not in ('', '.', '..'):
                report = _inside(root, profile_root / build / 'index.html')
            if report is None or not report.is_file():
                child = record.get('result', {})
                post = record.get('postprocessing', {})
                document = child.get('performance_report', {}) if phase == 'train' else post.get('report', {})
                report = _inside(root, document.get('interactive_output') or document.get('output')) if isinstance(document, dict) else None
            locations[(path, phase)] = {'run_id': run_id, 'report': report, 'isolated': base / 'isolated' / run_id}
    for phase, groups in result.get('grouped', {}).items():
        for path, group in groups.items():
            output = _inside(root, group.get('result', {}).get('output'))
            if output is None or not output.exists():
                candidates = sorted((root / phase / path / 'grouped').glob('*/metadata.json'))
                output = candidates[0].parent if len(candidates) == 1 else None
            locations[(path, phase, 'grouped')] = {'report': output}
    return locations


def prepare_experiment_views(plan, result, root):
    root = Path(root).resolve()
    for item in report_locations(plan, result, root).values():
        for key in ('report', 'isolated'):
            candidate = item.get(key)
            if candidate:
                publish_report_view(root, candidate)


def build_experiment_index(plan, result, export_root):
    root = Path(export_root).resolve()
    nodes = {node['path']: node for node in plan['nodes']}
    selected = set(plan.get('selected_nodes', nodes))
    locations = report_locations(plan, result, root)
    labels, run_links = {}, {}

    def href(candidate):
        page = report_page(root, candidate)
        return quote(page.relative_to(root).as_posix(), safe='/') if page else None

    def link(candidate, text, primary=False):
        target = href(candidate)
        return (f'<a class="button {"primary" if primary else ""}" href="{_escape(target)}">{_escape(text)}</a>' if target
                else f'<span class="unavailable">{_escape(text)} unavailable</span>')

    for (key, item) in locations.items():
        if len(key) == 2:
            path, _ = key
            labels[item['run_id']] = (path or 'Root') + ' · ' + nodes[path].get('name', nodes[path]['id'])
            target = href(item.get('report'))
            if target:
                run_links[item['run_id']] = target

    def phase_failure(path, phase):
        return next((failure for failure in result.get('failures', [])
                     if failure.get('node') == path and failure.get('phase') == phase), None)

    def phase_status(record):
        return str(record.get('result', {}).get('status') or record.get('status') or 'completed').lower()

    def descendant_counts(path, phase):
        descendants = [other for other in nodes if other != path and (not path or other.startswith(path + '/'))]
        planned = sum(phase in nodes[other].get('execute', []) and other in selected for other in descendants)
        completed = sum(phase_status(record) in ('completed', 'success', 'succeeded', 'verified')
                        for other in descendants
                        if (record := result.get('nodes', {}).get(other, {}).get(phase)))
        return completed, max(planned, completed)

    def phase_html(node, phase):
        path = node['path']
        record = result.get('nodes', {}).get(path, {}).get(phase)
        heading = 'Training' if phase == 'train' else 'Evaluation'
        if record:
            item = locations.get((path, phase), {})
            failure = phase_failure(path, phase)
            recorded_status = phase_status(record)
            completed = recorded_status in ('completed', 'success', 'succeeded', 'verified')
            status = 'Run completed · reporting issue' if failure and completed else 'Failed' if failure else recorded_status.replace('_', ' ').title()
            tone = 'warning' if failure and completed else 'failed' if failure or recorded_status in ('failed', 'interrupted', 'error') else 'completed' if completed else 'pending'
            return (f'<article class="phase"><p class="phase-title">{heading} <span class="badge" data-status="{tone}">{_escape(status)}</span></p>'
                    f'<p class="run-id">{_escape(record.get("run_id", "Unknown"))}</p><div class="actions">'
                    + link(item.get('report'), 'Interactive report', True) + link(item.get('isolated'), 'Figures & tables')
                    + '</div>' + (f'<p class="failure">{_escape(failure.get("message", ""))}</p>' if failure else '') + '</article>')
        descendants, planned = descendant_counts(path, phase)
        failure = phase_failure(path, phase)
        if failure:
            return f'<p class="phase-note failure"><strong>{heading} failed:</strong> {_escape(failure.get("message", "No completed run was returned."))}</p>'
        if phase in node.get('execute', []):
            status = 'Outside selected subtree' if path not in selected else 'Waiting to run' if result.get('status') == 'running' else 'No completed result recorded'
        elif planned:
            status = f'{descendants} of {planned} child runs completed.'
        else:
            status = f'{heading} is not configured at this node.'
        return f'<p class="phase-note"><strong>{heading}:</strong> {_escape(status)}</p>'

    def comparison_html(path, depth):
        parts = []
        for phase in ('train', 'evaluate'):
            group = result.get('grouped', {}).get(phase, {}).get(path)
            if not group:
                continue
            item = locations.get((path, phase, 'grouped'), {})
            output = item.get('report')
            if output and output.is_file():
                output = output.parent
            metadata = _read(output / 'metadata.json') if output else {}
            summary = _read(output / 'quick-comparison.json') if output else {}
            comparison = metadata.get('comparison', {}) if phase == 'evaluate' else _read(output / 'comparison.json') if output else {}
            status = summary.get('status') or comparison.get('status', 'unavailable')
            heading = min(depth + 3, 6)
            parts.append(f'<div class="grouped"><h{heading}>{"Evaluation" if phase == "evaluate" else "Training"} comparison <span class="badge">{_escape(str(status).replace("_", " "))}</span></h{heading}>')
            if summary:
                parts.append(render_quick_comparison(summary, labels=labels, links=run_links,
                    scope_label=(path or 'Root') + ' · ' + phase, heading_level=min(heading + 1, 6)))
            else:
                parts.append('<p class="phase-note">Quick comparison is not available in this older export. Regenerate the grouped export to calculate it.</p>')
            parts.append('<div class="actions">' + link(output, 'Open grouped report', True))
            if output and (output / 'metadata.json').is_file():
                meta = quote((output / 'metadata.json').relative_to(root).as_posix(), safe='/')
                parts.append(f'<a class="data-link" href="{_escape(meta)}">Metadata JSON</a>')
            parts.append('</div></div>')
        return ''.join(parts)

    def render_node(path, depth=0):
        node = nodes[path]
        anchor = 'node-' + hashlib.sha256(path.encode()).hexdigest()[:12]
        children = [f'{path}/{child}'.strip('/') for child in node.get('children', [])]
        execute = node.get('execute', [])
        role = 'Experiment group' if not execute and children else 'Run & variants' if children else 'Experiment run'
        heading = min(depth + 2, 6)
        phases = [phase_html(node, phase) for phase in ('train', 'evaluate')]
        cards = ''.join(part for part in phases if part.startswith('<article'))
        notes = ''.join(part for part in phases if not part.startswith('<article'))
        child_badge = f'<span class="badge">{len(children)} child {"node" if len(children) == 1 else "nodes"}</span>' if children else ''
        branch_navigation = ('<nav class="branch-nav" aria-label="Child nodes of ' + _escape(path) + '">'
            + ''.join(f'<a href="#node-{hashlib.sha256(child.encode()).hexdigest()[:12]}">{_escape(nodes[child]["id"])}</a>' for child in children)
            + '</nav>') if children and path else ''
        header = (f'<header class="node-header"><div><p class="eyebrow">{role}</p>'
                f'<h{heading}>{_escape(node.get("name") or node["id"])}</h{heading}><p class="path">{_escape(path or "Root")}</p></div>'
                f'{child_badge}</header>')
        if not path and not execute:
            scope = f'Root · {len(children)} child {"branch" if len(children) == 1 else "branches"}' if children else 'Root scope'
            header = f'<header class="node-header root-header"><h2>Experiment results</h2><span class="badge">{scope}</span></header>'
        return (f'<section class="node depth-{depth}" id="{anchor}">{header}{branch_navigation}'
                + (f'<div class="phases">{cards}</div>' if cards else '')
                + (f'<div class="phase-notes">{notes}</div>' if notes else '') + comparison_html(path, depth)
                + ('<ol class="children">' + ''.join('<li>' + render_node(child, depth + 1) + '</li>' for child in children) + '</ol>' if children else '') + '</section>')

    evaluation = sum('evaluate' in x for x in result.get('nodes', {}).values())
    training = sum('train' in x for x in result.get('nodes', {}).values())
    groups = sum(len(x) for x in result.get('grouped', {}).values())
    top_paths = ['', *nodes[''].get('children', [])]
    navigation = ''.join(f'<a href="#node-{hashlib.sha256(path.encode()).hexdigest()[:12]}">{_escape(path or "Overview")}</a>' for path in top_paths)
    failure = result.get('failure')
    failure_html = '<p class="failure">' + _escape(failure.get('message', '')) + '</p>' if isinstance(failure, dict) else ''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(plan.get('name', plan.get('id', 'Experiment')))} · OVLAB experiment</title><style>
:root{{color-scheme:light;--ink:#132b43;--muted:#53677c;--line:#d6e1ec;--blue:#175fb4}}*{{box-sizing:border-box}}body{{margin:0;font:16px/1.5 system-ui,sans-serif;color:var(--ink);background:#f2f6fa}}a{{color:var(--blue)}}a:focus-visible,select:focus-visible{{outline:3px solid #f2a738;outline-offset:3px}}main{{max-width:1480px;margin:auto;padding:2rem 1.25rem 5rem}}h1{{font-size:clamp(1.8rem,4vw,2.8rem);line-height:1.18;margin:.4rem 0}}h2,h3,h4,h5,h6{{margin:.2rem 0;font-size:1.2rem}}.eyebrow{{text-transform:uppercase;font-size:.72rem;letter-spacing:.1em;color:var(--muted);margin:0}}.identity,.path,.run-id{{overflow-wrap:anywhere;font: .8rem/1.6 ui-monospace,monospace;color:var(--muted)}}.intro{{max-width:80ch;color:var(--muted)}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.7rem;margin:1.5rem 0}}.summary>div{{background:white;padding:1rem;border:1px solid var(--line);border-radius:10px}}.summary strong{{display:block;font-size:1.5rem}}.summary span{{font-size:.85rem;color:var(--muted)}}.tree-nav{{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0 1.5rem}}.tree-nav a{{background:#e5edf7;border-radius:6px;padding:.3rem .6rem;text-decoration:none;font-size:.8rem}}.node{{background:white;border:1px solid var(--line);border-radius:14px;padding:1.2rem;margin:1rem 0;min-width:0}}.node-header{{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}}.node-header .path{{margin:.25rem 0 .8rem}}.badge{{display:inline-block;border-radius:5px;background:#e9f2fc;color:#315675;padding:.2rem .45rem;font-size:.72rem;font-weight:500;white-space:nowrap}}.phases{{display:flex;flex-wrap:wrap;gap:.7rem;align-items:flex-start}}.phase{{flex:1 1 280px;background:#f6f9fc;border:1px solid var(--line);border-radius:9px;padding:1rem}}.phase h3{{font-size:1rem}}.phase-note{{color:var(--muted);font-size:.85rem;flex:1 1 250px;margin:.4rem 0}}.actions{{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-top:.6rem}}.button{{display:inline-block;padding:.45rem .7rem;border:1px solid #afc5dd;border-radius:7px;text-decoration:none;font-size:.85rem;font-weight:600;background:white}}.button.primary{{background:#155eb3;color:white;border-color:#155eb3}}.data-link{{font-size:.8rem}}.unavailable{{color:var(--muted);font-size:.85rem}}.children{{list-style:none;border-left:2px solid #d7e3f0;padding-left:1rem;margin:1.25rem 0 0}}.children>li{{min-width:0}}.grouped{{border-top:1px solid var(--line);margin-top:1.2rem;padding-top:1rem}}.failure{{background:#fff0ec;color:#94301e;padding:.8rem;border-left:3px solid #bd4327}}@media(max-width:640px){{main{{padding:.9rem .6rem 3rem}}.node{{padding:.8rem}}.children{{padding-left:.45rem}}.node-header{{display:block}}.badge{{white-space:normal}}}}
{COMPARISON_CSS}
{OVERVIEW_POLISH_CSS}
</style></head><body><a class="skip-link" href="#experiment-tree">Skip to experiment results</a><main id="experiment-top"><header><p class="eyebrow">OVLAB / Experiment evidence</p><h1>{_escape(plan.get('name', plan.get('id', 'Experiment')))}</h1><p class="identity">{_escape(result['experiment_id'])}</p><p class="intro">Compare shared tasks, then open a run for its full evidence. Group nodes organize child runs; they do not add another evaluation.</p></header>
<div class="summary"><div><strong>{_escape(result.get('status','unknown')).title()}</strong><span>Workflow status</span></div><div><strong>{evaluation}</strong><span>Evaluation runs</span></div><div><strong>{training}</strong><span>Training runs</span></div><div><strong>{groups}</strong><span>Comparisons</span></div></div>
<p><a href="experiment.json">Complete result JSON</a> · {len(nodes)} nodes · {len(plan.get('steps', []))} planned phases</p>{failure_html}<nav class="tree-nav" aria-label="Experiment nodes">{navigation}</nav><div id="experiment-tree" tabindex="-1">{render_node('')}</div><footer><a class="button" href="#experiment-top">Back to top</a><p>Offline report · Full run details load only when opened · Missing measurements remain unavailable.</p></footer>
</main>{COMPARISON_SCRIPT}</body></html>'''


OVERVIEW_POLISH_CSS = """
html{scroll-behavior:smooth}button,input,select{font:inherit}.skip-link{position:fixed;top:-100px;left:1rem;background:white;z-index:20;padding:.75rem 1rem}.skip-link:focus{top:1rem}
.node{scroll-margin-top:1rem}.phase-title{font-weight:700;margin:0 0 .5rem}.phase-title .badge{margin-left:.35rem}.phases .phase{min-width:0}
.intro{font-size:.9rem;line-height:1.55;margin:.5rem 0}.summary{margin:1rem 0}.summary>div{padding:.8rem}.node-header .path{margin-bottom:.4rem}
.branch-nav{display:flex;gap:.5rem;flex-wrap:wrap;margin:0 0 .7rem}.branch-nav a{display:inline-flex;align-items:center;min-height:40px;padding:.25rem .65rem;background:#edf3fa;border-radius:6px;text-decoration:none;font-size:.8rem}
.root-header{align-items:center;margin-bottom:.65rem}.root-header h2{font-size:1rem;margin:0}
.phase-notes{display:flex;flex-wrap:wrap;gap:.4rem 1.3rem;margin-top:.75rem;padding:.6rem .8rem;border-left:2px solid #c8d9ea;background:#f6f9fc;border-radius:0 7px 7px 0}
.phase-notes .phase-note{margin:0;flex:1 1 300px;font-size:.8rem;line-height:1.65}.phase .run-id{font-size:.76rem;line-height:1.7;margin:.6rem 0}
.badge[data-status=completed]{background:#e2f1e8;color:#1e5e3c}.badge[data-status=failed]{background:#fbe7e4;color:#8b2922}.badge[data-status=warning]{background:#fff0d8;color:#79501a}
.button,.data-link{min-height:44px;display:inline-flex;align-items:center}.button:hover{background:#eaf2fc}.button.primary:hover{background:#104f98}.tree-nav a{min-height:36px;display:inline-flex;align-items:center}
.grouped>h3,.grouped>h4,.grouped>h5,.grouped>h6{font-size:1.1rem;line-height:1.5}.quick-comparison h3,.quick-comparison h4,.quick-comparison h5,.quick-comparison h6{font-size:1.05rem;margin:0 0 .5rem}
footer{margin-top:1.5rem;color:#465e76;font-size:.8rem}.actions .unavailable{padding:.5rem 0}
@media(min-width:900px){.tree-nav{position:sticky;top:0;z-index:8;background:#f2f6fa;padding:.65rem 0;border-bottom:1px solid #ccdbe9}.node{scroll-margin-top:95px}}
@media(max-width:640px){.node{padding:.85rem .65rem}.children{margin-top:1rem;padding-left:.45rem}.phase{padding:.8rem}.phase-notes{padding:.6rem}.summary{grid-template-columns:repeat(2,minmax(0,1fr))}.summary strong{font-size:1.2rem}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:white}main{padding:0;max-width:none}.tree-nav,.skip-link,footer{display:none}.node{padding:.7rem;break-inside:auto}.phase-notes{font-size:.75rem}.button{border:0;padding:.2rem}}
"""
