"""Small, offline comparison summaries; no episode records reach the overview UI."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from statistics import mean

SCHEMA = "ovlab.quick-comparison/v1"


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rows(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_evaluation_comparison(root: Path, metadata, models=None, labels=None):
    """Compare equal-weight shared tasks, preserving missing and incompatible data."""
    models, labels = models or {}, labels or {}
    source = metadata.get("source", {})
    comparison = metadata.get("comparison", {})
    runs = source.get("runs", [])
    tasks = comparison.get("compared_task_ids", [])
    comparable = comparison.get("status") == "comparable" and bool(tasks)
    rows = [{"run_id": run, "label": labels.get(run, run), "values": {}, "contracts": {}, "episodes": 0} for run in runs]
    metrics = []
    notes = ["Shared tasks have equal weight; variants are shown individually, never averaged into a model family.",
             "Percent deltas are relative to the selected reference; success uses percentage points (pp). Zero references have no relative percent delta.",
             "Performance describes these recorded runs, not a controlled hardware benchmark. VRAM is PyTorch allocated memory, not whole-device utilization.",
             "Small episode counts are descriptive evidence, not a statistically established ranking."]
    # Matching task IDs alone do not justify deltas when the recorded trial
    # seeds, initial states or episode budgets differ.
    trials = {}
    for run in runs:
        model = models.get(run, {})
        if not model:
            continue
        selected = [task for task in model.get("tasks", []) if task.get("task_id") in tasks]
        trials[run] = json.dumps([(task["task_id"], sorted([
            [episode.get("seed"), episode.get("maximum_step_count"), episode.get("initial_state_identity")]
            for episode in task.get("episodes", [])], key=lambda x: json.dumps(x, sort_keys=True)))
            for task in sorted(selected, key=lambda x: x["task_id"])], sort_keys=True)
    if len(set(trials.values())) > 1:
        notes.append("Percent deltas are withheld between runs with different recorded trial seeds, initial states or episode budgets.")
    if not comparable:
        return {"schema_version": SCHEMA, "status": comparison.get("status", "unavailable"),
                "task_ids": tasks, "metrics": [], "rows": rows, "notes": notes}
    csv_metrics = _rows(root / "tables/metric-summary.csv")
    episode_rows = _rows(root / "tables/episode-summary.csv")
    for row in rows:
        row["episodes"] = sum(x.get("run_id") == row["run_id"] and x.get("task_id") in tasks for x in episode_rows)
    for metric_id, label, unit, delta in (
        ("task.success_rate", "Success", "%", "pp"),
        ("action.smoothness_1", "Action change", "action/step", "percent"),
        ("failure.invalid_prediction_rate", "Invalid predictions", "%", "pp"),
    ):
        key = metric_id
        for row in rows:
            by_task = {x["task_id"]: x for x in csv_metrics if x.get("run_id") == row["run_id"]
                       and x.get("task_id") in tasks and x.get("metric_id") == metric_id}
            values = []
            signatures = []
            for task in tasks:
                item = by_task.get(task, {})
                value = None
                if item.get("status") == "available":
                    try:
                        raw = json.loads(item.get("value") or "null")
                    except ValueError:
                        raw = None
                    if isinstance(raw, dict):
                        raw = raw.get("mean")
                    value = _number(raw)
                    signatures.append((task, item.get("metric_version"), item.get("metric_config_hash"), item.get("unit")))
                values.append(value)
            row["values"][key] = mean(values) * (100 if unit == "%" else 1) if values and all(x is not None for x in values) else None
            row["contracts"][key] = json.dumps([signatures, trials.get(row["run_id"])], sort_keys=True) if row["values"][key] is not None else None
        if any(row["values"][key] is not None for row in rows):
            metrics.append({"key": key, "label": label, "unit": unit, "delta": delta,
                            "comparable": True, "source": "Canonical task metrics on the shared task set"})
    for key, label, unit, mode in (
        ("inference_ms_per_action", "Inference / action", "ms", "inference"),
        ("allocated_gib", "Max observed allocated VRAM", "GiB", "memory"),
        ("gflops_per_action", "Estimated compute / action", "GFLOPs", "compute"),
    ):
        for row in rows:
            methods = set()
            model = models.get(row["run_id"], {})
            by_task = {x["task_id"]: x for x in model.get("tasks", [])}
            values = []
            for task in tasks:
                episodes = by_task.get(task, {}).get("episodes", [])
                per_episode = []
                total, steps = 0., 0
                for episode in episodes:
                    trace = episode.get("trace_view", {})
                    count = _number(episode.get("executed_step_count"))
                    if mode == "memory":
                        value = _number(trace.get("vram_statistics_mib", {}).get("allocated", {}).get("maximum"))
                        per_episode.append(value / 1024 if value is not None else None)
                    else:
                        stats = trace.get("inference_latency_statistics_ms" if mode == "inference" else "estimated_gflops_statistics", {})
                        avg, n = _number(stats.get("mean")), _number(stats.get("n"))
                        if mode == "compute":
                            methods.add((trace.get("estimated_compute_identity") or {}).get("method"))
                        if avg is None or n is None or n <= 0 or count is None or count <= 0:
                            per_episode.append(None)
                        else:
                            total += avg * n
                            steps += count
                            per_episode.append(avg * n / count)
                value = None
                if per_episode and all(x is not None for x in per_episode):
                    value = max(per_episode) if mode == "memory" else total / steps
                values.append(value)
            row["values"][key] = (max(values) if mode == "memory" else mean(values)) if values and all(x is not None for x in values) else None
            row["contracts"][key] = (json.dumps([mode, trials.get(row["run_id"]), sorted(methods)], sort_keys=True)
                                     if row["values"][key] is not None and (mode != "compute" or len(methods) == 1 and None not in methods) else None)
        if any(row["values"][key] is not None for row in rows):
            metrics.append({"key": key, "label": label, "unit": unit, "delta": "percent",
                            "comparable": True,
                            "source": "Shared-task recorded prediction statistics; inference and compute amortized over applied actions"})
    # Quality, speed, memory and workload in an easy-to-scan order.
    order = {name: i for i, name in enumerate(("task.success_rate", "inference_ms_per_action", "allocated_gib", "gflops_per_action", "action.smoothness_1", "failure.invalid_prediction_rate"))}
    metrics.sort(key=lambda x: order[x["key"]])
    return {"schema_version": SCHEMA, "status": "comparable", "task_ids": tasks,
            "metrics": metrics, "rows": rows, "notes": notes}


def build_training_comparison(comparison):
    rows = [{"run_id": x["run_id"], "label": x.get("profile_id") or x["run_id"], "episodes": None,
             "values": {"loss": _number(x.get("final_training_loss")),
                        "vram": (_number(x.get("peak_allocated_bytes")) or 0) / 1024**3 if _number(x.get("peak_allocated_bytes")) is not None else None}}
            for x in comparison.get("dataset_coverage", [])]
    metrics = [{"key": "loss", "label": "Final training loss", "unit": "loss", "delta": "percent", "comparable": True},
               {"key": "vram", "label": "Peak allocated VRAM", "unit": "GiB", "delta": "percent", "comparable": True}]
    if comparison.get("status") != "comparable":
        metrics = []
    return {"schema_version": SCHEMA, "status": comparison.get("status", "unavailable"), "task_ids": [],
            "metrics": metrics, "rows": rows, "notes": [comparison.get("basis", "Training compatibility was not recorded."),
            "Training and evaluation are never mixed. Performance deltas describe recorded execution; no overall score is inferred."]}


def _escape(value):
    return html.escape(str(value), quote=True)


def _contract_reason(contract, reference, key):
    try:
        left, right = json.loads(contract), json.loads(reference)
        if key == 'gflops_per_action' and left[-1] != right[-1]:
            return 'Compute estimators differ'
        if key in ('task.success_rate', 'action.smoothness_1', 'failure.invalid_prediction_rate') and left[0] != right[0]:
            return 'Metric definitions differ'
    except (ValueError, TypeError, IndexError):
        pass
    return 'Trial settings or definitions differ'


def _change(value, baseline, metric, contract, reference_contract, is_reference=False):
    if value is None:
        return 'No recorded value', 'unavailable'
    if is_reference:
        return 'Reference', 'reference'
    if baseline is None:
        return 'Reference has no value', 'unavailable'
    if not metric.get('comparable', True) or contract is None or contract != reference_contract:
        return 'Not comparable · ' + _contract_reason(contract, reference_contract, metric['key']), 'unavailable'
    if value == baseline:
        return 'No change', 'neutral'
    if metric['delta'] != 'pp' and baseline == 0:
        return 'No %: reference is zero', 'unavailable'
    amount = value - baseline if metric['delta'] == 'pp' else 100 * (value - baseline) / abs(baseline)
    text = f'{amount:+.2g}' if abs(amount) < .05 else f'{amount:+.1f}'
    preferred = _direction(metric['key'])
    tone = 'favorable' if preferred and amount * preferred > 0 else 'unfavorable' if preferred else 'neutral'
    return text + (' pp' if metric['delta'] == 'pp' else '%'), tone


def _direction(key):
    if key == 'task.success_rate':
        return 1
    if key in ('inference_ms_per_action', 'allocated_gib', 'gflops_per_action', 'failure.invalid_prediction_rate', 'vram'):
        return -1
    return 0


def render_quick_comparison(summary, *, labels=None, links=None, scope_label=None, heading_level=3):
    import hashlib
    labels, links = labels or {}, links or {}
    rows, metrics = summary.get('rows', []), summary.get('metrics', [])
    scope_label = scope_label or 'This group'
    heading = min(6, max(2, heading_level))
    identity = 'comparison-' + hashlib.sha256((scope_label + json.dumps([row['run_id'] for row in rows])).encode()).hexdigest()[:12]
    if not metrics or len(rows) < 2:
        reasons = {
            'no_shared_tasks': 'These runs have no shared tasks. Their results remain available separately.',
            'different_benchmark_suites': 'These runs use different benchmark suites.',
            'incompatible_metrics': 'The recorded metric definitions are incompatible.',
            'not_comparable': 'Dataset, task coverage or training objective differs or was not recorded.',
            'comparable': 'There are not enough recorded numeric values for a percentage comparison.',
        }
        reason = reasons.get(summary.get('status'), 'Comparable data is not available for this selection.')
        return f'<section class="quick-comparison"><h{heading}>Quick comparison</h{heading}><p>' + _escape(reason) + '</p><p class="coverage">Use the report tables and metadata for the complete evidence.</p></section>'
    raw_labels = [labels.get(row['run_id'], row.get('label') or row['run_id']) for row in rows]
    def display_label(row):
        label = labels.get(row['run_id'], row.get('label') or row['run_id'])
        return label + ' · ' + row['run_id'][-8:] if raw_labels.count(label) > 1 else label
    options = ''.join(f'<option value="{i}">{_escape(display_label(row))}</option>' for i, row in enumerate(rows))
    hints = {'task.success_rate': 'Higher success', 'inference_ms_per_action': 'Lower time', 'allocated_gib': 'Lower allocation',
             'gflops_per_action': 'Lower estimated cost', 'action.smoothness_1': 'No preferred direction',
             'failure.invalid_prediction_rate': 'Lower failure rate', 'vram': 'Lower allocation', 'loss': 'Training objective'}
    head = ''.join(f'<th scope="col">{_escape(metric["label"])}<small>{_escape(metric["unit"])}</small><small class="metric-direction">{_escape(hints.get(metric["key"], "Recorded value"))}</small></th>' for metric in metrics)
    body = []
    baseline = rows[0]['values']
    for index, row in enumerate(rows):
        label = _escape(display_label(row))
        if row['run_id'] in links:
            label = f'<a href="{_escape(links[row["run_id"]])}">{label}</a>'
        cells = []
        for metric in metrics:
            value = _number(row['values'].get(metric['key']))
            contract = row.get('contracts', {}).get(metric['key'], 'legacy')
            reference_contract = rows[0].get('contracts', {}).get(metric['key'], 'legacy')
            change, tone = _change(value, _number(baseline.get(metric['key'])), metric, contract, reference_contract, index == 0)
            display = f'{value:,.6g}' if value is not None else 'Not recorded'
            cells.append(f'<td title="Recorded value: {value if value is not None else "Not recorded"}" data-metric="{_escape(metric["key"])}" data-direction="{_direction(metric["key"])}" data-contract="{_escape(contract)}" data-value="{value if value is not None else ""}" data-kind="{metric["delta"]}" data-comparable="{str(metric.get("comparable", True) and contract is not None).lower()}"><strong>{_escape(display)}</strong><small class="delta" data-tone="{tone}">{_escape(change)}</small></td>')
        reference_badge = '<span class="reference-badge">Reference</span>' if index == 0 else '<span class="reference-badge" hidden>Reference</span>'
        body.append(f'<tr data-run="{index}" class="{"is-reference" if index == 0 else ""}"><th scope="row">{label}{reference_badge}<small class="comparison-run-id">{_escape(row["run_id"])}</small>' + (f'<small>{row["episodes"]} episodes</small>' if row.get('episodes') is not None else '') + '</th>' + ''.join(cells) + '</tr>')
    tasks = summary.get('task_ids', [])
    scope = f'{len(tasks)} shared tasks · {len(rows)} runs' if tasks else f'Training scope · {len(rows)} runs'
    task_ids = f'<span class="task-ids">{_escape(", ".join(tasks))}</span>' if tasks else ''
    episode_counts = [row['episodes'] for row in rows if isinstance(row.get('episodes'), int)]
    warning = ''
    if episode_counts and min(episode_counts) < max(5, len(tasks) * 5):
        low, high = min(episode_counts), max(episode_counts)
        samples = str(low) if low == high else f'{low}–{high}'
        warning = f'<p class="sample-warning"><strong>Small evaluation sample:</strong> {samples} episodes per run. Treat differences as descriptive; this is not an established ranking.</p>'
    notes = ''.join(f'<li>{_escape(note)}</li>' for note in summary.get('notes', []))
    initial = _escape(display_label(rows[0]))
    return (f'<section class="quick-comparison" data-comparison aria-label="Quick comparison: {_escape(scope_label)}"><h{heading}>Quick comparison</h{heading}>'
            f'<p class="coverage">{scope}{task_ids}</p>{warning}'
            f'<div class="comparison-controls"><label for="{identity}-reference">Compare against<select id="{identity}-reference" aria-label="Reference for percentage comparison: {_escape(scope_label)}" aria-describedby="{identity}-help" disabled>{options}</select></label>'
            f'<p id="{identity}-help" class="comparison-help">Changes are relative to the reference. <abbr title="percentage points">pp</abbr> compares rates; % compares other values. Color describes numeric direction, not an overall model score.</p></div>'
            '<noscript><p class="comparison-help">Reference switching requires JavaScript. The recorded values and changes against the first run remain visible below.</p></noscript>'
            f'<p class="comparison-status" role="status" aria-live="polite" aria-atomic="true">Reference: {initial}</p>'
            f'<p id="{identity}-scroll" class="comparison-scroll-hint" data-scroll-hint>More columns: scroll the table horizontally. Keyboard: focus the table and use Left/Right.</p>'
            f'<div class="comparison-scroll" tabindex="0" role="region" aria-label="Comparison values: {_escape(scope_label)}" aria-describedby="{identity}-scroll"><table><caption>Recorded values and changes versus the reference</caption><thead><tr><th scope="col">Run / variant</th>{head}</tr></thead><tbody>'
            + ''.join(body) + '</tbody></table></div><ul class="comparison-notes">' + notes + '</ul></section>')


COMPARISON_CSS = """
.quick-comparison{margin:1rem 0;padding:1.15rem;border:1px solid #bfcfe0;border-radius:12px;background:#f7faff;min-width:0}
.quick-comparison h3{margin:0 0 .5rem}.comparison-controls{display:flex;flex-wrap:wrap;align-items:flex-end;gap:.75rem 1.5rem}
.quick-comparison label{display:block;font-weight:650;font-size:.88rem;max-width:100%}
.quick-comparison select{display:block;width:min(100%,430px);max-width:100%;min-height:44px;margin-top:.35rem;padding:.6rem;border:1px solid #8ba6bf;border-radius:7px;background:white;color:#18344b;font:inherit}
.quick-comparison :focus-visible{outline:3px solid #b85808;outline-offset:3px}.comparison-help{flex:1 1 260px;max-width:65ch;font-size:.8rem;color:#415b74;margin:.3rem 0}
.comparison-scroll{overflow:auto;max-height:680px;margin-top:.5rem;border:1px solid #ccdbe9;border-radius:8px;position:relative;overscroll-behavior-inline:contain}
.quick-comparison table{border-collapse:separate;border-spacing:0;width:100%;font-size:.86rem;line-height:1.45}
.quick-comparison caption{text-align:left;padding:.6rem .8rem;background:#eaf1f9;color:#3a536b;font-size:.78rem}
.quick-comparison th,.quick-comparison td{text-align:left;padding:.8rem;border:0;border-bottom:1px solid #d3dfeb;min-width:135px;vertical-align:top;white-space:normal}
.quick-comparison thead th{position:sticky;top:0;z-index:2;background:#eaf1f9;font-weight:650;color:#203c55}
.quick-comparison th:first-child{position:sticky;left:0;min-width:230px;max-width:310px;width:26%;z-index:1;background:#f7faff;box-shadow:1px 0 #d3dfeb}
.quick-comparison thead th:first-child{z-index:3;background:#eaf1f9}
.quick-comparison small{display:block;color:#465e76;font-weight:400;overflow-wrap:anywhere;font-size:.76rem;white-space:normal}
.quick-comparison .metric-direction{font-size:.7rem;margin-top:.2rem;color:#546b80}
.quick-comparison .comparison-run-id{font: .7rem/1.5 ui-monospace,monospace;margin:.35rem 0;overflow-wrap:anywhere;word-break:break-word}
.quick-comparison td strong{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:700}
.quick-comparison .delta{font-variant-numeric:tabular-nums;margin-top:.35rem;line-height:1.5;max-width:23ch}
.quick-comparison .delta[data-tone=favorable]{color:#17613f}.quick-comparison .delta[data-tone=unfavorable]{color:#913a22}.quick-comparison .delta[data-tone=reference]{font-weight:650;color:#204f88}
.quick-comparison tr.is-reference>th,.quick-comparison tr.is-reference>td{background:#e3eefc}
.quick-comparison tbody tr:hover>th,.quick-comparison tbody tr:hover>td{background:#edf4fc}
.reference-badge{display:inline-block;padding:.1rem .35rem;margin:.3rem 0 0 .4rem;border-radius:4px;background:#d0e1f7;color:#204c7b;font-size:.68rem;font-weight:600}.reference-badge[hidden]{display:none}
.comparison-notes{font-size:.8rem;color:#465e76;padding-left:1.3rem;margin:1rem 0 0}.comparison-notes li{margin:.35rem 0}
.coverage{color:#415b74;font-size:.86rem}.task-ids{display:block;font: .74rem/1.6 ui-monospace,monospace;margin-top:.25rem;overflow-wrap:anywhere}
.sample-warning{padding:.65rem .8rem;background:#fff5e5;border-left:3px solid #b67923;color:#6d471b;font-size:.82rem;line-height:1.5;margin:.75rem 0 1rem}
.comparison-status{font-size:.78rem;color:#415b74;overflow-wrap:anywhere;margin:.65rem 0 .3rem}
.comparison-scroll-hint{font-size:.76rem;color:#415b74;margin:.3rem 0}.comparison-scroll-hint[hidden]{display:none}
@media(max-width:640px){.quick-comparison{padding:.85rem .6rem}.quick-comparison th,.quick-comparison td{padding:.6rem;min-width:120px}.quick-comparison th:first-child{min-width:155px;max-width:180px;width:155px}.comparison-controls{display:block}.quick-comparison label,.quick-comparison select{width:100%}.comparison-help{margin-top:.7rem}.comparison-scroll{max-height:620px}}
@media print{.comparison-scroll{overflow:visible;max-height:none}.quick-comparison th,.quick-comparison td{min-width:0!important;position:static!important}.quick-comparison table{font-size:9px}.comparison-controls select{border:0}.quick-comparison{break-inside:auto}.quick-comparison th:first-child{max-width:none}}
"""
COMPARISON_SCRIPT = """<script>
document.querySelectorAll('[data-comparison]').forEach(section=>{
 const select=section.querySelector('select'),rows=[...section.querySelectorAll('tbody tr')],status=section.querySelector('[role=status]');
 function reason(cell,ref){try{const a=JSON.parse(cell.dataset.contract),b=JSON.parse(ref.dataset.contract);
 if(cell.dataset.metric==='gflops_per_action'&&JSON.stringify(a.at(-1))!==JSON.stringify(b.at(-1)))return 'Compute estimators differ';
 if(['task.success_rate','action.smoothness_1','failure.invalid_prediction_rate'].includes(cell.dataset.metric)&&JSON.stringify(a[0])!==JSON.stringify(b[0]))return 'Metric definitions differ';}catch(error){}return 'Trial settings or definitions differ';}
 function update(announce=false){const ref=rows[Number(select.value)],baseline=[...ref.querySelectorAll('td')];
 rows.forEach(row=>{const selected=row===ref;row.classList.toggle('is-reference',selected);row.querySelector('.reference-badge').hidden=!selected;
 [...row.querySelectorAll('td')].forEach((cell,i)=>{const base=baseline[i],raw=cell.dataset.value,referenceRaw=base.dataset.value;let text,tone='unavailable';
 if(raw==='')text='No recorded value';else if(selected){text='Reference';tone='reference';}
 else if(referenceRaw==='')text='Reference has no value';
 else if(cell.dataset.comparable!=='true'||cell.dataset.contract!==base.dataset.contract)text='Not comparable · '+reason(cell,base);
 else {const value=Number(raw),reference=Number(referenceRaw);if(value===reference){text='No change';tone='neutral';}
 else if(cell.dataset.kind!=='pp'&&reference===0)text='No %: reference is zero';
 else {const change=cell.dataset.kind==='pp'?value-reference:100*(value-reference)/Math.abs(reference);
 const formatted=Math.abs(change)<.05?(change>0?'+':'-')+Math.abs(change).toPrecision(2):(change>0?'+':'')+change.toFixed(1);
 text=formatted+(cell.dataset.kind==='pp'?' pp':'%');const direction=Number(cell.dataset.direction);tone=direction?(change*direction>0?'favorable':'unfavorable'):'neutral';}}
 const delta=cell.querySelector('.delta');delta.textContent=text;delta.dataset.tone=tone;});});
 if(announce)status.textContent='Reference: '+select.selectedOptions[0].textContent+'. Changes recalculated for '+rows.length+' runs.';}
 const scroll=section.querySelector('.comparison-scroll'),hint=section.querySelector('[data-scroll-hint]');
 const resize=()=>{hint.hidden=scroll.scrollWidth<=scroll.clientWidth+2;};
 if(typeof ResizeObserver!=='undefined'){new ResizeObserver(resize).observe(scroll);}else window.addEventListener('resize',resize);
 select.addEventListener('change',()=>update(true));select.disabled=false;update();resize();});
</script>"""
