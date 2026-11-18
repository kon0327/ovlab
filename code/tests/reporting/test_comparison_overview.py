"""Comparable summaries and short offline paths must preserve scope and evidence."""
import csv
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ovlab_runner.comparison_index import build_evaluation_comparison, render_quick_comparison
from ovlab_runner.report_views import publish_report_view, report_page


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metadata():
    return {'source': {'runs': ['a', 'b']}, 'comparison': {'status': 'comparable', 'compared_task_ids': ['shared']}}


def _metrics(root, *, mismatch=False):
    _csv(root / 'tables/metric-summary.csv', [
        {'run_id': run, 'task_id': task, 'metric_id': 'task.success_rate', 'status': 'available', 'value': value,
         'unit': 'ratio', 'metric_version': '1', 'metric_config_hash': 'other' if mismatch and run == 'b' else 'same'}
        for run, task, value in [('a', 'shared', 0), ('b', 'shared', .5), ('a', 'excluded', 9999)]])


def _model(latency, n, steps):
    return {'tasks': [{'task_id': 'shared', 'episodes': [{'executed_step_count': steps, 'trace_view': {
        'inference_latency_statistics_ms': {'mean': latency, 'n': n},
        'vram_statistics_mib': {'allocated': {'maximum': 1024}},
        'estimated_gflops_statistics': {'mean': 25, 'n': n},
        'estimated_compute_identity': {'method': 'proxy-v1'},
    }}]}]}


def test_quick_comparison_uses_only_shared_tasks_and_amortizes_chunked_calls(tmp_path):
    _metrics(tmp_path)
    summary = build_evaluation_comparison(tmp_path, _metadata(), models={'a': _model(100, 2, 10), 'b': _model(50, 1, 10)})
    values = {row['run_id']: row['values'] for row in summary['rows']}
    assert values['a']['task.success_rate'] == 0
    assert values['b']['task.success_rate'] == 50
    assert values['a']['inference_ms_per_action'] == 20
    assert values['b']['inference_ms_per_action'] == 5
    assert values['a']['allocated_gib'] == 1
    markup = render_quick_comparison(summary)
    assert '+50.0 pp' in markup
    assert '-75.0%' in markup
    assert 'No %: reference is zero' not in markup  # success is a pp delta, even with a zero reference.
    assert '9999' not in markup


def test_missing_or_incompatible_values_never_produce_percentage_claims(tmp_path):
    _metrics(tmp_path, mismatch=True)
    summary = build_evaluation_comparison(tmp_path, _metadata(), models={'a': _model(100, 2, 10)})
    assert summary['rows'][0]['contracts']['task.success_rate'] != summary['rows'][1]['contracts']['task.success_rate']
    assert summary['rows'][1]['values']['inference_ms_per_action'] is None
    markup = render_quick_comparison(summary)
    assert '+50.0 pp' not in markup
    assert 'Not comparable' in markup and 'Not recorded' in markup
    metadata = _metadata()
    metadata['comparison'] = {'status': 'no_shared_tasks', 'compared_task_ids': []}
    assert build_evaluation_comparison(tmp_path, metadata)['metrics'] == []


def test_different_trial_budgets_withhold_deltas_and_missing_peer_does_not_block_others(tmp_path):
    _metrics(tmp_path)
    models = {'a': _model(100, 2, 10), 'b': _model(50, 1, 10)}
    models['a']['tasks'][0]['episodes'][0]['maximum_step_count'] = 520
    models['b']['tasks'][0]['episodes'][0]['maximum_step_count'] = 2
    summary = build_evaluation_comparison(tmp_path, _metadata(), models=models)
    assert '+50.0 pp' not in render_quick_comparison(summary)
    metadata = _metadata()
    metadata['source']['runs'].append('missing')
    models['b']['tasks'][0]['episodes'][0]['maximum_step_count'] = 520
    summary = build_evaluation_comparison(tmp_path, metadata, models=models)
    assert '+50.0 pp' in render_quick_comparison(summary)
    assert '-75.0%' in render_quick_comparison(summary)


def test_relative_delta_handles_zero_reference_and_escapes_labels():
    summary = {'status': 'comparable', 'metrics': [{'key': 'x', 'label': 'Value', 'unit': 'ms', 'delta': 'percent', 'comparable': True}],
               'rows': [{'run_id': run, 'label': '<script>bad</script>', 'values': {'x': value}} for run, value in [('a', 0), ('b', 1)]]}
    markup = render_quick_comparison(summary)
    assert 'No %: reference is zero' in markup
    assert '<script>bad</script>' not in markup


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls.extend(value for key, value in attrs if key in ('href', 'src') and value)


def test_short_static_view_rewrites_long_assets_and_preserves_source(tmp_path):
    root = tmp_path / 'experiment'
    original = root / 'evaluate/model/isolated' / ('run-' + 'x' * 90)
    asset = original / 'episodes/figures' / ('episode-' + 'a' * 100 + '.png')
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'png')
    metadata = original / 'metadata.json'
    metadata.write_text('{}')
    page = original / 'index.html'
    page.write_text(f'<a href="metadata.json">Metadata</a><img src="{asset.relative_to(original)}">')
    before = {path: path.read_bytes() for path in original.rglob('*') if path.is_file()}
    view = publish_report_view(root, original)
    assert view == report_page(root, original)
    assert view.is_relative_to(root / 'v')
    parser = _Links()
    parser.feed(view.read_text())
    for url in parser.urls:
        path = view.parent / unquote(urlsplit(url).path)
        assert path.is_file()
        assert len(path.relative_to(root).as_posix()) < 60
    assert all(path.read_bytes() == contents for path, contents in before.items())
    stamp = view.stat().st_mtime_ns
    assert publish_report_view(root, original).stat().st_mtime_ns == stamp
    page.write_text('<h1>Updated</h1>')
    assert report_page(root, original) == page
    publish_report_view(root, original)
    assert '<h1>Updated</h1>' == view.read_text()


def test_interactive_view_retains_relative_scripts_and_task_navigation(tmp_path):
    root = tmp_path / 'experiment'
    source = root / 'evaluate/branch/interactive/run/profile/build'
    (source / 'assets').mkdir(parents=True)
    (source / 'assets/dashboard.js').write_text('window.ready=true')
    (source / 'tasks/task').mkdir(parents=True)
    (source / 'tasks/task/index.html').write_text('<a href="../../index.html">Run</a>')
    (source / 'index.html').write_text('<script src="assets/dashboard.js"></script><a href="tasks/task/index.html">Task</a>')
    (source / 'manifest.json').write_text('{}')
    target = publish_report_view(root, source)
    assert (target.parent / 'assets/dashboard.js').read_bytes() == (source / 'assets/dashboard.js').read_bytes()
    assert (target.parent / 'tasks/task/index.html').is_file()
    (target.parent / 'assets/dashboard.js').unlink()
    assert report_page(root, source) == source / 'index.html'
    publish_report_view(root, source)
    assert (target.parent / 'assets/dashboard.js').is_file()
    assert publish_report_view(root, tmp_path / 'outside') is None


def test_comparison_accessibility_reference_and_small_values():
    summary = {'status': 'comparable', 'task_ids': ['task'], 'metrics': [
        {'key': 'inference_ms_per_action', 'label': 'Inference', 'unit': 'ms', 'delta': 'percent', 'comparable': True}],
        'rows': [{'run_id': 'a', 'label': 'Reference run', 'episodes': 1, 'values': {'inference_ms_per_action': .000002}},
                 {'run_id': 'b', 'label': 'Other run', 'episodes': 1, 'values': {'inference_ms_per_action': .0000020001}}]}
    markup = render_quick_comparison(summary, scope_label='family A')
    assert 'class="reference-badge">Reference' in markup
    assert 'data-tone="reference">Reference' in markup
    assert '+0.005%' in markup
    assert 'Recorded value: 2e-06' in markup
    assert 'Small evaluation sample' in markup
    assert 'role="region"' in markup and 'tabindex="0"' in markup
    assert '<caption>' in markup and 'aria-live="polite"' in markup
    assert 'family A' in markup
    assert '0 shared tasks' not in render_quick_comparison({**summary, 'task_ids': []})


def test_empty_numeric_summary_explains_missing_data_without_claiming_incompatibility():
    markup = render_quick_comparison({'status': 'comparable', 'metrics': [], 'rows': []})
    assert 'not enough recorded numeric values' in markup
    assert 'unavailable: comparable' not in markup


def test_phase_progress_counts_and_reporting_failures_remain_distinct(tmp_path):
    from ovlab_benchctl.experiment_overview import build_experiment_index
    plan = {'id': 'study', 'name': 'Study', 'selected_nodes': ['', 'a', 'b'], 'nodes': [
        {'path': '', 'id': 'study', 'children': ['a', 'b'], 'execute': []},
        {'path': 'a', 'id': 'a', 'children': [], 'execute': ['evaluate']},
        {'path': 'b', 'id': 'b', 'children': [], 'execute': ['evaluate']}]}
    result = {'experiment_id': 'study', 'status': 'running', 'nodes': {}, 'grouped': {}}
    markup = build_experiment_index(plan, result, tmp_path)
    assert '0 of 2 child runs completed' in markup
    assert 'Waiting to run' in markup
    assert '0 child nodes' not in markup and 'Not scheduled' not in markup
    result['nodes'] = {'a': {'evaluate': {'run_id': 'run-a', 'result': {'status': 'completed'}}},
                       'b': {'evaluate': {'run_id': 'run-b', 'result': {'status': 'failed'}}}}
    result['failures'] = [{'node': 'a', 'phase': 'evaluate', 'message': 'Report publication failed'}]
    markup = build_experiment_index(plan, result, tmp_path)
    assert '1 of 2 child runs completed' in markup
    assert 'Run completed · reporting issue' in markup
    assert 'Report publication failed' in markup
    assert 'data-status="failed"' in markup
    assert 'Skip to experiment results' in markup


def test_short_views_offer_experiment_return_from_run_and_task_and_rebase_video(tmp_path):
    import os
    root = tmp_path / 'data/exports/study'
    root.mkdir(parents=True)
    (root / 'index.html').write_text('<h1>Study</h1>')
    source = root / 'evaluate/model/interactive/run/profile/build'
    (source / 'assets').mkdir(parents=True)
    (source / 'assets/dashboard.js').write_text('window.ready=true')
    (source / 'tasks/task').mkdir(parents=True)
    document = '<html><body><aside><div class="related"><p>Navigation</p></div></aside><main><header>Report</header></main></body></html>'
    (source / 'index.html').write_text(document)
    (source / 'tasks/task/index.html').write_text(document)
    canonical = tmp_path / 'data/runs/study/evaluate/model/run/video.mp4'
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b'canonical-video')
    (source / 'data').mkdir()
    payload = 'window.receive(' + json.dumps({'video_href': os.path.relpath(canonical, source)}) + ');'
    (source / 'data/episode.js').write_text(payload)
    (source / 'manifest.json').write_text('{}')
    target = publish_report_view(root, source)
    for page in (target, target.parent / 'tasks/task/index.html'):
        parser = _Links()
        parser.feed(page.read_text())
        assert '← Experiment overview' in page.read_text()
        assert any((page.parent / unquote(urlsplit(url).path)).resolve() == root / 'index.html' for url in parser.urls)
    rewritten = (target.parent / 'data/episode.js').read_text()
    assert 'a/video-' in rewritten
    assert next((target.parent / 'a').glob('*.mp4')).read_bytes() == b'canonical-video'
    assert (source / 'data/episode.js').read_text() == payload
