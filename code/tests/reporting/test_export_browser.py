"""Optional offline UX regressions; run with Playwright and Chromium installed."""
import base64
import json

import pytest

playwright = pytest.importorskip('playwright.sync_api')

from ovlab_runner.comparison_index import COMPARISON_CSS, COMPARISON_SCRIPT, render_quick_comparison
from ovlab_runner.isolated_index import render_isolated_index


@pytest.fixture(scope='module')
def browser():
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch(headless=True, args=['--allow-file-access-from-files'])
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip('Install Chromium with: python -m playwright install chromium')
            raise
        yield instance
        instance.close()


@pytest.fixture
def comparison_page(tmp_path, browser):
    summary = {'status': 'comparable', 'task_ids': ['task-a'], 'metrics': [
        {'key': 'task.success_rate', 'label': 'Success', 'unit': '%', 'delta': 'pp', 'comparable': True},
        {'key': 'inference_ms_per_action', 'label': 'Inference', 'unit': 'ms', 'delta': 'percent', 'comparable': True}],
        'rows': [{'run_id': run, 'label': label, 'episodes': 1,
                  'values': {'task.success_rate': success, 'inference_ms_per_action': speed}}
                 for run, label, success, speed in [('run-a', 'First run', 0, 0), ('run-b', 'Second run', 50, 1), ('run-c', 'Missing data', None, None)]]}
    target = tmp_path / 'index.html'
    target.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Comparison</title><style>body{margin:0;font:16px system-ui}main{max-width:1100px;margin:auto;min-width:0}' + COMPARISON_CSS + '</style></head><body><main><h1>Experiment</h1>'
                      + render_quick_comparison(summary, scope_label='Model variants') + '</main>' + COMPARISON_SCRIPT + '</body></html>')
    pages = []
    def open_page(*, width=1200, javascript=True):
        page = browser.new_page(viewport={'width': width, 'height': 900}, java_script_enabled=javascript, reduced_motion='reduce')
        page.goto(target.as_uri())
        pages.append(page)
        return page
    yield open_page
    for page in pages:
        page.close()


def test_reference_changes_work_with_keyboard_and_missing_data_stays_missing(comparison_page):
    page = comparison_page()
    rows = page.locator('tbody tr')
    assert rows.nth(0).locator('.delta').first.inner_text() == 'Reference'
    assert rows.nth(1).locator('.delta').nth(0).inner_text() == '+50.0 pp'
    assert rows.nth(1).locator('.delta').nth(1).inner_text() == 'No %: reference is zero'
    control = page.get_by_role('combobox', name='Reference for percentage comparison: Model variants')
    control.focus()
    page.keyboard.press('ArrowDown')
    assert rows.nth(1).locator('.reference-badge').is_visible()
    assert rows.nth(0).locator('.delta').nth(0).inner_text() == '-50.0 pp'
    assert rows.nth(0).locator('.delta').nth(1).inner_text() == '-100.0%'
    assert rows.nth(2).locator('.delta').nth(1).inner_text() == 'No recorded value'
    assert 'Second run' in page.get_by_role('status').inner_text()


def test_mobile_table_keeps_variant_name_visible_and_scrolls_with_keyboard(comparison_page):
    page = comparison_page(width=320)
    region = page.get_by_role('region', name='Comparison values: Model variants')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    region.focus()
    page.keyboard.press('ArrowRight')
    page.wait_for_function("document.querySelector('.comparison-scroll').scrollLeft>0")
    first = page.locator('tbody tr th').first.bounding_box()
    bounds = region.bounding_box()
    assert abs(first['x'] - bounds['x']) < 3
    assert page.locator('[data-scroll-hint]').is_visible()


def test_no_javascript_keeps_values_visible_and_disables_reference_switching(comparison_page):
    page = comparison_page(javascript=False)
    assert page.get_by_role('combobox').is_disabled()
    assert page.locator('tbody tr').count() == 3
    assert 'Reference switching requires JavaScript' in page.inner_text('body')
    assert '+50.0 pp' in page.inner_text('body')


def test_figure_preview_closes_with_escape_and_restores_keyboard_focus(tmp_path, browser):
    figure = tmp_path / 'overview/figures/example.png'
    figure.parent.mkdir(parents=True)
    figure.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='))
    metadata = {'export_type': 'isolated', 'source': {'run': 'run-a', 'scope': 'run', 'episodes': []}}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    target = tmp_path / 'index.html'
    target.write_text(render_isolated_index(tmp_path, category='evaluate', metadata=metadata))
    page = browser.new_page()
    try:
        page.goto(target.as_uri())
        trigger = page.locator('a[data-preview]').first
        trigger.focus()
        page.keyboard.press('Enter')
        dialog = page.get_by_role('dialog')
        assert dialog.is_visible()
        assert dialog.get_by_role('button', name='Close preview').evaluate('el=>el===document.activeElement')
        page.keyboard.press('Escape')
        assert not dialog.is_visible()
        assert trigger.evaluate('el=>el===document.activeElement')
    finally:
        page.close()
