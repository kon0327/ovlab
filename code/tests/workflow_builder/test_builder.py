"""Offline authoring regressions and parity with the actual workflow resolver."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from ovlab_benchctl.experiment_workflow import ExperimentWorkflow
from ovlab_benchctl.strict_yaml import loads

ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / 'code/apps/workflow-builder'
NODE = os.environ.get('OVLAB_NODE') or shutil.which('node')


@pytest.mark.skipif(not NODE, reason='Node.js is needed for browser-core/resolver parity')
def test_exported_repository_workflows_match_the_real_resolver(tmp_path):
    script = r'''
const fs=require('node:fs'),vm=require('node:vm'),C=require('./core.js');
const context={window:{}};
vm.runInNewContext(fs.readFileSync('./catalog.js','utf8'),context);
const catalog=JSON.parse(JSON.stringify(context.window.OVLAB_BUILDER_CATALOG));
console.log(JSON.stringify(catalog.workflows.map(source=>{
 const analysis=C.analyze(source.document,catalog);
 return {ref:source.ref,yaml:C.dumpYaml(source.document),
   steps:analysis.rows.map(row=>({node:row.path,phase:row.phase})),
   resolved:analysis.entries.flatMap(entry=>['train','evaluate'].filter(phase=>entry.state[phase]?.profile).map(phase=>{
     const resolved=C.resolve(entry,phase,catalog);
     return {path:entry.path,phase,document:resolved.document,components:resolved.components};
   }))};
})));
'''
    result = subprocess.run([NODE, '-e', script], cwd=BUILDER, capture_output=True, text=True, check=True)
    sources = json.loads(result.stdout)
    assert len(sources) >= 6
    for index, source in enumerate(sources):
        path = tmp_path / f'workflow-{index}.yaml'
        path.write_text(source['yaml'])
        assert loads(source['yaml'])['kind'] == 'workflow'
        plan = ExperimentWorkflow(ROOT).plan(path)
        assert source['steps'] == plan['steps'], source['ref']
        nodes = {node['path']: node for node in plan['nodes']}
        for selected in source['resolved']:
            configuration = nodes[selected['path']]['configuration'][selected['phase']]
            assert selected['document'] == configuration['document'], (source['ref'], selected['path'])
            for kind, component in configuration.get('component_documents', {}).items():
                assert selected['components'][kind] == component['document']


@pytest.fixture(scope='module')
def browser():
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch(headless=True)
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip('Install a local Chromium browser with python -m playwright install chromium')
            raise
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
    current = context.new_page()
    errors, remote = [], []
    current.on('pageerror', lambda error: errors.append(str(error)))
    current.on('request', lambda request: remote.append(request.url) if request.url.startswith(('http:', 'https:')) else None)
    current.goto((BUILDER / 'index.html').as_uri())
    current.wait_for_load_state('networkidle')
    yield current
    context.close()
    assert errors == []
    assert remote == [], 'The editor must work without a CDN, backend or network requests'


def document(page):
    return page.evaluate('OVLABStudio.getDocument()')


def choose(page, path):
    page.locator(f'#outline [data-select="{path}"]').click()


def edit(page, selector, value):
    element = page.locator(selector)
    element.fill(str(value))
    element.press('Tab')


def export_plan(page, tmp_path):
    with page.expect_download() as download:
        page.get_by_role('button', name='Export YAML', exact=True).click()
    target = tmp_path / download.value.suggested_filename
    download.value.save_as(target)
    return ExperimentWorkflow(ROOT).plan(target)


def test_offline_default_export_uses_its_own_training_checkpoint(page, tmp_path):
    assert page.locator('#plan-count').inner_text() == '2'
    assert page.locator('#validation-status').inner_text() == 'Browser checks passed'
    page.get_by_role('tab', name='Execution plan').click()
    assert page.locator('.plan-table tbody tr').count() == 2
    assert '<root>:train' in page.locator('.plan-table').inner_text()
    plan = export_plan(page, tmp_path)
    evaluation = plan['nodes'][1]['configuration']['evaluate']
    assert evaluation['checkpoint_from'] == ''
    assert plan['steps'] == [{'node': '', 'phase': 'train'}, {'node': 'eval-checkpoint', 'phase': 'evaluate'}]


def test_graphical_build_rename_move_undo_and_export_keep_handoff(page, tmp_path):
    page.get_by_role('button', name='New', exact=True).click()
    page.get_by_role('button', name='Add child node', exact=True).first.click()
    page.locator('[data-template="handoff"]').click()
    edit(page, '#node-id', 'adaptation')
    assert document(page)['children'][0]['children'][0]['config']['evaluate']['checkpoint_from'] == 'adaptation'
    choose(page, '')
    page.get_by_role('button', name='Add child node', exact=True).first.click()
    page.locator('[data-template="group"]').click()
    edit(page, '#node-id', 'models')
    choose(page, 'adaptation')
    page.locator('#node-parent').select_option('models')
    model = document(page)
    assert model['children'][0]['children'][0]['children'][0]['config']['evaluate']['checkpoint_from'] == 'models/adaptation'
    page.get_by_role('button', name='Undo', exact=True).click()
    assert document(page)['children'][0]['id'] == 'adaptation'
    page.get_by_role('button', name='Redo', exact=True).click()
    assert page.locator('#inspector .inspector-header p').inner_text() == 'models/adaptation'
    plan = export_plan(page, tmp_path)
    assert [step['node'] for step in plan['steps']] == ['models/adaptation', 'models/adaptation/full']


def test_duplicate_parent_execution_has_actionable_warning(page):
    page.locator('#inspector [data-execute="evaluate"]').check()
    assert page.locator('#plan-count').inner_text() == '3'
    assert 'same evaluation' in page.locator('#issues').inner_text()
    page.locator('#inspector [data-execute="evaluate"]').uncheck()
    assert 'same evaluation' not in page.locator('#issues').inner_text()
    assert page.locator('#plan-count').inner_text() == '2'


def test_quic_options_and_invalid_advanced_patch_block_export(page):
    choose(page, 'eval-checkpoint')
    quantization = page.locator('[data-select-field$=".quantization"]')
    assert quantization.locator('option[value="4bit"]').count() == 0
    page.locator('details[data-section="advanced"] summary').click()
    config = document(page)['children'][0].get('config', {})
    config.setdefault('evaluate', {}).setdefault('component_patches', {})['policy'] = {'settings': {'runtime': {'quantization': '4bit'}}}
    page.locator('#advanced-config').fill(json.dumps(config))
    page.get_by_role('button', name='Apply local configuration', exact=True).click()
    assert 'full precision' in page.locator('#issues').inner_text()
    assert page.get_by_role('button', name='Export YAML', exact=True).is_disabled()
    page.get_by_role('button', name='Undo', exact=True).click()
    assert not page.get_by_role('button', name='Export YAML', exact=True).is_disabled()


def test_yaml_import_preserves_advanced_fields_and_rejects_duplicate_keys(page, tmp_path):
    before = document(page)
    page.get_by_role('tab', name='YAML', exact=True).click()
    page.locator('#yaml-source').fill('id: one\nid: two\n')
    page.get_by_role('button', name='Apply source', exact=True).click()
    assert 'Duplicate key' in page.locator('#source-error').inner_text()
    assert document(page) == before
    before['config']['train']['patch']['training'] = {'max_steps': 2, 'seed': 19}
    before['name'] = '<img src=x onerror="window.polluted=true">'
    page.locator('#yaml-source').fill(json.dumps(before))
    page.get_by_role('button', name='Apply source', exact=True).click()
    assert page.locator('#workflow-title').inner_text() == before['name']
    assert page.locator('#workflow-title img').count() == 0
    assert not page.evaluate('Boolean(window.polluted)')
    plan = export_plan(page, tmp_path)
    training = plan['nodes'][0]['configuration']['train']['document']['training']
    assert training['seed'] == 19 and training['max_steps'] == 2


@pytest.mark.parametrize('width', [390, 768, 1440])
def test_responsive_views_fit_and_support_keyboard_tabs(page, width):
    page.set_viewport_size({'width': width, 'height': 900})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.get_by_role('tab', name='Builder', exact=True).focus()
    page.keyboard.press('ArrowRight')
    assert page.get_by_role('tab', name='Execution plan').get_attribute('aria-selected') == 'true'
    page.get_by_role('tab', name='YAML', exact=True).click()
    assert page.locator('#yaml-source').is_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')


def test_import_file_and_local_draft_restore_without_a_server(page, tmp_path):
    source = ROOT / 'experiments/workflows/libero10-lora-4bit-smoke.yaml'
    page.locator('#import-file').set_input_files(source)
    page.wait_for_function("OVLABStudio.getDocument().id === 'libero10-lora-4bit-smoke'")
    assert document(page)['id'] == 'libero10-lora-4bit-smoke'
    edit(page, '[data-field="name"]', 'Local draft')
    page.reload()
    assert document(page)['name'] == 'Local draft'
    assert export_plan(page, tmp_path)['steps'] == [{'node': '', 'phase': 'evaluate'}]


def test_unapplied_local_json_survives_navigation_and_reload(page, tmp_path):
    choose(page, 'eval-checkpoint')
    page.locator('details[data-section="advanced"] summary').click()
    draft = json.dumps({'evaluate': {'component_patches': {'protocol': {'execution': {'maximum_episode_steps': 5}}}}})
    page.locator('#advanced-config').fill(draft)
    assert page.get_by_role('button', name='Export YAML', exact=True).is_disabled()
    choose(page, '')
    page.get_by_role('tab', name='YAML', exact=True).click()
    assert page.locator('#yaml-source').get_attribute('readonly') is not None
    page.reload()
    assert page.locator('#draft-notice').is_visible()
    page.get_by_role('button', name='Open draft', exact=True).click()
    assert page.locator('#advanced-config').input_value() == draft
    assert page.locator('#inspector .inspector-header p').inner_text() == 'eval-checkpoint'
    page.get_by_role('button', name='Apply local configuration', exact=True).click()
    assert not page.locator('#draft-notice').is_visible()
    plan = export_plan(page, tmp_path)
    assert plan['nodes'][1]['configuration']['evaluate']['component_documents']['protocol']['document']['execution']['maximum_episode_steps'] == 5


def test_unapplied_yaml_survives_tabs_reload_and_node_selection(page, tmp_path):
    before = document(page)
    source = {**before, 'name': 'Recovered source draft'}
    page.get_by_role('tab', name='YAML', exact=True).click()
    page.locator('#yaml-source').fill(json.dumps(source))
    page.get_by_role('tab', name='Builder', exact=True).click()
    choose(page, 'eval-checkpoint')
    assert page.locator('#node-id').is_disabled()
    assert document(page) == before
    page.reload()
    page.get_by_role('button', name='Open draft', exact=True).click()
    assert json.loads(page.locator('#yaml-source').input_value())['name'] == 'Recovered source draft'
    page.get_by_role('button', name='Apply source', exact=True).click()
    assert export_plan(page, tmp_path)['name'] == 'Recovered source draft'


def test_pending_json_error_remains_editable_and_can_be_discarded(page):
    page.locator('details[data-section="advanced"] summary').click()
    page.locator('#advanced-config').fill('{')
    page.get_by_role('button', name='Apply local configuration', exact=True).click()
    assert page.locator('#advanced-error').is_visible()
    assert page.locator('#advanced-config').is_editable()
    page.get_by_role('button', name='Discard edits', exact=True).last.click()
    assert not page.get_by_role('button', name='Export YAML', exact=True).is_disabled()
    assert page.locator('#draft-notice').is_hidden()


def test_one_pointer_click_after_typing_commits_and_activates_the_target(page, tmp_path):
    name = page.locator('[data-field="name"]')
    name.fill('Edited before clicking a phase')
    box = page.locator('[data-execute="evaluate"]').bounding_box()
    page.mouse.click(box['x'] + box['width']/2, box['y'] + box['height']/2)
    assert document(page)['name'] == 'Edited before clicking a phase'
    assert document(page)['execute'] == ['train', 'evaluate']
    page.locator('[data-field="name"]').fill('Edited before selecting a node')
    card = page.locator('.node-card[data-select="eval-checkpoint"]').bounding_box()
    page.mouse.click(card['x'] + card['width']/2, card['y'] + card['height']/2)
    assert page.locator('#inspector .inspector-header p').inner_text() == 'eval-checkpoint'
    assert document(page)['name'] == 'Edited before selecting a node'
    page.locator('#issues [data-fix="remove-evaluate"]').click()
    assert document(page)['execute'] == ['train']
    assert len(export_plan(page, tmp_path)['steps']) == 2


def test_new_evaluation_uses_the_nearest_quic_training_context(page, tmp_path):
    page.locator('details[data-section="advanced"] summary').click()
    page.locator('#advanced-config').fill(json.dumps({'train': {'profile': 'openvla-libero10-quic-smoke.yaml'}}))
    page.get_by_role('button', name='Apply local configuration', exact=True).click()
    # The existing child now needs a profile; the newly added child gets one
    # from its training context and binds to the root rather than choosing LoRA.
    choose(page, 'eval-checkpoint')
    page.get_by_role('button', name='Delete node', exact=True).click()
    page.locator('#confirm-dialog').get_by_role('button', name='Delete subtree', exact=True).click()
    page.get_by_role('button', name='Add child node', exact=True).first.click()
    page.locator('[data-template="evaluate"]').click()
    evaluation = document(page)['children'][0]['config']['evaluate']
    assert evaluation['profile'].endswith('libero10-quic-adapter-rpc-smoke.yaml')
    assert evaluation['checkpoint_from'] == ''
    assert export_plan(page, tmp_path)['steps'][-1]['phase'] == 'evaluate'


def test_escape_does_not_repeat_a_previous_delete_confirmation(page):
    page.get_by_role('button', name='New', exact=True).click()
    page.get_by_role('button', name='Add child node', exact=True).first.click()
    page.locator('[data-template="group"]').click()
    page.get_by_role('button', name='Duplicate node', exact=True).click()
    page.get_by_role('button', name='Delete node', exact=True).click()
    page.locator('#confirm-dialog').get_by_role('button', name='Delete subtree', exact=True).click()
    choose(page, 'group')
    page.get_by_role('button', name='Delete node', exact=True).click()
    page.keyboard.press('Escape')
    assert [node['id'] for node in document(page)['children']] == ['group']


def test_task_shortcuts_and_invalid_component_types_are_recoverable(page, tmp_path):
    page.locator('[data-task-action="first-two"]').click()
    assert len(document(page)['config']['train']['patch']['dataset']['partition']['groups']) == 2
    choose(page, 'eval-checkpoint')
    page.locator('[data-eval-tasks="0,1"]').click()
    page.locator('details[data-section="advanced"] summary').click()
    config = document(page)['children'][0]['config']
    config['evaluate']['component_patches']['policy'] = {'type': 7}
    page.locator('#advanced-config').fill(json.dumps(config))
    page.get_by_role('button', name='Apply local configuration', exact=True).click()
    assert 'Unknown policy type' in page.locator('#issues').inner_text()
    assert page.get_by_role('button', name='Export YAML', exact=True).is_disabled()
    page.get_by_role('button', name='Undo', exact=True).click()
    plan = export_plan(page, tmp_path)
    assert plan['nodes'][1]['configuration']['evaluate']['component_documents']['benchmark']['document']['settings']['task_indices'] == [0, 1]


def test_mobile_new_and_edit_node_controls_are_reachable(page):
    page.set_viewport_size({'width': 390, 'height': 850})
    assert page.get_by_role('button', name='New', exact=True).is_visible()
    page.get_by_role('button', name='New', exact=True).click()
    page.get_by_role('button', name='Add child node', exact=True).first.click()
    page.locator('[data-template="handoff"]').click()
    page.get_by_role('button', name='Edit node ↓', exact=True).click()
    assert page.locator('#node-id').evaluate('element => element === document.activeElement')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')


def test_cycle_plan_keeps_blocked_phases_and_honest_counts(page):
    source = document(page)
    source['depends_on'] = [{'node': 'eval-checkpoint', 'phase': 'evaluate'}]
    page.get_by_role('tab', name='YAML', exact=True).click()
    page.locator('#yaml-source').fill(json.dumps(source))
    page.get_by_role('button', name='Apply source', exact=True).click()
    page.get_by_role('tab', name='Execution plan').click()
    assert page.locator('#plan-count').inner_text() == '2'
    assert page.locator('.plan-table tbody tr').count() == 2
    assert page.locator('.plan-table').inner_text().count('Blocked by dependencies') == 2
    assert page.get_by_role('button', name='Export YAML', exact=True).is_disabled()


def test_inherited_lora_handoff_creates_matching_training_and_evaluation(page, tmp_path):
    source = {'schema_version': 'ovlab.experiment/v1', 'kind': 'workflow', 'id': 'lora-branches',
              'name': 'LoRA branches', 'execution': {'max_workers': 1},
              'config': {'train': {'profile': 'openvla-lora-libero10-smoke.yaml'}}}
    page.get_by_role('tab', name='YAML', exact=True).click()
    page.locator('#yaml-source').fill(json.dumps(source))
    page.get_by_role('button', name='Apply source', exact=True).click()
    page.get_by_role('tab', name='Builder', exact=True).click()
    page.get_by_role('button', name='Add child node', exact=True).first.click()
    page.locator('[data-template="handoff"]').click()
    assert document(page)['children'][0]['id'] == 'lora'
    plan = export_plan(page, tmp_path)
    assert plan['steps'] == [{'node': 'lora', 'phase': 'train'}, {'node': 'lora/full', 'phase': 'evaluate'}]
    assert plan['nodes'][2]['configuration']['evaluate']['checkpoint_from'] == 'lora'


def test_replacing_then_undoing_restores_the_unapplied_source_draft(page):
    original = document(page)
    page.get_by_role('tab', name='YAML', exact=True).click()
    text = json.dumps({**original, 'name': 'Still a draft'})
    page.locator('#yaml-source').fill(text)
    page.get_by_role('button', name='New', exact=True).click()
    page.keyboard.press('Escape')
    assert document(page) == original
    assert page.locator('#yaml-source').input_value() == text
    page.get_by_role('button', name='New', exact=True).click()
    page.locator('#confirm-dialog').get_by_role('button', name='Replace workflow', exact=True).click()
    page.wait_for_function("OVLABStudio.getDocument().id === 'new-experiment'")
    assert document(page)['id'] == 'new-experiment'
    page.get_by_role('button', name='Undo', exact=True).click()
    assert document(page) == original
    page.get_by_role('button', name='Open draft', exact=True).click()
    assert page.locator('#yaml-source').input_value() == text


def test_large_tree_can_be_panned_and_navigated_without_changing_the_workflow(page):
    page.locator('#preset').select_option('experiments/workflows/libero10-two-task-training-smoke.yaml')
    page.wait_for_function("OVLABStudio.getDocument().id === 'libero10-two-task-training-smoke'")
    original = document(page)
    assert page.locator('#plan-count').inner_text() == '9'
    root = page.locator('#outline [data-select=""]')
    root.focus()
    page.keyboard.press('ArrowRight')
    page.keyboard.press('ArrowRight')
    assert page.locator('#inspector .inspector-header p').inner_text() == 'lora/full'
    page.keyboard.press('ArrowLeft')
    assert page.locator('#inspector .inspector-header p').inner_text() == 'lora'
    area = page.locator('#canvas-scroll')
    area.evaluate('element => element.scrollTop = 100')
    bounds = area.bounding_box()
    before = area.evaluate('element => element.scrollTop')
    page.mouse.move(bounds['x'] + 10, bounds['y'] + 180)
    page.mouse.down()
    page.mouse.move(bounds['x'] + 10, bounds['y'] + 60, steps=4)
    page.mouse.up()
    assert area.evaluate('element => element.scrollTop') > before
    page.get_by_role('button', name='Fit', exact=True).click()
    assert int(page.locator('#zoom-label').inner_text().strip('%')) <= 100
    assert document(page) == original
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
