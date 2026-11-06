"""Optional offline dashboard regressions; install Playwright and Chromium to run."""

import pytest
from pathlib import Path

playwright = pytest.importorskip("playwright.sync_api")

from ovlab_runner.interactive import _assets, _page


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch(headless=True)
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip("Install Chromium with: python -m playwright install chromium")
            raise
        yield instance
        instance.close()


@pytest.fixture
def dashboard(tmp_path, browser):
    pages = []

    def open_dashboard(metrics=None, *, mobile=False):
        rows = metrics if metrics is not None else [
            {"global_step": 1, "training_loss": 0.000002, "action_token_accuracy": None},
            {"global_step": 2, "training_loss": 0.25, "action_token_accuracy": None},
        ]
        model = {
            "mode": "train", "run_id": "browser-regression", "status": "completed",
            "tasks": [], "metrics": rows,
            "counts": {"optimizer_steps": len(rows), "dataset_records": 10, "examples_seen": 2},
            "hyperparameters": {"training": {"learning_rate": 0.000002, "precision": "bf16"}},
            "notes": [], "parameter_counts": {"total": 100, "trainable": 10},
            "performance": {"statistics": {"loss": {"minimum": 0.000002}}},
            "evidence": {
                "microbatch_index": 0, "batch_size": 1,
                "input_batch": {"shape": [1, 4]},
                "input_record": {"instruction": "Move the block", "input_ids": [1, 2, 3, 4]},
                "output_batch": {"shape": [1, 4, 16]},
                "output_record": {"loss": 0.000002, "predicted_tokens": [2, 3, 4]},
            },
        }
        assets = tmp_path / "assets"
        assets.mkdir(exist_ok=True)
        for name, value in _assets().items():
            (assets / name).write_bytes(value)
        target = tmp_path / "index.html"
        target.write_text(_page(model, "", [], "report.html"), encoding="utf-8")
        page = browser.new_page(viewport={"width": 390 if mobile else 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        pages.append((page, errors))
        page.goto(target.as_uri())
        page.wait_for_load_state("networkidle")
        return page

    yield open_dashboard
    for page, errors in pages:
        page.close()
        assert not errors, errors


def test_exact_small_values_survive_settings_and_chart_sample_inspection(dashboard):
    page = dashboard()
    page.get_by_role("searchbox").fill("learning_rate")
    assert page.locator("#parameter-rows td").nth(1).inner_text() == "0.000002"
    loss = page.locator(".chart-card").first
    assert "0.000002 loss" in loss.locator(".exact-sample").inner_text()
    assert loss.locator(".series-data tbody tr").first.locator("td").nth(1).inner_text() == "0.000002"
    loss.get_by_label("Inspect an exact recorded sample").press("End")
    assert "0.25 loss" in loss.locator(".exact-sample").inner_text()


def test_sparse_long_series_and_unavailable_series_keep_every_sample(dashboard):
    # Index 1,000 is omitted by uniformly sampling 2,001 positions down to 2,000.
    rows = [{"global_step": index + 1, "training_loss": 0.000002 if index == 1000 else None,
             "action_token_accuracy": None} for index in range(2001)]
    page = dashboard(rows)
    loss, accuracy = page.locator(".chart-card").nth(0), page.locator(".chart-card").nth(1)
    assert "1 valid / 2,001 recorded samples" in loss.locator(".chart-summary").inner_text()
    assert "no finite recorded samples" not in loss.inner_text()
    assert loss.locator(".series-data tbody tr").count() == 100
    for _ in range(10):
        loss.get_by_role("button", name="Next samples").click()
    assert loss.locator(".series-data tbody tr").first.locator("td").nth(1).inner_text() == "0.000002"
    sample = loss.get_by_label("Inspect an exact recorded sample")
    sample.fill("1000")
    sample.dispatch_event("input")
    assert "0.000002 loss" in loss.locator(".exact-sample").inner_text()
    assert "no finite recorded samples" in accuracy.inner_text()
    assert accuracy.locator(".series-data tbody tr").count() == 100
    assert "Unavailable fraction" in accuracy.locator(".exact-sample").inner_text()


def test_parameter_search_explains_no_matches_and_recovers_all_settings(dashboard):
    page = dashboard()
    original_count = page.locator("#parameter-rows tr").count()
    page.get_by_role("searchbox").fill("absent-setting-123")
    assert "No matching settings" in page.locator("#parameter-rows").inner_text()
    assert page.locator("#parameter-count").inner_text().startswith("0 of ")
    page.get_by_role("searchbox").fill("")
    assert page.locator("#parameter-rows tr").count() == original_count


def test_keyboard_model_selection_reveals_mobile_stage_without_collapsing_evidence(dashboard):
    page = dashboard(mobile=True)
    assert page.locator("details").count() > 0
    assert page.locator("details").evaluate_all("items => items.every(item => item.open)")

    button = page.get_by_role("button", name="Optimizer & checkpoint", exact=True)
    button.focus()
    page.keyboard.press("Enter")
    assert button.get_attribute("aria-pressed") == "true"
    diagram = page.get_by_role("region", name="Interactive model diagram", exact=True)
    labels_fit = diagram.locator(".model-stage").evaluate_all("""stages => stages.every(stage => {
        const label = stage.querySelector('text').getBoundingClientRect();
        const frame = stage.querySelector('.stage-frame').getBoundingClientRect();
        return label.left >= frame.left - 1 && label.right <= frame.right + 1;
    })""")
    assert labels_fit, "Diagram stage labels must fit their frames on mobile"
    geometry = diagram.evaluate("""host => {
        const bounds = host.getBoundingClientRect();
        const selected = host.querySelector('.selected .stage-frame').getBoundingClientRect();
        return {left: selected.left - bounds.left, right: selected.right - bounds.left,
                width: bounds.width, scroll: host.scrollLeft};
    }""")
    assert geometry["scroll"] > 0
    assert geometry["left"] >= -1 and geometry["right"] <= geometry["width"] + 1
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator("details").evaluate_all("items => items.every(item => item.open)")

def test_real_evaluation_loads_only_the_selected_episode_offline(tmp_path, browser):
    from test_reports_exports import _canonical_run
    from ovlab_runner import DerivedReportEngine

    runs = tmp_path / "runs"
    run_id, _ = _canonical_run(runs, name="lazy-browser", rollouts=2, alternating=True, maximum_steps=5)
    generated = DerivedReportEngine(runs, tmp_path / "derived").generate(run_id)
    target = Path(generated["interactive_output"])
    page = browser.new_page()
    requests, errors = [], []
    page.on("request", lambda request: requests.append(request.url))
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto((target / "index.html").as_uri())
        page.wait_for_load_state("networkidle")
        assert not any("/data/episodes/" in url for url in requests)
        assert page.locator("#record .card").count() == 0
        assert page.get_by_role("button", name="Open run 1, mock-task-0").count() == 1
        page.get_by_role("button", name="Open run 1, mock-task-0").click()
        page.wait_for_function("document.querySelectorAll('#record .card').length === 2")
        assert len([url for url in requests if "/data/episodes/" in url]) == 1
        assert "Rollout 1" in page.locator("#batch").inner_text()
        assert page.locator("#record img").count() > 0
        assert page.locator("#selectors input[type=range]").get_attribute("max") == "4"
        page.locator("#selectors input[type=range]").press("End")
        assert "Prediction 5 / 5" in page.locator("#selectors").inner_text()
        task_page = next(target.glob("tasks/*/index.html"))
        requests.clear()
        page.goto(task_page.as_uri())
        page.wait_for_load_state("networkidle")
        assert not any("/data/episodes/" in url for url in requests)
        page.get_by_role("button", name="Load run details").click()
        page.wait_for_function("document.querySelectorAll('#record .card').length === 2")
        assert len([url for url in requests if "/data/episodes/" in url]) == 1
        assert page.locator("#selectors input[type=range]").get_attribute("max") == "4"
        assert not errors
    finally:
        page.close()


def test_real_training_loads_metrics_only_when_requested(tmp_path, browser):
    import runpy
    from ovlab_runner import TrainingDerivedReportEngine

    helper = Path(__file__).resolve().parents[1] / "unit/training/test_performance_reporting.py"
    run = runpy.run_path(str(helper))["_training_run"](tmp_path / "runs/train")
    generated = TrainingDerivedReportEngine(run.parent, tmp_path / "exports/train/derived").generate(run.name)
    target = Path(generated["interactive_output"])
    page = browser.new_page()
    requests, errors = [], []
    page.on("request", lambda request: requests.append(request.url))
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto((target / "index.html").as_uri())
        page.wait_for_load_state("networkidle")
        assert not any("/data/training.js" in url for url in requests)
        assert page.locator(".chart-card").count() == 0
        assert "0.5" in page.locator("#highlights").inner_text()
        page.get_by_role("button", name="Load training results").click()
        page.wait_for_function("document.querySelectorAll('.chart-card').length === 4")
        assert len([url for url in requests if "/data/training.js" in url]) == 1
        assert "All recorded training metrics" in page.locator("#metric-table").inner_text()
        assert not errors
    finally:
        page.close()
