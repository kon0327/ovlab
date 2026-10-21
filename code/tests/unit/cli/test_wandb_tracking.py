"""W&B receives canonical training and evaluation metrics without changing runs."""

import json
from types import SimpleNamespace
import sys

from ovlab_benchctl.wandb_tracking import track_run


class _Run:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.artifacts = []
        self.url = "https://wandb.example/run"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def log(self, data, **kwargs):
        self.logged.append((data, kwargs))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class _Artifact:
    def __init__(self, **kwargs):
        self.files = []

    def add_file(self, path, name):
        self.files.append(name)


def _wandb(monkeypatch):
    run = _Run()
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Artifact=_Artifact))
    return run, calls


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_training_tracks_steps_and_checkpoint(tmp_path, monkeypatch):
    run, calls = _wandb(monkeypatch)
    root = tmp_path / "train-one"
    root.mkdir()
    _json(root / "training-plan.json", {"scientific": {"profile": {"training": {"max_steps": 2}}}})
    _json(root / "result.json", {"status": "completed", "checkpoint_id": "checkpoint-abc"})
    (root / "metrics.jsonl").write_text(
        '\n'.join(json.dumps(row) for row in (
            {"global_step": 1, "training_loss": 2.5, "learning_rate": 0.001},
            {"global_step": 2, "training_loss": 1.5, "gradient_norm": float("nan")},
        )) + '\n', encoding="utf-8",
    )

    result = track_run("train", root, {"OVLAB_WANDB_PROJECT": "thesis"})

    assert result == {"status": "synced", "url": run.url}
    assert calls[0]["job_type"] == "train"
    assert calls[0]["config"]["scientific"]["profile"]["training"]["max_steps"] == 2
    assert run.logged[0] == ({"train/global_step": 1, "train/training_loss": 2.5, "train/learning_rate": 0.001}, {"step": 1, "commit": True})
    assert "train/gradient_norm" not in run.logged[1][0]
    assert run.summary["ovlab/checkpoint_id"] == "checkpoint-abc"
    assert run.artifacts[0].files == ["training-plan.json", "result.json"]


def test_evaluation_tracks_episodes_and_task_summary(tmp_path, monkeypatch):
    run, calls = _wandb(monkeypatch)
    root = tmp_path / "eval-one"
    _json(root / "plan.json", {"selected_task_ids": ["task-a"]})
    _json(root / "manifest.completed.json", {"status": "completed", "episode_count": 2, "task_count": 1})
    episodes = root / "tasks" / "task-a" / "episodes"
    for number, success in enumerate((True, False)):
        _json(episodes / f"episode-{number}" / "metrics.episode.json", [
            {"metric_id": "task.success", "status": "available", "value": success},
            {"metric_id": "invalid", "status": "unavailable", "value": None},
        ])
    _json(root / "tasks" / "task-a" / "metrics.task.json", [
        {"metric_id": "task.success_rate", "status": "available", "value": 0.5},
        {"metric_id": "system.latency", "status": "available", "value": {"mean": 12.0, "max": 20.0}},
    ])

    result = track_run("eval", root, {"OVLAB_WANDB_PROJECT": "thesis"})

    assert result["status"] == "synced"
    assert calls[0]["job_type"] == "eval"
    assert [row["eval/episode/task.success"] for row, _ in run.logged] == [1, 0]
    assert run.summary["eval/mean/task.success_rate"] == 0.5
    assert run.summary["eval/task/task-a/system.latency/mean"] == 12.0
    assert run.artifacts[0].files == ["plan.json", "manifest.completed.json"]


def test_tracking_is_opt_in_and_errors_do_not_change_run(tmp_path, monkeypatch):
    assert track_run("train", tmp_path, {}) is None
    run, _ = _wandb(monkeypatch)
    result = track_run("train", tmp_path, {"OVLAB_WANDB_PROJECT": "thesis"})
    assert result["status"] == "failed"
    assert run.logged == []


def test_standard_wandb_project_environment_is_supported(tmp_path, monkeypatch):
    _, calls = _wandb(monkeypatch)
    root = tmp_path / "eval-one"
    _json(root / "plan.json", {})
    _json(root / "manifest.completed.json", {"status": "completed"})
    result = track_run("eval", root, {"WANDB_PROJECT": "standard", "WANDB_ENTITY": "team"})
    assert result["status"] == "synced"
    assert calls[0]["project"] == "standard"
    assert calls[0]["entity"] == "team"
