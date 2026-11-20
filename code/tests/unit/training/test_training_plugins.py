"""A standalone project can train and finalize without a built-in model import."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap
import uuid

import pytest

from ovlab_benchctl.policy_plugins import registry_scope
from ovlab_benchctl.training_plugins import (backend_for_profile, preparation_for_checkpoint,
                                            training_image, finalizer_image)
from ovlab_benchctl.training_profiles import TrainingProfile, TrainingPlanner
from ovlab_benchctl.training_runs import CheckpointBundleStore, TrainingRunContext
from ovlab_benchctl.training_errors import CheckpointBundleError, TrainingProfileError
from ovlab_benchctl.training_worker import _train


def registered_trainer(tmp_path, *, include_training=True):
    root = tmp_path / "checkout"
    project = root / "plugins" / "fixture"
    module = "ovlab_fixture_training_" + uuid.uuid4().hex
    source = project / "src" / (module + ".py")
    source.parent.mkdir(parents=True)
    source.write_text(textwrap.dedent('''
        import json
        from pathlib import Path
        from ovlab_policy_sdk.training import TrainingPlugin, CheckpointPreparationPlugin
        from ovlab_policy_sdk.training_errors import TrainingProfileError, CheckpointBundleError
        from ovlab_policy_sdk.training_support import TrainerAdapter
        from ovlab_policy_sdk.training_services import host_services

        class Trainer(TrainerAdapter):
            def capabilities(self): return {"id": "fixture-trainer"}
            def validate_profile(self, profile): pass
            def resolve_plan(self, profile, model, dataset): return {}
            def preflight(self, plan): return {"compatible": True}
            def initialize(self, plan, context): self.context = context
            def train(self, plan, context):
                weights = context.staging_checkpoint / "weights-or-adapter"
                weights.mkdir(parents=True)
                (weights / "payload.json").write_text('{"weight": 3}')
                return {"status": "checkpointing"}
            def finalize(self, context): return {"status": "checkpointing"}
            def interrupt(self, context): pass
            def close(self):
                self.context.close_count += 1
                (self.context.root / "closed.json").write_text(json.dumps({"count": self.context.close_count}))

        class Extension(TrainingPlugin):
            def validate_profile(self, document):
                if type(document.get("custom_width")) is not int:
                    raise TrainingProfileError("custom_width must be an integer")
                return dict(document)
            def estimated_vram(self, profile): return float(profile.document["custom_width"])
            def plan(self, services, profile, **options):
                assert host_services().checkpoint_store is not None
                return {"profile_id": profile.profile_id, "custom_width": profile.document["custom_width"],
                        "data_root": str(services.model_data_root), "options": options}
            def create_trainer(self): return Trainer()
            def validate_staging(self, context):
                if not (context.staging_checkpoint / "weights-or-adapter/payload.json").is_file():
                    raise CheckpointBundleError("fixture checkpoint is missing")
                return None, []
            def checkpoint_format(self, scientific): return "json"
            def expected_loader(self, scientific): return "fixture-loader"
            def verify_checkpoint(self, root, checkpoint):
                if json.loads((root / "weights-or-adapter/payload.json").read_text()) != {"weight": 3}:
                    raise CheckpointBundleError("fixture checkpoint is invalid")

        class Preparer(CheckpointPreparationPlugin):
            def prepare(self, base, checkpoint, output, *, dtype):
                output.mkdir(exist_ok=True)
                (output / "prepared.json").write_text(json.dumps({"dtype": dtype, "base": str(base)}))

        training = Extension()
        preparation = Preparer()
    '''))
    manifest = {"schema_version": "ovlab.plugin/v1", "api_version": "1", "kind": "policy",
                "id": "fixture", "version": "1.0.0", "sources": {}, "builds": {},
                "plugins": {"fixture_policy": {"id": "fixture", "family": "fixture",
                            "runtime_status": "unavailable", "sources": []}}}
    if include_training:
        manifest["training"] = {"entrypoint": module + ":training", "implementations": ["fixture-trainer"],
                                "expected_loaders": ["fixture-loader"], "images": {
                                    "fixture-trainer": {"role": "fixture-train", "image": "fixture-train:local", "image_env": "FIXTURE_TRAIN_IMAGE"}},
                                "finalizer": {"role": "fixture-finalize", "image": "fixture-finalize:local", "image_env": "FIXTURE_FINALIZER_IMAGE"}}
        manifest["preparation"] = {"entrypoint": module + ":preparation", "expected_loaders": ["fixture-loader"],
                                   "converter_version": "fixture-converter/v1"}
    (project / "plugin.json").write_text(json.dumps(manifest))
    (root / "plugins/registry.json").write_text(json.dumps({"schema_version": "ovlab.plugins/v2",
        "plugins": {"fixture": {"source": {"kind": "local", "path": "plugins/fixture"}}}}))
    (root / "configs").mkdir()
    profile = {"schema_version": "ovlab.training-profile/v1", "kind": "training", "id": "fixture-run",
               "trainer": {"id": "fixture", "implementation": "fixture-trainer"}, "custom_width": 3}
    return root, profile, module


def test_registered_project_controls_profile_plan_images_and_validation(tmp_path):
    root, document, _ = registered_trainer(tmp_path)
    before = set(sys.modules)
    with registry_scope(root, {}):
        profile = TrainingProfile.from_document(document)
        plan = TrainingPlanner(root, tmp_path / "data").plan(profile, image_identity="fixture:image")
        assert plan["custom_width"] == 3
        assert plan["options"]["image_identity"] == "fixture:image"
        assert TrainingPlanner._estimated_vram(profile) == 3.0
        assert training_image(document)["image"] == "fixture-train:local"
        assert finalizer_image(document)["image"] == "fixture-finalize:local"
        with pytest.raises(TrainingProfileError, match="custom_width"):
            TrainingProfile.from_document({**document, "custom_width": "invalid"})
    assert not any(name.startswith("ovlab_openvla") for name in set(sys.modules) - before)


def test_worker_and_finalizer_use_registered_trainer_and_artifact_hooks(tmp_path):
    root, profile, _ = registered_trainer(tmp_path)
    run = tmp_path / "runs/train/fixture-run"
    run.mkdir(parents=True)
    plan = {"scientific_training_id": "training-plan-fixture", "execution_plan_id": "execution-fixture",
            "scientific": {"profile": profile, "training": {}, "output": {"output_kind": "full"},
                           "base_checkpoint": {"resource_id": "base"}}}
    (run / "training-plan.json").write_text(json.dumps(plan))
    for filename in ("events.jsonl", "metrics.jsonl"):
        (run / filename).write_text("")
    with registry_scope(root, {}):
        assert _train(run) == 0
        assert json.loads((run / "closed.json").read_text())["count"] == 1
        context = TrainingRunContext(run.name, run, plan)
        store = CheckpointBundleStore(tmp_path / "data")
        result = store.finalize(context)
        assert result["checkpoint"]["expected_loader"] == "fixture-loader"
        assert result["checkpoint"]["output_format"] == "json"
        assert store.verify(result["checkpoint_id"])["status"] == "verified"


def test_checkpoint_preparation_is_optional_and_loader_selected(tmp_path):
    root, _, _ = registered_trainer(tmp_path)
    with registry_scope(root, {}):
        preparer, metadata = preparation_for_checkpoint({"expected_loader": "fixture-loader"})
        output = tmp_path / "prepared"
        preparer.prepare(tmp_path / "base", tmp_path / "bundle", output, dtype="fixture-dtype")
        assert json.loads((output / "prepared.json").read_text())["dtype"] == "fixture-dtype"
        assert metadata["converter_version"] == "fixture-converter/v1"
        with pytest.raises(CheckpointBundleError, match="registered provider"):
            preparation_for_checkpoint({"expected_loader": "unregistered-loader"})
    empty, _, _ = registered_trainer(tmp_path / "without-training", include_training=False)
    with registry_scope(empty, {}):
        with pytest.raises(TrainingProfileError, match="registered provider"):
            backend_for_profile({"trainer": {"implementation": "fixture-trainer"}})


def test_planner_keeps_its_registry_after_profile_loading_scope_ends(tmp_path):
    root, document, _ = registered_trainer(tmp_path)
    with registry_scope(root, {}):
        profile = TrainingProfile.from_document(document)
    result = TrainingPlanner(root, tmp_path / 'data').plan(profile)
    assert result['custom_width'] == 3
