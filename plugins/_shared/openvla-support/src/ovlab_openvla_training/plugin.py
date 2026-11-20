"""OpenVLA optional training hooks for the OVLAB plugin contract."""
from pathlib import Path

from ovlab_policy_sdk.training import CheckpointPreparationPlugin, TrainingPlugin
from ovlab_policy_sdk.training_errors import CheckpointBundleError


class OpenVlaTrainingPlugin(TrainingPlugin):
    def validate_profile(self, document):
        from .training_profiles import TrainingProfile
        return dict(TrainingProfile.from_document(document).document)

    def validate_profile_template(self, document):
        from .training_profiles import TrainingProfile
        return dict(TrainingProfile.from_document(document, partial=True).document)

    def estimated_vram(self, profile):
        from .training_profiles import TrainingPlanner
        return TrainingPlanner._estimated_vram(profile)

    def plan(self, services, profile, **options):
        from .training_profiles import TrainingPlanner
        return TrainingPlanner.plan(services, profile, **options)

    def create_trainer(self):
        from .training_runs import OpenVlaTrainerAdapter
        return OpenVlaTrainerAdapter()

    def container_resources(self, plan, dataset):
        from ovlab_policy_sdk.training_errors import TrainingRuntimeError
        name = dataset.get("source_metadata", {}).get("tfds_name")
        if not isinstance(name, str) or not name or "/" in name or ".." in name:
            raise TrainingRuntimeError("prepared OpenVLA dataset requires a portable TFDS name")
        return {"environment": {"OVLAB_TRAINING_DATA_ROOT": "/datasets/resolved",
                                "OVLAB_TRAINING_DATASET_NAME": name},
                "mounts": [{"source": str(Path(dataset["host_path"]) / "prepared"),
                            "target": f"/datasets/resolved/{name}/1.0.0"}]}

    def validate_ablation(self, profile, factors):
        from ovlab_policy_sdk.training_errors import TrainingProfileError
        from .training_data import validate_partition
        allowed = {"training.peft.adapter", "training.peft.placement", "training.peft.targets", "training.peft.max_trainable_parameters"}
        if any(not any(factor == prefix or factor.startswith(prefix + ".") for prefix in allowed) for factor in factors):
            raise TrainingProfileError("OpenVLA ablation factors must select training.peft adapter/placement/targets/budget paths")
        if "partition" not in profile.get("dataset", {}):
            raise TrainingProfileError("ablation base profile requires explicit shared dataset.partition")
        validate_partition(profile["dataset"]["partition"])

    def validate_staging(self, context):
        scientific = context.plan["scientific"]
        if scientific["output"]["output_kind"] != "adapter":
            return None, []
        weights = context.staging_checkpoint / "weights-or-adapter"
        training = scientific["training"]
        if training.get("peft", {}).get("method") == "quic":
            from .training_quic import validate_staged_adapter
            return validate_staged_adapter(weights, scientific, run_root=context.root)
        from .training_runs import _validate_lora_staging
        return _validate_lora_staging(weights, training)

    def expected_loader(self, scientific):
        if scientific["output"]["output_kind"] != "adapter":
            return "ovlab-openvla-vanilla"
        return "ovlab-openvla-quic" if scientific["training"].get("peft", {}).get("method") == "quic" else "ovlab-openvla-peft"

    def verify_checkpoint(self, root, checkpoint):
        if checkpoint.get("kind") != "peft_adapter":
            return
        from ovlab_policy_sdk.training_support import inspect_safetensors
        files = sorted((root / "weights-or-adapter").glob("*.safetensors"))
        if checkpoint.get("expected_loader") == "ovlab-openvla-quic":
            expected = {"adapter.safetensors"}
            adapter = checkpoint.get("adapter") or {}
            metadata = adapter.get("metadata") or {} if isinstance(adapter, dict) else {}
            if isinstance(metadata, dict) and metadata.get("action_head") is not None:
                expected.add("action_head.safetensors")
            if {path.name for path in files} != expected:
                raise CheckpointBundleError("QuIC checkpoint has an invalid weight inventory")
        elif len(files) != 1:
            raise CheckpointBundleError("adapter checkpoint has an invalid weight inventory")
        for path in files:
            inspect_safetensors(path)

    def validate_checkpoint_binding(self, training, policy):
        from .workflow_checkpoints import validate_checkpoint_binding
        return validate_checkpoint_binding(training, policy)

    def bind_training_checkpoint(self, *args, **kwargs):
        from .workflow_checkpoints import bind_training_checkpoint
        return bind_training_checkpoint(*args, **kwargs)

    def prepare_dense_checkpoint(self, *args, **kwargs):
        from .workflow_checkpoints import prepare_dense_checkpoint
        return prepare_dense_checkpoint(*args, **kwargs)


class OpenVlaCheckpointPreparation(CheckpointPreparationPlugin):
    def prepare(self, base, checkpoint, output, *, dtype):
        from .checkpoint_preparation import merge_lora
        merge_lora(Path(base), Path(checkpoint), Path(output), dtype=dtype)


plugin = OpenVlaTrainingPlugin()
preparer = OpenVlaCheckpointPreparation()
