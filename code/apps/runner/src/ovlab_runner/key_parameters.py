"""Select recorded, decision-relevant settings for both report renderers."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _item(label, sources, *candidates):
    for source, path in candidates:
        value = sources.get(source)
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if (isinstance(value, (str, int, float, bool)) and value != ""
                and (not isinstance(value, float) or math.isfinite(value))):
            return {"label": label, "value": value, "source": f"{source if source == 'report.json' else 'run/' + source}:{path}"}
    return {"label": label, "value": None, "source": None}


def _group(title, stage, items):
    return {"title": title, "stage": stage, "items": items}


def evaluation_parameters(plan, connection, complexity):
    sources = {"plan.json": plan, "connection.json": connection,
               "report.json": {"performance": {"model_complexity": complexity}}}
    item = lambda label, *paths: _item(label, sources, *paths)
    return [
        _group("Inputs", "inputs", [
            item("Benchmark", ("connection.json", "benchmark.name")),
            item("Camera", ("plan.json", "metadata.policy_configuration.input.camera")),
            item("Action normalization key", ("plan.json", "metadata.policy_configuration.unnorm_key")),
        ]),
        _group("Evaluation setup", "representation", [
            item("Episodes per task", ("plan.json", "rollout_count_per_task")),
            item("Base episode seed", ("plan.json", "base_episode_seed")),
            item("Maximum steps per episode", ("plan.json", "default_maximum_episode_steps")),
        ]),
        _group("Model / policy", "policy", [
            item("Policy", ("connection.json", "policy.name")),
            item("Checkpoint", ("plan.json", "metadata.policy_configuration.checkpoint_id"), ("plan.json", "metadata.deployment.checkpoint_id")),
            item("Method", ("report.json", "performance.model_complexity.method_family"), ("report.json", "performance.model_complexity.method_id")),
            item("Configured precision", ("plan.json", "metadata.policy_configuration.runtime.dtype")),
            item("Configured quantization", ("plan.json", "metadata.policy_configuration.runtime.quantization"), ("report.json", "performance.model_complexity.quantization")),
            item("Runtime parameters", ("report.json", "performance.model_complexity.parameter_counts.total")),
        ]),
        _group("Predicted action", "outputs", [
            item("Action codec", ("plan.json", "metadata.policy_configuration.action.codec"), ("connection.json", "metadata.policy_capabilities.action_codec")),
            item("Action interface", ("plan.json", "metadata.policy_configuration.action.interface_ref")),
        ]),
        _group("Action execution", "result", [
            item("Execution mode", ("plan.json", "action_execution_policy.mode")),
            item("Replan interval", ("plan.json", "action_execution_policy.replan_interval")),
        ]),
    ]


def training_parameters(plan, result, manifest, evidence=None):
    sources = {"training-plan.json": plan, "result.json": result,
               "manifest.json": manifest, "data-flow.json": evidence or {}}
    item = lambda label, *paths: _item(label, sources, *paths)
    return [
        _group("Inputs", "inputs", [
            item("Dataset", ("training-plan.json", "scientific.profile.dataset.ref")),
            item("Dataset ID", ("training-plan.json", "scientific.dataset.dataset_id")),
            item("Dataset split", ("training-plan.json", "scientific.profile.dataset.split")),
        ]),
        _group("Batch & supervision", "representation", [
            item("Batch size per device", ("training-plan.json", "scientific.training.per_device_batch_size"), ("training-plan.json", "scientific.profile.training.per_device_batch_size")),
            item("Gradient accumulation steps", ("training-plan.json", "scientific.training.gradient_accumulation_steps"), ("training-plan.json", "scientific.profile.training.gradient_accumulation_steps")),
            item("Configured precision", ("training-plan.json", "scientific.training.precision"), ("training-plan.json", "scientific.profile.training.precision")),
        ]),
        _group("Model / policy", "policy", [
            item("Base checkpoint", ("training-plan.json", "scientific.base_checkpoint.resource_id"), ("training-plan.json", "scientific.profile.model.base_checkpoint")),
            item("Adaptation method", ("training-plan.json", "scientific.training.peft.method"), ("training-plan.json", "scientific.profile.training.peft.method")),
            item("Configured quantization", ("training-plan.json", "scientific.training.quantization"), ("training-plan.json", "scientific.profile.training.quantization")),
            item("Total parameters", ("result.json", "performance.parameter_counts.total"), ("result.json", "total_parameter_count")),
            item("Trainable parameters", ("result.json", "performance.parameter_counts.trainable"), ("result.json", "trainable_parameter_count")),
        ]),
        _group("Prediction & objective", "outputs", [
            item("Training objective", ("training-plan.json", "scientific.trainer_contract.objective"), ("data-flow.json", "objective")),
            item("Action horizon", ("training-plan.json", "scientific.trainer_contract.action_semantics.horizon"), ("training-plan.json", "scientific.model_variant.action_head.horizon")),
        ]),
        _group("Optimizer & checkpoint", "result", [
            item("Learning rate", ("training-plan.json", "scientific.training.learning_rate"), ("training-plan.json", "scientific.profile.training.learning_rate")),
            item("Captured optimizer LR", ("data-flow.json", "optimizer.lr")),
            item("Optimizer", ("training-plan.json", "scientific.trainer_contract.optimizer")),
            item("Maximum optimizer steps", ("training-plan.json", "scientific.training.max_steps"), ("training-plan.json", "scientific.profile.training.max_steps")),
            item("Training seed", ("training-plan.json", "scientific.training.seed"), ("training-plan.json", "scientific.profile.training.seed")),
            item("Output checkpoint", ("manifest.json", "checkpoint_id")),
        ]),
    ]
