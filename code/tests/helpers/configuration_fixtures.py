"""Small test-owned configurations composed from supported public defaults."""
from copy import deepcopy
from pathlib import Path

from ovlab_benchctl.strict_yaml import load

ROOT = Path(__file__).resolve().parents[3]


def training_config():
    return deepcopy(load(ROOT / "examples/experiments/plugin-pipeline-smoke.yaml")["config"]["train"])


def evaluation_config():
    return deepcopy(load(ROOT / "examples/experiments/openvla-eval.yaml")["config"]["evaluate"])


def training_document():
    from ovlab_benchctl.experiment_workflow import ExperimentWorkflow
    return ExperimentWorkflow(ROOT).plan(ROOT / "examples/experiments/plugin-pipeline-smoke.yaml")["nodes"][0]["configuration"]["train"]["document"]


def evaluation_configuration(kind="openvla"):
    from ovlab_benchctl.experiment_workflow import ExperimentWorkflow
    from ovlab_benchctl.config_bundle import ConfigBundleBuilder
    names = {"openvla": "openvla-eval", "vanilla": "openvla-eval", "oft": "openvla-oft-eval",
             "quic": "plugin-pipeline-smoke", "mock": "mock-smoke"}
    plan = ExperimentWorkflow(ROOT).plan(ROOT / "examples/experiments" / (names[kind] + ".yaml"))
    index = 1 if kind == "quic" else 0
    configuration = deepcopy(plan["nodes"][index]["configuration"]["evaluate"])
    configuration.pop("checkpoint_from", None)
    if kind == "vanilla":
        configuration["document"]["components"]["policy"] = "plugin://openvla/policy/default.yaml"
        configuration["component_documents"].pop("policy", None)
    if kind == "quic":
        policy = configuration["component_documents"]["policy"]["document"]
        policy["settings"]["quic"].update({"adapter_path": "checkpoints/test-adapter",
            "adapter_manifest_sha256": "a" * 64, "adapter_tensor_sha256": "b" * 64})
    return configuration


def evaluation_document(kind="openvla"):
    return evaluation_configuration(kind)["document"]
