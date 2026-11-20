"""Small explicit user configurations derived from the maintained plugin examples.

Tests exercise the real namespace resolver. They do not need a production preset
for every quantization/checkpoint combination.
"""
from copy import deepcopy

from ovlab_benchctl.experiment_workflow import ExperimentWorkflow
from ovlab_benchctl.strict_yaml import dumps, load


def write_evaluation_fixtures(root, destination):
    destination.mkdir(parents=True, exist_ok=False)
    references = {}
    for name, source in (("openvla", "openvla-eval"), ("vanilla", "openvla-eval"),
                         ("oft", "openvla-oft-eval"), ("quic", "plugin-pipeline-smoke")):
        plan = ExperimentWorkflow(root).plan(f"examples/experiments/{source}.yaml")
        state = next(node["configuration"]["evaluate"] for node in plan["nodes"]
                     if "evaluate" in node["execute"])
        document = deepcopy(state["document"])
        if name == "vanilla":
            document["components"]["policy"] = "plugin://openvla/policy/default.yaml"
        components = deepcopy(state.get("component_documents", {}))
        if name == "quic":
            components["policy"]["document"]["settings"]["quic"].update({
                "adapter_path": "/fixture/adapter", "adapter_manifest_sha256": "a" * 64,
                "adapter_tensor_sha256": "b" * 64,
            })
        for kind, component in components.items():
            path = destination / "components" / f"{name}-{kind}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps(component["document"]))
            document["components"][kind] = path.relative_to(root / "configs").as_posix()
        path = destination / "evaluation" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(document))
        references[name] = path.relative_to(root).as_posix()
    mock = destination / "evaluation/mock.yaml"
    mock.write_text(dumps(load(root / "examples/experiments/mock-smoke.yaml")))
    references["mock"] = mock.relative_to(root).as_posix()
    deploy = destination / "deploy/openvla.yaml"
    deploy.parent.mkdir()
    deploy.write_text(dumps({
        "schema_version": "ovlab.deployment/v1", "kind": "deployment", "id": "test-openvla",
        "extends": "plugin://openvla/deploy/default.yaml",
        "components": {"policy": "plugin://openvla/policy/lora.yaml"},
    }))
    from helpers.training_profiles import write_training_profile
    write_training_profile(destination / "train/lora.yaml")
    return references
