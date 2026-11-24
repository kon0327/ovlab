"""Independent benchmark control contracts and a CPU runner/report smoke."""
from dataclasses import dataclass, replace
import hashlib
import json
import subprocess
import sys

import pytest

from helpers.contexts import make_run_context
from helpers.mock_benchmark import MockBenchmark
from helpers.mock_policy import MockPolicy
from helpers.mock_specs import mock_action_spec, mock_observation_spec
from ovlab_benchmarks import (
    BenchmarkDeploymentSpec, BenchmarkPlugin, BenchmarkPluginCompatibilityError,
    BenchmarkPluginReferenceError, BenchmarkPluginValidationError, BenchmarkResetResult,
    BenchmarkRuntimeMount, CapabilityRefinement, TaskCatalog, TaskDescriptor,
    create_benchmark_adapter, validate_runtime_capabilities,
)
from ovlab_core.contracts import (
    OVLAB_CONTRACT_VERSION, AdapterState, BenchmarkCapabilities, BenchmarkContext,
    BenchmarkIdentity, BenchmarkResolutionContext, ContractValidationError,
    SignalAccess, SignalRegistry, SignalSpec, TaskId,
)
from ovlab_core.datasets import DatasetProvider, DatasetRequest, DatasetRequestError, DatasetResolution
from ovlab_policy_sdk import PolicyPlugin, PolicyPluginV2


@dataclass(frozen=True)
class ToySettings:
    maximum_steps: int


class ToyPlugin(BenchmarkPlugin):
    def validate_benchmark(self, document, path):
        if set(document) != {"maximum_steps"} or type(document["maximum_steps"]) is not int:
            raise BenchmarkPluginValidationError(f"{path}: expected integer maximum_steps")
        if document["maximum_steps"] <= 0:
            raise BenchmarkPluginValidationError(f"{path}: maximum_steps must be positive")

    def resolve_settings(self, document, context):
        self.validate_benchmark(document, "toy")
        return ToySettings(min(document["maximum_steps"], context.maximum_episode_steps))

    def identity(self, settings):
        return BenchmarkIdentity("toy-project", "toy", "1", "fixture-source-sha256")

    def describe_capabilities(self, settings):
        return BenchmarkCapabilities(
            "toy", "1", OVLAB_CONTRACT_VERSION, mock_observation_spec(), mock_action_spec(),
            SignalRegistry((
                SignalSpec("benchmark.task_success", "bool", (), "", SignalAccess.EVALUATION_ONLY, "Success"),
                SignalSpec("hidden_target", "float32", (2,), "m", SignalAccess.PRIVILEGED, "Target"),
            )), True, False, True, ("toy-suite",),
        )

    def task_catalog(self, settings):
        return TaskCatalog("toy-tasks", "immutable-fixture-v1", tuple(
            TaskDescriptor("toy-suite", TaskId(task_id), task_id, index, "Move once", settings.maximum_steps)
            for task_id, index in (("toy/push", 4), ("toy/hold", 17), ("toy/finish", 31))
        ))

    def create_adapter(self, settings):
        return ToyAdapter(self, settings)


class ToyAdapter(MockBenchmark):
    """CPU-only execution fixture with authoritative opaque catalog IDs."""

    def __init__(self, plugin, settings):
        super().__init__(maximum_steps=settings.maximum_steps)
        self.plugin, self.settings = plugin, settings

    def _initialize(self, context):
        return self.plugin.describe_capabilities(self.settings)

    def _list_tasks(self):
        return self.plugin.task_catalog(self.settings).tasks

    def _reset_episode(self, context):
        self._step_index, self._seed, self._terminal_outcome = 0, context.seed, "success"
        observation = self._observation(context, 0)
        return BenchmarkResetResult(context, observation, self._signals(observation.step_id, 0, False), 0)


def test_sdk_imports_do_not_load_runtime_or_orchestration():
    script = '''
import importlib.abc, sys
blocked = ("libero", "mujoco", "robosuite", "torch", "ovlab_benchctl", "ovlab_runner")
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("forbidden import: " + fullname)
sys.meta_path.insert(0, Reject())
import ovlab_benchmarks.plugin, ovlab_core.datasets
from ovlab_core.contracts import BenchmarkContext
from ovlab_policy_sdk import PolicyPluginV2
assert not any(name.split(".")[0] in blocked for name in sys.modules)
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_catalog_selection_is_deterministic_opaque_and_ordered():
    plugin, settings = ToyPlugin(), ToySettings(3)
    catalog = plugin.task_catalog(settings)
    assert catalog.sha256 == plugin.task_catalog(settings).sha256
    selection = catalog.select((TaskId("toy/finish"), TaskId("toy/push")))
    assert selection.task_ids == (TaskId("toy/finish"), TaskId("toy/push"))
    assert selection.identity.catalog_sha256 == catalog.sha256
    assert catalog.select(suite_names=("toy-suite",)).tasks == catalog.tasks
    changed = replace(catalog, tasks=(replace(catalog.tasks[0], natural_language_instruction="Changed"), *catalog.tasks[1:]))
    assert changed.sha256 != catalog.sha256
    with pytest.raises(BenchmarkPluginReferenceError, match="unknown"):
        catalog.select((TaskId("unregistered"),))
    with pytest.raises(BenchmarkPluginValidationError, match="unique"):
        catalog.select((TaskId("toy/push"), TaskId("toy/push")))
    with pytest.raises(BenchmarkPluginValidationError, match="non-empty"):
        catalog.select(())
    with pytest.raises(BenchmarkPluginValidationError, match="unique"):
        replace(catalog, tasks=(catalog.tasks[0], catalog.tasks[0]))


def test_peer_context_has_immutable_metadata_and_validated_suite_identity():
    plugin, settings = ToyPlugin(), ToySettings(2)
    source = {"toy/observation@1": {"components": ["position"]}}
    context = BenchmarkContext(plugin.identity(settings), plugin.describe_capabilities(settings),
                               plugin.task_catalog(settings).select().identity, source)
    source["toy/observation@1"]["components"].append("unexpected")
    assert context.method_claims["toy/observation@1"]["components"] == ("position",)
    with pytest.raises(TypeError):
        context.method_claims["new"] = True
    with pytest.raises(ContractValidationError, match="selected suites"):
        replace(context, task_set=replace(context.task_set, suite_ids=("unregistered",)))
    resources = {"primary": "cpu"}
    resolution = BenchmarkResolutionContext(4, 8, devices=resources, execution={"custom": [1]})
    resources["primary"] = "cuda:0"
    assert resolution.devices["primary"] == "cpu"
    assert resolution.execution["custom"] == (1,)
    with pytest.raises(ContractValidationError):
        BenchmarkResolutionContext(True, 8)


def test_runtime_parity_permits_only_exact_optional_signal_additions():
    declared = ToyPlugin().describe_capabilities(ToySettings(2))
    extra = SignalSpec("toy.state", "float64", (5,), "native", SignalAccess.PRIVILEGED, "Replay state", optional=True)
    actual = replace(declared, signal_registry=SignalRegistry((*declared.signal_registry, extra)))
    with pytest.raises(BenchmarkPluginCompatibilityError, match="undeclared"):
        validate_runtime_capabilities(declared, actual)
    validate_runtime_capabilities(declared, actual, CapabilityRefinement((extra,)))
    with pytest.raises(BenchmarkPluginCompatibilityError, match="action_spec"):
        validate_runtime_capabilities(declared, replace(declared, action_spec=replace(declared.action_spec, units=("other",) * 3)))
    with pytest.raises(BenchmarkPluginCompatibilityError, match="signal differs"):
        validate_runtime_capabilities(declared, replace(declared, signal_registry=SignalRegistry()))
    with pytest.raises(BenchmarkPluginValidationError, match="optional"):
        CapabilityRefinement((replace(extra, optional=False),))
    with pytest.raises(BenchmarkPluginValidationError, match="optional"):
        CapabilityRefinement((replace(extra, access=SignalAccess.POLICY_VISIBLE),))
    with pytest.raises(BenchmarkPluginCompatibilityError, match="undeclared"):
        validate_runtime_capabilities(declared, actual, CapabilityRefinement((replace(extra, shape=(6,)),)))


def test_factory_requires_created_adapter_and_deployment_is_data():
    plugin, settings = ToyPlugin(), ToySettings(1)
    adapter = create_benchmark_adapter(plugin, settings)
    assert adapter.state is AdapterState.CREATED
    adapter.close()

    class BadFactory(ToyPlugin):
        def create_adapter(self, settings):
            result = super().create_adapter(settings)
            result.initialize(make_run_context())
            return result

    with pytest.raises(BenchmarkPluginCompatibilityError, match="CREATED"):
        create_benchmark_adapter(BadFactory(), settings)
    spec = BenchmarkDeploymentSpec("benchmark-toy", "toy:local", "OVLAB_TOY_IMAGE", environment={"MODE": "cpu"})
    assert not spec.requires_gpu and not spec.mounts
    with pytest.raises(TypeError):
        spec.environment["MODE"] = "gpu"
    with pytest.raises(BenchmarkPluginValidationError, match="absolute"):
        BenchmarkRuntimeMount("/host", "relative")
    with pytest.raises(BenchmarkPluginValidationError, match="inside"):
        BenchmarkRuntimeMount("/host", "/data", required_files=("../escape",))


def test_dataset_resolution_preserves_portable_v1_identity():
    metadata = {"source": {"parts": ["a", "b"]}}
    resolution = DatasetResolution("toy", "set", "revision", "fixture", "fixture://set", "toy", "1", None, "native", metadata)
    metadata["source"]["parts"].append("c")
    assert resolution.source_metadata["source"]["parts"] == ("a", "b")
    document = resolution.identifying_document()
    # Existing published dataset identities include this trailing newline.
    canonical = (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    assert resolution.resolution_id == "dsr-" + hashlib.sha256(canonical).hexdigest()[:24]
    assert resolution.as_dict()["state"] == "resolved"
    assert DatasetProvider.supports_resume is False
    with pytest.raises(DatasetRequestError):
        DatasetRequest("toy", "../escape")
    with pytest.raises(DatasetRequestError):
        replace(resolution, expected_digest="invalid")


def test_policy_api_v2_is_additive_and_has_no_fabricated_standalone_peer():
    assert PolicyPlugin.api_version == "1" and PolicyPluginV2.api_version == "2"
    assert PolicyPluginV2.deployment_context(None, {}, {"input": {"width": 64}}) is None


def test_cpu_plugin_runs_through_metrics_canonical_verification_and_report(tmp_path):
    from ovlab_runner import (
        DeterministicClock, ExperimentPlan, ExperimentRunner, FilesystemRunArtifactStore,
        RunConfigurationSnapshot, TraceRecordingPolicy, recompute_run_metrics, verify_run,
    )
    from ovlab_runner.reporting import build_run_report

    plugin = ToyPlugin()
    settings = plugin.resolve_settings({"maximum_steps": 3}, BenchmarkResolutionContext(7, 3))
    selected = plugin.task_catalog(settings).select((TaskId("toy/finish"), TaskId("toy/push")))
    benchmark = create_benchmark_adapter(plugin, settings)
    plan = ExperimentPlan(
        make_run_context(run_id="sdk-toy-cpu"), selected.task_ids, 1, 7, 3,
        enabled_metric_ids=("task.success", "action.variance", "system.inference_latency"),
        trace_recording_policy=TraceRecordingPolicy(record_raw_policy_output=True),
    )
    store = FilesystemRunArtifactStore(tmp_path)
    source = "kind: scientific_experiment\nbenchmark: toy\n"
    scientific_hash = hashlib.sha256(source.encode()).hexdigest()
    execution_hash = hashlib.sha256(b"toy-cpu-execution").hexdigest()
    resolved = f"scientific_config_hash: {scientific_hash}\nexecution_config_hash: {execution_hash}\n"
    snapshot = RunConfigurationSnapshot(source, resolved, scientific_hash, execution_hash)
    runner = ExperimentRunner(plan, benchmark, MockPolicy(), store, clock=DeterministicClock(),
                              configuration_snapshot=snapshot)
    try:
        connection = runner.connect()
        validate_runtime_capabilities(plugin.describe_capabilities(settings), benchmark.capabilities)
        runner.run()
    finally:
        runner.close()
    assert tuple(task.task_id for task in connection.selected_tasks) == selected.task_ids
    run_path = store._run_path(plan.run_context.run_id)
    assert verify_run(run_path)["integrity"] == "verified"
    recomputed = recompute_run_metrics(run_path)
    assert recomputed["all_results_agree"] and recomputed["episode_count"] == 2
    report = build_run_report(run_path)
    assert report["status"] == "completed" and len(report["episodes"]) == 2
    assert all(episode["task_id"].startswith("toy/") for episode in report["episodes"])


def test_static_preflight_checks_opaque_tasks_without_initializing_adapters():
    from helpers.runner_fixtures import runner_plan
    from ovlab_metrics import MetricRegistry
    from ovlab_runner import preflight_components, ConnectionError
    plugin, settings = ToyPlugin(), ToySettings(2)
    catalog = plugin.task_catalog(settings)
    # Capabilities can be supplied as control metadata; no initialize call occurs here.
    policy = MockPolicy()
    policy_capabilities = policy._initialize(make_run_context())
    plan = runner_plan(selected_task_ids=(TaskId("toy/push"),))
    report = preflight_components(plan, plugin.describe_capabilities(settings), policy_capabilities,
                                  catalog.tasks, MetricRegistry.default())
    assert report.selected_tasks[0].task_id == TaskId("toy/push")
    assert policy.state is AdapterState.CREATED
    with pytest.raises(ConnectionError, match="do not exist"):
        preflight_components(runner_plan(selected_task_ids=(TaskId("unknown/task"),)),
                             plugin.describe_capabilities(settings), policy_capabilities,
                             catalog.tasks, MetricRegistry.default())
