"""Thin application services used by the unified OVLAB command line."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from enum import Enum
import hashlib
import os
from pathlib import Path
import re
import time
import uuid

from ovlab_core.contracts import OVLAB_VERSION

from .catalog import registered_policies
from .errors import ConfigCompatibilityError, ConfigReferenceError
from .models import MockPolicySettings, ResolvedExperimentConfig
from .resolver import ConfigResolver
from .strict_yaml import dumps, load
from .versioning import CLI_VERSION, repository_revision


CLI_SCHEMA_VERSION = "ovlab-cli/1.0.0"


def _readable_run_id(
    experiment_name: str,
    created_wall_time_utc_ns: int,
    entropy: str,
    *,
    timezone=None,
) -> str:
    """Build a unique, host-local, human-readable run identifier."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_name).strip("._-")
    slug = slug[:180] or "experiment"
    seconds = created_wall_time_utc_ns // 1_000_000_000
    moment = datetime.fromtimestamp(seconds, tz=timezone)
    if timezone is None:
        moment = moment.astimezone()
    timestamp = moment.strftime("%Y-%m-%d_%H-%M-%S")
    digest = hashlib.sha256(
        f"{experiment_name}\0{created_wall_time_utc_ns}\0{entropy}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{slug}_{timestamp}_{digest}"


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _action_spec(spec) -> dict[str, object]:
    return {
        "dimension": spec.dimension,
        "representation": spec.representation.value,
        "translation_indices": list(spec.translation_indices),
        "rotation_indices": list(spec.rotation_indices),
        "gripper_indices": list(spec.gripper_indices),
        "rotation_representation": spec.rotation_representation.value,
        "gripper_convention": spec.gripper_convention.value,
        "units": list(spec.units),
        "minimum": None if spec.minimum is None else spec.minimum.tolist(),
        "maximum": None if spec.maximum is None else spec.maximum.tolist(),
        "dtype": spec.dtype,
        "control_frequency_hz": spec.control_frequency_hz,
    }


def _observation_requirements(requirements) -> dict[str, object]:
    return {
        "images": [
            {
                "name": item.name,
                "shapes": [list(shape) for shape in item.shapes],
                "dtype": item.dtype,
                "encodings": [value.value for value in item.encodings],
                "color_spaces": [value.value for value in item.color_spaces],
                "minimum_count": item.minimum_count,
                "maximum_count": item.maximum_count,
                "required": item.required,
            }
            for item in requirements.images
        ],
        "proprioception": [
            {
                "name": item.name,
                "shapes": [list(shape) for shape in item.shapes],
                "dtype": item.dtype,
                "units": list(item.units),
                "required": item.required,
            }
            for item in requirements.proprioception
        ],
        "minimum_image_count": requirements.minimum_image_count,
        "maximum_image_count": requirements.maximum_image_count,
        "minimum_proprioception_count": requirements.minimum_proprioception_count,
        "maximum_proprioception_count": requirements.maximum_proprioception_count,
    }


class OvlabApplication:
    """Orchestrate owner APIs; it contains no model, rollout, metric, or trace logic."""

    def __init__(self, repository_root: str | Path | None = None, *, environment=None) -> None:
        values = os.environ if environment is None else environment
        configured_root = values.get("OVLAB_ROOT")
        root = Path(repository_root or configured_root or Path(__file__).resolve().parents[5]).resolve()
        self.repository_root = root
        self.config_root = root / "configs"
        self.environment = values
        self._resolver = None

    @property
    def resolver(self) -> ConfigResolver:
        """Construct the experiment resolver only for configuration operations.

        Dataset and finalized-artifact inspection are valid in purpose-built
        images that intentionally contain no portable experiment tree.
        """
        if self._resolver is None:
            self._resolver = ConfigResolver(self.config_root, repository_root=self.repository_root)
        return self._resolver

    def _config_path(self, config: str | Path) -> Path:
        path = Path(config).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        path = path.resolve()
        if not path.is_file():
            raise ConfigReferenceError(f"configuration does not exist: {path}")
        if not path.is_relative_to(self.config_root):
            raise ConfigReferenceError("CLI configurations must be inside configs/")
        return path

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.config_root))

    def _local_profile(self) -> Path:
        configured = self.environment.get("OVLAB_LOCAL_PROFILE")
        path = Path(configured).expanduser() if configured else self.config_root / "local/profile.yaml"
        if not path.is_absolute():
            path = self.repository_root / path
        if not path.is_file():
            raise ConfigReferenceError(
                "runtime resolution requires OVLAB_LOCAL_PROFILE or configs/local/profile.yaml"
            )
        return path.resolve()

    def _data_roots(self, *, runs_root=None):
        data_root = Path(
            self.environment.get("OVLAB_DATA_ROOT", self.repository_root.parent / "ovlab-data")
        ).expanduser().resolve()
        runs = Path(
            runs_root or self.environment.get("OVLAB_RUNS_ROOT", data_root / "runs")
        ).expanduser().resolve()
        derived = Path(
            self.environment.get("OVLAB_DERIVED_ROOT", runs.parent / "derived")
        ).expanduser().resolve()
        exports = Path(
            self.environment.get("OVLAB_EXPORTS_ROOT", runs.parent / "exports")
        ).expanduser().resolve()
        return runs, derived, exports

    def _model_data_root(self) -> Path:
        """Host-visible model data; never part of a scientific identity."""
        return Path(
            self.environment.get(
                "OVLAB_MODEL_DATA_ROOT",
                self.environment.get("OVLAB_DATA_ROOT", self.repository_root.parent / "ovlab-data"),
            )
        ).expanduser().resolve()

    def _training_profile(self, reference):
        from .training_profiles import TrainingProfile

        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        if not path.is_file():
            raise ConfigReferenceError(f"training profile does not exist: {path.resolve()}")
        return path.resolve(), TrainingProfile.from_document(load(path.resolve()))

    def _report_profile(self, reference):
        from ovlab_runner import ArtifactError, ReportProfile, builtin_profile
        if reference == "libero-task-default":
            return builtin_profile(reference)
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        if not path.is_file():
            raise ConfigReferenceError(f"report profile does not exist: {path.resolve()}")
        try:
            return ReportProfile.from_mapping(load(path.resolve()), template_base=path.resolve().parent)
        except ArtifactError as exc:
            from .errors import ConfigSchemaError
            raise ConfigSchemaError(f"invalid report profile: {exc}") from exc

    def _quic_descriptor(self, path: Path, *, runtime: bool):
        from ovlab_openvla_quic import descriptor_from_document

        document = self.resolver.load_component(self._relative(path), "quic_policy_descriptor")
        descriptor = descriptor_from_document(document)
        if runtime:
            descriptor.require_runtime_ready()
        return document, descriptor

    def resolve(self, config: str | Path, *, mode: str = "descriptor"):
        if mode not in {"descriptor", "runtime"}:
            raise ValueError("mode must be descriptor or runtime")
        path = self._config_path(config)
        header = load(path)
        if header.get("kind") == "quic_policy_descriptor":
            document, descriptor = self._quic_descriptor(path, runtime=mode == "runtime")
            return {
                "schema_version": CLI_SCHEMA_VERSION,
                "kind": "resolved_policy_descriptor",
                "source": self._relative(path),
                "descriptor": descriptor.as_metadata(),
                "scientific_config_hash": descriptor.scientific_hash,
                "execution_config_hash": None,
                "runtime_ready": mode == "runtime",
            }
        if header.get("kind") != "experiment":
            kind = header.get("kind")
            document = self.resolver.load_component(self._relative(path), kind)
            return document
        execution = self.environment.get("OVLAB_EXECUTION_PROFILE")
        return self.resolver.resolve(
            path,
            local_profile=self._local_profile(),
            execution_profile=execution,
            environment=self.environment,
        )

    def validate(self, config: str | Path, *, mode: str) -> dict[str, object]:
        resolved = self.resolve(config, mode=mode)
        if isinstance(resolved, ResolvedExperimentConfig):
            return {
                "valid": True,
                "mode": mode,
                "kind": "experiment",
                "experiment_id": resolved.experiment_id,
                "scientific_config_hash": resolved.scientific_config_hash,
                "execution_config_hash": resolved.execution_config_hash,
            }
        return {
            "valid": True,
            "mode": mode,
            "kind": resolved.get("kind"),
            "scientific_config_hash": resolved.get("scientific_config_hash"),
            "execution_config_hash": resolved.get("execution_config_hash"),
        }

    def resolved_document(self, config: str | Path, *, mode: str):
        value = self.resolve(config, mode=mode)
        return _plain(value.document() if isinstance(value, ResolvedExperimentConfig) else value)

    @staticmethod
    def policy_list() -> list[dict[str, object]]:
        return registered_policies()

    def policy_describe(self, config: str | Path) -> dict[str, object]:
        path = self._config_path(config)
        header = load(path)
        if header.get("kind") == "quic_policy_descriptor":
            _, descriptor = self._quic_descriptor(path, runtime=False)
            metadata = descriptor.as_metadata()
            return {
                "method": descriptor.variant.value,
                "family": descriptor.family,
                "artifact": metadata["artifact_identity"],
                "capabilities": metadata["capability_identity"],
                "readiness": {
                    "runtime_ready": False,
                    "implementation_status": metadata["implementation_status"],
                    "openvla_integration_status": metadata["openvla_integration_status"],
                },
                "profile": metadata["profile"],
                "scientific_config_hash": descriptor.scientific_hash,
                "execution_config_hash": None,
                "unavailable_fields": metadata["unavailable_fields"],
            }
        resolved = self.resolve(path, mode="descriptor")
        if not isinstance(resolved, ResolvedExperimentConfig):
            raise ConfigCompatibilityError("policy describe requires an experiment or QuIC descriptor")
        policy = resolved.scientific_config["components"]["policy"]
        settings = policy["settings"]
        return {
            "method": policy["type"],
            "family": {
                "openvla_vanilla": "openvla",
                "openvla_lora_merged": "lora",
                "openvla_oft": "openvla_oft",
                "mock": "mock",
            }[policy["type"]],
            "artifact": {"checkpoint_id": settings.get("checkpoint_id", "unavailable")},
            "capabilities": {
                "action_spec": _action_spec(resolved.action_spec),
                "input": settings["input"],
            },
            "readiness": {"runtime_ready": policy["type"] != "mock", "provider_loaded": False},
            "profile": None,
            "scientific_config_hash": resolved.scientific_config_hash,
            "execution_config_hash": resolved.execution_config_hash,
            "unavailable_fields": [],
        }

    def default_socket(self, config: str | Path) -> Path:
        configured = self.environment.get("OVLAB_POLICY_SOCKET")
        if configured:
            return Path(configured)
        path = self._config_path(config)
        raw = load(path)
        identity = raw.get("id") or raw.get("experiment", {}).get("id") or path.stem
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity).strip(".-") or "policy"
        return Path("/tmp") / f"ovlab-{os.getuid()}" / f"{slug}.sock"

    def _resolved_experiment(self, config: str | Path) -> ResolvedExperimentConfig:
        resolved = self.resolve(config, mode="runtime")
        if not isinstance(resolved, ResolvedExperimentConfig):
            raise ConfigCompatibilityError("operation requires a complete experiment configuration")
        return resolved

    @staticmethod
    def _policy_adapter(settings):
        if isinstance(settings, MockPolicySettings):
            raise ConfigCompatibilityError("mock policy adapters are test-only and unavailable to production CLI")
        from ovlab_openvla_oft import OpenVlaOftAdapter, OpenVlaOftSettings
        if isinstance(settings, OpenVlaOftSettings):
            return OpenVlaOftAdapter(settings)
        from ovlab_openvla_common import OpenVlaMethodFamily
        if settings.method_descriptor.family is OpenVlaMethodFamily.LORA:
            from ovlab_openvla_lora_merged import OpenVlaMergedLoraAdapter
            return OpenVlaMergedLoraAdapter(settings)
        from ovlab_openvla_vanilla import OpenVlaVanillaAdapter
        return OpenVlaVanillaAdapter(settings)

    @staticmethod
    def _identity_provider(capabilities):
        if capabilities.component_name == "ovlab-openvla-vanilla":
            from ovlab_remote_policy.openvla_service import _identity_provider
            identity = _identity_provider(capabilities)
            return OvlabApplication._with_deployment_identity(identity)
        if capabilities.component_name == "ovlab-openvla-lora-merged":
            from ovlab_openvla_lora_merged.service import _identity_provider
            identity = _identity_provider(capabilities)
            return OvlabApplication._with_deployment_identity(identity)
        if capabilities.component_name == "ovlab-openvla-oft":
            from ovlab_openvla_oft.service import _identity
            identity = _identity(capabilities)
            return OvlabApplication._with_deployment_identity(identity)
        if capabilities.component_name not in {"mock-policy", "handshake-only-policy", "qualification-test-policy"}:
            raise ConfigCompatibilityError(
                f"no registered service identity provider for {capabilities.component_name!r}"
            )
        metadata = capabilities.metadata
        checkpoint = dict(metadata.get("checkpoint_identity", metadata.get("runtime", {}).get("verified_artifact", {})))
        method = metadata.get("method_descriptor")
        return OvlabApplication._with_deployment_identity({
            "model_identity": checkpoint or {"availability": "unavailable"},
            "normalization_identity": {
                "unnorm_key": checkpoint.get("unnorm_key", "unavailable"),
                "action_statistics_identity": checkpoint.get("action_statistics_identity", "unavailable"),
            },
            "prompt_template_identity": metadata.get("prompt_template", "unavailable"),
            "action_codec_identity": {
                "identifier": metadata.get("action_codec", "unavailable"),
                "conversion_owner": metadata.get("action_codec_owner", "unavailable"),
                "application_count": 1,
                "output_gripper_convention": capabilities.output_action_spec.gripper_convention.value,
            },
            "runtime_versions": {
                "policy_component": f"{capabilities.component_name}@{capabilities.component_version}",
                "protocol_component": f"ovlab-remote-policy@{OVLAB_VERSION}",
            },
            **({"method_descriptor": _plain(method)} if method is not None else {}),
        })

    @staticmethod
    def _deployment_provenance(environment=None) -> dict[str, str]:
        values = os.environ if environment is None else environment
        mapping = {
            "image_role": "OVLAB_IMAGE_ROLE",
            "image_reference": "OVLAB_IMAGE_REFERENCE",
            "image_digest": "OVLAB_IMAGE_DIGEST",
            "source_manifest_sha256": "OVLAB_SOURCE_MANIFEST_SHA256",
            "source_dirty": "OVLAB_SOURCE_DIRTY",
            "dependency_lock_sha256": "OVLAB_LOCK_SHA256",
            "dockerfile_sha256": "OVLAB_DOCKERFILE_SHA256",
            "build_target": "OVLAB_BUILD_TARGET",
            "python_version": "OVLAB_PYTHON_VERSION",
            "cuda_runtime_version": "OVLAB_CUDA_RUNTIME_VERSION",
            "deployment_manifest_sha256": "OVLAB_DEPLOYMENT_MANIFEST_SHA256",
            "config_bundle_sha256": "OVLAB_CONFIG_BUNDLE_SHA256",
            "container_runtime_version": "OVLAB_CONTAINER_RUNTIME_VERSION",
            "service_topology": "OVLAB_SERVICE_TOPOLOGY",
            "offline_mode": "OVLAB_OFFLINE_MODE",
            "mount_contract": "OVLAB_MOUNT_CONTRACT",
            "checkpoint_id": "OVLAB_CHECKPOINT_ID",
            "checkpoint_source_kind": "OVLAB_CHECKPOINT_SOURCE_KIND",
            "checkpoint_host_path": "OVLAB_CHECKPOINT_HOST_PATH",
            "checkpoint_repository": "OVLAB_CHECKPOINT_REPOSITORY",
            "checkpoint_revision": "OVLAB_CHECKPOINT_REVISION",
            "checkpoint_sha256": "OVLAB_CHECKPOINT_SHA256",
        }
        return {
            key: values[name]
            for key, name in mapping.items()
            if isinstance(values.get(name), str) and values[name]
        }

    @staticmethod
    def _with_deployment_identity(identity):
        result = dict(identity)
        runtime = dict(result["runtime_versions"])
        for key, value in OvlabApplication._deployment_provenance().items():
            runtime[f"deployment_{key}"] = value
        result["runtime_versions"] = runtime
        return result

    def serve(self, config: str | Path, *, socket_path: str | Path | None = None, adapter_factory=None) -> dict[str, object]:
        path = self._config_path(config)
        if load(path).get("kind") == "quic_policy_descriptor":
            self._quic_descriptor(path, runtime=True)
            raise AssertionError("runtime-ready QuIC descriptor did not produce an experiment")
        resolved = self._resolved_experiment(path)
        adapter = (adapter_factory or self._policy_adapter)(resolved.policy_settings)
        socket = Path(socket_path) if socket_path is not None else self.default_socket(path)
        socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(socket.parent, 0o700)
        from ovlab_core.contracts import RunContext, RunId
        from ovlab_remote_policy.service import PolicyService
        startup_context = RunContext(
            RunId(f"service-{uuid.uuid4().hex}"),
            time.time_ns(),
            resolved.experiment_id,
            resolved.protocol_settings.base_seed,
        )
        PolicyService(
            socket,
            adapter,
            identity_provider=self._identity_provider,
            startup_context=startup_context,
        ).serve()
        return {"socket": str(socket), "closed": True}

    @staticmethod
    def service_health(socket_path: str | Path) -> dict[str, object]:
        from ovlab_remote_policy import UnixPolicyClient
        from ovlab_remote_policy.errors import RemotePolicyServiceError

        client = UnixPolicyClient(socket_path, request_timeout_s=5.0)
        try:
            client.connect()
            result = client.health()
        finally:
            client.close_socket()
        if result["state"] != "ready":
            raise RemotePolicyServiceError(
                f"policy service is not ready (state={result['state']!r})"
            )
        return {**result, "ready": True, "prediction_count": 0, "trace_created": False}

    def connect(self, config: str | Path, *, socket_path: str | Path | None = None) -> dict[str, object]:
        from ovlab_core import negotiate_capabilities
        from ovlab_core.contracts import RunContext, RunId
        from ovlab_remote_policy import RemotePolicyAdapter, UnixPolicyClient
        from ovlab_benchmarks.libero import LiberoAdapterSettings, configured_capabilities

        resolved = self._resolved_experiment(config)
        if not isinstance(resolved.benchmark_settings, LiberoAdapterSettings):
            raise ConfigCompatibilityError("connect requires a configured isolated runtime policy and benchmark")
        socket = Path(socket_path) if socket_path is not None else self.default_socket(config)
        policy = RemotePolicyAdapter(UnixPolicyClient(socket))
        context = RunContext(RunId(f"connect-{uuid.uuid4().hex}"), time.time_ns(), resolved.experiment_id, 0)
        try:
            policy_capabilities = policy.initialize(context)
            report = negotiate_capabilities(configured_capabilities(resolved.benchmark_settings), policy_capabilities)
            handshake = policy.handshake
            return {
                "policy": handshake.get("model_identity"),
                "protocol_version": handshake["protocol_version"],
                "observation_requirements": _observation_requirements(policy_capabilities.observation_requirements),
                "supports_single_action": policy_capabilities.supports_single_action,
                "supports_action_chunks": policy_capabilities.supports_action_chunks,
                "minimum_action_horizon": policy_capabilities.minimum_action_horizon,
                "maximum_action_horizon": policy_capabilities.maximum_action_horizon,
                "action_spec": _action_spec(policy_capabilities.output_action_spec),
                "normalization_identity": handshake["normalization_identity"],
                "compatible": report.compatible,
                "compatibility_issues": [
                    {"code": item.code, "severity": item.severity.value, "path": item.path, "message": item.message}
                    for item in report.issues
                ],
                "scientific_config_hash": resolved.scientific_config_hash,
                "execution_config_hash": resolved.execution_config_hash,
                "prediction_count": 0,
                "trace_created": False,
            }
        finally:
            policy.close()

    def execution_plan(self, config: str | Path, *, output_root: str | Path | None = None) -> dict[str, object]:
        resolved = self._resolved_experiment(config)
        output = resolved.artifact_settings.root if output_root is None else str(Path(output_root).expanduser().resolve())
        scientific = resolved.scientific_config
        return {
            "experiment_id": resolved.experiment_id,
            "policy": scientific["components"]["policy"]["type"],
            "benchmark": scientific["components"]["benchmark"]["type"],
            "metrics": list(resolved.metric_settings.enabled_metric_ids),
            "scientific_config_hash": resolved.scientific_config_hash,
            "execution_config_hash": resolved.execution_config_hash,
            "service_mode": "external-af-unix",
            "socket": str(self.default_socket(config)),
            "output_root": output,
            "side_effects_performed": False,
        }

    @staticmethod
    def _selected_tasks(resolved: ResolvedExperimentConfig):
        from ovlab_benchmarks.libero import LiberoAdapterSettings
        from ovlab_core.contracts import TaskId
        settings = resolved.benchmark_settings
        if not isinstance(settings, LiberoAdapterSettings):
            raise ConfigCompatibilityError("production CLI run currently requires LIBERO")
        slug = {
            "LIBERO-10": "10", "LIBERO-Spatial": "spatial", "LIBERO-Object": "object", "LIBERO-Goal": "goal",
        }
        indices = settings.task_indices if settings.task_indices is not None else tuple(range(10))
        return tuple(TaskId(f"libero/{slug[suite]}/{index}") for suite in settings.suite_names for index in indices)

    def run(self, config: str | Path, *, output_root: str | Path | None = None) -> dict[str, object]:
        from ovlab_benchmarks.libero import LiberoBenchmarkAdapter
        from ovlab_core.contracts import RunContext, RunId
        from ovlab_remote_policy import RemotePolicyAdapter, UnixPolicyClient
        from ovlab_runner import (
            AutomaticDerivedReporter, DerivedReportEngine, ExperimentRunner, ExportEngine,
            FilesystemRunArtifactStore,
        )

        resolved = self._resolved_experiment(config)
        created_wall_time_utc_ns = time.time_ns()
        run_id = RunId(_readable_run_id(
            resolved.experiment_id,
            created_wall_time_utc_ns,
            uuid.uuid4().hex,
        ))
        context = RunContext(
            run_id,
            created_wall_time_utc_ns,
            resolved.experiment_id,
            resolved.protocol_settings.base_seed,
        )
        plan = resolved.create_plan(context, self._selected_tasks(resolved))
        plan = replace(plan, metadata={
            **dict(plan.metadata),
            **({"qualification": "test-provider"} if isinstance(resolved.policy_settings, MockPolicySettings) else {}),
            "cli": {
                "schema_version": CLI_SCHEMA_VERSION,
                "version": CLI_VERSION,
                "repository_revision": repository_revision(self.repository_root) or "unavailable",
                "command": "run",
                "resolved_config_identity": resolved.experiment_id,
                "scientific_config_hash": resolved.scientific_config_hash,
                "execution_config_hash": resolved.execution_config_hash,
                "service_topology": "external-af-unix",
                "socket": "machine-local",
                "output_root_overridden": output_root is not None,
            },
            "deployment": self._deployment_provenance(self.environment),
        })
        root = resolved.artifact_settings.root if output_root is None else Path(output_root).expanduser().resolve()
        reporting = resolved.reporting_settings
        postprocessor = None
        postprocessing_mode = self.environment.get("OVLAB_POSTPROCESSING_MODE", "external")
        if postprocessing_mode not in {"external", "internal"}:
            raise ConfigCompatibilityError(
                "OVLAB_POSTPROCESSING_MODE must be 'external' or 'internal'"
            )
        if postprocessing_mode == "internal":
            _, derived_root, exports_root = self._data_roots(runs_root=root)
            report_engine = None
            if reporting.enabled:
                profile = self._report_profile(reporting.profile)
                report_engine = DerivedReportEngine(root, derived_root, profile)
            postprocessor = AutomaticDerivedReporter(
                report_engine,
                on_task_finalize=reporting.on_task_finalize,
                on_run_finalize=reporting.on_run_finalize,
                isolated_export_engine=ExportEngine(root, exports_root),
            )
        socket = self.default_socket(config)
        policy = RemotePolicyAdapter(UnixPolicyClient(socket))
        runner = ExperimentRunner(
            plan,
            LiberoBenchmarkAdapter(resolved.benchmark_settings),
            policy,
            FilesystemRunArtifactStore(root),
            configuration_snapshot=resolved.configuration_snapshot(),
            postprocessor=postprocessor,
            postprocessor_failure_policy=reporting.failure_policy,
        )
        try:
            report = runner.connect()
            runner.run()
        finally:
            runner.close()
        return {
            "run_id": str(run_id),
            "run_path": str(FilesystemRunArtifactStore(root)._run_path(run_id)),
            "compatible": report.compatibility_report.compatible,
            "scientific_config_hash": resolved.scientific_config_hash,
            "execution_config_hash": resolved.execution_config_hash,
            "status": "completed",
        }

    @staticmethod
    def inspect(path):
        from ovlab_runner import inspect_run
        return inspect_run(path)

    @staticmethod
    def verify(path):
        from ovlab_runner import verify_run
        return verify_run(path)

    @staticmethod
    def recompute_metrics(path):
        from ovlab_runner import recompute_run_metrics
        return recompute_run_metrics(path)

    @staticmethod
    def generate_report(path, output):
        from ovlab_runner import regenerate_report
        return regenerate_report(path, output)

    def report_generate(self, run, profile="libero-task-default", *, task_id=None):
        from ovlab_runner import DerivedReportEngine
        runs, derived, _ = self._data_roots()
        return DerivedReportEngine(runs, derived, self._report_profile(profile)).generate(run, task_id=task_id)

    def report_publish(self, run, profile="libero-task-default", *, report_enabled=True):
        """Publish regenerable outputs from one finalized canonical run."""
        from ovlab_runner import DerivedReportEngine, ExportEngine
        runs, derived, exports = self._data_roots()
        report = None
        if report_enabled:
            report = DerivedReportEngine(
                runs, derived, self._report_profile(profile)
            ).generate(run)
        isolated = ExportEngine(runs, exports).generate_isolated(run)
        return {
            "schema_version": "ovlab.postprocessing-result/v1",
            "run_id": str(run),
            "canonical_run_modified": False,
            "report": report,
            "isolated_export": isolated,
            "status": "completed",
        }

    def report_verify(self, run, profile="libero-task-default", *, build_id=None):
        from ovlab_runner import DerivedReportEngine
        runs, derived, _ = self._data_roots()
        return DerivedReportEngine(runs, derived, self._report_profile(profile)).verify(run, build_id=build_id)

    @staticmethod
    def report_profiles():
        from ovlab_runner import report_profiles
        return {"schema_version": "ovlab.report-profiles/v1", "profiles": list(report_profiles())}

    def export_generate(self, spec):
        from ovlab_runner import ArtifactError, ExportEngine, validate_export_spec
        path = Path(spec).expanduser()
        if not path.is_absolute():
            path = self.repository_root / path
        if not path.is_file():
            raise ConfigReferenceError(f"export specification does not exist: {path.resolve()}")
        runs, _, exports = self._data_roots()
        try:
            document = validate_export_spec(load(path.resolve()))
        except ArtifactError as exc:
            from .errors import ConfigSchemaError
            raise ConfigSchemaError(f"invalid export specification: {exc}") from exc
        return ExportEngine(runs, exports).generate(document)

    def export_isolated(self, run_id, *, episode_id=None, template="isolated-default-v1"):
        from ovlab_runner import ExportEngine
        runs, _, exports = self._data_roots()
        return ExportEngine(runs, exports).generate_isolated(
            run_id, episode_id=episode_id, template=template,
        )

    def export_grouped(
        self, group_name, *, all_runs=False, run_ids=(), same_model_as=None,
        suite=None, template="grouped-default-v1",
    ):
        from ovlab_runner import ExportEngine
        runs, _, exports = self._data_roots()
        return ExportEngine(runs, exports).generate_grouped(
            group_name, all_runs=all_runs, run_ids=run_ids,
            same_model_as=same_model_as, suite=suite, template=template,
        )

    def export_verify(self, kind, name):
        from ovlab_runner import ExportEngine
        runs, _, exports = self._data_roots()
        return ExportEngine(runs, exports).verify(kind, name)

    # Gate I dataset, training, and checkpoint application services. These
    # compose domain owners but never import model runtimes during read-only work.
    @staticmethod
    def dataset_providers():
        from .datasets import DatasetBridgeRegistry
        return {"schema_version": "ovlab.dataset-providers/v1", "providers": DatasetBridgeRegistry().providers()}

    def dataset_resolve(self, *, source, name, version="1", url=None, sha256=None, archive="auto", preparation=None, local_path=None, allow_local_http=False):
        from .datasets import DatasetRequest, DatasetStore
        request = DatasetRequest(
            source=source, name=name, version=version, url=url, sha256=sha256,
            archive=archive, preparation=preparation,
            local_path=None if local_path is None else Path(local_path),
            allow_local_http=allow_local_http,
        )
        return DatasetStore(self._model_data_root()).resolve(request).as_dict()

    def dataset_fetch(self, *, source, name, version="1", url=None, sha256=None, archive="auto", preparation=None, allow_dataset_download=False, allow_local_http=False):
        from .datasets import DatasetRequest, DatasetStore
        request = DatasetRequest(
            source=source, name=name, version=version, url=url, sha256=sha256,
            archive=archive, preparation=preparation, allow_local_http=allow_local_http,
        )
        return DatasetStore(self._model_data_root()).fetch(
            request, allow_download=allow_dataset_download,
            progress=lambda message: print(f"ovlab: dataset: {message}", file=__import__("sys").stderr),
        )

    def dataset_import(self, *, name, version, path, preparation=None):
        from .datasets import DatasetRequest, DatasetStore
        return DatasetStore(self._model_data_root()).import_local(DatasetRequest(
            source="local", name=name, version=version, local_path=Path(path), preparation=preparation,
        ))

    def dataset_prepare(self, dataset_id, preparation_format):
        from .datasets import DatasetStore
        return DatasetStore(self._model_data_root()).prepare(dataset_id, preparation_format)

    def dataset_list(self):
        from .datasets import DatasetStore
        return {"schema_version": "ovlab.dataset-list/v1", "datasets": DatasetStore(self._model_data_root()).list()}

    def dataset_inspect(self, dataset_id):
        from .datasets import DatasetStore
        return DatasetStore(self._model_data_root()).inspect(dataset_id)

    def dataset_verify(self, dataset_id):
        from .datasets import DatasetStore
        return DatasetStore(self._model_data_root()).verify(dataset_id)

    def train_profiles(self):
        from .training_profiles import TrainingProfile
        profiles = []
        for path in sorted((self.config_root / "training").glob("*.yaml")):
            try:
                profile = TrainingProfile.from_document(load(path))
            except Exception as exc:
                profiles.append({"path": str(path), "valid": False, "error": str(exc)})
            else:
                profiles.append({"id": profile.profile_id, "path": str(path), "valid": True})
        return {"schema_version": "ovlab.training-profiles/v1", "profiles": profiles}

    def train_validate(self, profile):
        from .training_identity import identity
        path, parsed = self._training_profile(profile)
        return {
            "schema_version": "ovlab.training-validation/v1",
            "valid": True,
            "profile_id": parsed.profile_id,
            "normalized_profile_id": identity("training-profile", parsed.document, 32),
            "source": str(path),
            "model_initialized": False,
            "network_used": False,
        }

    def train_plan(self, profile, *, allow_dataset_download=False):
        from .training_profiles import TrainingPlanner
        _, parsed = self._training_profile(profile)
        gpu_count = self.environment.get("OVLAB_AVAILABLE_GPU_COUNT")
        vram = self.environment.get("OVLAB_AVAILABLE_VRAM_GIB")
        return TrainingPlanner(self.repository_root, self._model_data_root()).plan(
            parsed,
            allow_dataset_download=allow_dataset_download,
            available_gpu_count=None if gpu_count is None else int(gpu_count),
            available_vram_gib=None if vram is None else float(vram),
            image_identity=self.environment.get("OVLAB_TRAINING_IMAGE_DIGEST", "unavailable"),
        )

    def train_inspect(self, run_id):
        from .training_runs import TrainingRunStore
        return TrainingRunStore(self._model_data_root()).inspect(run_id)

    def train_status(self, run_id):
        inspected = self.train_inspect(run_id)
        return {"schema_version": "ovlab.training-status/v1", "run_id": run_id, **inspected["result"]}

    def train_verify(self, run_id):
        from .training_runs import TrainingRunStore
        return TrainingRunStore(self._model_data_root()).verify(run_id)

    def checkpoint_list(self):
        from .training_runs import CheckpointBundleStore
        return {"schema_version": "ovlab.checkpoint-list/v1", "checkpoints": CheckpointBundleStore(self._model_data_root()).list()}

    def checkpoint_inspect(self, checkpoint_id):
        from .training_runs import CheckpointBundleStore
        return CheckpointBundleStore(self._model_data_root()).inspect(checkpoint_id)

    def checkpoint_verify(self, checkpoint_id):
        from .training_runs import CheckpointBundleStore
        return CheckpointBundleStore(self._model_data_root()).verify(checkpoint_id)
