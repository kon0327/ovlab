"""Unified dependency-light OVLAB command-line interface."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import traceback

from .versioning import CLI_VERSION, repository_revision

OUTPUT_SCHEMA_VERSION = "ovlab-cli-output/1.0.0"


class CliUsageError(ValueError):
    """Parsed arguments form an invalid command combination."""


class ExitCode:
    SUCCESS = 0
    USAGE = 2
    CONFIGURATION = 3
    POLICY_UNAVAILABLE = 4
    SERVICE = 5
    RUNTIME = 6
    INTEGRITY = 7
    METRICS = 8
    INTERRUPTED = 130


@contextmanager
def _configured_artifact_umask():
    """Apply a process-local artifact umask for one CLI operation."""
    raw = os.environ.get("OVLAB_ARTIFACT_UMASK")
    if raw is None:
        yield
        return
    if not isinstance(raw, str) or len(raw) not in {3, 4} or any(
        character not in "01234567" for character in raw
    ):
        raise CliUsageError("OVLAB_ARTIFACT_UMASK must be a three- or four-digit octal value")
    value = int(raw, 8)
    if value > 0o777:
        raise CliUsageError("OVLAB_ARTIFACT_UMASK must be between 0000 and 0777")
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


def _output_options(parser: argparse.ArgumentParser, *, detail: bool = True) -> None:
    """Add consistent human-detail and machine-readable output controls."""
    output = parser.add_mutually_exclusive_group()
    if detail:
        output.add_argument(
            "--detail", action="store_true",
            help="print the complete result document instead of the compact summary",
        )
    output.add_argument(
        "--json", action="store_true",
        help="print one stable machine-readable JSON envelope",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ovlab", description="OpenVLABenchmark orchestration CLI")
    parser.add_argument(
        "--version", dest="show_version", action="store_true",
        help="show OVLAB CLI and repository revision",
    )
    parser.add_argument("--debug", action="store_true", help="include a traceback for unexpected failures")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    config = commands.add_parser("config", help="validate or resolve configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="validate a configuration")
    validate.add_argument("config")
    validate.add_argument("--mode", choices=("descriptor", "runtime"), default="descriptor")
    _output_options(validate)
    resolve = config_commands.add_parser("resolve", help="print deterministic resolved configuration")
    resolve.add_argument("config")
    resolve.add_argument("--mode", choices=("descriptor", "runtime"), default="descriptor")
    resolve.add_argument("--format", choices=("yaml", "json"), default="yaml")

    policy = commands.add_parser("policy", help="inspect registered policy identities")
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    listing = policy_commands.add_parser("list", help="list policy variants without loading providers")
    _output_options(listing)
    describe = policy_commands.add_parser("describe", help="describe a configured policy without loading it")
    describe.add_argument("config")
    _output_options(describe)

    service = commands.add_parser("service", help="run an isolated policy service")
    service_commands = service.add_subparsers(dest="service_command", required=True)
    serve = service_commands.add_parser("serve", help="serve one policy over a foreground AF_UNIX socket")
    serve.add_argument("config")
    serve.add_argument("--socket")
    _output_options(serve)
    health = service_commands.add_parser("health", help="probe protocol readiness without model inference")
    health.add_argument("--socket", required=True)
    _output_options(health)

    connect = commands.add_parser("connect", help="probe policy handshake and capability compatibility")
    connect.add_argument("config")
    _output_options(connect)

    deploy = commands.add_parser("deploy", help="orchestrate isolated OVLAB containers")
    deploy_commands = deploy.add_subparsers(dest="deploy_command", required=True)
    deploy_run = deploy_commands.add_parser("run", help="run one experiment through Docker Compose")
    deploy_run.add_argument("experiment")
    deploy_run.add_argument(
        "--profile", choices=("openvla", "oft"),
        help="override deployment.profile from the experiment",
    )
    deploy_run.add_argument(
        "--renderer", choices=("egl", "glfw"),
        help="override deployment.renderer from the experiment",
    )
    deploy_run.add_argument("--env-file", default="deploy/compose/.env")
    deploy_run.add_argument(
        "--local-profile",
        help="gitignored host profile containing optional checkpoint local_path overrides",
    )
    deploy_run.add_argument(
        "--offline",
        action="store_true",
        help="reject a missing checkpoint instead of downloading its pinned revision",
    )
    deploy_run.add_argument("--project-name")
    deploy_run.add_argument("--dry-run", action="store_true")
    _output_options(deploy_run)

    run = commands.add_parser("run", help="execute, inspect, or verify a run")
    run.add_argument("target", help="CONFIG, or 'inspect'/'verify'")
    run.add_argument("path", nargs="?", help="RUN_PATH, RUN_ID, or RUN_HASH for inspect or verify")
    run.add_argument("--output-root")
    run.add_argument("--dry-run", action="store_true")
    _output_options(run)

    metrics = commands.add_parser("metrics", help="offline metric operations")
    metrics_commands = metrics.add_subparsers(dest="metrics_command", required=True)
    recompute = metrics_commands.add_parser("recompute", help="recompute metrics from immutable traces")
    recompute.add_argument("run_path", metavar="RUN_PATH_OR_REFERENCE")
    _output_options(recompute)

    report = commands.add_parser("report", help="generate and verify offline reports from immutable runs")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    generate = report_commands.add_parser(
        "generate", help="build a reproducible report (example: ovlab report generate --run RUN_ID --profile libero-task-default)",
    )
    generate.add_argument(
        "run_path", nargs="?",
        help="legacy canonical RUN_PATH, RUN_ID, or displayed RUN_HASH",
    )
    generate.add_argument("--run", dest="run_id", help="canonical run ID or displayed run hash")
    generate.add_argument("--task", help="optional canonical task ID")
    generate.add_argument("--profile", default="libero-task-default", help="built-in profile ID or local YAML path")
    generate.add_argument("--output", help="legacy output path; cannot be combined with --run")
    _output_options(generate)
    publish = report_commands.add_parser(
        "publish", help="publish the HTML report and isolated export for one finalized run",
    )
    publish.add_argument("--run", dest="run_id", required=True, help="canonical run ID or displayed run hash")
    publish.add_argument("--profile", default="libero-task-default")
    publish.add_argument(
        "--report-enabled", choices=("true", "false"), default="true",
        help="generate derived HTML in addition to the always-generated isolated export",
    )
    _output_options(publish)
    verify_report = report_commands.add_parser("verify", help="verify a derived report and its canonical inputs")
    verify_report.add_argument(
        "--run", dest="run_id", required=True,
        help="canonical run ID or displayed run hash",
    )
    verify_report.add_argument("--profile", default="libero-task-default")
    verify_report.add_argument("--build")
    _output_options(verify_report)
    profiles = report_commands.add_parser("profiles", help="list built-in report profiles")
    _output_options(profiles)

    export = commands.add_parser("export", help="generate readable isolated or grouped exports from canonical runs")
    export_commands = export.add_subparsers(dest="export_command", required=True)
    isolated = export_commands.add_parser("isolated", help="export one complete run or one episode")
    isolated.add_argument("--run", dest="run_id", required=True, help="canonical run ID or displayed run hash")
    isolated.add_argument("--episode", dest="episode_id", help="optional canonical episode ID")
    isolated.add_argument("--template", default="isolated-default-v1")
    _output_options(isolated)
    grouped = export_commands.add_parser("grouped", help="compare an explicitly selected group of runs")
    grouped.add_argument("--name", required=True, help="portable group name")
    selectors = grouped.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--all-runs", action="store_true", help="select every compatible completed run")
    selectors.add_argument("--same-model-as", metavar="RUN_REFERENCE", help="select runs with the same model/checkpoint as RUN_ID or RUN_HASH")
    selectors.add_argument("--runs", nargs="+", metavar="RUN_REFERENCE", help="select these run IDs or displayed hashes manually")
    grouped.add_argument("--suite", help="optional benchmark-suite filter")
    grouped.add_argument("--template", default="grouped-default-v1")
    _output_options(grouped)
    export_generate = export_commands.add_parser("generate", help="legacy: generate a grouped export from ovlab.export-spec/v1")
    export_generate.add_argument("--spec", required=True)
    _output_options(export_generate)
    export_verify = export_commands.add_parser("verify", help="verify an export build and source checksums")
    export_verify.add_argument("--kind", choices=("isolated", "grouped"), default="grouped")
    export_verify.add_argument(
        "--name", dest="export_name",
        help="group name, or isolated run ID/displayed run hash",
    )
    export_verify.add_argument("--export", dest="legacy_export_id", help="legacy alias for --kind grouped --name")
    _output_options(export_verify)

    data = commands.add_parser(
        "data", help="list, archive, and safely delete runs, reports, and exports"
    )
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_list = data_commands.add_parser(
        "list", help="list active or archived runs, reports, and exports"
    )
    data_list.add_argument(
        "--kind", choices=("runs", "reports", "exports", "all"), default="all"
    )
    data_list.add_argument("--archived", action="store_true", help="list OVLAB_DATA_ROOT/archive instead of active data")
    _output_options(data_list)
    for action in ("archive", "delete"):
        command = data_commands.add_parser(action, help=f"{action} selected canonical data")
        selectors = command.add_mutually_exclusive_group(required=True)
        selectors.add_argument("--run", dest="run_id", metavar="RUN_ID_OR_HASH")
        selectors.add_argument(
            "--report", dest="report_id", metavar="REPORT_ID",
            help="benchmark RUN_ID/RUN_HASH, or training:TRAINING_RUN_ID/RUN_HASH",
        )
        selectors.add_argument(
            "--export", dest="export_id", metavar="EXPORT_ID",
            help="isolated:RUN_ID_OR_HASH or grouped:GROUP_ID",
        )
        selectors.add_argument(
            "--all", dest="all_data", action="store_true",
            help=(
                "select all runs, reports, and exports; incomplete entries are refused"
                + (" unless --force is supplied" if action == "delete" else "")
            ),
        )
        command.add_argument("--dry-run", action="store_true", help="show the exact selection without changing files")
        command.add_argument("--yes", action="store_true", help="confirm the operation non-interactively")
        if action == "delete":
            command.add_argument(
                "--force", action="store_true",
                help="allow deletion of active or incomplete artifacts; path and permission safety checks remain enforced",
            )
        _output_options(command)

    dataset = commands.add_parser("dataset", help="resolve, acquire, prepare, and verify immutable datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_providers = dataset_commands.add_parser("providers", help="list registered dataset bridges without model imports")
    _output_options(dataset_providers)
    dataset_resolve = dataset_commands.add_parser("resolve", help="resolve a known benchmark dataset without downloading")
    dataset_resolve.add_argument("--benchmark", choices=("libero",), required=True)
    dataset_resolve.add_argument("--suite", required=True)
    _output_options(dataset_resolve)
    dataset_fetch = dataset_commands.add_parser("fetch", help="explicitly acquire and prepare a verified dataset")
    dataset_fetch.add_argument("--source", choices=("libero", "url"), required=True)
    dataset_fetch.add_argument("--name", required=True)
    dataset_fetch.add_argument("--version", default="1")
    dataset_fetch.add_argument("--url")
    dataset_fetch.add_argument("--sha256")
    dataset_fetch.add_argument("--archive", default="auto", choices=("auto", "none", "zip", "tar", "tar.gz", "tgz", "tar.zst"))
    dataset_fetch.add_argument("--format", dest="preparation")
    dataset_fetch.add_argument("--allow-local-http", action="store_true", help=argparse.SUPPRESS)
    _output_options(dataset_fetch)
    dataset_import = dataset_commands.add_parser("import", help="copy and register an existing local dataset immutably")
    dataset_import.add_argument("--name", required=True)
    dataset_import.add_argument("--version", required=True)
    dataset_import.add_argument("--path", required=True)
    dataset_import.add_argument("--format", dest="preparation")
    _output_options(dataset_import)
    dataset_prepare = dataset_commands.add_parser("prepare", help="verify the selected immutable preparation")
    dataset_prepare.add_argument("--dataset", dest="dataset_id", required=True)
    dataset_prepare.add_argument("--format", dest="preparation", required=True)
    _output_options(dataset_prepare)
    dataset_list = dataset_commands.add_parser("list", help="list locally ready immutable datasets")
    _output_options(dataset_list)
    dataset_inspect = dataset_commands.add_parser("inspect", help="inspect one immutable dataset manifest")
    dataset_inspect.add_argument("--dataset", dest="dataset_id", required=True)
    _output_options(dataset_inspect)
    dataset_verify = dataset_commands.add_parser("verify", help="verify dataset bytes without network access")
    dataset_verify.add_argument("--dataset", dest="dataset_id", required=True)
    _output_options(dataset_verify)

    train = commands.add_parser("train", help="validate, plan, execute, and inspect isolated training")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_profiles = train_commands.add_parser("profiles", help="list portable training profiles")
    _output_options(train_profiles)
    train_validate = train_commands.add_parser("validate", help="strictly validate a profile without resolving resources")
    train_validate.add_argument("--profile", required=True)
    _output_options(train_validate)
    train_plan = train_commands.add_parser("plan", help="resolve immutable resources without model initialization")
    train_plan.add_argument("--profile", required=True)
    _output_options(train_plan)
    train_run = train_commands.add_parser("run", help="run an isolated offline trainer and finalize its checkpoint")
    train_run.add_argument("--profile", required=True)
    train_run.add_argument("--allow-dataset-download", action="store_true")
    _output_options(train_run)
    for command_name in ("status", "inspect", "verify"):
        command = train_commands.add_parser(command_name, help=f"{command_name} a canonical training run")
        command.add_argument(
            "--run", dest="run_id", required=True,
            help="training run ID or displayed run hash",
        )
        _output_options(command)
    train_report = train_commands.add_parser(
        "report", help="generate or verify an offline system-performance report",
    )
    train_report.add_argument(
        "--run", dest="run_id", required=True,
        help="training run ID or displayed run hash",
    )
    train_report.add_argument("--verify", action="store_true", help="verify the selected/latest build instead of generating")
    train_report.add_argument("--build", help="derived build ID used with --verify")
    _output_options(train_report)

    checkpoint = commands.add_parser("checkpoint", help="inspect finalized immutable training checkpoints")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_list = checkpoint_commands.add_parser("list", help="list finalized training checkpoints")
    _output_options(checkpoint_list)
    checkpoint_inspect = checkpoint_commands.add_parser("inspect", help="inspect checkpoint identity and compatibility")
    checkpoint_inspect.add_argument("--checkpoint", dest="checkpoint_id", required=True)
    _output_options(checkpoint_inspect)
    checkpoint_verify = checkpoint_commands.add_parser("verify", help="verify checkpoint files and tensor structure")
    checkpoint_verify.add_argument("--checkpoint", dest="checkpoint_id", required=True)
    _output_options(checkpoint_verify)
    return parser


def _repository_root() -> Path:
    return Path(os.environ.get("OVLAB_ROOT", Path(__file__).resolve().parents[5])).resolve()


def _application():
    from .application import OvlabApplication
    return OvlabApplication(_repository_root())


def _command_name(args) -> str:
    parts = [args.command]
    for name in (
        "config_command", "policy_command", "service_command", "deploy_command",
        "metrics_command", "report_command", "export_command",
        "data_command", "dataset_command", "train_command", "checkpoint_command",
    ):
        value = getattr(args, name, None)
        if value:
            parts.append(value)
    if args.command == "run" and args.target in {"inspect", "verify"}:
        parts.append(args.target)
    return " ".join(item for item in parts if item)


def _json_output(command: str, status: str, result=None, errors=()) -> None:
    document = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "command": command,
        "status": status,
        "result": result,
        "errors": list(errors),
    }
    sys.stdout.write(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")


def _human(value) -> None:
    if isinstance(value, str):
        if value:
            sys.stdout.write(value.rstrip() + "\n")
    else:
        sys.stdout.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


_MISSING = object()


def _nested(document, path: str):
    value = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _first(document, paths: tuple[str, ...]):
    for path in paths:
        value = _nested(document, path)
        if value is not _MISSING and value is not None:
            return value
    return _MISSING


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)) and all(not isinstance(item, (dict, list, tuple)) for item in value):
        return ", ".join(_scalar(item) for item in value) if value else "none"
    if isinstance(value, dict):
        for key in ("id", "checkpoint_id", "resource_id", "name", "repository"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float)):
                return str(candidate)
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return "none"
    return str(value)


def _columns(rows, fields: tuple[tuple[str, ...], ...], *, empty: str) -> str:
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            lines.append(_scalar(row))
            continue
        values = []
        for alternatives in fields:
            value = _first(row, alternatives)
            values.append("-" if value is _MISSING else _scalar(value))
        lines.append(" ".join(values))
    return "\n".join(lines) if lines else empty


_SUMMARY_FIELDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "config validate": (
        ("valid", ("valid",)), ("kind", ("kind",)), ("mode", ("mode",)),
        ("experiment", ("experiment_id",)),
        ("scientific hash", ("scientific_config_hash",)),
        ("execution hash", ("execution_config_hash",)),
    ),
    "policy describe": (
        ("method", ("method",)), ("family", ("family",)),
        ("checkpoint", ("artifact.checkpoint_id", "artifact.id")),
        ("runtime ready", ("readiness.runtime_ready",)),
        ("implementation", ("readiness.implementation_status",)),
        ("scientific hash", ("scientific_config_hash",)),
        ("execution hash", ("execution_config_hash",)),
    ),
    "service serve": (("socket", ("socket",)), ("closed", ("closed",))),
    "service health": (
        ("state", ("state",)), ("ready", ("ready",)),
        ("protocol", ("protocol_version",)), ("predictions", ("prediction_count",)),
    ),
    "connect": (
        ("compatible", ("compatible",)), ("protocol", ("protocol_version",)),
        ("policy", ("policy",)), ("action horizon", ("minimum_action_horizon",)),
        ("compatibility issues", ("compatibility_issues",)),
        ("scientific hash", ("scientific_config_hash",)),
        ("execution hash", ("execution_config_hash",)),
    ),
    "deploy run": (
        ("status", ("status",)), ("deployment", ("deployment",)),
        ("experiment", ("experiment",)), ("profile", ("profile",)),
        ("renderer", ("renderer",)), ("run", ("run_path",)),
        ("cleanup", ("cleanup",)), ("side effects", ("side_effects_performed",)),
    ),
    "run": (
        ("status", ("status",)), ("run", ("run_id",)), ("path", ("run_path",)),
        ("experiment", ("experiment_id",)), ("policy", ("policy",)),
        ("benchmark", ("benchmark",)), ("compatible", ("compatible",)),
        ("output", ("output_root",)), ("side effects", ("side_effects_performed",)),
        ("scientific hash", ("scientific_config_hash",)),
        ("execution hash", ("execution_config_hash",)),
    ),
    "run inspect": (
        ("run", ("run_id",)), ("status", ("status",)), ("policy", ("policy",)),
        ("benchmark", ("benchmark",)), ("tasks", ("task_count",)),
        ("rollouts", ("rollout_count",)), ("failure", ("failure_type",)),
    ),
    "run verify": (
        ("run", ("run_id",)), ("status", ("status",)), ("integrity", ("integrity",)),
        ("episodes", ("verified_episode_count",)), ("files", ("verified_file_count",)),
        ("failure", ("failure_type",)),
    ),
    "metrics recompute": (
        ("results agree", ("all_results_agree",)), ("episodes", ("episode_count",)),
        ("trace modified", ("original_trace_modified",)), ("metric API", ("metric_api",)),
    ),
    "report generate": (
        ("run", ("run_id", "source_run")), ("profile", ("profile_id",)),
        ("build", ("derived_build_id",)), ("status", ("status",)),
        ("integrity", ("integrity",)), ("output", ("output",)), ("reused", ("reused",)),
    ),
    "report publish": (
        ("status", ("status",)), ("run", ("run_id",)),
        ("report", ("report.output",)), ("export", ("isolated_export.output",)),
        ("canonical run modified", ("canonical_run_modified",)),
    ),
    "report verify": (
        ("run", ("run_id",)), ("profile", ("profile_id",)),
        ("build", ("derived_build_id",)), ("status", ("status",)),
        ("integrity", ("integrity",)), ("files", ("verified_file_count",)),
        ("output", ("output",)),
    ),
    "export isolated": (
        ("type", ("export_type",)), ("name", ("name",)),
        ("integrity", ("integrity",)), ("files", ("verified_file_count",)),
        ("runs", ("source_run_ids",)), ("output", ("output",)),
    ),
    "export grouped": (
        ("type", ("export_type",)), ("name", ("name",)),
        ("integrity", ("integrity",)), ("files", ("verified_file_count",)),
        ("runs", ("source_run_ids",)), ("output", ("output",)),
    ),
    "export generate": (
        ("type", ("export_type",)), ("name", ("name",)),
        ("integrity", ("integrity",)), ("files", ("verified_file_count",)),
        ("runs", ("source_run_ids",)), ("output", ("output",)),
    ),
    "export verify": (
        ("type", ("export_type",)), ("name", ("name",)),
        ("integrity", ("integrity",)), ("files", ("verified_file_count",)),
        ("runs", ("source_run_ids",)), ("output", ("output",)),
    ),
    "data archive": (
        ("status", ("status",)), ("targets", ("target_count",)),
        ("archive", ("archive_root",)), ("dry run", ("dry_run",)),
    ),
    "data delete": (
        ("status", ("status",)), ("targets", ("target_count",)),
        ("dry run", ("dry_run",)),
    ),
    "dataset resolve": (
        ("provider", ("provider",)), ("dataset", ("logical_name",)),
        ("revision", ("source_revision",)), ("resolution", ("resolution_id",)),
        ("format", ("preparation_format",)), ("state", ("state",)),
    ),
    "dataset fetch": (
        ("dataset", ("logical_name",)), ("version", ("dataset_version",)),
        ("id", ("dataset_id",)), ("state", ("state",)),
        ("path", ("host_path",)), ("reused", ("reused",)),
        ("samples", ("sample_count",)),
    ),
    "dataset import": (
        ("dataset", ("logical_name",)), ("version", ("dataset_version",)),
        ("id", ("dataset_id",)), ("state", ("state",)),
        ("path", ("host_path",)), ("reused", ("reused",)),
    ),
    "dataset prepare": (
        ("dataset", ("dataset_id",)), ("status", ("status",)),
        ("format", ("preparation_format",)), ("files", ("verified_file_count",)),
        ("path", ("host_path",)), ("reused", ("reused",)),
    ),
    "dataset inspect": (
        ("dataset", ("logical_name",)), ("version", ("dataset_version",)),
        ("id", ("dataset_id",)), ("provider", ("provider",)),
        ("state", ("state",)), ("format", ("preparation.format",)),
        ("path", ("host_path",)), ("revision", ("source_revision",)),
    ),
    "dataset verify": (
        ("dataset", ("dataset_id",)), ("status", ("status",)),
        ("files", ("verified_file_count",)), ("path", ("host_path",)),
    ),
    "train validate": (
        ("valid", ("valid",)), ("profile", ("profile_id",)),
        ("profile identity", ("normalized_profile_id",)), ("source", ("source",)),
        ("model initialized", ("model_initialized",)), ("network used", ("network_used",)),
    ),
    "train plan": (
        ("profile", ("profile_id",)), ("compatible", ("capabilities.compatible",)),
        ("mode", ("capabilities.training_mode",)),
        ("scientific plan", ("scientific_training_id",)),
        ("execution plan", ("execution_plan_id",)),
        ("estimated VRAM GiB", ("execution.estimated_vram_gib",)),
        ("network", ("execution.network",)),
    ),
    "train run": (
        ("status", ("status", "result.status")), ("run", ("run_id",)),
        ("checkpoint", ("checkpoint_id", "result.checkpoint_id", "checkpoint.checkpoint_id")),
        ("path", ("checkpoint_path", "host_path")), ("reused", ("reused",)),
    ),
    "train status": (
        ("run", ("run_id",)), ("status", ("status",)),
        ("checkpoint", ("checkpoint_id",)), ("failure", ("failure",)),
    ),
    "train inspect": (
        ("run", ("run_id",)), ("status", ("result.status",)),
        ("checkpoint", ("result.checkpoint_id",)), ("path", ("host_path",)),
        ("profile", ("plan.profile_id",)),
    ),
    "train verify": (
        ("run", ("run_id",)), ("status", ("status",)),
        ("files", ("verified_file_count",)),
    ),
    "train report": (
        ("training run", ("training_run_id",)), ("build", ("derived_build_id",)),
        ("integrity", ("integrity",)), ("output", ("output",)), ("reused", ("reused",)),
    ),
    "checkpoint inspect": (
        ("checkpoint", ("checkpoint.checkpoint_id", "manifest.checkpoint_id")),
        ("kind", ("checkpoint.kind", "manifest.kind")),
        ("state", ("manifest.state",)), ("merge status", ("checkpoint.merge_status", "manifest.merge_status")),
        ("base checkpoint", ("checkpoint.base_checkpoint.resource_id", "manifest.base_checkpoint.resource_id")),
        ("path", ("host_path",)),
    ),
    "checkpoint verify": (
        ("checkpoint", ("checkpoint_id",)), ("status", ("status",)),
        ("files", ("verified_file_count",)), ("path", ("host_path",)),
    ),
}


def _compact(command: str, result) -> str:
    if command == "policy list":
        return _columns(
            result, (("id",), ("family",), ("config_type",), ("quic_profile",)),
            empty="No policy variants found.",
        )
    if command == "data list":
        rows = result.get("items", ()) if isinstance(result, dict) else ()
        lines = []
        for row in rows:
            if isinstance(row, dict) and row.get("kind") == "run":
                lines.append(
                    f"run {_scalar(row.get('id'))} ({_scalar(row.get('run_hash'))}) "
                    f"{_scalar(row.get('state'))} {_scalar(row.get('path'))}"
                )
            elif isinstance(row, dict):
                lines.append(
                    " ".join(_scalar(row.get(key, "-")) for key in ("kind", "id", "state", "path"))
                )
            else:
                lines.append(_scalar(row))
        return "\n".join(lines) if lines else "No runs, reports, or exports found."
    list_contracts = {
        "report profiles": (
            "profiles", (("id",), ("source",), ("template",)), "No report profiles found.",
        ),
        "dataset providers": (
            "providers", (("id",), ("version",)), "No dataset providers found.",
        ),
        "dataset list": (
            "datasets", (("logical_name",), ("dataset_version",), ("path",)), "No datasets found.",
        ),
        "train profiles": (
            "profiles", (("id",), ("valid",), ("path",)), "No training profiles found.",
        ),
        "checkpoint list": (
            "checkpoints", (("checkpoint_id",), ("kind",), ("merge_status",), ("host_path",)),
            "No checkpoints found.",
        ),
    }
    if command in list_contracts:
        key, fields, empty = list_contracts[command]
        rows = result.get(key, ()) if isinstance(result, dict) else ()
        return _columns(rows, fields, empty=empty)
    fields = _SUMMARY_FIELDS.get(command)
    if fields is None or not isinstance(result, dict):
        return json.dumps(result, indent=2, sort_keys=True)
    rendered = []
    for label, paths in fields:
        value = _first(result, paths)
        if value is _MISSING:
            continue
        if label == "compatibility issues" and isinstance(value, list):
            value = len(value)
        rendered.append((label, _scalar(value)))
    if not rendered:
        return json.dumps(result, indent=2, sort_keys=True)
    width = max(len(label) for label, _ in rendered)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rendered)


def _error_context(exc: BaseException) -> dict[str, object]:
    context = {}
    for name in (
        "variant", "expected_package", "expected_source", "implementation_status",
        "next_implementation_gate",
    ):
        if hasattr(exc, name):
            context[name] = getattr(exc, name)
    return context


def _classify(exc: BaseException) -> tuple[int, str]:
    name = type(exc).__name__
    module = type(exc).__module__
    if name == "KeyboardInterrupt":
        return ExitCode.INTERRUPTED, "interrupted"
    if name == "CliUsageError":
        return ExitCode.USAGE, "usage_error"
    if name in {"MetricRecomputationError"}:
        return ExitCode.METRICS, "metric_recomputation_error"
    if name == "ReportingSourceUnavailableError":
        return ExitCode.POLICY_UNAVAILABLE, "source_unavailable"
    if name == "DataSourceUnavailableError":
        return ExitCode.POLICY_UNAVAILABLE, "data_unavailable"
    if name == "RunReferenceUnavailableError":
        return ExitCode.POLICY_UNAVAILABLE, "source_unavailable"
    if name == "RunReferenceAmbiguousError":
        return ExitCode.USAGE, "ambiguous_run_reference"
    if name == "RunReferenceError":
        return ExitCode.INTEGRITY, "run_reference_error"
    if name == "DataSafetyError":
        return ExitCode.INTEGRITY, "data_safety_error"
    if name == "DataManagementError":
        return ExitCode.RUNTIME, "data_management_error"
    if name == "ReportingRendererError":
        return ExitCode.RUNTIME, "report_renderer_error"
    if name in {"DatasetIntegrityError", "CheckpointBundleError"}:
        return ExitCode.INTEGRITY, "artifact_integrity_error"
    if name in {"DatasetRequestError", "TrainingProfileError"}:
        return ExitCode.CONFIGURATION, "configuration_error"
    if name in {"DatasetUnavailableError"}:
        return ExitCode.POLICY_UNAVAILABLE, "dataset_unavailable"
    if name in {"DatasetInterruptedError", "TrainingInterruptedError"}:
        return ExitCode.INTERRUPTED, "interrupted"
    if name.startswith("Training"):
        return ExitCode.RUNTIME, "training_error"
    if name in {"RunIntegrityError", "ArtifactError"}:
        return ExitCode.INTEGRITY, "run_integrity_error"
    if name.startswith("QuIC") and ("Incomplete" in name or "Unavailable" in name or "Provider" in name):
        return ExitCode.POLICY_UNAVAILABLE, "policy_unavailable"
    if name.startswith("RemotePolicy") or name in {"ContractCompatibilityError", "ConnectionError"}:
        return ExitCode.SERVICE, "service_or_capability_error"
    if name.startswith("Config") or name in {"StrictYamlError", "ResolvedConfigWriteError"}:
        return ExitCode.CONFIGURATION, "configuration_error"
    if "benchmark" in module or name in {"ExperimentExecutionError", "RunnerLifecycleError"}:
        return ExitCode.RUNTIME, "runtime_error"
    return ExitCode.RUNTIME, "runtime_error"


def _install_interrupt_handlers():
    previous = {}
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt()
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    return previous


def _restore_handlers(previous):
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _confirm_data_operation(args, preview: dict[str, object]) -> bool:
    if args.dry_run or args.yes:
        return True
    if args.json:
        raise CliUsageError("destructive JSON-mode data operations require --yes or --dry-run")
    if not sys.stdin.isatty():
        raise CliUsageError("data operation requires an interactive terminal, --yes, or --dry-run")
    action = str(preview["action"])
    phrase = f"{action.upper()} ALL" if args.all_data else action.upper()
    sys.stderr.write(
        f"ovlab: {action} will affect {preview['target_count']} item(s). "
        f"Type {phrase!r} to continue: "
    )
    sys.stderr.flush()
    return sys.stdin.readline().strip() == phrase


def _dispatch(args):
    if args.command == "config":
        app = _application()
        if args.config_command == "validate":
            return app.validate(args.config, mode=args.mode), args.json, None
        document = app.resolved_document(args.config, mode=args.mode)
        if args.format == "json":
            return json.dumps(document, sort_keys=True, separators=(",", ":")), False, "raw"
        from .strict_yaml import dumps
        return dumps(document), False, "raw"
    if args.command == "policy":
        if args.policy_command == "list":
            from .catalog import registered_policies
            result = registered_policies()
        else:
            result = _application().policy_describe(args.config)
        return result, args.json, None
    if args.command == "service":
        if args.service_command == "health":
            return _application().service_health(args.socket), args.json, None
        previous = _install_interrupt_handlers()
        try:
            result = _application().serve(args.config, socket_path=args.socket)
        finally:
            _restore_handlers(previous)
        return result, args.json, None
    if args.command == "connect":
        return _application().connect(args.config), args.json, None
    if args.command == "deploy":
        from .deployment import ComposeDeployment
        deployment = ComposeDeployment(_repository_root())
        plan = deployment.plan(
            args.experiment,
            profile=args.profile,
            renderer=args.renderer,
            env_file=args.env_file,
            local_profile=args.local_profile,
            offline=args.offline,
            project_name=args.project_name,
        )
        previous = _install_interrupt_handlers()
        try:
            result = deployment.run(plan, dry_run=args.dry_run)
        finally:
            _restore_handlers(previous)
        return result, args.json, None
    if args.command == "run":
        app = _application()
        if args.target in {"inspect", "verify"}:
            if args.path is None:
                raise CliUsageError(f"run {args.target} requires RUN_PATH, RUN_ID, or RUN_HASH")
            result = app.inspect(args.path) if args.target == "inspect" else app.verify(args.path)
        else:
            if args.path is not None:
                raise CliUsageError("run CONFIG accepts no second positional argument")
            if args.dry_run:
                result = app.execution_plan(args.target, output_root=args.output_root)
            else:
                previous = _install_interrupt_handlers()
                try:
                    result = app.run(args.target, output_root=args.output_root)
                finally:
                    _restore_handlers(previous)
        return result, args.json, None
    if args.command == "metrics":
        return _application().recompute_metrics(args.run_path), args.json, None
    if args.command == "report":
        app = _application()
        if args.report_command == "profiles":
            return app.report_profiles(), args.json, None
        if args.report_command == "publish":
            return app.report_publish(
                args.run_id,
                args.profile,
                report_enabled=args.report_enabled == "true",
            ), args.json, None
        if args.report_command == "verify":
            return app.report_verify(args.run_id, args.profile, build_id=args.build), args.json, None
        if args.run_id:
            if args.run_path is not None or args.output is not None:
                raise CliUsageError("report generate --run cannot be combined with legacy RUN_PATH/--output")
            return app.report_generate(args.run_id, args.profile, task_id=args.task), args.json, None
        if args.run_path is None or args.output is None:
            raise CliUsageError("report generate requires --run RUN_ID or legacy RUN_PATH --output PATH")
        if args.task is not None or args.profile != "libero-task-default":
            raise CliUsageError("--task/--profile require the --run form")
        return app.generate_report(args.run_path, args.output), args.json, None
    if args.command == "export":
        app = _application()
        if args.export_command == "isolated":
            return app.export_isolated(args.run_id, episode_id=args.episode_id, template=args.template), args.json, None
        if args.export_command == "grouped":
            return app.export_grouped(
                args.name, all_runs=args.all_runs, run_ids=args.runs or (),
                same_model_as=args.same_model_as, suite=args.suite, template=args.template,
            ), args.json, None
        if args.export_command == "generate":
            return app.export_generate(args.spec), args.json, None
        name = args.export_name or args.legacy_export_id
        if name is None:
            raise CliUsageError("export verify requires --name NAME")
        if args.export_name is not None and args.legacy_export_id is not None:
            raise CliUsageError("export verify accepts either --name or legacy --export, not both")
        kind = "grouped" if args.legacy_export_id is not None else args.kind
        return app.export_verify(kind, name), args.json, None
    if args.command == "data":
        app = _application()
        if args.data_command == "list":
            return app.data_list(
                kind=args.kind, archived=args.archived, detail=args.detail,
            ), args.json, None
        selector = {
            "run_id": args.run_id, "report_id": args.report_id,
            "export_id": args.export_id,
            "all_data": args.all_data,
        }
        force = args.force if args.data_command == "delete" else False
        preview = app.data_preview(args.data_command, force=force, **selector)
        if args.dry_run:
            return preview, args.json, None
        if not _confirm_data_operation(args, preview):
            return {**preview, "status": "cancelled", "dry_run": False}, False, None
        result = (
            app.data_archive(**selector)
            if args.data_command == "archive" else app.data_delete(force=force, **selector)
        )
        return result, args.json, None
    if args.command == "dataset":
        app = _application()
        if args.dataset_command == "providers":
            return app.dataset_providers(), args.json, None
        if args.dataset_command == "resolve":
            return app.dataset_resolve(source=args.benchmark, name=args.suite), args.json, None
        if args.dataset_command == "fetch":
            return app.dataset_fetch(
                source=args.source, name=args.name, version=args.version,
                url=args.url, sha256=args.sha256, archive=args.archive,
                preparation=args.preparation, allow_dataset_download=True,
                allow_local_http=args.allow_local_http,
            ), args.json, None
        if args.dataset_command == "import":
            return app.dataset_import(
                name=args.name, version=args.version, path=args.path,
                preparation=args.preparation,
            ), args.json, None
        if args.dataset_command == "prepare":
            return app.dataset_prepare(args.dataset_id, args.preparation), args.json, None
        if args.dataset_command == "list":
            return app.dataset_list(), args.json, None
        if args.dataset_command == "inspect":
            return app.dataset_inspect(args.dataset_id), args.json, None
        return app.dataset_verify(args.dataset_id), args.json, None
    if args.command == "train":
        app = _application()
        if args.train_command == "profiles":
            return app.train_profiles(), args.json, None
        if args.train_command == "validate":
            return app.train_validate(args.profile), args.json, None
        if args.train_command == "plan":
            return app.train_plan(args.profile), args.json, None
        if args.train_command == "run":
            from .training_deployment import TrainingDeployment
            return TrainingDeployment(_repository_root()).run(
                args.profile, allow_dataset_download=args.allow_dataset_download,
            ), args.json, None
        if args.train_command == "status":
            return app.train_status(args.run_id), args.json, None
        if args.train_command == "inspect":
            return app.train_inspect(args.run_id), args.json, None
        if args.train_command == "report":
            if args.build is not None and not args.verify:
                raise CliUsageError("train report --build requires --verify")
            return app.train_report(
                args.run_id, verify=args.verify, build_id=args.build,
            ), args.json, None
        return app.train_verify(args.run_id), args.json, None
    if args.command == "checkpoint":
        app = _application()
        if args.checkpoint_command == "list":
            return app.checkpoint_list(), args.json, None
        if args.checkpoint_command == "inspect":
            return app.checkpoint_inspect(args.checkpoint_id), args.json, None
        return app.checkpoint_verify(args.checkpoint_id), args.json, None
    raise AssertionError("unhandled CLI command")


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.show_version:
        value = repository_revision(_repository_root())
        _human(f"ovlab {CLI_VERSION} (revision {value or 'unavailable'})")
        return ExitCode.SUCCESS
    if args.command is None:
        parser.print_help()
        return ExitCode.SUCCESS
    command = _command_name(args)
    wants_json = bool(getattr(args, "json", False))
    try:
        with _configured_artifact_umask():
            result, json_mode, render = _dispatch(args)
        if json_mode:
            _json_output(command, "success", result, ())
        elif render == "raw" or bool(getattr(args, "detail", False)):
            _human(result)
        else:
            _human(_compact(command, result))
        return ExitCode.SUCCESS
    except KeyboardInterrupt as exc:
        code, category = _classify(exc)
        error = {"code": category, "type": type(exc).__name__, "message": "operation interrupted", "context": {}}
        if wants_json:
            _json_output(command, "error", None, (error,))
        else:
            sys.stderr.write("ovlab: operation interrupted\n")
        return code
    except Exception as exc:
        code, category = _classify(exc)
        error = {
            "code": category,
            "type": type(exc).__name__,
            "message": str(exc),
            "context": _error_context(exc),
        }
        if wants_json:
            _json_output(command, "error", None, (error,))
        else:
            sys.stderr.write(f"ovlab: {category}: {exc}\n")
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
