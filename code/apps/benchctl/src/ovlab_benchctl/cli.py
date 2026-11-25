"""Unified dependency-light OVLAB command-line interface."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import signal
import shutil
import textwrap
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


def _primary_argument(parser, flag, dest, metavar, *, help=None, choices=None):
    """Accept the primary operand positionally or through its established flag."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(f"_pos_{dest}", nargs="?", metavar=metavar, choices=choices, help=help)
    group.add_argument(flag, dest=dest, metavar=metavar, choices=choices, help=f"alias for {metavar}")


class _RootParser(argparse.ArgumentParser):
    """Normalize the shorthand report entry point."""

    def parse_args(self, args=None, namespace=None):
        arguments = list(sys.argv[1:] if args is None else args)
        if (len(arguments) > 1 and arguments[0] == "report"
                and not arguments[1].startswith("-")
                and arguments[1] not in {"auto", "generate", "publish", "verify", "profiles", "export", "recompute"}):
            arguments.insert(1, "auto")
        return super().parse_args(arguments, namespace)


def _parser() -> argparse.ArgumentParser:
    parser = _RootParser(prog="ovlab", description="OpenVLABenchmark orchestration CLI")
    parser.add_argument(
        "-v", "--version", dest="show_version", action="store_true",
        help="show OVLAB CLI and repository revision",
    )
    parser.add_argument("--debug", action="store_true", help="include a traceback for unexpected failures")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    editor = commands.add_parser(
        "editor", help="open the visual workflow editor in a browser",
        description="Open the local Workflow Studio in the default browser. No server or Docker image is needed.",
    )
    _output_options(editor)

    doctor = commands.add_parser("doctor", help="diagnose host deployment readiness without changing files")
    doctor_experiment = doctor.add_mutually_exclusive_group()
    doctor_experiment.add_argument("_pos_experiment", nargs="?", metavar="EXPERIMENT", help="also check one deployable experiment and its images")
    doctor_experiment.add_argument("--experiment", help="alias for EXPERIMENT")
    doctor.add_argument("--env-file", default="deploy/compose/.env", help="Compose environment file")
    _output_options(doctor)

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

    plugin = commands.add_parser("plugin", help="register, inspect, check, sync and build plugin projects")
    plugin_commands = plugin.add_subparsers(dest="plugin_command", required=True)
    register = plugin_commands.add_parser("register", help="register a standalone local or pinned Git plugin project")
    register.add_argument("source", help="local project directory or Git repository URL")
    register.add_argument("--commit", help="full immutable Git commit for a remote project")
    register.add_argument("--registry", help="destination user registry (default configs/plugins/registry.json)")
    register.add_argument("--dry-run", action="store_true")
    _output_options(register)
    for name, help_text in (("list", "list registered projects without model imports"), ("inspect", "inspect project variants, capabilities and sources"), ("check", "check project templates and SDK control API without loading a model"), ("sync", "materialize development sources into external/"), ("build", "build project images from registered recipes")):
        command = plugin_commands.add_parser(name, help=help_text)
        if name != "list": command.add_argument("plugin", nargs="?" if name in {"sync", "build"} else None, help="plugin project ID" + ("; omit to select all" if name in {"sync", "build"} else ""))
        command.add_argument("--registry", help="registry JSON path (also OVLAB_PLUGIN_REGISTRY)")
        if name in {"check", "sync", "build"}:
            command.add_argument("--offline", action="store_true")
        if name in {"sync", "build"}:
            command.add_argument("--dry-run", action="store_true")
        if name == "build":
            command.add_argument("--config", help="experiment/deployment configuration to resolve and include in the image")
            command.add_argument("--execution-profile", help="execution profile included with --config")
            sources = command.add_mutually_exclusive_group()
            sources.add_argument("--isolated-sources", dest="isolated_sources", action="store_true", default=True,
                                 help="fetch/copy registered sources into a temporary context (default)")
            sources.add_argument("--workspace-sources", dest="isolated_sources", action="store_false",
                                 help="sync and build from the development external/ workspace")
        _output_options(command)

    module = commands.add_parser("module", help="register, inspect and check functionality modules")
    module_commands = module.add_subparsers(dest="module_command", required=True)
    for operation in ("list", "inspect", "check", "register"):
        command = module_commands.add_parser(operation)
        if operation == "register":
            command.add_argument("source")
            command.add_argument("--commit")
            command.add_argument("--dry-run", action="store_true")
        elif operation != "list":
            command.add_argument("module")
        command.add_argument("--registry")
        _output_options(command)

    experiment = commands.add_parser("experiment", help="plan and run hierarchical YAML or JSON experiments")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    for operation in ("plan", "run"):
        command = experiment_commands.add_parser(operation, help=f"{operation} an experiment tree")
        _primary_argument(command, "--spec", "spec", "SPEC", help="ovlab.experiment/v1 YAML or JSON file")
        command.add_argument("--node", help="slash-separated node path; omit for the root")
        command.add_argument("--node-only", action="store_true", help="run only the selected node, without descendants")
        execution = command.add_mutually_exclusive_group()
        execution.add_argument("--sequential", action="store_true", help="run one phase at a time")
        execution.add_argument("--max-workers", type=int, metavar="N", help="maximum concurrent phases (default: 2; 1–32)")
        if operation == "run":
            command.add_argument("--dry-run", action="store_true", help="resolve and validate without writing or starting workloads")
        _output_options(command)

    evaluation = commands.add_parser("eval", help="run, inspect, and verify evaluation experiments")
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_commands.add_parser("run", help="run one experiment through Docker Compose")
    eval_run.add_argument("target", metavar="TARGET", help="evaluation configuration path or plugin:// reference")
    eval_run.add_argument(
        "--renderer", help="select an execution alias declared by the benchmark plugin",
    )
    eval_run.add_argument("--execution-profile", help="explicit plugin or local benchmark execution configuration")
    eval_run.add_argument(
        "--quantization", help="override with a quantization profile supported by the selected plugin",
    )
    eval_run.add_argument(
        "--offline",
        action="store_true",
        help="reject a missing checkpoint instead of downloading a pinned, remote/main or latest revision",
    )
    eval_run.add_argument("--project-name")
    eval_run.add_argument("--plan", "--dry-run", dest="dry_run", action="store_true", help="resolve and show the evaluation plan without starting workloads")
    _output_options(eval_run)

    for operation in ("inspect", "verify"):
        command = eval_commands.add_parser(operation, help=f"{operation} a canonical evaluation run")
        command.add_argument("target", help="evaluation run path, ID, or hash")
        _output_options(command)

    deploy = commands.add_parser("deploy", help="run, connect to and verify model services")
    deploy_commands = deploy.add_subparsers(dest="deploy_command", required=True)
    serve = deploy_commands.add_parser("serve", help="serve a configured policy in the foreground")
    serve.add_argument("config", help="deployment/evaluation configuration path or plugin:// reference")
    transport = serve.add_mutually_exclusive_group()
    transport.add_argument("--socket")
    transport.add_argument("--port", type=int)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--persistent", action="store_true", help="keep the model ready between client sessions")
    _output_options(serve)
    connect = deploy_commands.add_parser("connect", help="check a running policy against an experiment without inference")
    connect.add_argument("target", nargs="?", help="managed deployment ID or configuration; omit for a direct endpoint")
    connect.add_argument("--experiment", required=True, help="evaluation configuration defining the expected interface")
    transport = connect.add_mutually_exclusive_group()
    transport.add_argument("--socket", help="direct Unix socket; defaults to the experiment socket")
    transport.add_argument("--port", type=int, help="direct TCP endpoint")
    connect.add_argument("--host", help="TCP host (default: 127.0.0.1)")
    _output_options(connect)
    deploy_run = deploy_commands.add_parser("run", help="build a policy image and start a model service")
    deploy_run.add_argument("target", help="deployment configuration path or plugin:// reference")
    deploy_run.add_argument("--profile", help="local checkpoint profile (overrides OVLAB_LOCAL_PROFILE)")
    deploy_run.add_argument("--quantization", help="quantization profile supported by the selected plugin")
    deploy_run.add_argument("--port", type=int, help="override the published model RPC port")
    deploy_run.add_argument("--offline", action="store_true", help="resolve checkpoints without downloading")
    deploy_run.add_argument("--dry-run", action="store_true", help="show the plan without building or starting containers")
    _output_options(deploy_run)
    for operation in ("inspect", "verify", "stop"):
        command = deploy_commands.add_parser(operation, help=f"{operation} a deployed model service")
        command.add_argument("target", help="deployment ID or YAML name/path")
        if operation == "stop":
            command.add_argument("--dry-run", action="store_true")
        _output_options(command)

    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    health = worker_commands.add_parser("health", help="internal policy protocol readiness probe")
    transport = health.add_mutually_exclusive_group(required=True)
    transport.add_argument("--socket")
    transport.add_argument("--port", type=int)
    health.add_argument("--host", default="127.0.0.1")
    _output_options(health)

    run = commands.add_parser("_native-run", help=argparse.SUPPRESS)
    run.add_argument("target", nargs="?", help="evaluation YAML in experiments/evaluation, or 'inspect'/'verify'")
    run.add_argument("path", nargs="?", help="RUN_PATH, RUN_ID, or RUN_HASH for inspect or verify")
    run.add_argument(
        "--profile", help="local machine YAML (overrides OVLAB_LOCAL_PROFILE and configs/local/profile.yaml)",
    )
    run.add_argument("--output-root")
    run.add_argument("--dry-run", action="store_true")
    _output_options(run)

    report = commands.add_parser(
        "report", help="create and verify reports and exports",
        usage="%(prog)s RUN_OR_EXPERIMENT_ID [--detail | --json]\n       %(prog)s {publish,profiles,verify,recompute} ...",
        epilog="Export a run or experiment automatically: ovlab report RUN_OR_EXPERIMENT_ID",
    )
    report_commands = report.add_subparsers(dest="report_command", required=True, metavar="{publish,profiles,verify,recompute}")
    recompute = report_commands.add_parser("recompute", help="audit recorded episode metrics by recomputing from immutable traces")
    recompute.add_argument("run_path", metavar="RUN", help="evaluation run ID, path or hash")
    _output_options(recompute)
    auto = report_commands.add_parser("auto", help="export a run or all runs in an experiment")
    auto.add_argument("reference", help="run ID, run hash, or completed experiment ID")
    _output_options(auto)
    publish = report_commands.add_parser(
        "publish", help="publish the HTML report and isolated export for one finalized run",
    )
    _primary_argument(publish, "--run", "run_id", "RUN", help="canonical run ID, path or hash")
    publish.add_argument("--profile", default="libero-task-default")
    publish.add_argument("--task", help="optional canonical task ID for the report")
    publish.add_argument("--output", help="legacy machine-report output path")
    publish.add_argument(
        "--report-enabled", choices=("true", "false"), default="true",
        help="generate derived HTML in addition to the always-generated isolated export",
    )
    _output_options(publish)
    verify_report = report_commands.add_parser("verify", help="verify a derived report and its canonical inputs")
    _primary_argument(verify_report, "--run", "run_id", "RUN", help="canonical run ID, path or hash")
    verify_report.add_argument("--profile", default="libero-task-default")
    verify_report.add_argument("--build")
    _output_options(verify_report)
    profiles = report_commands.add_parser("profiles", help="list built-in report profiles")
    _output_options(profiles)

    export = commands.add_parser("export", help="generate readable isolated or grouped exports from canonical runs")
    export_commands = export.add_subparsers(dest="export_command", required=True)
    isolated = export_commands.add_parser("isolated", help="export one complete run or one episode")
    _primary_argument(isolated, "--run", "run_id", "RUN", help="canonical run ID, path or hash")
    isolated.add_argument("--category", choices=("evaluate", "train"), default="evaluate")
    isolated.add_argument("--episode", dest="episode_id", help="optional canonical episode ID")
    isolated.add_argument("--template", default="isolated-default-v1")
    _output_options(isolated)
    grouped = export_commands.add_parser("grouped", help="compare an explicitly selected group of runs")
    grouped.add_argument("group_targets", nargs="*", metavar="TARGET", help="group name followed by run IDs")
    grouped.add_argument("--name", help="group name; otherwise the first positional target")
    grouped.add_argument("--category", choices=("evaluate", "train"), default="evaluate")
    selectors = grouped.add_mutually_exclusive_group()
    selectors.add_argument("--all-runs", action="store_true", help="select every compatible completed run")
    selectors.add_argument("--same-model-as", metavar="RUN_REFERENCE", help="select runs with the same model/checkpoint as RUN_ID or RUN_HASH")
    selectors.add_argument("--runs", nargs="+", metavar="RUN_REFERENCE", help="select these run IDs or displayed hashes manually")
    grouped.add_argument("--suite", help="optional benchmark-suite filter")
    grouped.add_argument("--template", default="grouped-default-v1")
    _output_options(grouped)
    export_generate = export_commands.add_parser("generate", help="legacy: generate a grouped export from ovlab.export-spec/v1")
    _primary_argument(export_generate, "--spec", "spec", "SPEC", help="export specification")
    _output_options(export_generate)
    export_verify = export_commands.add_parser("verify", help="verify an export build and source checksums")
    export_verify.add_argument("--kind", choices=("isolated", "grouped"), default="grouped")
    export_verify.add_argument("--category", choices=("evaluate", "train"), default="evaluate")
    export_selector = export_verify.add_mutually_exclusive_group(required=True)
    export_selector.add_argument("_pos_export_name", nargs="?", metavar="NAME", help="group name or isolated run ID")
    export_selector.add_argument("--name", dest="export_name", help="alias for NAME")
    export_selector.add_argument("--export", dest="legacy_export_id", help="legacy grouped export alias")
    _output_options(export_verify)
    # Compatibility entry points remain available to existing workflows, while
    # the public interface automatically selects exports via `report TARGET`.
    report_commands._choices_actions = [action for action in report_commands._choices_actions
                                        if action.dest in {"publish", "profiles", "verify", "recompute"}]

    data = commands.add_parser(
        "data", help="list, archive, and safely delete runs, reports, and exports"
    )
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_list = data_commands.add_parser(
        "list", help="list active or archived runs, reports, and exports"
    )
    data_kind = data_list.add_mutually_exclusive_group()
    data_kind.add_argument("_pos_kind", nargs="?", metavar="KIND", choices=("runs", "reports", "exports", "all"))
    data_kind.add_argument("--kind", choices=("runs", "reports", "exports", "all"), default="all")
    data_list.add_argument("--archived", action="store_true", help="list OVLAB_DATA_ROOT/archive instead of active data")
    _output_options(data_list)
    for action in ("archive", "delete"):
        command = data_commands.add_parser(action, help=f"{action} selected canonical data")
        command.add_argument("data_target", nargs="?", metavar="TARGET", help="typed selector: run:ID, report:ID or export:TYPE:ID")
        selectors = command.add_mutually_exclusive_group()
        selectors.add_argument(
            "--run", dest="run_id", metavar="RUN_ID_OR_HASH",
            help="run ID/hash; prefix train: or evaluate: when IDs collide",
        )
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
            help="select all runs, reports, and exports; incomplete entries require --force",
        )
        command.add_argument("--dry-run", action="store_true", help="show the exact selection without changing files")
        command.add_argument("--yes", action="store_true", help="confirm the operation non-interactively")
        command.add_argument(
            "--force", action="store_true",
            help=f"allow {action} of active or incomplete artifacts; path safety checks remain enforced",
        )
        _output_options(command)

    dataset = commands.add_parser("dataset", help="resolve, acquire, prepare, and verify immutable datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_providers = dataset_commands.add_parser("providers", help="list registered dataset bridges without model imports")
    _output_options(dataset_providers)
    dataset_resolve = dataset_commands.add_parser("resolve", help="resolve a known benchmark dataset without downloading")
    dataset_resolve.add_argument("--benchmark", required=True, help="registered dataset provider ID")
    _primary_argument(dataset_resolve, "--suite", "suite", "SUITE")
    _output_options(dataset_resolve)
    dataset_fetch = dataset_commands.add_parser("fetch", help="explicitly acquire and prepare a verified dataset")
    dataset_fetch.add_argument("--source", required=True, help="registered dataset provider ID")
    _primary_argument(dataset_fetch, "--name", "name", "NAME")
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
    _primary_argument(dataset_import, "--path", "path", "PATH")
    dataset_import.add_argument("--format", dest="preparation")
    _output_options(dataset_import)
    dataset_prepare = dataset_commands.add_parser("prepare", help="verify the selected immutable preparation")
    _primary_argument(dataset_prepare, "--dataset", "dataset_id", "DATASET")
    dataset_prepare.add_argument("--format", dest="preparation", required=True)
    _output_options(dataset_prepare)
    dataset_list = dataset_commands.add_parser("list", help="list locally ready immutable datasets")
    _output_options(dataset_list)
    dataset_inspect = dataset_commands.add_parser("inspect", help="inspect one immutable dataset manifest")
    _primary_argument(dataset_inspect, "--dataset", "dataset_id", "DATASET")
    _output_options(dataset_inspect)
    dataset_verify = dataset_commands.add_parser("verify", help="verify dataset bytes without network access")
    _primary_argument(dataset_verify, "--dataset", "dataset_id", "DATASET")
    _output_options(dataset_verify)

    train = commands.add_parser("train", help="validate, plan, execute, and inspect isolated training")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_profiles = train_commands.add_parser("profiles", help="list portable training profiles")
    _output_options(train_profiles)
    train_validate = train_commands.add_parser("validate", help="strictly validate a profile without resolving resources")
    _primary_argument(train_validate, "--profile", "profile", "PROFILE")
    _output_options(train_validate)
    train_plan = train_commands.add_parser("plan", help="resolve immutable resources without model initialization")
    _primary_argument(train_plan, "--profile", "profile", "PROFILE")
    _output_options(train_plan)
    train_run = train_commands.add_parser("run", help="run an isolated offline trainer and finalize its checkpoint")
    train_run.add_argument("target", metavar="TARGET", help="training YAML name/path in experiments/training")
    train_run.add_argument("--allow-dataset-download", action="store_true")
    train_run.add_argument("--dry-run", action="store_true", help="resolve the training plan without starting a trainer")
    for command in (train_validate, train_plan, train_run):
        command.add_argument("--fit-action-normalization", action="store_true", help="fit frozen action q01/q99 from training episodes only (requires dataset.partition)")
    _output_options(train_run)
    ablate = train_commands.add_parser("ablate", help="validate, resolve, or materialize a shared ablation suite")
    ablate_commands = ablate.add_subparsers(dest="ablate_command", required=True)
    for operation in ("validate", "plan", "materialize", "run"):
        command = ablate_commands.add_parser(operation, help=f"{operation} an ablation suite of ordinary training profiles")
        _primary_argument(command, "--suite", "suite", "SUITE")
        command.add_argument("--fit-action-normalization", action="store_true", help="fit one consistent train-only action normalization contract across all variants")
        if operation in {"materialize", "run"}:
            command.add_argument("--output", required=True, help="new directory for immutable resolved profiles and plan")
        _output_options(command)
    for command_name in ("status", "inspect", "verify"):
        command = train_commands.add_parser(command_name, help=f"{command_name} a canonical training run")
        command.add_argument("target", nargs="?", help="training run path, ID, or hash")
        command.add_argument("--run", dest="run_id", help="alias for positional TARGET")
        _output_options(command)
    train_report = train_commands.add_parser(
        "report", help="generate or verify an offline system-performance report",
    )
    _primary_argument(train_report, "--run", "run_id", "RUN", help="training run ID, path or hash")
    train_report.add_argument("--verify", action="store_true", help="verify the selected/latest build instead of generating")
    train_report.add_argument("--build", help="derived build ID used with --verify")
    train_report.add_argument("--parameters", help="JSON configuration of highlighted training parameters")
    _output_options(train_report)

    checkpoint = commands.add_parser("checkpoint", help="inspect local and registered checkpoints")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_list = checkpoint_commands.add_parser("list", help="list local training checkpoints or registered resources")
    checkpoint_list.add_argument("--source", choices=("local", "registry"), default="local", help="checkpoint catalogue (default: local)")
    network = checkpoint_list.add_mutually_exclusive_group()
    network.add_argument("--offline", action="store_true", help="inspect registry and local cache without network (default)")
    network.add_argument("--online", action="store_true", help="also check registered revisions on Hugging Face")
    checkpoint_list.add_argument("--profile", help="local profile containing checkpoint path overrides (registry only)")
    _output_options(checkpoint_list)
    checkpoint_export = checkpoint_commands.add_parser("export", help="export a verified checkpoint bundle")
    _primary_argument(checkpoint_export, "--checkpoint", "checkpoint_id", "CHECKPOINT")
    checkpoint_export.add_argument("--format", choices=("bundle", "tar", "onnx"), required=True,
                                   help="bundle directory, tar archive, or ONNX (not yet implemented)")
    checkpoint_export.add_argument("--output", required=True, help="new output directory or tar file")
    _output_options(checkpoint_export)
    checkpoint_inspect = checkpoint_commands.add_parser("inspect", help="inspect checkpoint identity and compatibility")
    _primary_argument(checkpoint_inspect, "--checkpoint", "checkpoint_id", "CHECKPOINT")
    _output_options(checkpoint_inspect)
    checkpoint_verify = checkpoint_commands.add_parser("verify", help="verify checkpoint files and tensor structure")
    _primary_argument(checkpoint_verify, "--checkpoint", "checkpoint_id", "CHECKPOINT")
    _output_options(checkpoint_verify)

    unified = commands.add_parser(
        "run", help="run any training, evaluation, or hierarchical experiment",
        description="Resolve an experiment configuration and execute its complete tree. "
                    "Model, runtime, and scheduling settings belong in the configuration.",
    )
    unified.add_argument("target", metavar="EXPERIMENT", help="YAML path, workflow JSON, or unique experiment name; run or experiment ID/hash with --verify")
    mode = unified.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="resolve and display the complete plan without starting workloads")
    mode.add_argument("--verify", action="store_true", help="validate a configuration, or verify the canonical artifacts of an existing run or experiment")
    _output_options(unified)
    # Phase-specific parsers remain private worker interfaces. Public execution
    # always uses the experiment owner, including one-phase configurations.
    commands._choices_actions = [action for action in commands._choices_actions
                                 if action.dest not in {"eval", "train", "experiment", "export", "_native-run", "_worker"}]
    return parser


def _repository_root() -> Path:
    return Path(os.environ.get("OVLAB_ROOT", Path(__file__).resolve().parents[5])).resolve()


def _application():
    from .application import OvlabApplication
    return OvlabApplication(_repository_root())


def _command_name(args) -> str:
    parts = [args.command]
    for name in (
        "config_command", "plugin_command", "module_command", "worker_command", "experiment_command", "eval_command", "deploy_command",
        "report_command", "export_command",
        "data_command", "dataset_command", "train_command", "ablate_command", "checkpoint_command",
    ):
        value = getattr(args, name, None)
        if value:
            parts.append(value)
    if args.command == "_native-run" and args.target in {"inspect", "verify"}:
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


def _table(rows, columns: tuple[tuple[str, tuple[str, ...]], ...], *, empty: str) -> str:
    """Render labeled, terminal-width-aware rows without dropping values."""
    rows = list(rows)
    if not rows:
        return empty
    headers = [label for label, _ in columns]
    values = []
    for row in rows:
        if not isinstance(row, dict):
            row = {columns[0][1][0]: row}
        cells = []
        for _, alternatives in columns:
            value = _first(row, alternatives)
            cells.append("-" if value is _MISSING or value is None or value == [] else _scalar(value))
        values.append(cells)

    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    widths = [max(len(headers[i]), *(max(map(len, cell.splitlines()), default=0)
                                    for cell in (row[i] for row in values)))
              for i in range(len(headers))]
    # Keep common identifiers and status words intact. If they cannot fit,
    # records are easier to scan than a table that splits every word.
    minimum = [max(len(label), min(40, max(
        (len(word) for row in values for word in row[i].split()), default=0)))
        for i, label in enumerate(headers)]
    if sum(minimum) + 2 * (len(headers) - 1) > terminal_width:
        blocks = []
        label_width = max(map(len, headers))
        cell_width = max(1, terminal_width - label_width - 2)
        for row in values:
            lines = []
            for label, cell in zip(headers, row):
                parts = []
                for part in cell.split("\n"):
                    parts.extend(textwrap.wrap(part, width=cell_width, break_long_words=True,
                                               break_on_hyphens=False) or [""])
                lines.append(f"{label:<{label_width}}  {parts[0]}")
                lines.extend(" " * (label_width + 2) + part for part in parts[1:])
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
    while sum(widths) + 2 * (len(headers) - 1) > terminal_width:
        candidates = [i for i in range(len(widths)) if widths[i] > minimum[i]]
        if not candidates:
            break
        index = max(candidates, key=lambda i: widths[i] - minimum[i])
        widths[index] -= 1

    def line(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [line(headers), line(["-" * width for width in widths])]
    for row in values:
        wrapped = []
        for cell, width in zip(row, widths):
            parts = []
            for part in cell.split("\n"):
                parts.extend(textwrap.wrap(part, width=width, break_long_words=True,
                                           break_on_hyphens=False) or [""])
            wrapped.append(parts)
        for index in range(max(map(len, wrapped))):
            lines.append(line([cell[index] if index < len(cell) else "" for cell in wrapped]))
    return "\n".join(lines)


def _training_check_rows(document: dict) -> list[tuple[str, str]]:
    """Explain run health without treating a short validation as rollout proof."""
    result = document.get("result")
    if not isinstance(result, dict):
        return []
    if result.get("status") != "completed" or not result.get("checkpoint_id"):
        return [("check", "[CHECK] training or checkpoint finalization is incomplete")]
    steps = result.get("optimizer_steps")
    summary = "[OK] checkpoint finalized"
    if isinstance(steps, int):
        summary += f"; {steps} optimizer steps"
    if "adapter_parameter_count" in result:
        changed = result.get("changed_adapter_tensor_count")
        if isinstance(changed, int) and changed > 0 and result.get("identity_logit_max_absolute_error") == 0 and result.get("reload_logit_max_absolute_error") == 0:
            summary += f"; {changed} adapter tensors updated; exact save/reload"
        else:
            summary = "[CHECK] adapter update or save/reload evidence is incomplete"
    rows = [("check", summary)]
    validation = result.get("adapter_validation")
    if isinstance(validation, dict):
        accuracy, tokens = validation.get("action_token_accuracy"), validation.get("action_tokens")
        if isinstance(accuracy, (int, float)) and isinstance(tokens, int) and tokens > 0:
            prefix = "[CHECK]" if accuracy == 0 else "[INFO]"
            detail = f"{prefix} {accuracy:.1%} on {tokens} action tokens"
            entropy = validation.get("action_token_cross_entropy")
            if isinstance(entropy, (int, float)):
                detail += f"; CE={entropy:.3f}"
            detail += "; teacher-forced, rollout not measured"
            rows.append(("validation", detail))
    return rows


_SUMMARY_FIELDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "editor": (("status", ("status",)), ("editor", ("editor",)), ("opener", ("opener",))),
    "config validate": (
        ("valid", ("valid",)), ("kind", ("kind",)), ("mode", ("mode",)),
        ("experiment", ("experiment_id",)),
        ("scientific hash", ("scientific_config_hash",)),
        ("execution hash", ("execution_config_hash",)),
    ),
    "plugin check": (("plugin", ("id",)), ("valid", ("valid",)), ("checks", ("checks",)), ("runtime tested", ("runtime_tested",))),
    "deploy serve": (("socket", ("socket",)), ("closed", ("closed",))),
    "_worker health": (
        ("state", ("state",)), ("ready", ("ready",)),
        ("protocol", ("protocol_version",)), ("predictions", ("prediction_count",)),
    ),
    "deploy connect": (
        ("compatible", ("compatible",)), ("protocol", ("protocol_version",)),
        ("policy", ("policy",)), ("action horizon", ("minimum_action_horizon",)),
        ("compatibility issues", ("compatibility_issues",)),
        ("scientific hash", ("scientific_config_hash",)),
        ("execution hash", ("execution_config_hash",)),
    ),
    "deploy run": (
        ("deployment", ("deployment_id",)), ("status", ("status",)),
        ("image", ("image",)), ("host", ("host",)), ("port", ("port",)),
        ("quantization", ("quantization",)), ("side effects", ("side_effects_performed",)),
    ),
    "deploy inspect": (("deployment", ("deployment_id",)), ("state", ("state",)), ("ports", ("ports",))),
    "deploy verify": (("deployment", ("deployment_id",)), ("verified", ("verified",))),
    "deploy stop": (("deployment", ("deployment_id",)), ("status", ("status",))),
    "eval inspect": (("run", ("run_id",)), ("status", ("status",))),
    "eval verify": (("integrity", ("integrity",)), ("run", ("run_id",))),
    "eval run": (
        ("status", ("status",)), ("deployment", ("deployment",)),
        ("experiment", ("experiment",)), ("profile", ("profile",)),
        ("renderer", ("renderer",)), ("quantization", ("quantization",)),
        ("run", ("run_path",)),
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
    "report recompute": (
        ("results agree", ("all_results_agree",)), ("episodes", ("episode_count",)),
        ("trace modified", ("original_trace_modified",)), ("metric API", ("metric_api",)),
    ),
    "report publish": (
        ("status", ("status",)), ("run", ("run_id",)),
        ("report", ("report.output",)), ("export", ("isolated_export.output",)),
        ("interactive", ("report.interactive_output",)),
        ("canonical run modified", ("canonical_run_modified",)),
    ),
    "report verify": (
        ("run", ("run_id",)), ("profile", ("profile_id",)),
        ("build", ("derived_build_id",)), ("status", ("status",)),
        ("integrity", ("integrity",)), ("files", ("verified_file_count",)),
        ("output", ("output",)),
        ("interactive", ("interactive_output",)), ("interactive integrity", ("interactive_integrity",)),
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
        ("forced", ("force",)),
    ),
    "data delete": (
        ("status", ("status",)), ("targets", ("target_count",)),
        ("dry run", ("dry_run",)), ("forced", ("force",)),
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
    "train ablate validate": (
        ("valid", ("valid",)), ("suite", ("suite_id",)),
        ("variants", ("variant_count",)), ("ablation", ("ablation_id",)),
        ("model initialized", ("model_initialized",)), ("network used", ("network_used",)),
    ),
    "train ablate plan": (
        ("suite", ("suite_id",)), ("variants", ("variant_count",)),
        ("ablation", ("ablation_id",)),
        ("partition contract", ("scientific.partition_contract_id",)),
    ),
    "train ablate materialize": (
        ("suite", ("suite_id",)), ("variants", ("variant_count",)),
        ("ablation", ("ablation_id",)), ("output", ("output",)),
    ),
    "train ablate run": (
        ("suite", ("suite_id",)), ("status", ("status",)),
        ("ablation", ("ablation_id",)), ("shared split", ("shared_manifest_id",)),
        ("output", ("output",)),
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
        ("interactive", ("interactive_output",)), ("interactive integrity", ("interactive_integrity",)),
    ),
    "checkpoint export": (
        ("checkpoint", ("checkpoint_id",)), ("format", ("format",)),
        ("kind", ("kind",)), ("integrity", ("integrity",)), ("output", ("output",)),
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
    if command in {"run", "experiment plan", "experiment run"} and isinstance(result, dict):
        lines = [f"experiment  {result.get('experiment_id', result.get('id', '-'))}"]
        if "status" in result:
            lines.append(f"status      {result['status']}")
        if "source_type" in result:
            lines.append(f"source      {result['source_type']}")
        if "verified_runs" in result:
            lines.append(f"verified    {len(result['verified_runs'])} runs")
        if "nodes" in result:
            lines.append(f"nodes       {len(result['nodes'])}")
        if "steps" in result:
            lines.append(f"phases      {len(result['steps'])}")
        if "selected_node" in result:
            lines.append(f"selection   {result['selected_node'] or '<root>'}")
        if isinstance(result.get("execution"), dict):
            execution = result["execution"]
            lines.append(f"execution   {execution.get('mode', '-')} (max {execution.get('max_workers', '-')} workers)")
        for key in ("overview", "runs_root", "exports_root"):
            if key in result:
                lines.append(f"{key:<11} {result[key]}")
        if isinstance(result.get("nodes"), list):
            rows = []
            for node in result["nodes"]:
                for phase in node.get("execute", []):
                    configuration = node.get("configuration", {}).get(phase, {})
                    rows.append({"node": node["path"] or "<root>", "phase": phase,
                                 "profile": configuration.get("profile", "-"),
                                 "checkpoint": ((configuration['checkpoint_from'] or '<root>') + ':train'
                                                if 'checkpoint_from' in configuration else '-')})
            if rows:
                columns = (("NODE", ("node",)), ("PHASE", ("phase",)), ("CONFIGURATION", ("profile",)))
                if any(row['checkpoint'] != '-' for row in rows):
                    columns += (("CHECKPOINT FROM", ("checkpoint",)),)
                lines.extend(("", _table(rows, columns, empty="No executable phases.")))
        return "\n".join(lines)
    if command == "report auto" and isinstance(result, dict):
        lines = [f"source  {result.get('source_type', '-')}  {result.get('source', '-')}"]
        for item in result.get("exports", []):
            lines.append("")
            lines.append(f"{item.get('category', '-')}  {item.get('type', '-')} export")
            if item.get("node") is not None:
                lines.append(f"  node       {item['node'] or '<root>'}")
            lines.append(f"  runs       {', '.join(item.get('runs', []))}")
            lines.append(f"  output     {item.get('output', '-')}")
            if item.get("integrity") is not None:
                lines.append(f"  integrity  {item['integrity']}")
            if item.get("reused") is not None:
                lines.append(f"  reused     {_scalar(item['reused'])}")
        return "\n".join(lines)
    if command == "doctor":
        counts = result["counts"]
        lines = [
            "OVLAB doctor: " + ", ".join(f"{counts[key]} {key}" for key in ("ok", "warning", "error", "skipped"))
        ]
        for check in result["checks"]:
            lines.append(f"[{check['status'].upper()}] {check['name']}: {check['message']}")
            if check["remedy"]:
                lines.append(f"  Fix: {check['remedy']}")
        return "\n".join(lines)
    if command in {"plugin list", "module list"}:
        return _table(result, (
            ("ID", ("id",)), ("KIND", ("kind",)), ("VERSION", ("version",)),
            ("SOURCE", ("source.kind",)), ("LOCATION", ("source.url", "source.path")),
            ("COMMIT", ("source.commit",)), ("VARIANTS", ("variants",)),
        ), empty="No plugin projects found.")
    if command in {"plugin inspect", "module inspect"}:
        lines = ["\n".join(f"{label}  {_scalar(result[label])}" for label in ("id", "kind", "version", "source", "capabilities"))]
        lines.append(_table(result["variants"], (
            ("VARIANT", ("id",)), ("CONFIG TYPE", ("config_type",)), ("RUNTIME", ("runtime_status",)),
            ("PROFILES", ("supported_profiles", "supported_quic_profiles", "deployment.profile")),
            ("QUANTIZATION", ("deployment.quantization",)),
        ), empty="No policy variants found."))
        return "\n\n".join(lines)
    if command == "data list":
        rows = result.get("items", ()) if isinstance(result, dict) else ()
        return _table(rows, (
            ("KIND", ("kind",)), ("ID", ("id",)), ("HASH", ("run_hash",)),
            ("STATE", ("state",)), ("TYPE", ("run_type",)), ("PATH", ("path",)),
        ), empty="No runs, reports, or exports found.")
    list_contracts = {
        "report profiles": ("profiles", (("ID", ("id",)), ("SOURCE", ("source",)),
                                         ("TEMPLATE", ("template",))), "No report profiles found."),
        "dataset providers": ("providers", (("ID", ("id",)), ("VERSION", ("version",))),
                              "No dataset providers found."),
        "dataset list": ("datasets", (("DATASET", ("logical_name",)), ("VERSION", ("dataset_version",)),
                                      ("PATH", ("path",))), "No datasets found."),
        "train profiles": ("profiles", (("PROFILE", ("id",)), ("TEMPLATE", ("template",)), ("VALID", ("valid",)),
                                        ("PATH", ("path",))), "No training profiles found."),
        "checkpoint list": ("checkpoints", (("CHECKPOINT", ("checkpoint_id",)), ("KIND", ("kind",)),
                                            ("MERGE STATUS", ("merge_status",)), ("PATH", ("host_path",))),
                            "No checkpoints found."),
    }
    if command in list_contracts:
        if command == "checkpoint list" and isinstance(result, dict) and result.get("source") == "registry":
            return _table(result.get("checkpoints", ()), (
                ("CHECKPOINT", ("checkpoint_id",)), ("REPOSITORY", ("repo_id",)),
                ("REVISION", ("revision",)), ("LOCAL", ("local_status",)),
                ("REMOTE", ("remote_status",)),
            ), empty="No registered checkpoints found.")
        key, columns, empty = list_contracts[command]
        rows = result.get(key, ()) if isinstance(result, dict) else ()
        return _table(rows, columns, empty=empty)
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
    if command == "train run":
        rendered.extend(_training_check_rows(result))
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
    if name == "WorkflowEditorError":
        return ExitCode.RUNTIME, "editor_unavailable"
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
    if name in {"DatasetRequestError", "TrainingProfileError", "CheckpointExportError"}:
        return ExitCode.CONFIGURATION, "configuration_error"
    if name in {"DatasetUnavailableError"}:
        return ExitCode.POLICY_UNAVAILABLE, "dataset_unavailable"
    if name in {"DatasetInterruptedError", "TrainingInterruptedError"}:
        return ExitCode.INTERRUPTED, "interrupted"
    if name.startswith("Training"):
        return ExitCode.RUNTIME, "training_error"
    if name in {"RunIntegrityError", "ArtifactError", "ExperimentIntegrityError"}:
        return ExitCode.INTEGRITY, "run_integrity_error"
    from ovlab_policy_sdk.plugin import PolicyPluginUnavailableError
    if isinstance(exc, PolicyPluginUnavailableError):
        return ExitCode.POLICY_UNAVAILABLE, "policy_unavailable"
    if name.startswith("RemotePolicy") or name in {"ContractCompatibilityError", "ConnectionError"}:
        return ExitCode.SERVICE, "service_or_capability_error"
    if name.startswith("Config") or name in {"StrictYamlError", "ResolvedConfigWriteError", "ExperimentSpecError"}:
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


def _validate_port(port):
    if port is not None and not 1 <= port <= 65535:
        raise CliUsageError("port must be in 1..65535")


def _dispatch(args):
    for name, value in vars(args).copy().items():
        if name.startswith("_pos_") and value is not None:
            setattr(args, name[len("_pos_"):], value)
    if args.command == "export" and args.export_command == "grouped":
        targets = list(args.group_targets)
        if args.name is None:
            if not targets:
                raise CliUsageError("export grouped requires a group NAME")
            args.name = targets.pop(0)
        if targets:
            if args.runs or args.all_runs or args.same_model_as:
                raise CliUsageError("use positional RUNs or a selector flag, not both")
            args.runs = targets
        if not (args.runs or args.all_runs or args.same_model_as):
            raise CliUsageError("export grouped requires RUNs or a selector flag")
    if args.command == "data" and args.data_command in {"archive", "delete"}:
        selected = (args.run_id, args.report_id, args.export_id, args.all_data)
        if args.data_target is not None:
            if any(selected):
                raise CliUsageError("use a typed TARGET or a selector flag, not both")
            kind, separator, value = args.data_target.partition(":")
            if not separator or not value or kind not in {"run", "report", "export"}:
                raise CliUsageError("data TARGET must be run:ID, report:ID or export:TYPE:ID")
            setattr(args, {"run": "run_id", "report": "report_id", "export": "export_id"}[kind], value)
        elif not any(selected):
            raise CliUsageError("data operation requires a typed TARGET, a selector flag or --all")
    if args.command == "editor":
        from .workflow_editor import open_workflow_editor
        return open_workflow_editor(_repository_root()), args.json, None
    if args.command == "doctor":
        from .doctor import Doctor
        return Doctor(_repository_root()).run(experiment=args.experiment, env_file=args.env_file), args.json, None
    if args.command == "config":
        app = _application()
        if args.config_command == "validate":
            return app.validate(args.config, mode=args.mode), args.json, None
        document = app.resolved_document(args.config, mode=args.mode)
        if args.format == "json":
            return json.dumps(document, sort_keys=True, separators=(",", ":")), False, "raw"
        from .strict_yaml import dumps
        return dumps(document), False, "raw"
    if args.command == "module":
        from .plugin_catalog import list_projects, inspect_project, check_project
        from .plugin_projects import register_extension_project
        from .policy_plugins import registry_scope
        root = _repository_root()
        values = dict(os.environ)
        if args.registry:
            values["OVLAB_EXTENSION_REGISTRY"] = args.registry
        with registry_scope(root, values):
            if args.module_command == "register":
                result = register_extension_project(root, args.source, commit=args.commit,
                    registry_path=args.registry, dry_run=args.dry_run, category="module")
            elif args.module_command == "list":
                result = list_projects(root, category="module")
            elif args.module_command == "inspect":
                result = inspect_project(root, args.module, category="module")
            else:
                result = check_project(root, args.module, category="module")
        return result, args.json, None
    if args.command == "plugin":
        if args.plugin_command == "register":
            from .plugin_projects import register_extension_project
            return register_extension_project(_repository_root(), args.source, commit=args.commit,
                                    registry_path=args.registry, dry_run=args.dry_run), args.json, None
        from .plugin_catalog import check_project, inspect_project, list_projects, project_roles, project_sources
        from .plugin_sources import sync_sources
        from .plugin_build import build_images
        from .policy_plugins import registry_document, registry_scope
        root = _repository_root()
        values = dict(os.environ)
        if args.registry:
            path = Path(args.registry).expanduser()
            values["OVLAB_EXTENSION_REGISTRY"] = str(path if path.is_absolute() else root / path)
        with registry_scope(root, values, offline=getattr(args, "offline", False) or getattr(args, "dry_run", False)):
            selection = getattr(args, "plugin", None)
            if args.plugin_command == "list":
                result = list_projects(root)
            elif args.plugin_command == "inspect":
                result = inspect_project(root, selection)
            elif args.plugin_command == "check":
                result = check_project(root, selection)
            else:
                registry = registry_document(root)
                if args.plugin_command == "sync":
                    sources = project_sources(registry, selection) if selection else None
                    result = sync_sources(root, sources, offline=args.offline, dry_run=args.dry_run)
                else:
                    roles = project_roles(registry, selection)
                    if selection and not roles:
                        raise CliUsageError(f"plugin project {selection!r} has no runnable image recipe")
                    result = build_images(root, roles, offline=args.offline, dry_run=args.dry_run,
                                          isolated_sources=args.isolated_sources, config=args.config,
                                          execution_profile=args.execution_profile) if roles else []
            return result, args.json, None
    if args.command == "_worker":
        _validate_port(args.port)
        options = {"host": args.host, "port": args.port} if args.port is not None else {}
        return _application().service_health(args.socket, **options), args.json, None
    if args.command == "run":
        from .experiment_workflow import ExperimentWorkflow
        workflow = ExperimentWorkflow(_repository_root())
        if args.verify:
            return workflow.verify(args.target), args.json, None
        if args.plan:
            return workflow.plan(args.target), args.json, None
        previous = _install_interrupt_handlers()
        try:
            result = workflow.run(args.target)
        finally:
            _restore_handlers(previous)
        return result, args.json, None
    if args.command == "experiment":
        from .experiment_workflow import ExperimentWorkflow
        workflow = ExperimentWorkflow(_repository_root())
        if args.experiment_command == "plan":
            return workflow.plan(args.spec, node=args.node, node_only=args.node_only,
                                 sequential=args.sequential, max_workers=args.max_workers), args.json, None
        previous = _install_interrupt_handlers()
        try:
            result = workflow.run(args.spec, node=args.node, node_only=args.node_only, dry_run=args.dry_run,
                                  sequential=args.sequential, max_workers=args.max_workers)
        finally:
            _restore_handlers(previous)
        return result, args.json, None
    if args.command == "eval":
        if args.eval_command in {"inspect", "verify"}:
            app = _application()
            result = app.inspect(args.target) if args.eval_command == "inspect" else app.verify(args.target)
            return result, args.json, None
        from .deployment import ComposeDeployment
        deployment = ComposeDeployment(_repository_root())
        plan = deployment.plan(
            args.target,
            renderer=args.renderer,
            execution_profile=args.execution_profile,
            quantization=args.quantization,
            offline=True if args.offline else None,
            project_name=args.project_name,
        )
        previous = _install_interrupt_handlers()
        try:
            result = deployment.run(plan, dry_run=args.dry_run)
        finally:
            _restore_handlers(previous)
        return result, args.json, None
    if args.command == "deploy":
        if args.deploy_command == "serve":
            _validate_port(args.port)
            previous = _install_interrupt_handlers()
            try:
                options = {"host": args.host, "port": args.port, "persistent": args.persistent} if args.port is not None or args.persistent else {}
                result = _application().serve(args.config, socket_path=args.socket, **options)
            finally:
                _restore_handlers(previous)
            return result, args.json, None
        if args.deploy_command == "connect":
            _validate_port(args.port)
            if args.target and any(value is not None for value in (args.socket, args.port, args.host)):
                raise CliUsageError("deploy connect accepts a managed deployment or a direct endpoint, not both")
            if args.host is not None and args.port is None:
                raise CliUsageError("deploy connect --host requires --port")
            options = {"socket_path": args.socket} if args.socket is not None else {}
            if args.port is not None:
                options = {"host": args.host or "127.0.0.1", "port": args.port}
            if args.target:
                from .model_deployment import ModelDeployment
                options = ModelDeployment(_repository_root()).connection_endpoint(args.target)
            result = _application().connect(args.experiment, **options)
            if args.target:
                result["deployment"] = args.target
            return result, args.json, None
        from .model_deployment import ModelDeployment
        deployment = ModelDeployment(_repository_root())
        if args.deploy_command == "run":
            plan = deployment.plan(args.target, quantization=args.quantization, port=args.port,
                                   offline=args.offline, local_profile=args.profile)
            previous = _install_interrupt_handlers()
            try:
                result = deployment.run(plan, dry_run=args.dry_run)
            finally:
                _restore_handlers(previous)
        elif args.deploy_command == "inspect":
            result = deployment.inspect(args.target)
        elif args.deploy_command == "verify":
            result = deployment.verify(args.target)
        else:
            result = deployment.stop(args.target, dry_run=args.dry_run)
        return result, args.json, None
    if args.command == "_native-run":
        app = _application()
        if args.target in {"inspect", "verify"}:
            if args.profile is not None:
                raise CliUsageError("run inspect/verify cannot be combined with --profile")
            if args.path is None:
                raise CliUsageError(f"run {args.target} requires RUN_PATH, RUN_ID, or RUN_HASH")
            result = app.inspect(args.path) if args.target == "inspect" else app.verify(args.path)
        else:
            if args.path is not None:
                raise CliUsageError("run TARGET accepts no second positional argument")
            if args.target is None:
                raise CliUsageError("run requires an experiment YAML TARGET")
            from .config_paths import resolve_config_path
            config = resolve_config_path(args.target, _repository_root(), "experiments/evaluation")
            profile_options = {"local_profile": args.profile} if args.profile is not None else {}
            if args.dry_run:
                result = app.execution_plan(config, output_root=args.output_root, **profile_options)
            else:
                previous = _install_interrupt_handlers()
                try:
                    result = app.run(config, output_root=args.output_root, **profile_options)
                finally:
                    _restore_handlers(previous)
                from .wandb_tracking import track_run
                tracking = track_run("eval", result["run_path"]) if "run_path" in result else None
                if tracking:
                    result["wandb"] = tracking
        return result, args.json, None
    if args.command == "report":
        if args.report_command == "auto":
            from .report_auto import generate_report
            return generate_report(args.reference, repository_root=_repository_root()), args.json, None
        app = _application()
        if args.report_command == "recompute":
            return app.recompute_metrics(args.run_path), args.json, None
        if args.report_command == "profiles":
            return app.report_profiles(), args.json, None
        if args.report_command == "publish":
            options = {"report_enabled": args.report_enabled == "true"}
            if args.output is not None:
                if args.task is not None or args.profile != "libero-task-default" or not options["report_enabled"]:
                    raise CliUsageError("report publish --output cannot be combined with --task, --profile or disabled reporting")
                options["output"] = args.output
            if args.task is not None:
                options["task_id"] = args.task
            return app.report_publish(args.run_id, args.profile, **options), args.json, None
        if args.report_command == "verify":
            return app.report_verify(args.run_id, args.profile, build_id=args.build), args.json, None
        raise AssertionError("unhandled report command")
    if args.command == "export":
        app = _application()
        if getattr(args, "category", "evaluate") == "train":
            from ovlab_core.experiment_scope import ExperimentScope
            from ovlab_runner.training_exports import TrainingExportEngine
            runs, _, exports = app._data_roots()
            if args.export_command == "isolated":
                if args.episode_id is not None or args.template != "isolated-default-v1":
                    raise CliUsageError("training isolated exports do not accept --episode or --template")
                selected = app._training_run(args.run_id)
                scope = ExperimentScope.from_run_path(runs, selected.path, "train")
                engine = TrainingExportEngine(runs, exports, scope=scope)
                return engine.generate_isolated(selected.run_id), args.json, None
            if args.export_command == "grouped":
                if not args.runs or args.all_runs or args.same_model_as or args.suite or args.template != "grouped-default-v1":
                    raise CliUsageError("training grouped exports require --runs and do not accept --suite or --template")
                selected = [app._training_run(value) for value in args.runs]
                run_ids = [run.run_id for run in selected]
                scope = ExperimentScope.from_environment()
                if scope is None:
                    scopes = [ExperimentScope.from_run_path(runs, run.path, "train") for run in selected]
                    if any(item is None for item in scopes) and any(item is not None for item in scopes):
                        raise CliUsageError("training grouped export cannot mix legacy and experiment runs")
                    if all(item is not None for item in scopes):
                        if len({item.experiment_id for item in scopes}) != 1:
                            raise CliUsageError("training grouped export requires runs from one experiment")
                        common = list(scopes[0].node_parts)
                        for item in scopes[1:]:
                            while common and tuple(common) != item.node_parts[:len(common)]:
                                common.pop()
                        scope = ExperimentScope(scopes[0].experiment_id, tuple(common))
                engine = TrainingExportEngine(runs, exports, scope=scope)
                return engine.generate_grouped(args.name, run_ids=run_ids), args.json, None
            if args.export_command == "verify":
                if args.legacy_export_id is not None or args.export_name is None:
                    raise CliUsageError("training export verify requires --name NAME")
                scope = ExperimentScope.from_environment()
                if scope is None and args.kind == "isolated":
                    selected = app._training_run(args.export_name)
                    scope = ExperimentScope.from_run_path(runs, selected.path, "train")
                if scope is None and args.kind == "grouped":
                    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.export_name) is None:
                        raise CliUsageError("training grouped export name must be a portable identifier")
                    candidates = [path for path in exports.glob(f"*/train/**/grouped/{args.export_name}/metadata.json") if path.is_file()]
                    if len(candidates) > 1:
                        raise CliUsageError("training grouped export name is ambiguous; set OVLAB_EXPERIMENT_ID and OVLAB_EXPERIMENT_NODE")
                    if candidates:
                        parts = candidates[0].parent.relative_to(exports).parts
                        scope = ExperimentScope(parts[0], parts[2:-2])
                engine = TrainingExportEngine(runs, exports, scope=scope)
                return engine.verify(args.kind, args.export_name), args.json, None
            raise CliUsageError("legacy export generate supports evaluation only")
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
        force = args.force
        preview = app.data_preview(args.data_command, force=force, **selector)
        if args.dry_run:
            return preview, args.json, None
        if not _confirm_data_operation(args, preview):
            return {**preview, "status": "cancelled", "dry_run": False}, False, None
        result = (
            app.data_archive(force=force, **selector)
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
        if args.train_command == "ablate":
            return app.train_ablate(
                args.suite, resolve=args.ablate_command != "validate",
                output=args.output if args.ablate_command in {"materialize", "run"} else None,
                execute=args.ablate_command == "run",
                fit_action_normalization=args.fit_action_normalization,
            ), args.json, None
        if args.train_command == "profiles":
            return app.train_profiles(), args.json, None
        if args.train_command == "validate":
            return app.train_validate(args.profile, fit_action_normalization=args.fit_action_normalization), args.json, None
        if args.train_command == "plan":
            return app.train_plan(args.profile, fit_action_normalization=args.fit_action_normalization), args.json, None
        if args.train_command == "run":
            if args.dry_run:
                plan = app.train_plan(args.target, fit_action_normalization=args.fit_action_normalization)
                return {**plan, "side_effects_performed": False}, args.json, None
            from .training_deployment import TrainingDeployment
            return TrainingDeployment(_repository_root()).run(
                args.target, allow_dataset_download=args.allow_dataset_download,
                fit_action_normalization=args.fit_action_normalization,
                interactive_progress=sys.stderr.isatty(),
            ), args.json, None
        if args.train_command in {"status", "inspect", "verify"}:
            if args.target is not None and args.run_id is not None:
                raise CliUsageError("use TARGET or --run, not both")
            reference = args.target or args.run_id
            if reference is None:
                raise CliUsageError(f"train {args.train_command} requires a run TARGET")
            method = {"status": app.train_status, "inspect": app.train_inspect, "verify": app.train_verify}[args.train_command]
            return method(reference), args.json, None
        if args.train_command == "report":
            if args.build is not None and not args.verify:
                raise CliUsageError("train report --build requires --verify")
            return app.train_report(
                args.run_id, verify=args.verify, build_id=args.build,
                **({"parameters": args.parameters} if args.parameters is not None else {}),
            ), args.json, None
        raise AssertionError("unhandled training command")
    if args.command == "checkpoint":
        app = _application()
        if args.checkpoint_command == "list":
            if args.source == "local" and (args.online or args.offline or args.profile):
                raise CliUsageError("--online, --offline and --profile require --source registry")
            if args.source == "local":
                return app.checkpoint_list(), args.json, None
            return app.checkpoint_registry_list(online=args.online, profile=args.profile), args.json, None
        if args.checkpoint_command == "export":
            from .checkpoint_export import export_checkpoint
            return export_checkpoint(app._model_data_root(), args.checkpoint_id,
                                     format=args.format, output=Path(args.output)), args.json, None
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
        from .policy_plugins import registry_scope
        with _configured_artifact_umask(), registry_scope(_repository_root(), os.environ):
            result, json_mode, render = _dispatch(args)
        diagnosis_failed = command == "doctor" and result["status"] == "error"
        if json_mode:
            diagnosis_errors = (
                tuple({
                    "code": "diagnostic_failure", "type": "DiagnosticCheck",
                    "message": check["message"],
                    "context": {"check": check["name"], "remedy": check["remedy"]},
                } for check in result["checks"] if check["status"] == "error")
                if diagnosis_failed else ()
            )
            _json_output(command, "error" if diagnosis_failed else "success", result, diagnosis_errors)
        elif render == "raw" or bool(getattr(args, "detail", False)):
            _human(result)
        else:
            _human(_compact(command, result))
        return ExitCode.RUNTIME if diagnosis_failed else ExitCode.SUCCESS
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
