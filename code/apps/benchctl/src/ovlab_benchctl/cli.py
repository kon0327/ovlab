"""Unified dependency-light OVLAB command-line interface."""

from __future__ import annotations

import argparse
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ovlab", description="OpenVLABenchmark orchestration CLI")
    parser.add_argument("--version", action="store_true", help="show OVLAB CLI and repository revision")
    parser.add_argument("--debug", action="store_true", help="include a traceback for unexpected failures")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    config = commands.add_parser("config", help="validate or resolve configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="validate a configuration")
    validate.add_argument("config")
    validate.add_argument("--mode", choices=("descriptor", "runtime"), default="descriptor")
    validate.add_argument("--json", action="store_true")
    resolve = config_commands.add_parser("resolve", help="print deterministic resolved configuration")
    resolve.add_argument("config")
    resolve.add_argument("--mode", choices=("descriptor", "runtime"), default="descriptor")
    resolve.add_argument("--format", choices=("yaml", "json"), default="yaml")

    policy = commands.add_parser("policy", help="inspect registered policy identities")
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    listing = policy_commands.add_parser("list", help="list policy variants without loading providers")
    listing.add_argument("--json", action="store_true")
    describe = policy_commands.add_parser("describe", help="describe a configured policy without loading it")
    describe.add_argument("config")
    describe.add_argument("--json", action="store_true")

    service = commands.add_parser("service", help="run an isolated policy service")
    service_commands = service.add_subparsers(dest="service_command", required=True)
    serve = service_commands.add_parser("serve", help="serve one policy over a foreground AF_UNIX socket")
    serve.add_argument("config")
    serve.add_argument("--socket")
    health = service_commands.add_parser("health", help="probe protocol readiness without model inference")
    health.add_argument("--socket", required=True)
    health.add_argument("--json", action="store_true")

    connect = commands.add_parser("connect", help="probe policy handshake and capability compatibility")
    connect.add_argument("config")
    connect.add_argument("--json", action="store_true")

    deploy = commands.add_parser("deploy", help="orchestrate isolated OVLAB containers")
    deploy_commands = deploy.add_subparsers(dest="deploy_command", required=True)
    deploy_run = deploy_commands.add_parser("run", help="run one experiment through Docker Compose")
    deploy_run.add_argument("experiment")
    deploy_run.add_argument("--profile", choices=("openvla", "oft"), required=True)
    deploy_run.add_argument("--renderer", choices=("egl", "glfw"), default="egl")
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
    deploy_run.add_argument("--json", action="store_true")

    run = commands.add_parser("run", help="execute, inspect, or verify a run")
    run.add_argument("target", help="CONFIG, or 'inspect'/'verify'")
    run.add_argument("path", nargs="?", help="RUN_PATH for inspect or verify")
    run.add_argument("--output-root")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--json", action="store_true")

    metrics = commands.add_parser("metrics", help="offline metric operations")
    metrics_commands = metrics.add_subparsers(dest="metrics_command", required=True)
    recompute = metrics_commands.add_parser("recompute", help="recompute metrics from immutable traces")
    recompute.add_argument("run_path")
    recompute.add_argument("--json", action="store_true")

    report = commands.add_parser("report", help="regenerate deterministic reports from immutable runs")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    generate = report_commands.add_parser("generate", help="write a derived report outside the canonical run")
    generate.add_argument("run_path")
    generate.add_argument("--output", required=True)
    generate.add_argument("--json", action="store_true")
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
        "metrics_command", "report_command",
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
        sys.stdout.write(value.rstrip() + "\n")
    else:
        sys.stdout.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
        return result, False, None
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
        return deployment.run(plan, dry_run=args.dry_run), args.json, None
    if args.command == "run":
        app = _application()
        if args.target in {"inspect", "verify"}:
            if args.path is None:
                raise CliUsageError(f"run {args.target} requires RUN_PATH")
            result = app.inspect(args.path) if args.target == "inspect" else app.verify(args.path)
        else:
            if args.path is not None:
                raise CliUsageError("run CONFIG accepts no second positional argument")
            result = (
                app.execution_plan(args.target, output_root=args.output_root)
                if args.dry_run else app.run(args.target, output_root=args.output_root)
            )
        return result, args.json, None
    if args.command == "metrics":
        return _application().recompute_metrics(args.run_path), args.json, None
    if args.command == "report":
        return _application().generate_report(args.run_path, args.output), args.json, None
    raise AssertionError("unhandled CLI command")


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.version:
        value = repository_revision(_repository_root())
        _human(f"ovlab {CLI_VERSION} (revision {value or 'unavailable'})")
        return ExitCode.SUCCESS
    if args.command is None:
        parser.print_help()
        return ExitCode.SUCCESS
    command = _command_name(args)
    wants_json = bool(getattr(args, "json", False))
    try:
        result, json_mode, render = _dispatch(args)
        if json_mode:
            _json_output(command, "success", result, ())
        else:
            _human(result)
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
