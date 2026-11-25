"""Pinned Git projects freeze into a worker using installed control packages.

Existing tests cover local-source workers and pinned acquisition separately.
This regression joins those boundaries without GitHub, plugin PYTHONPATH,
editable core installs, ambient site-packages, a simulator or Docker.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
POLICY_REF = "plugin:policy:example-policy/constant"
BENCHMARK_REF = "plugin:benchmark:example-benchmark/planar"


def _run(command, *, cwd=None, environment=None):
    result = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _git_environment():
    return {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1"}


def _git_project(temporary, example):
    source = temporary / "upstreams" / example
    shutil.copytree(ROOT / "examples" / example, source,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "build", "dist"))
    environment = _git_environment()

    def git(*arguments):
        return _run(["git", "-C", str(source), *arguments], environment=environment)

    git("init", "--quiet", "--initial-branch=main")
    git("add", ".")
    git("-c", "user.name=OVLAB fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--quiet", "--message=pinned project")
    pinned = git("rev-parse", "HEAD")
    remote = temporary / "remotes" / f"{example}.git"
    remote.parent.mkdir(exist_ok=True)
    _run(["git", "clone", "--bare", str(source), str(remote)], environment=environment)

    # The branch moves before registration. Both declaration and executable code
    # at its head become unusable; only the explicit initial commit can work.
    manifest_path = source / "extension.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = "999.0.0"
    manifest_path.write_text(json.dumps(manifest))
    package = source / "src" / ('example_' + example.removesuffix('-plugin')) / "plugin.py"
    package.write_text("raise AssertionError('moving branch code was loaded')\n")
    git("add", ".")
    git("-c", "user.name=OVLAB fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--quiet", "--message=move branch beyond pin")
    git("push", remote.as_uri(), "HEAD:refs/heads/main")
    assert git("rev-parse", "HEAD") != pinned
    return remote.as_uri(), pinned


def _install_control_wheels(temporary):
    """Build copied distributions offline so the repository stays untouched."""
    projects = []
    for relative in ("code/apps/benchctl", "code/packages/ovlab-policy-sdk", "code/packages/ovlab-benchmarks"):
        destination = temporary / "wheel-sources" / Path(relative).name
        shutil.copytree(ROOT / relative, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "build", "dist"))
        projects.append(str(destination))
    environment = {**os.environ, "PIP_NO_INDEX": "1", "PIP_NO_CACHE_DIR": "1",
                   "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    wheels = temporary / "wheels"
    _run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
          "--wheel-dir", str(wheels), *projects], cwd=temporary, environment=environment)
    built = sorted(wheels.glob("*.whl"))
    assert len(built) == 3
    installed = temporary / "installed-control"
    _run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
          "--target", str(installed), *(str(wheel) for wheel in built)],
         cwd=temporary, environment=environment)
    return installed


def test_registered_pinned_git_projects_freeze_into_clean_installed_worker(tmp_path, monkeypatch):
    from ovlab_benchctl.config_bundle import ConfigBundleBuilder
    from ovlab_benchctl.extensions import ExtensionCatalog
    from ovlab_benchctl.plugin_projects import register_extension_project
    from ovlab_benchctl.policy_plugins import registry_scope

    # Other registration fixtures deliberately exercise changed source trees.
    # This host composition must start without their cached Python namespaces.
    for name in list(sys.modules):
        if name.startswith(('example_policy', 'example_benchmark')):
            monkeypatch.delitem(sys.modules, name)

    pins = {name: _git_project(tmp_path, name) for name in ("policy-plugin", "benchmark-plugin")}
    host = tmp_path / "host"
    shutil.copytree(ROOT / "config", host / "config")
    registry = host / "configs/extensions/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"schema_version": "ovlab.extensions/v1", "projects": {}}))
    environment = {"OVLAB_EXTENSION_REGISTRY": str(registry)}
    with registry_scope(host, environment):
        for remote, pin in pins.values():
            register_extension_project(host, remote, commit=pin, registry_path=registry)
        catalog = ExtensionCatalog(host, registry_path=registry)
        for project in ("example-policy", "example-benchmark"):
            assert catalog.project_manifest(project)["version"] == ('2.0.0' if project == 'example-policy' else '1.0.0')
        builder = ConfigBundleBuilder(host, environment=environment)
        bundle = builder.build("plugin://example-benchmark/evaluate.yaml", None)
        worker = tmp_path / "worker"
        with builder.materialize(bundle) as frozen:
            shutil.copytree(frozen, worker / "configs")

    lock = json.loads((worker / "configs/extensions/lock.json").read_text())["lock"]
    assert set(lock["providers"]) == {POLICY_REF, BENCHMARK_REF}
    for project, source in (("example-policy", "policy-plugin"), ("example-benchmark", "benchmark-plugin")):
        assert lock["sources"][f"{project}/self"]["commit"] == pins[source][1]

    installed = _install_control_wheels(tmp_path)
    # Remove every acquisition route, including original clones and the managed
    # cache, before creating the worker process. Only the frozen closure remains.
    for path in (host, tmp_path / "upstreams", tmp_path / "remotes"):
        shutil.rmtree(path)
    empty_home = tmp_path / "worker-home"
    empty_home.mkdir()
    worker_environment = {"HOME": str(empty_home), "PATH": "", "PYTHONDONTWRITEBYTECODE": "1",
        "OVLAB_ROOT": str(worker),
        "OVLAB_EXTENSION_REGISTRY": str(worker / "configs/extensions/registry.json"),
        # -I ignores this deliberately hostile inherited source path.
        "PYTHONPATH": os.pathsep.join((str(ROOT / "code/apps/benchctl/src"),
                                      str(ROOT / "examples/policy-plugin/src"),
                                      str(ROOT / "examples/benchmark-plugin/src")))}
    script = r'''
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

installed, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(installed))
assert sys.flags.isolated and sys.flags.no_site
assert importlib.util.find_spec('numpy') is None
assert importlib.util.find_spec('example_policy') is None
assert importlib.util.find_spec('example_benchmark') is None

def forbidden(*args, **kwargs):
    raise AssertionError('frozen control resolution attempted subprocess/network IO')
subprocess.run = forbidden
subprocess.Popen = forbidden
socket.create_connection = forbidden
socket.socket = forbidden

import ovlab_benchctl, ovlab_benchmarks, ovlab_policy_sdk
for package in (ovlab_benchctl, ovlab_benchmarks, ovlab_policy_sdk):
    assert Path(package.__file__).resolve().is_relative_to(installed), package.__file__
from ovlab_benchctl.extensions import ExtensionCatalog
from ovlab_benchctl.config_bundle import ConfigBundleBuilder
from ovlab_benchmarks.plugin import BenchmarkPlugin
from ovlab_policy_sdk.plugin import PolicyPlugin

catalog = ExtensionCatalog(root, registry_path=root/'configs/extensions/registry.json')
assert catalog.frozen
builder = ConfigBundleBuilder(root)
policy_doc = builder.load_component('plugin://example-policy/policy.yaml', 'policy')
benchmark_doc = builder.load_component('plugin://example-benchmark/benchmark.yaml', 'benchmark')
policy = catalog.load_provider(policy_doc['provider'], expected_kind='policy', api_version='2', contract=PolicyPlugin)
benchmark = catalog.load_provider(benchmark_doc['provider'], expected_kind='benchmark', contract=BenchmarkPlugin)
policy.validate_policy(policy_doc, 'frozen policy')
benchmark.validate_benchmark(benchmark_doc, 'frozen benchmark')
deployment = benchmark.resolve_deployment(benchmark_doc, execution_profile={}, local_execution={}, devices={}, environment={}, deployment={})
assert not deployment.requires_gpu and not deployment.mounts and not deployment.environment
import example_policy.plugin, example_benchmark.plugin
for module in (example_policy.plugin, example_benchmark.plugin):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
assert len(example_benchmark.plugin.catalog_data()['tasks']) == 3
assert 'example_policy.runtime' not in sys.modules and 'example_benchmark.runtime' not in sys.modules
assert all(not str(path).endswith('/src') for path in sys.path if 'code/' in str(path))
print(json.dumps({'policy_api':policy.api_version, 'benchmark_api':benchmark.api_version, 'frozen':True}))
'''
    output = _run([sys.executable, "-I", "-S", "-c", script, str(installed), str(worker)],
                  cwd=worker, environment=worker_environment)
    assert json.loads(output) == {"policy_api": "2", "benchmark_api": "1", "frozen": True}
