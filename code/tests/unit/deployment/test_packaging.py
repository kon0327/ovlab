"""Dependency-light invariants for Gate H deployment packaging."""

from __future__ import annotations

import json
from pathlib import Path
import re
import runpy
import subprocess

from ovlab_benchctl.application import OvlabApplication
from ovlab_benchctl.versioning import repository_revision


ROOT = Path(__file__).resolve().parents[4]
DOCKER = ROOT / "deploy/docker"
PLUGIN = ROOT / "plugins/openvla"
COMPOSE = (ROOT / "deploy/compose/compose.yaml").read_text(encoding="utf-8")
from ovlab_benchctl.deployment import ComposeDeployment


def _evaluation_services(profile="openvla", renderer="egl"):
    config = {"openvla": "libero10-lora-merged-rpc-smoke", "oft": "libero10-openvla-oft-rpc-smoke", "quic": "libero10-quic-adapter-rpc-smoke"}[profile]
    deployment = ComposeDeployment(ROOT)
    plan = deployment.plan(config, renderer=renderer, env_file="deploy/compose/.env.example")
    return deployment, plan, deployment.service_spec(plan)["services"]


def _dockerfile(name: str) -> str:
    path = DOCKER / name
    return (path if path.is_file() else PLUGIN / "docker" / name).read_text(encoding="utf-8")


def _lock(name):
    path = ROOT / "deploy/locks" / name
    return path if path.is_file() else PLUGIN / "locks" / name


def test_configuration_secret_layers_use_bundle_identity_in_their_cache_key():
    recipes = [*DOCKER.glob("Dockerfile.*"), *PLUGIN.joinpath("docker").glob("Dockerfile.*"),
               ROOT / "examples/policy-plugin/Dockerfile"]
    for path in recipes:
        text = path.read_text()
        if "id=ovlab_config_bundle" not in text:
            continue
        instructions = re.sub(r"\\\n\s*", " ", text).splitlines()
        installation = [index for index, instruction in enumerate(instructions)
                        if instruction.startswith("RUN ") and "id=ovlab_config_bundle" in instruction
                        and "bash /tmp/install-ovlab-packages.sh" in instruction]
        assert installation, path
        for index in installation:
            assert instructions[index - 1] == "ARG OVLAB_CONFIG_BUNDLE_SHA256=none", path
            assert 'test -n "$OVLAB_CONFIG_BUNDLE_SHA256"' in instructions[index], path
            assert instructions[index].index('test -n "$OVLAB_CONFIG_BUNDLE_SHA256"') < instructions[index].index(
                "bash /tmp/install-ovlab-packages.sh"
            ), path


def _service(name: str, next_name: str | None = None) -> str:
    block = COMPOSE.split(f"  {name}:\n", 1)[1]
    if next_name is not None:
        return block.split(f"  {next_name}:\n", 1)[0]
    return block.split("\nvolumes:\n", 1)[0]


def test_production_images_are_digest_pinned_non_root_and_cli_only():
    for name in (
        "Dockerfile.benchmark", "Dockerfile.reporting", "Dockerfile.openvla",
        "Dockerfile.openvla-oft",
    ):
        text = _dockerfile(name)
        assert all(
            "@sha256:" in line
            for line in text.splitlines()
            if line.startswith("FROM ")
        )
        assert 'USER 10001:10001' in text
        assert 'ENTRYPOINT ["ovlab"]' in text
        assert "ENTRYPOINT [\"/bin/sh\"" not in text
        assert "--mount=type=secret,id=ovlab_source_manifest" in text
        assert "OVLAB_SOURCE_MANIFEST_B64" not in text
        assert "io.github.ovlab.dependency-lock.sha256" in text
        assert 'org.opencontainers.image.authors="Jiří Konečný, VSB - Technical University of Ostrava"' in text
        assert 'org.opencontainers.image.licenses="NOASSERTION"' in text
        assert 'io.github.ovlab.build-target="production"' in text
        expected_contract = (
            "resolved-checkpoint-quantization-config-bundle-v2"
            if name in {"Dockerfile.openvla", "Dockerfile.openvla-oft"}
            else "canonical-readonly-reporting-v1"
            if name == "Dockerfile.reporting"
            else "native-run-config-bundle-v3"
        )
        assert f'io.github.ovlab.deployment.contract="{expected_contract}"' in text


def test_role_closures_do_not_cross_heavy_runtime_boundaries():
    benchmark = _dockerfile("Dockerfile.benchmark")
    reporting = _dockerfile("Dockerfile.reporting")
    openvla = _dockerfile("Dockerfile.openvla")
    oft = _dockerfile("Dockerfile.openvla-oft")
    assert "COPY external/libero " in benchmark
    assert "PYTHONPATH=/opt/ovlab/external/libero" in benchmark
    assert "pip install" not in "\n".join(
        line for line in benchmark.splitlines() if "/opt/ovlab/external/libero" in line
    )
    assert "external/openvla " not in benchmark
    assert "external/openvla-oft " not in benchmark
    assert "COPY external/openvla " in openvla
    assert "COPY external/libero " not in openvla
    assert "COPY external/openvla-oft " in oft
    assert "COPY external/libero " not in oft
    assert "COPY external/" not in reporting
    assert "mujoco" not in reporting.lower()
    assert "resolved-checkpoint" not in reporting
    assert "COPY external/libero " not in _dockerfile("Dockerfile.openvla-quic")


def test_oft_quantization_stack_pins_the_dispatch_compatible_accelerate_version():
    source = _lock("openvla-oft.in").read_text(encoding="utf-8")
    requirements = _lock("openvla-oft.requirements.txt").read_text(
        encoding="utf-8"
    )
    for document in (source, requirements):
        assert "accelerate==0.30.1" in document
        assert "bitsandbytes==0.43.1" in document


def test_compose_is_socket_only_offline_and_least_privilege():
    assert not any(value in COMPOSE for value in ("ports:", "privileged:", "/var/run/docker.sock"))
    for profile in ("openvla", "oft", "quic"):
        _, _, services = _evaluation_services(profile)
        policy, benchmark = services.values()
        for service in services.values():
            assert service["network_mode"] == "none"
            assert service["read_only"] is True
            assert service["user"] == "10001:10001"
            assert service["cap_drop"] == ["ALL"]
            assert "ports" not in service
            assert service["environment"]["OVLAB_PLUGIN_REGISTRY"] == "/opt/ovlab/configs/plugins/registry.json"
        assert policy["environment"]["HF_HUB_OFFLINE"] == "1"
        assert policy["environment"]["TRANSFORMERS_OFFLINE"] == "1"
        assert policy["healthcheck"]["test"][:3] == ["CMD", "ovlab", "service"]
        assert all(mount["read_only"] for mount in policy["volumes"] if isinstance(mount, dict))
        assert benchmark["environment"]["MUJOCO_GL"] == "egl"
        assert list(benchmark["depends_on"].values())[0]["condition"] == "service_healthy"
        assert any(mount.get("target") == "/datasets" and mount["read_only"] for mount in benchmark["volumes"] if isinstance(mount, dict))


def test_host_artifact_mounts_separate_canonical_runs_from_unified_exports():
    for profile in ("openvla", "oft", "quic"):
        _, _, services = _evaluation_services(profile)
        benchmark = list(services.values())[1]
        mounts = {m["target"]: m for m in benchmark["volumes"] if isinstance(m, dict)}
        assert mounts["/var/lib/ovlab/runs"]["read_only"] is False
        assert "/var/lib/ovlab/exports" not in mounts
        assert benchmark["stop_grace_period"] == "5m"
        assert benchmark["environment"]["OVLAB_POSTPROCESSING_MODE"] == "external"
        assert "${OVLAB_HOST_ARTIFACT_GID:-1000}" in benchmark["group_add"]
    for name, successor in (("reporting", "export"), ("export", "dataset-acquire")):
        block = _service(name, successor)
        assert "target: /var/lib/ovlab/runs\n        read_only: true" in block
        assert "target: /var/lib/ovlab/exports" in block
        assert "gpus:" not in block
    assert "target: /etc/ovlab/export.yaml\n        read_only: true" in _service("export", "dataset-acquire")


def test_gate_i_compose_services_preserve_dataset_training_and_finalization_boundaries():
    dataset = _service("dataset-acquire", "training-openvla")
    assert "profiles: [dataset-acquire]" in dataset
    assert "image: ${OVLAB_DATASET_IMAGE:-ovlab-dataset:local}" in dataset
    assert "network_mode: none" not in dataset
    assert "OVLAB_MOUNT_CONTRACT: datasets-rw" in dataset
    assert "target: /var/lib/ovlab/model-data/datasets" in dataset
    assert "target: /var/lib/ovlab/model-data/datasets\n        read_only: true" not in dataset
    assert "/var/lib/ovlab/runs" not in dataset
    assert "/var/lib/ovlab/checkpoints" not in dataset
    assert "gpus:" not in dataset

    training = _service("training-openvla", "checkpoint-finalizer")
    assert "profiles: [training]" in training
    assert "<<: *policy-security" in training
    assert "gpus: all" in training
    assert "target: /checkpoints/base\n        read_only: true" in training
    assert "target: /datasets/resolved/${OVLAB_TRAINING_DATASET_NAME:-unset}/1.0.0\n        read_only: true" in training
    assert "target: /var/lib/ovlab/training-run" in training
    assert "target: /var/lib/ovlab/model-data/checkpoints" not in training
    assert "/var/lib/ovlab/runs" not in training

    finalizer = _service("checkpoint-finalizer")
    assert "profiles: [training-finalize]" in finalizer
    assert "<<: *policy-security" in finalizer
    assert "gpus:" not in finalizer
    assert "target: /var/lib/ovlab/training-run" in finalizer
    assert "target: /var/lib/ovlab/model-data/checkpoints" in finalizer
    assert "/checkpoints/base" not in finalizer
    assert "/datasets/resolved" not in finalizer
    assert "/var/lib/ovlab/runs" not in finalizer


def test_gate_i_images_keep_dataset_and_training_dependency_closures_separate():
    dataset = _dockerfile("Dockerfile.dataset")
    training = _dockerfile("Dockerfile.training-openvla")
    assert "COPY external/" not in dataset
    assert "reporting.requirements" not in dataset
    assert "matplotlib" not in (ROOT / "deploy/locks/dataset.requirements.txt").read_text(encoding="utf-8")
    assert "COPY external/openvla " in training
    assert "COPY external/libero " not in training
    assert "COPY external/openvla-oft " not in training
    assert "COPY external/openvla-quic " not in training
    assert "io.github.ovlab.dependency-lock.sha256" in dataset
    assert "io.github.ovlab.dependency-lock.sha256" in training


def test_hash_locks_and_immutable_vcs_revision_are_checked_in():
    locks = ROOT / "deploy/locks"
    for role in ("benchmark", "reporting", "openvla", "openvla-oft", "flash-attn"):
        pylock = _lock(f"{role}.pylock.toml").read_text(encoding="utf-8")
        requirements = _lock(f"{role}.requirements.txt").read_text(encoding="utf-8")
        assert 'lock-version = "1.0"' in pylock
        assert 'sha256 = "' in pylock
        assert "--hash=sha256:" in requirements
    assert "rich==15.0.0" in _lock("openvla.requirements.txt").read_text(encoding="utf-8")
    openvla_lock = _lock("openvla.requirements.txt").read_text(encoding="utf-8")
    assert (
        "bitsandbytes==0.43.1 "
        "--hash=sha256:a81c826d576d6d691c7b4a7491c8fdc0f37f769795d6ca2e54afa605d2c260a3"
    ) in openvla_lock
    assert "bitsandbytes=0.43.1" in _dockerfile("Dockerfile.openvla")
    assert "rich==15.0.0" in _lock("openvla-oft.requirements.txt").read_text(encoding="utf-8")
    assert "imageio-ffmpeg==0.6.0" in (locks / "benchmark.requirements.txt").read_text(encoding="utf-8")
    assert "jinja2==3.1.6" in (locks / "benchmark.requirements.txt").read_text(encoding="utf-8")
    assert "markupsafe==3.0.3" in (locks / "benchmark.requirements.txt").read_text(encoding="utf-8")
    oft = _lock("openvla-oft.pylock.toml").read_text(encoding="utf-8")
    assert oft.count('commit-id = "bc339d9ad707454c0c115970db43c260067c61ab"') == 1
    assert oft.count('commit-id = "040105d256bd28866cc6620621a3d5f7b6b91b46"') == 1
    oft_input = _lock("openvla-oft.in").read_text(encoding="utf-8")
    assert "tensorflow==2.15.0" in oft_input
    assert "tensorflow-datasets==4.9.3" in oft_input
    assert "tensorflow-metadata==1.15.0" in oft_input
    assert "jsonlines==4.0.0" in oft_input
    assert "wandb==0.16.6" in oft_input
    assert "dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46" in oft_input
    assert "040105d256bd28866cc6620621a3d5f7b6b91b46" in _dockerfile("Dockerfile.openvla-oft")
    build_script = (ROOT / "deploy/scripts/build-images.sh").read_text(encoding="utf-8")
    assert "build_images.py" in build_script
    assert "id=ovlab_source_manifest,src=" in (ROOT / "code/apps/benchctl/src/ovlab_benchctl/plugin_build.py").read_text()
    assert "OVLAB_SOURCE_MANIFEST_B64" not in build_script


def test_dataset_pyyaml_lock_uses_the_verified_platform_wheel_hash():
    """Keep the small dataset lock aligned with the accepted runtime locks."""
    expected = "9c7708761fccb9397fe64bbc0395abcae8c4bf7b0eac081e12b809bf47700d0b"
    locks = ROOT / "deploy/locks"
    dataset_requirement = (locks / "dataset.requirements.txt").read_text(encoding="utf-8")
    dataset_pylock = (locks / "dataset.pylock.toml").read_text(encoding="utf-8")
    accepted_openvla_lock = _lock("openvla.requirements.txt").read_text(encoding="utf-8")
    assert f"pyyaml==6.0.3 --hash=sha256:{expected}" in dataset_requirement
    assert f'sha256 = "{expected}"' in dataset_pylock
    assert f"pyyaml==6.0.3 --hash=sha256:{expected}" in accepted_openvla_lock


def test_training_image_python_label_matches_the_pinned_pytorch_base_runtime():
    training = _dockerfile("Dockerfile.training-openvla")
    assert 'io.github.ovlab.python.version="3.10.13"' in training
    assert 'io.github.ovlab.python.version="3.10.20"' not in training


def test_builds_disable_unlocked_pep517_isolation():
    installer = (DOCKER / "install-ovlab-packages.sh").read_text(encoding="utf-8")
    assert "--no-build-isolation" in installer
    for name in (
        "Dockerfile.benchmark", "Dockerfile.reporting", "Dockerfile.openvla",
        "Dockerfile.openvla-oft",
    ):
        for line in _dockerfile(name).splitlines():
            if "python -m pip install" in line or "python -m pip wheel" in line:
                assert "--no-build-isolation" in line


def test_pinned_base_networkx_overlay_and_numba_cache_are_normalized():
    for name in ("Dockerfile.benchmark", "Dockerfile.openvla", "Dockerfile.openvla-oft"):
        text = _dockerfile(name)
        assert "/site-packages/networkx-*.dist-info" in text
        assert "networkx==3.4.2" in _lock(
            "benchmark.requirements.txt" if name == "Dockerfile.benchmark" else
            "openvla-oft.requirements.txt" if name == "Dockerfile.openvla-oft" else
            "openvla.requirements.txt"
        ).read_text(encoding="utf-8")
    assert "NUMBA_CACHE_DIR=/tmp/ovlab-numba-cache" in _dockerfile("Dockerfile.benchmark")
    assert "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6" in _dockerfile("Dockerfile.benchmark")


def test_benchmark_renderer_is_selected_at_runtime_and_libero_paths_are_portable():
    benchmark = _dockerfile("Dockerfile.benchmark")
    assert "MUJOCO_GL=egl" not in benchmark
    assert "MUJOCO_EGL_DEVICE_ID=0" not in benchmark
    assert "LIBERO_CONFIG_PATH=/etc/ovlab/libero" in benchmark
    assert "MPLCONFIGDIR=/tmp/ovlab-matplotlib" in benchmark
    assert "install -d -o root -g root -m 0755 /etc/ovlab/libero" in benchmark
    assert "deploy/config/libero/config.yaml /etc/ovlab/libero/config.yaml" in benchmark

    config = (ROOT / "deploy/config/libero/config.yaml").read_text(encoding="utf-8")
    assert "datasets: /datasets" in config
    assert "/opt/ovlab/external/libero/libero/libero" in config
    assert "/home/" not in config


def test_glfw_overlay_requires_a_display_and_removes_egl_device_configuration():
    for profile in ("openvla", "oft", "quic"):
        _, plan, services = _evaluation_services(profile, "glfw")
        benchmark = services[plan.benchmark_service]
        assert benchmark["environment"]["MUJOCO_GL"] == "glfw"
        assert "MUJOCO_EGL_DEVICE_ID" not in benchmark["environment"]
        assert benchmark["environment"]["DISPLAY"] == "${OVLAB_DISPLAY:-:0}"
        assert any(m.get("target") == "/tmp/.X11-unix" and m["read_only"] for m in benchmark["volumes"] if isinstance(m, dict))
        assert "ports" not in benchmark and "privileged" not in benchmark


def test_source_manifest_is_deterministic_and_records_dirty_state():
    command = ["python", "deploy/scripts/source_manifest.py", "--root", "."]
    first = subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout
    second = subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout
    assert first == second
    document = json.loads(first)
    assert re.fullmatch(r"[0-9a-f]{40}", document["repository_revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", document["source_content_sha256"])
    assert document["submodules"] == []
    assert {row["id"] for row in document["registered_sources"]} == {
        "libero", "plugin-openvla", "openvla", "openvla-oft", "openvla-quic", "quic-peft-vla",
    }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    source_manifest = runpy.run_path(str(ROOT / "deploy/scripts/source_manifest.py"))
    included_status = tuple(
        line for line in status.splitlines()
        if source_manifest["_included_status"](line)
    )
    assert document["dirty"] is bool(included_status)
    assert not any(row["path"].startswith("configs/") for row in document["files"])
    assert source_manifest["_included"]("code/apps/benchctl/pyproject.toml") is True
    assert source_manifest["_included"]("experiments/evaluation/new.yaml") is False
    assert source_manifest["_included"]("RUN.md") is False
    assert source_manifest["_included"]("code/tests/unit/example.py") is False
    assert source_manifest["_included"]("code/packages/ovlab-core/README.md") is True


def test_production_images_require_runtime_configuration_bundle():
    for name in (
        "Dockerfile.benchmark", "Dockerfile.reporting", "Dockerfile.openvla",
        "Dockerfile.openvla-oft",
    ):
        assert "COPY configs " not in _dockerfile(name)


def test_portable_config_change_does_not_invalidate_image_source_hash(tmp_path):
    repository = tmp_path / "repository"
    (repository / "code").mkdir(parents=True)
    (repository / "experiments/evaluation").mkdir(parents=True)
    source = repository / "code/runtime.py"
    config = repository / "experiments/evaluation/test.yaml"
    guide = repository / "RUN.md"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("name: first\n", encoding="utf-8")
    guide.write_text("Initial guide\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "ovlab@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "OVLAB Test"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    build_manifest = runpy.run_path(
        str(ROOT / "deploy/scripts/source_manifest.py")
    )["build_manifest"]

    registry = {"schema_version": "ovlab.plugins/v1", "sources": {}, "plugins": {}, "builds": {}}
    initial = build_manifest(repository, registry=registry)["source_content_sha256"]
    config.write_text("name: second\n", encoding="utf-8")
    config_only = build_manifest(repository, registry=registry)["source_content_sha256"]
    guide.write_text("Updated guide\n", encoding="utf-8")
    guide_only = build_manifest(repository, registry=registry)["source_content_sha256"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    dirty_source = build_manifest(repository, registry=registry)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "source change"], cwd=repository, check=True)
    committed_source = build_manifest(repository, registry=registry)

    assert config_only == initial
    assert guide_only == initial
    assert dirty_source["source_content_sha256"] != initial
    assert committed_source["source_content_sha256"] == dirty_source["source_content_sha256"]
    assert committed_source["repository_revision"] != dirty_source["repository_revision"]
    assert dirty_source["dirty"] is True
    assert committed_source["dirty"] is False


def test_dirty_submodule_manifest_distinguishes_local_source_bytes_and_deletions(tmp_path):
    source = tmp_path / "adapter.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=ovlab@example.invalid", "-c", "user.name=OVLAB Test", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=ovlab@example.invalid", "-c", "user.name=OVLAB Test", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    changes = runpy.run_path(str(ROOT / "deploy/scripts/source_manifest.py"))["_submodule_changes"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    first = changes(tmp_path)
    source.write_text("VALUE = 3\n", encoding="utf-8")
    second = changes(tmp_path)
    assert first[0]["sha256"] != second[0]["sha256"]
    source.unlink()
    assert changes(tmp_path) == [{"path": "adapter.py", "deleted": True}]


def test_quic_runtime_uses_dense_token_host_and_separate_adapter_mount():
    policy = _dockerfile("Dockerfile.openvla-quic")
    training = _dockerfile("Dockerfile.training-openvla-quic")
    lock = _lock("quic-openvla.requirements.txt").read_text()
    for dockerfile in (policy, training):
        assert "/opt/ovlab/external/quic-peft-vla" in dockerfile
        assert "COPY external/quic-peft-vla " in dockerfile
        assert "COPY external/openvla-quic " in dockerfile
        assert "COPY external/libero " not in dockerfile
        assert 'USER 10001:10001' in dockerfile
        assert "--require-hashes" in dockerfile
    assert "transformers==4.40.1" in lock
    assert "tensorflow-metadata==1.17.2 --hash=sha256:" in lock
    assert "protobuf==4.21.12 --hash=sha256:" in lock
    assert "transformers-openvla-oft" not in lock
    assert not any(name in lock.lower() for name in ("robosuite", "mujoco", "libero"))
    installer = (DOCKER / "install-ovlab-packages.sh").read_text()
    assert "quic-petf-vla" not in installer
    deployment, plan, services = _evaluation_services("quic")
    plugin = deployment._plugin(plan)
    mounts = plugin.implementation.runtime_mounts(deployment._policy_document(plan), ROOT, {})
    spec = deployment.service_spec(plan, mounts)
    policy = spec["services"]["policy-openvla-quic"]
    assert policy["environment"]["OVLAB_QUIC_ADAPTER_PATH"] == "/checkpoints/quic-adapter"
    assert any(isinstance(m, dict) and m["target"] == "/checkpoints/quic-adapter" and m["read_only"] for m in policy["volumes"])
    assert all(not isinstance(m, dict) or m["target"] != "/checkpoints/quic-adapter" for m in services["benchmark-quic"]["volumes"])


def test_container_root_revision_and_deployment_provenance_are_explicit(monkeypatch):
    monkeypatch.setenv("OVLAB_ROOT", str(ROOT))
    monkeypatch.setenv("OVLAB_REVISION", "a" * 40)
    app = OvlabApplication()
    assert app.repository_root == ROOT
    assert repository_revision(Path("/nonexistent")) == "a" * 40

    values = {
        "OVLAB_IMAGE_ROLE": "benchmark-libero",
        "OVLAB_IMAGE_DIGEST": "sha256:image",
        "OVLAB_SOURCE_MANIFEST_SHA256": "sha256:source",
        "IGNORED_SECRET": "must-not-leak",
    }
    assert app._deployment_provenance(values) == {
        "image_role": "benchmark-libero",
        "image_digest": "sha256:image",
        "source_manifest_sha256": "sha256:source",
    }
    first = app._deployment_provenance({"OVLAB_IMAGE_DIGEST": "sha256:first"})
    second = app._deployment_provenance({"OVLAB_IMAGE_DIGEST": "sha256:second"})
    assert first != second
    assert app._deployment_provenance({"OVLAB_CONFIG_BUNDLE_SHA256": "sha256:bundle"}) == {
        "config_bundle_sha256": "sha256:bundle"
    }


def test_test_provider_is_excluded_from_every_production_dockerfile():
    for name in ("Dockerfile.benchmark", "Dockerfile.openvla", "Dockerfile.openvla-oft"):
        assert "deploy/smoke" not in _dockerfile(name)
    smoke = _dockerfile("Dockerfile.smoke")
    assert "deploy/smoke/sitecustomize.py" in smoke
    assert 'io.github.ovlab.production="false"' in smoke


def test_image_metadata_has_no_stale_institutional_namespace():
    dockerfiles = tuple(path for path in DOCKER.glob("Dockerfile.*") if path.is_file())
    assert dockerfiles
    for path in dockerfiles:
        assert "cz.cvut.ovlab" not in path.read_text(encoding="utf-8")


def test_internal_python_packages_identify_the_project_author():
    project_files = tuple((ROOT / "code").glob("**/pyproject.toml"))
    assert project_files
    for path in project_files:
        text = path.read_text(encoding="utf-8")
        assert 'authors = [{ name = "Jiří Konečný" }]' in text
