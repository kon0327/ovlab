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
COMPOSE = (ROOT / "deploy/compose/compose.yaml").read_text(encoding="utf-8")
GLFW_COMPOSE = (ROOT / "deploy/compose/compose.glfw.yaml").read_text(encoding="utf-8")


def _dockerfile(name: str) -> str:
    return (DOCKER / name).read_text(encoding="utf-8")


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
        assert "cz.cvut.ovlab.dependency-lock.sha256" in text
        assert 'org.opencontainers.image.licenses="NOASSERTION"' in text
        assert 'cz.cvut.ovlab.build-target="production"' in text
        expected_contract = (
            "resolved-checkpoint-quantization-config-bundle-v2"
            if name == "Dockerfile.openvla"
            else "canonical-readonly-reporting-v1"
            if name == "Dockerfile.reporting"
            else "resolved-checkpoint-config-bundle-v2"
        )
        assert f'cz.cvut.ovlab.deployment.contract="{expected_contract}"' in text


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
    assert "openvla-quic" not in COMPOSE.lower()


def test_compose_is_socket_only_offline_and_least_privilege():
    forbidden = ("ports:", "privileged:", "network_mode: host", "/var/run/docker.sock")
    assert not any(value in COMPOSE for value in forbidden)
    assert COMPOSE.count("network_mode: none") == 1
    assert COMPOSE.count("<<: *policy-security") == 6
    assert "read_only: true" in COMPOSE
    assert COMPOSE.count("cap_drop: [ALL]") >= 1
    assert "HF_HUB_OFFLINE: \"1\"" in COMPOSE
    assert "TRANSFORMERS_OFFLINE: \"1\"" in COMPOSE
    assert "MUJOCO_GL: egl" in COMPOSE
    assert "condition: service_healthy" in COMPOSE
    assert "OVLAB_DEPLOYMENT_MANIFEST_SHA256" in COMPOSE
    assert "OVLAB_CONTAINER_RUNTIME_VERSION" in COMPOSE
    assert "OVLAB_MOUNT_CONTRACT" in COMPOSE
    assert COMPOSE.count("OVLAB_DEPLOYMENT_PROFILE: ${OVLAB_DEPLOYMENT_PROFILE:-") == 4
    assert COMPOSE.count(
        "OVLAB_EXECUTION_PROFILE: ${OVLAB_EXECUTION_PROFILE:?configuration bundle resolver"
    ) == 4
    assert "service, health, --socket, /run/ovlab/policy.sock" in COMPOSE
    assert COMPOSE.count("source: ${OVLAB_RESOLVED_CHECKPOINT_PATH:?") == 2
    assert "../../checkpoints" not in COMPOSE
    assert COMPOSE.count("target: /checkpoints/resolved/${OVLAB_RESOLVED_CHECKPOINT_ID}") == 2
    assert "OVLAB_RESOLVED_CHECKPOINT_CONTAINER_PATH" in COMPOSE
    assert COMPOSE.count("source: ${OVLAB_DATASETS_PATH:?") == 2
    assert "../../datasets" not in COMPOSE
    assert "target: /datasets\n        read_only: true" in COMPOSE
    assert COMPOSE.count("${OVLAB_EXPERIMENT_CONFIG:?configuration bundle resolver") == 4
    assert COMPOSE.count("OVLAB_CONFIG_BUNDLE_SHA256: ${OVLAB_CONFIG_BUNDLE_SHA256:?") == 4
    assert "source: ${OVLAB_CONFIG_BUNDLE_PATH:?" in COMPOSE
    assert "target: /opt/ovlab/configs" in COMPOSE
    assert "config-bundle-ro" in COMPOSE


def test_host_artifact_mounts_separate_canonical_runs_from_derived_outputs():
    assert "OVLAB_RUNS_PATH" not in COMPOSE
    assert COMPOSE.count("source: ${OVLAB_RUNS_ROOT:-../../../ovlab-data/runs}") == 4
    assert COMPOSE.count("target: /var/lib/ovlab/runs") == 4
    assert COMPOSE.count("--output-root, /var/lib/ovlab/runs") == 2
    assert COMPOSE.count("target: /var/lib/ovlab/runs\n        read_only: true") == 2
    assert COMPOSE.count("source: ${OVLAB_DERIVED_ROOT:-../../../ovlab-data/derived}") == 1
    assert COMPOSE.count("target: /var/lib/ovlab/derived") == 1
    assert "OVLAB_MOUNT_CONTRACT: configs-ro,runs-ro,derived-rw,exports-rw" in COMPOSE
    assert COMPOSE.count("source: ${OVLAB_CONFIGS_ROOT:-../../configs}") == 2
    assert COMPOSE.count("target: /opt/ovlab/configs\n        read_only: true") == 2
    assert COMPOSE.count("source: ${OVLAB_EXPORTS_ROOT:-../../../ovlab-data/exports}") == 2
    assert COMPOSE.count("target: /var/lib/ovlab/exports") == 2
    assert COMPOSE.count("OVLAB_MOUNT_CONTRACT: configs-ro,runs-ro,derived-rw,exports-rw") == 1
    assert COMPOSE.count("OVLAB_MOUNT_CONTRACT: configs-ro,runs-ro,exports-rw,export-spec-ro") == 1
    assert COMPOSE.count("${OVLAB_HOST_ARTIFACT_GID:-1000}") == 4
    reporting = COMPOSE.split("  reporting:\n", 1)[1].split("\nvolumes:\n", 1)[0]
    assert "profiles: [reporting]" in reporting
    assert "image: ${OVLAB_REPORTING_IMAGE:-ovlab-reporting:local}" in reporting
    assert "report\n      - publish" in reporting
    assert "target: /var/lib/ovlab/runs\n        read_only: true" in reporting
    assert "target: /var/lib/ovlab/derived" in reporting
    assert "target: /var/lib/ovlab/exports" in reporting
    assert "gpus:" not in reporting
    assert "rpc-openvla" not in reporting and "rpc-oft" not in reporting
    export = COMPOSE.split("  export:\n", 1)[1].split("\nvolumes:\n", 1)[0]
    assert "profiles: [export]" in export
    assert "image: ${OVLAB_REPORTING_IMAGE:-ovlab-reporting:local}" in export
    assert "OVLAB_MOUNT_CONTRACT: configs-ro,runs-ro,exports-rw,export-spec-ro" in export
    assert "target: /var/lib/ovlab/runs\n        read_only: true" in export
    assert "target: /etc/ovlab/export.yaml\n        read_only: true" in export
    assert "gpus:" not in export and "rpc-openvla" not in export and "rpc-oft" not in export
    assert "runs:" not in COMPOSE.split("\nvolumes:\n", 1)[1]

    profile = (ROOT / "deploy/config/container-profile.yaml").read_text(encoding="utf-8")
    assert "runs_root: /var/lib/ovlab/runs" in profile
    benchmark = _dockerfile("Dockerfile.benchmark")
    assert "install -d -o 10001 -g 10001 -m 0700 /run/ovlab /var/lib/ovlab/runs" in benchmark
    assert COMPOSE.count("OVLAB_POSTPROCESSING_MODE: external") == 2
    benchmark_services = COMPOSE.split("  benchmark-openvla:\n", 1)[1].split("  policy-openvla-oft:\n", 1)[0] + COMPOSE.split("  benchmark-oft:\n", 1)[1].split("  reporting:\n", 1)[0]
    assert "/var/lib/ovlab/derived" not in benchmark_services
    assert "/var/lib/ovlab/exports" not in benchmark_services


def test_hash_locks_and_immutable_vcs_revision_are_checked_in():
    locks = ROOT / "deploy/locks"
    for role in ("benchmark", "reporting", "openvla", "openvla-oft", "flash-attn"):
        pylock = (locks / f"{role}.pylock.toml").read_text(encoding="utf-8")
        requirements = (locks / f"{role}.requirements.txt").read_text(encoding="utf-8")
        assert 'lock-version = "1.0"' in pylock
        assert 'sha256 = "' in pylock
        assert "--hash=sha256:" in requirements
    assert "rich==15.0.0" in (locks / "openvla.requirements.txt").read_text(encoding="utf-8")
    openvla_lock = (locks / "openvla.requirements.txt").read_text(encoding="utf-8")
    assert (
        "bitsandbytes==0.43.1 "
        "--hash=sha256:a81c826d576d6d691c7b4a7491c8fdc0f37f769795d6ca2e54afa605d2c260a3"
    ) in openvla_lock
    assert "bitsandbytes=0.43.1" in _dockerfile("Dockerfile.openvla")
    assert "rich==15.0.0" in (locks / "openvla-oft.requirements.txt").read_text(encoding="utf-8")
    assert "imageio-ffmpeg==0.6.0" in (locks / "benchmark.requirements.txt").read_text(encoding="utf-8")
    assert "jinja2==3.1.6" in (locks / "benchmark.requirements.txt").read_text(encoding="utf-8")
    assert "markupsafe==3.0.3" in (locks / "benchmark.requirements.txt").read_text(encoding="utf-8")
    oft = (locks / "openvla-oft.pylock.toml").read_text(encoding="utf-8")
    assert oft.count('commit-id = "bc339d9ad707454c0c115970db43c260067c61ab"') == 1
    assert oft.count('commit-id = "040105d256bd28866cc6620621a3d5f7b6b91b46"') == 1
    oft_input = (locks / "openvla-oft.in").read_text(encoding="utf-8")
    assert "tensorflow==2.15.0" in oft_input
    assert "tensorflow-datasets==4.9.3" in oft_input
    assert "tensorflow-metadata==1.15.0" in oft_input
    assert "jsonlines==4.0.0" in oft_input
    assert "wandb==0.16.6" in oft_input
    assert "dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46" in oft_input
    assert "040105d256bd28866cc6620621a3d5f7b6b91b46" in _dockerfile("Dockerfile.openvla-oft")
    build_script = (ROOT / "deploy/scripts/build-images.sh").read_text(encoding="utf-8")
    assert '--secret "id=ovlab_source_manifest,src=$manifest"' in build_script
    assert "OVLAB_SOURCE_MANIFEST_B64" not in build_script


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
        assert "networkx==3.4.2" in (ROOT / "deploy/locks" / (
            "benchmark.requirements.txt" if name == "Dockerfile.benchmark" else
            "openvla-oft.requirements.txt" if name == "Dockerfile.openvla-oft" else
            "openvla.requirements.txt"
        )).read_text(encoding="utf-8")
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
    assert GLFW_COMPOSE.count("OVLAB_EXECUTION_PROFILE: profiles/libero-playground-glfw.yaml") == 2
    assert GLFW_COMPOSE.count("MUJOCO_GL: glfw") == 2
    assert GLFW_COMPOSE.count("MUJOCO_EGL_DEVICE_ID: !reset null") == 2
    assert GLFW_COMPOSE.count("target: /tmp/.X11-unix") == 2
    assert "privileged:" not in GLFW_COMPOSE
    assert "ports:" not in GLFW_COMPOSE


def test_source_manifest_is_deterministic_and_records_dirty_state():
    command = ["python", "deploy/scripts/source_manifest.py", "--root", "."]
    first = subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout
    second = subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout
    assert first == second
    document = json.loads(first)
    assert re.fullmatch(r"[0-9a-f]{40}", document["repository_revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", document["source_content_sha256"])
    assert {row["path"] for row in document["submodules"]} == {
        "external/libero", "external/openvla", "external/openvla-oft", "external/openvla-quic"
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
    assert source_manifest["_included"]("configs/experiments/new.yaml") is False


def test_production_images_require_runtime_configuration_bundle():
    for name in (
        "Dockerfile.benchmark", "Dockerfile.reporting", "Dockerfile.openvla",
        "Dockerfile.openvla-oft",
    ):
        assert "COPY configs " not in _dockerfile(name)


def test_portable_config_change_does_not_invalidate_image_source_hash(tmp_path):
    repository = tmp_path / "repository"
    (repository / "code").mkdir(parents=True)
    (repository / "configs/experiments").mkdir(parents=True)
    source = repository / "code/runtime.py"
    config = repository / "configs/experiments/test.yaml"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("name: first\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "ovlab@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "OVLAB Test"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    build_manifest = runpy.run_path(
        str(ROOT / "deploy/scripts/source_manifest.py")
    )["build_manifest"]

    initial = build_manifest(repository)["source_content_sha256"]
    config.write_text("name: second\n", encoding="utf-8")
    config_only = build_manifest(repository)["source_content_sha256"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    code_changed = build_manifest(repository)["source_content_sha256"]

    assert config_only == initial
    assert code_changed != initial


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
    assert 'cz.cvut.ovlab.production="false"' in smoke
