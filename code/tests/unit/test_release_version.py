"""Regression checks for the unified OVLAB monorepo release version."""

from __future__ import annotations

from pathlib import Path
import re

from ovlab_core.contracts import OVLAB_CONTRACT_VERSION, OVLAB_VERSION


REPOSITORY = Path(__file__).resolve().parents[3]
SEMANTIC_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def _project_version(pyproject: Path) -> str:
    match = re.search(
        r"(?ms)^\[project\]\s*$.*?^version\s*=\s*\"([^\"]+)\"\s*$",
        pyproject.read_text(encoding="utf-8"),
    )
    assert match is not None, f"missing [project] version in {pyproject}"
    return match.group(1)


def test_release_version_is_semantic_and_shared_by_public_surfaces() -> None:
    version = (REPOSITORY / "VERSION").read_text(encoding="utf-8").strip()
    assert SEMANTIC_VERSION.fullmatch(version)
    assert OVLAB_VERSION == version

    versioning = (
        REPOSITORY / "code/apps/benchctl/src/ovlab_benchctl/versioning.py"
    ).read_text(encoding="utf-8")
    assert f'CLI_VERSION = "{version}"' in versioning

    pyprojects = sorted((REPOSITORY / "code").glob("**/pyproject.toml"))
    assert len(pyprojects) == 12
    assert {_project_version(path) for path in pyprojects} == {version}


def test_internal_distribution_requirements_do_not_accept_an_older_release() -> None:
    version = (REPOSITORY / "VERSION").read_text(encoding="utf-8").strip()
    for pyproject in sorted((REPOSITORY / "code").glob("**/pyproject.toml")):
        source = pyproject.read_text(encoding="utf-8")
        project_name = re.search(r'(?m)^name\s*=\s*"(ovlab-[^"]+)"$', source)
        assert project_name is not None
        requirements = re.findall(r'"(ovlab-[A-Za-z0-9-]+[^\"]*)"', source)
        for requirement in requirements:
            if requirement == project_name.group(1):
                continue
            assert f">={version}" in requirement, (
                f"{pyproject} permits a mismatched internal release: {requirement}"
            )


def test_container_images_publish_the_same_release_version() -> None:
    version = (REPOSITORY / "VERSION").read_text(encoding="utf-8").strip()
    dockerfiles = sorted((REPOSITORY / "deploy/docker").glob("Dockerfile.*"))
    assert dockerfiles
    for dockerfile in dockerfiles:
        source = dockerfile.read_text(encoding="utf-8")
        assert f"ARG OVLAB_VERSION={version}" in source
        assert 'org.opencontainers.image.version="$OVLAB_VERSION"' in source

    build_script = (REPOSITORY / "deploy/scripts/build-images.sh").read_text(encoding="utf-8")
    assert "version=\"$(tr -d '[:space:]' < VERSION)\"" in build_script
    assert '--build-arg "OVLAB_VERSION=$version"' in build_script


def test_product_release_does_not_implicitly_bump_compatibility_contract() -> None:
    assert OVLAB_VERSION == "0.2.0"
    assert OVLAB_CONTRACT_VERSION == "0.1.0"
