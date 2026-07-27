"""Dependency-light renderer selection and import-order contracts."""

import os
import sys

import pytest

from helpers.contexts import make_run_context
from helpers.fake_libero import FakeLiberoBackend
from ovlab_benchmarks.libero import (
    LiberoAdapterSettings,
    LiberoBenchmarkAdapter,
    LiberoConfigurationError,
    LiberoDependencyError,
    LiberoRendererBackend,
    LiberoRendererRuntime,
    LiberoRendererSettings,
    resolve_renderer_settings,
)


def test_egl_and_glfw_environment_mapping_and_restoration(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    egl = LiberoRendererRuntime(LiberoRendererSettings(
        LiberoRendererBackend.EGL, LiberoRendererBackend.EGL, 2
    ))
    egl.activate()
    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "2"
    egl.close()
    assert "MUJOCO_GL" not in os.environ
    assert "MUJOCO_EGL_DEVICE_ID" not in os.environ

    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "9")
    glfw = LiberoRendererRuntime(LiberoRendererSettings(
        LiberoRendererBackend.GLFW, LiberoRendererBackend.GLFW, None
    ))
    glfw.activate()
    assert os.environ["MUJOCO_GL"] == "glfw"
    assert "MUJOCO_EGL_DEVICE_ID" not in os.environ
    glfw.close()
    assert "MUJOCO_GL" not in os.environ
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "9"


def test_diagnostic_environment_overrides_configuration() -> None:
    glfw = resolve_renderer_settings(
        "egl", 3, {"MUJOCO_GL": "glfw", "MUJOCO_EGL_DEVICE_ID": "7"}
    )
    assert glfw.requested_backend is LiberoRendererBackend.EGL
    assert glfw.resolved_backend is LiberoRendererBackend.GLFW
    assert glfw.device_id is None
    egl = resolve_renderer_settings(
        "glfw", None, {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "4"}
    )
    assert egl.requested_backend is LiberoRendererBackend.GLFW
    assert egl.resolved_backend is LiberoRendererBackend.EGL
    assert egl.device_id == 4


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"MUJOCO_GL": "osmesa"}, "MUJOCO_GL override"),
        ({"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "gpu0"}, "MUJOCO_EGL_DEVICE_ID"),
        ({"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "-1"}, "MUJOCO_EGL_DEVICE_ID"),
    ],
)
def test_invalid_diagnostic_renderer_overrides_fail_clearly(environment, message) -> None:
    with pytest.raises(LiberoConfigurationError, match=message):
        resolve_renderer_settings("egl", 0, environment)


def test_conflicting_already_imported_graphics_backend_fails(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mujoco.glfw", object())
    runtime = LiberoRendererRuntime(LiberoRendererSettings(
        LiberoRendererBackend.EGL, LiberoRendererBackend.EGL, 0
    ))
    with pytest.raises(LiberoDependencyError, match="conflicts with already imported glfw"):
        runtime.activate()


def test_renderer_is_applied_before_default_backend_construction(monkeypatch) -> None:
    import ovlab_benchmarks.libero.adapter as adapter_module

    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    backend = FakeLiberoBackend()
    constructed = []

    def backend_factory():
        assert os.environ["MUJOCO_GL"] == "egl"
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "5"
        assert "mujoco" not in sys.modules
        constructed.append(True)
        return backend

    monkeypatch.setattr(adapter_module, "PinnedLiberoBackend", backend_factory)
    settings = LiberoAdapterSettings(
        camera_width=5,
        camera_height=4,
        initialization_settling_steps=0,
        renderer=LiberoRendererSettings(
            LiberoRendererBackend.EGL, LiberoRendererBackend.EGL, 5
        ),
    )
    adapter = LiberoBenchmarkAdapter(settings)
    adapter.initialize(make_run_context())
    assert constructed == [True]
    adapter.close()
    assert "MUJOCO_GL" not in os.environ
    assert "MUJOCO_EGL_DEVICE_ID" not in os.environ
