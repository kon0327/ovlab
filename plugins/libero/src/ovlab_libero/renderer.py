"""Explicit process-local LIBERO renderer selection before native imports."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import os
import sys

from .errors import LiberoConfigurationError, LiberoDependencyError


class LiberoRendererBackend(str, Enum):
    EGL = "egl"
    GLFW = "glfw"


@dataclass(frozen=True, slots=True)
class LiberoRendererSettings:
    requested_backend: LiberoRendererBackend = LiberoRendererBackend.EGL
    resolved_backend: LiberoRendererBackend = LiberoRendererBackend.EGL
    device_id: int | None = None
    override_source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.requested_backend, LiberoRendererBackend):
            raise LiberoConfigurationError("requested renderer backend must be egl or glfw")
        if not isinstance(self.resolved_backend, LiberoRendererBackend):
            raise LiberoConfigurationError("resolved renderer backend must be egl or glfw")
        if self.device_id is not None and (type(self.device_id) is not int or self.device_id < 0):
            raise LiberoConfigurationError("renderer device_id must be a non-negative integer or None")
        if self.resolved_backend is LiberoRendererBackend.GLFW and self.device_id is not None:
            raise LiberoConfigurationError("renderer device_id is applicable only to EGL")

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend.value,
            "resolved_backend": self.resolved_backend.value,
            "device_id": self.device_id,
            "override_source": self.override_source,
        }


def resolve_renderer_settings(
    backend: str,
    device_id: int | None,
    environment: Mapping[str, str] | None = None,
) -> LiberoRendererSettings:
    try:
        requested = LiberoRendererBackend(backend)
    except ValueError as exc:
        raise LiberoConfigurationError("renderer backend must be one of: egl, glfw") from exc
    values = os.environ if environment is None else environment
    override = values.get("MUJOCO_GL")
    try:
        resolved = requested if override is None else LiberoRendererBackend(override)
    except ValueError as exc:
        raise LiberoConfigurationError("MUJOCO_GL override must be one of: egl, glfw") from exc
    source = None if override is None else "environment:MUJOCO_GL"
    resolved_device = device_id
    if resolved is LiberoRendererBackend.EGL and "MUJOCO_EGL_DEVICE_ID" in values:
        raw = values["MUJOCO_EGL_DEVICE_ID"]
        if not raw.isdigit():
            raise LiberoConfigurationError("MUJOCO_EGL_DEVICE_ID must be a non-negative integer")
        resolved_device = int(raw)
        source = "environment:MUJOCO_GL,MUJOCO_EGL_DEVICE_ID" if override is not None else "environment:MUJOCO_EGL_DEVICE_ID"
    if resolved is LiberoRendererBackend.GLFW:
        resolved_device = None
    return LiberoRendererSettings(requested, resolved, resolved_device, source)


def _imported_graphics_backend(modules: Mapping[str, object]) -> LiberoRendererBackend | None:
    egl = any(name == "mujoco.egl" or name.startswith("OpenGL.EGL") for name in modules)
    glfw = any(name == "mujoco.glfw" or name == "glfw" or name.startswith("glfw.") for name in modules)
    if egl and glfw:
        raise LiberoDependencyError("both EGL and GLFW graphics modules are already imported")
    if egl:
        return LiberoRendererBackend.EGL
    if glfw:
        return LiberoRendererBackend.GLFW
    return None


class LiberoRendererRuntime:
    def __init__(self, settings: LiberoRendererSettings) -> None:
        self.settings = settings
        self.effective_settings = settings
        self._previous: dict[str, str | None] | None = None

    def activate(self) -> None:
        if self._previous is not None:
            return
        if "MUJOCO_GL" in os.environ or "MUJOCO_EGL_DEVICE_ID" in os.environ:
            self.effective_settings = resolve_renderer_settings(
                self.settings.requested_backend.value,
                self.settings.device_id,
                os.environ,
            )
        imported = _imported_graphics_backend(sys.modules)
        native_loaded = any(
            name == "mujoco" or name == "robosuite" or name.startswith("libero.libero")
            for name in sys.modules
        )
        if imported is not None and imported is not self.effective_settings.resolved_backend:
            raise LiberoDependencyError(
                f"requested {self.effective_settings.resolved_backend.value} renderer conflicts with already imported "
                f"{imported.value} graphics modules"
            )
        if native_loaded and imported is None:
            raise LiberoDependencyError(
                "MuJoCo, Robosuite, or LIBERO was imported before renderer selection; "
                "the active graphics backend cannot be verified"
            )
        self._previous = {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        }
        os.environ["MUJOCO_GL"] = self.effective_settings.resolved_backend.value
        if self.effective_settings.resolved_backend is LiberoRendererBackend.EGL:
            if self.effective_settings.device_id is None:
                os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
            else:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.effective_settings.device_id)
        else:
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

    def close(self) -> None:
        if self._previous is None:
            return
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._previous = None


def detect_renderer() -> dict[str, object]:
    """Query the active context after native environment creation."""
    try:
        from OpenGL import GL

        def value(name):
            result = GL.glGetString(name)
            return None if result is None else result.decode("utf-8", errors="replace")

        return {
            "vendor": value(GL.GL_VENDOR),
            "renderer": value(GL.GL_RENDERER),
            "version": value(GL.GL_VERSION),
        }
    except Exception as exc:
        return {"vendor": None, "renderer": None, "version": None, "detection_error": type(exc).__name__}
