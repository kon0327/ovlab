"""Explicit configuration origins and their portable frozen-bundle addresses.

``ovlab://`` selects OVLAB-owned defaults, ``plugin://`` selects a registered
project, and plain references select user configurations. Relative inheritance
stays inside its owner; crossing origins requires an explicit URI.
"""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlsplit

from .errors import ConfigReferenceError


class ConfigReferences:
    def __init__(self, repository_root, config_root=None):
        self.repository_root = Path(repository_root).resolve()
        self.config_root = Path(config_root or self.repository_root / "configs").resolve()

    @property
    def is_frozen(self):
        return (self.config_root / ".ovlab-bundle.json").is_file()

    @property
    def builtin_root(self):
        return self.config_root / "ovlab" if self.is_frozen else self.repository_root / "config"

    def plugin_roots(self):
        from .policy_plugins import plugin_config_roots
        roots = {key: Path(value).resolve() for key, value in plugin_config_roots(self.repository_root).items()}
        frozen = self.config_root / "plugins/configs"
        if self.is_frozen and frozen.is_dir():
            roots.update({path.name: path.resolve() for path in frozen.iterdir() if path.is_dir()})
        return roots

    @staticmethod
    def _uri(reference, scheme):
        parsed = urlsplit(reference)
        if (parsed.scheme != scheme or not parsed.netloc or parsed.query or parsed.fragment
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", parsed.netloc)):
            raise ConfigReferenceError(f"invalid {scheme} configuration reference: {reference}")
        relative = parsed.path.lstrip("/")
        if ".." in Path(relative).parts or "\\" in relative or "%" in relative:
            raise ConfigReferenceError(f"invalid {scheme} configuration path: {reference}")
        return parsed.netloc, relative

    @classmethod
    def plugin_parts(cls, reference):
        name, relative = cls._uri(reference, "plugin")
        if not relative:
            raise ConfigReferenceError(f"invalid plugin configuration path: {reference}")
        return name, relative

    @classmethod
    def builtin_parts(cls, reference):
        name, relative = cls._uri(reference, "ovlab")
        return name + ("/" + relative if relative else "")

    def source(self, reference):
        reference = str(reference)
        if reference.startswith("plugin:"):
            from .policy_plugins import plugin_config_path
            name, relative = self.plugin_parts(reference)
            frozen = self.config_root / "plugins/configs" / name
            if self.is_frozen and frozen.is_dir():
                return self.inside(frozen / relative, frozen, "plugin configuration")
            return Path(plugin_config_path(reference, self.repository_root)).resolve()
        if reference.startswith("ovlab:"):
            return self.inside(self.builtin_root / self.builtin_parts(reference), self.builtin_root, "OVLAB configuration")
        if "://" in reference:
            raise ConfigReferenceError(f"unknown configuration reference scheme: {reference}")
        path = Path(reference)
        if path.is_absolute():
            raise ConfigReferenceError(f"configuration reference must be relative: {reference}")
        if path.parts[:2] == ("plugins", "configs") and len(path.parts) > 3:
            return self.source("plugin://" + path.parts[2] + "/" + "/".join(path.parts[3:]))
        if path.parts and path.parts[0] == "ovlab":
            return self.source("ovlab://" + "/".join(path.parts[1:]))
        reference = reference.removeprefix("configs/")
        path = Path(reference)
        if path.parts and path.parts[0] in {"experiments", "examples"}:
            frozen = self.config_root / path
            if self.is_frozen and frozen.is_file():
                return self.inside(frozen, self.config_root / path.parts[0], "experiment")
            return self.inside(self.repository_root / path, self.repository_root / path.parts[0], "experiment")
        return self.inside(self.config_root / path, self.config_root, "user configuration")

    @staticmethod
    def inside(path, root, label):
        path, root = Path(path).resolve(), Path(root).resolve()
        if not path.is_relative_to(root):
            raise ConfigReferenceError(f"{label} escapes its configuration root: {path}")
        return path

    def _frozen_reference(self, path):
        frozen = self.config_root / "plugins/configs"
        if self.is_frozen and path.is_relative_to(frozen):
            parts = path.relative_to(frozen).parts
            if len(parts) > 1:
                return parts[0], frozen / parts[0], "/".join(parts[1:])
        return None

    def origin(self, path):
        path = Path(path).resolve()
        frozen = self._frozen_reference(path)
        if frozen:
            return frozen[1]
        if path.is_relative_to(self.builtin_root):
            return self.builtin_root
        for root in (self.config_root, self.repository_root / "experiments", self.repository_root / "examples"):
            if path.is_relative_to(root):
                return root
        for root in self.plugin_roots().values():
            if path.is_relative_to(root):
                return root
        raise ConfigReferenceError(f"configuration is outside registered roots: {path}")

    def reference(self, path):
        path = Path(path).resolve()
        frozen = self._frozen_reference(path)
        if frozen:
            name, _, relative = frozen
            return f"plugin://{name}/{relative}"
        if path.is_relative_to(self.builtin_root):
            return "ovlab://" + path.relative_to(self.builtin_root).as_posix()
        for name in ("experiments", "examples"):
            if path.is_relative_to(self.repository_root / name):
                return path.relative_to(self.repository_root).as_posix()
        if path.is_relative_to(self.config_root):
            return path.relative_to(self.config_root).as_posix()
        for name, root in self.plugin_roots().items():
            if path.is_relative_to(root):
                return f"plugin://{name}/{path.relative_to(root).as_posix()}"
        raise ConfigReferenceError(f"configuration is outside registered roots: {path}")

    def portable(self, reference):
        reference = str(reference)
        if reference.startswith("plugin:"):
            name, relative = self.plugin_parts(reference)
            return f"plugins/configs/{name}/{relative}"
        if reference.startswith("ovlab:"):
            return "ovlab/" + self.builtin_parts(reference)
        return self.bundle_path(self.source(reference)) if not Path(reference).is_absolute() else self.bundle_path(reference)

    def bundle_path(self, path):
        reference = self.reference(path)
        if reference.startswith("plugin:"):
            name, relative = self.plugin_parts(reference)
            return f"plugins/configs/{name}/{relative}"
        if reference.startswith("ovlab:"):
            return "ovlab/" + self.builtin_parts(reference)
        return reference

    def parent(self, path, reference):
        if not isinstance(reference, str) or not reference:
            raise ConfigReferenceError(f"{path}.extends must be a non-empty relative path or configuration URI")
        if reference.startswith(("plugin:", "ovlab:")):
            return self.source(reference)
        if Path(reference).is_absolute() or "://" in reference:
            raise ConfigReferenceError(f"{path}.extends must be relative or an explicit configuration URI")
        return self.inside(Path(path).parent / reference, self.origin(path), "extends")
