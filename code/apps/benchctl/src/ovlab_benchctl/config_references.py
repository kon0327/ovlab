"""Address repository and registered project configuration without leaking host paths.

Components use configs-relative references or ``plugin://project/path.yaml``.
Only explicit plugin URIs may cross configuration origins; relative inheritance
is bounded to the directory tree that owns the referring document.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from .errors import ConfigReferenceError


class ConfigReferences:
    def __init__(self, repository_root, config_root=None):
        self.repository_root = Path(repository_root).resolve()
        self.config_root = Path(config_root or self.repository_root / "configs").resolve()

    def plugin_roots(self):
        from .policy_plugins import plugin_config_roots
        roots = {key: Path(value).resolve() for key, value in plugin_config_roots(self.repository_root).items()}
        frozen = self.config_root / "plugins/configs"
        if frozen.is_dir():
            roots.update({path.name: path.resolve() for path in frozen.iterdir() if path.is_dir()})
        return roots

    @staticmethod
    def plugin_parts(reference):
        parsed = urlsplit(reference)
        if parsed.scheme != "plugin" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ConfigReferenceError(f"invalid plugin configuration reference: {reference}")
        relative = parsed.path.lstrip("/")
        if not relative or ".." in Path(relative).parts or "\\" in relative or "%" in relative:
            raise ConfigReferenceError(f"invalid plugin configuration path: {reference}")
        return parsed.netloc, relative

    def source(self, reference):
        reference = str(reference)
        from .policy_plugins import plugin_config_alias, plugin_config_path
        if reference.startswith("plugin:"):
            name, relative = self.plugin_parts(reference)
            frozen = self.config_root / "plugins/configs" / name
            if frozen.is_dir():
                return self.inside(frozen / relative, frozen, "plugin configuration")
            return Path(plugin_config_path(reference, self.repository_root)).resolve()
        path = Path(reference)
        if path.is_absolute():
            raise ConfigReferenceError(f"configuration reference must be relative: {reference}")
        if path.parts[:2] == ("plugins", "configs") and len(path.parts) > 3:
            return self.source("plugin://" + path.parts[2] + "/" + "/".join(path.parts[3:]))
        reference = reference.removeprefix("configs/")
        alias = plugin_config_alias(reference, self.repository_root) if not (self.config_root / reference).is_file() else None
        if alias:
            return self.source(alias)
        path = Path(reference)
        root = self.repository_root if path.parts and path.parts[0] == "experiments" and (self.repository_root / "experiments").is_dir() else self.config_root
        return self.inside(root / path, root, "configuration")

    @staticmethod
    def inside(path, root, label):
        path, root = Path(path).resolve(), Path(root).resolve()
        if not path.is_relative_to(root):
            raise ConfigReferenceError(f"{label} escapes its configuration root: {path}")
        return path

    def _frozen_reference(self, path):
        frozen = self.config_root / "plugins/configs"
        if path.is_relative_to(frozen):
            parts = path.relative_to(frozen).parts
            if len(parts) > 1:
                return parts[0], frozen / parts[0], "/".join(parts[1:])
        return None

    def origin(self, path):
        path = Path(path).resolve()
        frozen = self._frozen_reference(path)
        if frozen:
            return frozen[1]
        for root in (self.config_root, self.repository_root / "experiments"):
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
        if path.is_relative_to(self.repository_root / "experiments"):
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
        return self.bundle_path(self.source(reference)) if not Path(reference).is_absolute() else self.bundle_path(reference)

    def bundle_path(self, path):
        reference = self.reference(path)
        if reference.startswith("plugin:"):
            name, relative = self.plugin_parts(reference)
            return f"plugins/configs/{name}/{relative}"
        return reference

    def parent(self, path, reference):
        if not isinstance(reference, str) or not reference:
            raise ConfigReferenceError(f"{path}.extends must be a non-empty relative path or plugin URI")
        if reference.startswith("plugin:"):
            return self.source(reference)
        if Path(reference).is_absolute():
            raise ConfigReferenceError(f"{path}.extends must be relative")
        return self.inside(Path(path).parent / reference, self.origin(path), "extends")
