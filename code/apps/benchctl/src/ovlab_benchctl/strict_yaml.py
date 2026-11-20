"""Core error boundary for the SDK's public strict configuration parser."""
from ovlab_policy_sdk import strict_yaml as _yaml
from .errors import StrictYamlError


def loads(text, *, source="<string>"):
    try:
        return _yaml.loads(text, source=source)
    except _yaml.StrictYamlError as exc:
        raise StrictYamlError(str(exc)) from exc


def load(path):
    try:
        return _yaml.load(path)
    except _yaml.StrictYamlError as exc:
        raise StrictYamlError(str(exc)) from exc


dumps = _yaml.dumps
