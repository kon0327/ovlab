"""Small strict YAML subset used by versioned OVLAB configuration files."""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from .errors import StrictYamlError


@dataclass(frozen=True, slots=True)
class _Line:
    number: int
    indent: int
    text: str


_INTEGER = re.compile(r"[-+]?(?:0|[1-9][0-9]*)\Z")
_FLOAT = re.compile(r"[-+]?(?:(?:[0-9]+\.[0-9]*)|(?:[0-9]*\.[0-9]+)|(?:[0-9]+[eE][-+]?[0-9]+)|(?:[0-9]+\.[0-9]*[eE][-+]?[0-9]+))\Z")
_SAFE_KEY = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _strip_comment(text: str, source: str, line: int) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None and (index == 0 or text[index - 1].isspace()):
            return text[:index].rstrip()
    if quote is not None:
        raise StrictYamlError(f"{source}:{line}: unterminated quoted scalar")
    return text.rstrip()


def _split_mapping(text: str, source: str, line: int) -> tuple[str, str]:
    quote = None
    depth = 0
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
        elif char in ("'", '"'):
            if quote is None: quote = char
            elif quote == char: quote = None
        elif quote is None:
            if char == "[": depth += 1
            elif char == "]": depth -= 1
            elif char == ":" and depth == 0:
                return text[:index].strip(), text[index + 1 :].strip()
    raise StrictYamlError(f"{source}:{line}: expected a mapping entry")


def _split_inline(value: str, source: str, line: int) -> list[str]:
    result, start, quote, escaped = [], 0, None, False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
        elif quote == '"' and char == "\\":
            escaped = True
        elif char in ("'", '"'):
            if quote is None: quote = char
            elif quote == char: quote = None
        elif char == "," and quote is None:
            result.append(value[start:index].strip()); start = index + 1
    result.append(value[start:].strip())
    if any(not item for item in result):
        raise StrictYamlError(f"{source}:{line}: empty inline sequence item")
    return result


def _scalar(value: str, source: str, line: int) -> Any:
    if not value:
        raise StrictYamlError(f"{source}:{line}: empty scalar")
    if value.startswith("["):
        if not value.endswith("]"):
            raise StrictYamlError(f"{source}:{line}: unterminated inline sequence")
        inner = value[1:-1].strip()
        return [] if not inner else [_scalar(item, source, line) for item in _split_inline(inner, source, line)]
    if value == "{}":
        return {}
    if value.startswith("{") or value.endswith("}"):
        raise StrictYamlError(f"{source}:{line}: inline mappings are not supported")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise StrictYamlError(f"{source}:{line}: invalid double-quoted scalar") from exc
        if not isinstance(parsed, str):
            raise StrictYamlError(f"{source}:{line}: quoted value must be a string")
        return parsed
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise StrictYamlError(f"{source}:{line}: invalid single-quoted scalar")
        return value[1:-1].replace("''", "'")
    lowered = value.lower()
    if lowered in ("null", "~"): return None
    if lowered == "true": return True
    if lowered == "false": return False
    if _INTEGER.fullmatch(value): return int(value)
    if _FLOAT.fullmatch(value): return float(value)
    if value[0] in "&*!|>" or value.startswith("<<") or re.search(r"(?:^|\s)[&*!][^\s]+", value):
        raise StrictYamlError(f"{source}:{line}: YAML anchors, aliases, tags, merges, and block scalars are forbidden")
    return value


class _Parser:
    def __init__(self, lines: list[_Line], source: str) -> None:
        self.lines, self.source = lines, source

    def parse(self) -> Any:
        if not self.lines:
            raise StrictYamlError(f"{self.source}: document is empty")
        if self.lines[0].indent != 0:
            raise StrictYamlError(f"{self.source}:{self.lines[0].number}: root must start at column zero")
        value, index = self._block(0, 0)
        if index != len(self.lines):
            line = self.lines[index]
            raise StrictYamlError(f"{self.source}:{line.number}: unexpected indentation")
        return value

    def _block(self, index: int, indent: int) -> tuple[Any, int]:
        is_sequence = self.lines[index].text.startswith("- ")
        result: Any = [] if is_sequence else {}
        while index < len(self.lines):
            line = self.lines[index]
            if line.indent < indent: break
            if line.indent > indent:
                raise StrictYamlError(f"{self.source}:{line.number}: indentation jumped without a parent key")
            if is_sequence:
                if not line.text.startswith("- "):
                    break
                item = line.text[2:].strip()
                if not item:
                    raise StrictYamlError(f"{self.source}:{line.number}: nested sequence items are not supported")
                result.append(_scalar(item, self.source, line.number)); index += 1
                continue
            if line.text.startswith("- "):
                break
            key, raw = _split_mapping(line.text, self.source, line.number)
            if not key or not _SAFE_KEY.fullmatch(key):
                raise StrictYamlError(f"{self.source}:{line.number}: mapping keys must match {_SAFE_KEY.pattern!r}")
            if key in result:
                raise StrictYamlError(f"{self.source}:{line.number}: duplicate key {key!r}")
            index += 1
            if raw:
                result[key] = _scalar(raw, self.source, line.number)
            else:
                if index >= len(self.lines) or self.lines[index].indent <= indent:
                    raise StrictYamlError(f"{self.source}:{line.number}: key {key!r} requires a nested value")
                if self.lines[index].indent != indent + 2:
                    raise StrictYamlError(f"{self.source}:{self.lines[index].number}: indentation must increase by two spaces")
                result[key], index = self._block(index, indent + 2)
        return result, index


def loads(text: str, *, source: str = "<string>") -> dict[str, Any]:
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw:
            raise StrictYamlError(f"{source}:{number}: tabs are forbidden")
        stripped = _strip_comment(raw, source, number)
        if not stripped.strip(): continue
        if stripped.lstrip().startswith(("---", "...", "%YAML", "%TAG")):
            raise StrictYamlError(f"{source}:{number}: directives and multi-document markers are forbidden")
        indent = len(stripped) - len(stripped.lstrip(" "))
        if indent % 2:
            raise StrictYamlError(f"{source}:{number}: indentation must use multiples of two spaces")
        lines.append(_Line(number, indent, stripped[indent:]))
    parsed = _Parser(lines, source).parse()
    if not isinstance(parsed, dict):
        raise StrictYamlError(f"{source}: root must be a mapping")
    return parsed


def load(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    try:
        return loads(file.read_text(encoding="utf-8"), source=str(file))
    except OSError as exc:
        raise StrictYamlError(f"cannot read configuration {file}") from exc


def dumps(value: Any) -> str:
    lines: list[str] = []

    def scalar(item: Any) -> str:
        if item is None: return "null"
        if item is True: return "true"
        if item is False: return "false"
        if isinstance(item, str): return json.dumps(item, ensure_ascii=False)
        if isinstance(item, (int, float)) and not isinstance(item, bool): return repr(item)
        raise TypeError(f"unsupported YAML scalar: {type(item).__name__}")

    def emit(item: Any, indent: int) -> None:
        prefix = " " * indent
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str) or not _SAFE_KEY.fullmatch(key):
                    raise TypeError(f"unsupported YAML key: {key!r}")
                if isinstance(nested, Mapping) and not nested:
                    lines.append(f"{prefix}{key}: {{}}")
                elif isinstance(nested, (list, tuple)) and not nested:
                    lines.append(f"{prefix}{key}: []")
                elif isinstance(nested, (Mapping, list, tuple)):
                    lines.append(f"{prefix}{key}:")
                    emit(nested, indent + 2)
                else:
                    lines.append(f"{prefix}{key}: {scalar(nested)}")
        elif isinstance(item, (list, tuple)):
            for nested in item:
                if isinstance(nested, (Mapping, list, tuple)):
                    raise TypeError("resolved YAML sequences must contain only scalar values")
                lines.append(f"{prefix}- {scalar(nested)}")
        else:
            raise TypeError("resolved YAML root values must be mappings or sequences")

    emit(value, 0)
    return "\n".join(lines) + "\n"
