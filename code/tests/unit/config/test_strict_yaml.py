import pytest

from ovlab_benchctl import StrictYamlError, dumps, loads


def test_strict_yaml_parses_supported_scalars_and_sequences():
    value = loads('''
schema_version: "0.1.0"
enabled: true
missing: null
count: 3
tolerance: 1.0e-5
tags: [libero, "openvla"]
required:
  - task.success
  - action.smoothness_1
''')
    assert value == {
        "schema_version": "0.1.0", "enabled": True, "missing": None, "count": 3,
        "tolerance": 1e-5, "tags": ["libero", "openvla"],
        "required": ["task.success", "action.smoothness_1"],
    }
    assert loads(dumps(value)) == value


def test_strict_yaml_round_trip_preserves_empty_nested_collections():
    value = {"empty_sequence": [], "empty_mapping": {}, "nested": {"indices": ()}}
    dumped = dumps(value)
    assert "empty_sequence: []" in dumped
    assert "empty_mapping: {}" in dumped
    assert loads(dumped) == {"empty_sequence": [], "empty_mapping": {}, "nested": {"indices": []}}


@pytest.mark.parametrize("text,match", [
    ("key: 1\nkey: 2\n", "duplicate key"),
    ("key:\n   nested: true\n", "multiples of two"),
    ("key:\n\tnested: true\n", "tabs"),
    ("key: &anchor value\n", "anchors"),
    ("---\nkey: value\n", "multi-document"),
    ("key: {nested: value}\n", "inline mappings"),
])
def test_strict_yaml_rejects_ambiguous_or_advanced_features(text, match):
    with pytest.raises(StrictYamlError, match=match):
        loads(text)
