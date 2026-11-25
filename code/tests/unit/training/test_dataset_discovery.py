"""Dataset declarations stay lazy; selected providers use the public CLI."""

import json
from pathlib import Path
import sys

import pytest

from ovlab_benchctl import cli
from ovlab_benchctl.datasets import DatasetBridgeRegistry, DatasetRequest, DatasetStore, dataset_image
from ovlab_benchctl.training_errors import DatasetRequestError


def _provider(root, name, *, alias=None, broken=False, implementation_id=None, version="1.0.0"):
    project = root / "projects" / name
    package = "discovery_" + name.replace("-", "_")
    (project / "src" / package).mkdir(parents=True)
    source = "raise AssertionError('unselected dataset provider imported')\n" if broken else f'''
from ovlab_core.datasets import DatasetProvider, DatasetResolution
class Provider(DatasetProvider):
    bridge_id = {implementation_id or alias or name!r}
    bridge_version = {version!r}
    def capabilities(self):
        return {{'implementation_loaded': True}}
    def resolve(self, request):
        return DatasetResolution(self.bridge_id, request.name, 'fixed-v1', 'fixture',
            'fixture://immutable/samples', self.bridge_id, self.bridge_version,
            None, 'toy', {{'fixture': True}})
    def acquire(self, resolution, destination, progress=None):
        destination.mkdir()
        (destination / 'samples.bin').write_bytes(b'independent-dataset')
    def prepare(self, resolution, acquired, destination):
        destination.mkdir()
        (destination / 'samples.bin').write_bytes((acquired / 'samples.bin').read_bytes())
        return {{**self.verify(resolution, destination), 'format': 'toy'}}
provider = Provider()
'''
    (project / "src" / package / "__init__.py").write_text(source, encoding="utf-8")
    manifest = {
        "schema_version": "ovlab.extension/v1", "id": name, "category": "plugin", "version": "0.2.0",
        "providers": {"data": {"kind": "dataset", "api_version": "1", "version": version,
            "entrypoint": f"{package}:provider", "provider_id": alias or name,
            "capabilities": {"declared": True},
            "deployment": {"role": "dataset", "image": f"{name}:local", "image_env": "OVLAB_TEST_DATASET_IMAGE"}}},
        "builds": {"dataset": {"dockerfile": "Dockerfile", "image": f"{name}:local", "sources": [], "locks": []}},
    }
    (project / "extension.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project, package


@pytest.fixture
def registry(tmp_path, monkeypatch):
    root = tmp_path / "ovlab"
    root.mkdir()
    path = root / "configs/extensions/registry.json"
    path.parent.mkdir(parents=True)
    monkeypatch.setenv("OVLAB_ROOT", str(root))
    monkeypatch.setenv("OVLAB_EXTENSION_REGISTRY", str(path))
    monkeypatch.delenv("OVLAB_PLUGIN_REGISTRY", raising=False)
    monkeypatch.setenv("OVLAB_MODEL_DATA_ROOT", str(tmp_path / "model-data"))
    projects = {}

    def add(name, **options):
        project, package = _provider(root, name, **options)
        projects[name] = {"source": {"kind": "local", "path": str(project)}}
        path.write_text(json.dumps({"schema_version": "ovlab.extensions/v1", "projects": projects}), encoding="utf-8")
        monkeypatch.delitem(sys.modules, package, raising=False)
        return project, package

    yield root, add
    for key in list(sys.modules):
        if key.startswith("discovery_"):
            monkeypatch.delitem(sys.modules, key, raising=False)


def test_inventory_and_store_construction_do_not_import_dataset_providers(registry, tmp_path):
    root, add = registry
    _, package = add("broken", broken=True)
    store = DatasetStore(tmp_path / "data")
    assert store.list() == []
    assert package not in sys.modules

    rows = {row["id"]: row for row in DatasetBridgeRegistry().providers()}

    assert rows["broken"] == {"id": "broken", "version": "1.0.0", "capabilities": {"declared": True},
                              "provider_ref": "plugin:dataset:broken/data"}
    assert package not in sys.modules


def test_selected_provider_loads_once_without_importing_unrelated_provider(registry):
    _, add = registry
    _, selected = add("selected")
    _, unrelated = add("unrelated", broken=True)
    providers = DatasetBridgeRegistry()

    first = providers.get("selected")
    second = providers.get("selected")

    assert first is second
    assert selected in sys.modules and unrelated not in sys.modules
    assert first.resolve(DatasetRequest(source="selected", name="samples")).provider == "selected"
    assert providers.providers()[1]["id"] == "selected"


@pytest.mark.parametrize("alias", ["url", "local", "duplicate"])
def test_duplicate_ids_fail_before_any_provider_import(registry, alias):
    _, add = registry
    _, first = add("first", alias=alias, broken=True)
    if alias == "duplicate":
        _, second = add("second", alias=alias, broken=True)
    else:
        second = first

    with pytest.raises(DatasetRequestError, match=f"ambiguous dataset provider: {alias}"):
        DatasetBridgeRegistry().get(alias)

    assert first not in sys.modules and second not in sys.modules


def test_selected_implementation_id_must_match_declared_provider_id(registry):
    _, add = registry
    add("misidentified", implementation_id="different")

    with pytest.raises(DatasetRequestError, match="declares ID.*implementation returns"):
        DatasetBridgeRegistry().get("misidentified")


def test_existing_local_data_can_be_inspected_without_unrelated_provider_import(registry, tmp_path):
    _, add = registry
    _, broken = add("broken", broken=True)
    source = tmp_path / "source"
    source.mkdir()
    (source / "samples.bin").write_bytes(b"local-dataset")
    store = DatasetStore(tmp_path / "data")
    published = store.import_local(DatasetRequest(source="local", name="samples", local_path=source))
    second = DatasetStore(tmp_path / "data")

    assert second.inspect(published["dataset_id"])["dataset_id"] == published["dataset_id"]
    assert second.verify(published["dataset_id"])["status"] == "verified"
    assert broken not in sys.modules


def test_acquisition_image_selection_is_declarative_and_project_scoped(registry):
    root, add = registry
    _, package = add("image", broken=True)

    assert dataset_image("image", root) == {"role": "image/dataset", "image": "image:local",
                                            "image_env": "OVLAB_TEST_DATASET_IMAGE"}
    assert package not in sys.modules


def test_public_cli_resolves_fetches_and_verifies_third_party_dataset(registry, capsys):
    _, add = registry
    add("third-party")
    assert cli.main(["dataset", "resolve", "--benchmark", "third-party", "--suite", "samples", "--json"]) == 0
    resolved = json.loads(capsys.readouterr().out)["result"]
    assert resolved["provider"] == "third-party"
    assert resolved["state"] == "resolved"

    assert cli.main(["dataset", "fetch", "--source", "third-party", "--name", "samples", "--json"]) == 0
    acquired = json.loads(capsys.readouterr().out)["result"]
    assert acquired["resolution_id"] == resolved["resolution_id"]
    assert acquired["state"] == "ready"

    assert cli.main(["dataset", "verify", acquired["dataset_id"], "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["status"] == "verified"
