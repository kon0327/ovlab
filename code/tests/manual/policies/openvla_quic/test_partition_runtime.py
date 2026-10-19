"""CPU data-runtime smoke; run explicitly in the pinned OpenVLA image/env."""
from pathlib import Path

import pytest


@pytest.mark.openvla
def test_tfds_partition_runtime_preserves_temporal_standardization(tmp_path, monkeypatch):
    # Select the migrated host rather than an unrelated editable OpenVLA in
    # the developer's model environment.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "external/openvla-quic"))
    tf = pytest.importorskip("tensorflow")
    tfds = pytest.importorskip("tensorflow_datasets")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS
    from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
    from ovlab_benchctl.training_data import make_partitioned_datasets, verify_episode_manifest
    tf.config.set_visible_devices([], "GPU")

    class OvlabEpisodeFixture(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        def _info(self):
            return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": {"image": tfds.features.Image(shape=(8, 8, 3)), "state": tfds.features.Tensor(shape=(8,), dtype=np.float32)},
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": {"file_path": tfds.features.Text()},
            }))
        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}
        def _generate_examples(self):
            for episode in range(4):
                yield episode, {"episode_metadata": {"file_path": "fixture_demo.hdf5"}, "steps": [{
                    "observation": {"image": np.full((8, 8, 3), frame, dtype=np.uint8), "state": np.zeros(8, dtype=np.float32)},
                    "action": np.asarray([frame / 10] + [0.] * 5 + [1.], dtype=np.float32),
                    "language_instruction": "move the fixture",
                } for frame in range(4)]}

    def temporal_standardizer(trajectory):
        # A one-frame call would leave the first value at zero, revealing a
        # broken port of source standardizers that inspect neighboring frames.
        trajectory["action"] = tf.concat([tf.roll(trajectory["action"][:, :1], -1, axis=0), trajectory["action"][:, 1:]], axis=1)
        return trajectory

    builder = OvlabEpisodeFixture(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setitem(OXE_DATASET_CONFIGS, builder.name, {"image_obs_keys": {"primary": "image"}})
    monkeypatch.setitem(OXE_STANDARDIZATION_TRANSFORMS, builder.name, temporal_standardizer)
    config = {"seed": 42, "validation_fraction": .25, "max_episodes_per_group": 4, "frame_stride": 2, "max_frames_per_episode": 2, "action_normalization": "host"}
    statistics = {"q01": [-1.] * 7, "q99": [1.] * 7, "mask": [True] * 6 + [False]}
    train, validation, manifest = make_partitioned_datasets(tmp_path, builder.name, config, lambda frame: frame, resize_resolution=(16, 16), action_statistics=statistics, manifest_path=tmp_path / "partition.json")
    verify_episode_manifest(manifest)
    assert len(train) == 6
    assert len(validation) == 2
    first = next(iter(train))
    assert first["observation"]["image_primary"].shape == (1, 16, 16, 3)
    assert first["action"].shape == (1, 7)
    np.testing.assert_allclose(first["action"][0, 0], .1, atol=1e-6)
    assert first["action"][0, -1] == 1.
    assert len(list(validation)) == 2
    repeated_train, _, repeated = make_partitioned_datasets(tmp_path, builder.name, config, lambda frame: frame, resize_resolution=(16, 16), action_statistics=statistics, manifest_path=tmp_path / "partition.json")
    assert repeated == manifest
    assert len(repeated_train) == len(train)


@pytest.mark.openvla
def test_fitted_normalization_excludes_validation_extremes_and_freezes_stats(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "external/openvla-quic"))
    tf = pytest.importorskip("tensorflow")
    tfds = pytest.importorskip("tensorflow_datasets")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    import json
    from prismatic.vla.datasets.rlds.oxe.configs import ActionEncoding, OXE_DATASET_CONFIGS
    from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
    from ovlab_benchctl.training_data import build_episode_manifest, make_partitioned_datasets, verify_episode_manifest
    from ovlab_benchctl.training_errors import TrainingRuntimeError
    tf.config.set_visible_devices([], "GPU")

    class OvlabNormalizationFixture(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        def _info(self):
            return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": {"image": tfds.features.Image(shape=(8, 8, 3)), "state": tfds.features.Tensor(shape=(8,), dtype=np.float32)},
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": {"file_path": tfds.features.Text(), "episode_index": tfds.features.Scalar(dtype=np.int64)},
            }))
        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}
        def _generate_examples(self):
            for episode, base in enumerate((10., 20., 30., 10000.)):
                yield episode, {"episode_metadata": {"file_path": "fit_fixture_demo.hdf5", "episode_index": episode}, "steps": [{
                    "observation": {"image": np.full((8, 8, 3), frame, dtype=np.uint8), "state": np.zeros(8, dtype=np.float32)},
                    "action": np.asarray([base + frame] + [0.] * 5 + [-1. if frame % 2 == 0 else 1.], dtype=np.float32),
                    "language_instruction": "move the normalization fixture",
                } for frame in range(4)]}

    builder = OvlabNormalizationFixture(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setitem(OXE_DATASET_CONFIGS, builder.name, {"image_obs_keys": {"primary": "image"}, "action_encoding": ActionEncoding.EEF_POS})
    # Exercise the source benchmark bridge's raw-gripper convention; fitting
    # must happen after this complete-trajectory transform, exactly once.
    monkeypatch.setitem(OXE_STANDARDIZATION_TRANSFORMS, builder.name, OXE_STANDARDIZATION_TRANSFORMS["libero_10_no_noops"])
    catalog, extreme_id = [], None
    for episode in builder.as_dataset(split="train", shuffle_files=False, read_config=tfds.ReadConfig(add_tfds_id=True)):
        record_id = episode["tfds_id"].numpy().decode()
        catalog.append({"id": record_id, "group": "fit_fixture", "frames": 4})
        if int(episode["episode_metadata"]["episode_index"].numpy()) == 3:
            extreme_id = record_id
    config = {"seed": 0, "validation_fraction": .25, "frame_stride": 1, "action_normalization": "train-q99"}
    for seed in range(100):
        config["seed"] = seed
        candidate = build_episode_manifest(catalog, config, source_identity={})
        if any(row["id"] == extreme_id and row["split"] == "validation" for row in candidate["episodes"]):
            break
    else:
        pytest.fail("could not construct the deterministic held-out extreme fixture")
    host_stats = {"q01": [-100000.] * 7, "q99": [100000.] * 7}
    manifest_path = tmp_path / "fitted-partition.json"
    train, validation, manifest = make_partitioned_datasets(tmp_path, builder.name, config, lambda frame: frame, action_statistics=host_stats, manifest_path=manifest_path)
    verify_episode_manifest(manifest)
    statistics = manifest["normalization"]["statistics"]
    expected = np.asarray([base + frame for base in (10., 20., 30.) for frame in range(4)], dtype=np.float32)
    assert statistics["q01"][0] == pytest.approx(float(np.quantile(expected, .01)))
    assert statistics["q99"][0] == pytest.approx(float(np.quantile(expected, .99)))
    assert statistics["max"][0] == 33.
    assert statistics["min"][-1] == 0.
    assert statistics["max"][-1] == 1.
    assert statistics["mask"] == [True] * 6 + [False]
    assert manifest["normalization"]["source"] == "train-q99"
    assert manifest["normalization"]["fit_split"] == "train"
    assert len(train) == 12 and len(validation) == 4
    assert train.dataset_statistics[builder.name]["action"] == validation.dataset_statistics[builder.name]["action"] == statistics
    before = manifest_path.read_bytes()
    held_out = list(validation)
    assert all(frame["action"][0, 0] == 1. for frame in held_out)  # clipped using training q99, not fitted on held-out extremes
    assert [float(frame["action"][0, -1]) for frame in held_out] == [1., 0., 1., 0.]
    assert all(frame["observation"]["image_primary"].shape == (1, 224, 224, 3) for frame in held_out)
    assert manifest_path.read_bytes() == before
    _, repeated_validation, repeated = make_partitioned_datasets(tmp_path, builder.name, config, lambda frame: frame, action_statistics=host_stats, manifest_path=manifest_path)
    assert repeated == manifest
    assert manifest_path.read_bytes() == before
    assert len(list(repeated_validation)) == 4
    # A caller may not reinterpret or overwrite an existing shared host-stat
    # manifest by toggling the normalization mode in place.
    with pytest.raises(TrainingRuntimeError, match="existing shared partition manifest differs"):
        make_partitioned_datasets(tmp_path, builder.name, dict(config, action_normalization="host"), lambda frame: frame, action_statistics=host_stats, manifest_path=manifest_path)
    assert manifest_path.read_bytes() == before
