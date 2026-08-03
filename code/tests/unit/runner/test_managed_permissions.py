from __future__ import annotations

from pathlib import Path

import pytest

from ovlab_runner import ArtifactError
from ovlab_runner.permissions import finalize_managed_tree


def test_finalized_managed_tree_is_host_group_deletable_but_files_are_read_only(tmp_path):
    root = tmp_path / "run"
    nested = root / "tasks/task/episodes/episode"
    nested.mkdir(parents=True)
    artifact = nested / "trace.json"
    artifact.write_text("{}", encoding="utf-8")

    finalize_managed_tree(root)

    assert all(
        path.stat().st_mode & 0o7777 == 0o2770
        for path in (root, root / "tasks", root / "tasks/task", nested.parent, nested)
    )
    assert artifact.stat().st_mode & 0o777 == 0o640


def test_permission_finalization_rejects_symbolic_links_before_chmod(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("immutable", encoding="utf-8")
    (root / "link").symlink_to(outside)

    with pytest.raises(ArtifactError, match="symbolic link"):
        finalize_managed_tree(root)
    assert outside.read_text(encoding="utf-8") == "immutable"
