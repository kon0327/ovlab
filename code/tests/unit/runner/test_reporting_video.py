"""Web-compatible canonical video contract tests."""

from pathlib import Path

import numpy as np
import pytest

from ovlab_runner import ArtifactError
from ovlab_runner.reporting import _encode_h264, _faststart_enabled, _h264_command


def test_h264_command_is_browser_compatible_and_faststart_enabled(tmp_path) -> None:
    target = tmp_path / "video.tmp.mp4"
    command = _h264_command(
        "/runtime/ffmpeg", target, width=256, height=256, fps=20.0,
    )
    pairs = tuple(zip(command, command[1:]))
    assert ("-c:v", "libx264") in pairs
    assert ("-pix_fmt", "yuv420p") in pairs
    assert ("-tag:v", "avc1") in pairs
    assert ("-movflags", "+faststart") in pairs
    assert ("-pixel_format", "rgb24") in pairs
    assert "mp4v" not in command


def test_h264_encoder_rejects_dimensions_incompatible_with_yuv420p(tmp_path) -> None:
    frame = np.zeros((255, 256, 3), dtype=np.uint8)
    with pytest.raises(ArtifactError, match="even frame dimensions"):
        _encode_h264("/unused/ffmpeg", tmp_path / "video.mp4", [frame], fps=20.0)


def test_faststart_check_requires_moov_before_media_data(tmp_path) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"....ftyp....moov....mdat....")
    assert _faststart_enabled(target)
    target.write_bytes(b"....ftyp....mdat....moov....")
    assert not _faststart_enabled(target)
