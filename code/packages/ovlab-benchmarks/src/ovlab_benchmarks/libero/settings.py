"""Immutable settings for the LIBERO benchmark adapter."""

from dataclasses import dataclass, field
from enum import Enum

from ovlab_core.contracts import Metadata, normalize_metadata

from .errors import LiberoConfigurationError


class LiberoObservationProfile(str, Enum):
    PRIMARY_RGB = "primary_rgb"
    DUAL_RGB = "dual_rgb"
    RGB_PROPRIOCEPTION = "rgb_proprioception"


class InitialStateSelection(str, Enum):
    ROLLOUT_INDEX = "rollout_index"
    SEEDED = "seeded"


class LiberoRenderMode(str, Enum):
    HEADLESS = "headless"


@dataclass(frozen=True, slots=True)
class LiberoAdapterSettings:
    suite_names: tuple[str, ...] = ("LIBERO-Spatial",)
    task_indices: tuple[int, ...] | None = None
    camera_names: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")
    camera_width: int = 256
    camera_height: int = 256
    observation_profile: LiberoObservationProfile = LiberoObservationProfile.PRIMARY_RGB
    maximum_episode_steps: int = 220
    initialization_settling_steps: int = 10
    initial_state_selection: InitialStateSelection = InitialStateSelection.ROLLOUT_INDEX
    base_seed: int = 0
    render_mode: LiberoRenderMode = LiberoRenderMode.HEADLESS
    render_gpu_device_id: int | None = None
    controller_configuration_override: Metadata | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        suites = tuple(self.suite_names)
        cameras = tuple(self.camera_names)
        if not suites or any(not isinstance(value, str) or not value.strip() for value in suites):
            raise LiberoConfigurationError("suite_names must contain non-empty names")
        if len(suites) != len(set(suites)):
            raise LiberoConfigurationError("suite_names must be unique")
        if not cameras or any(not isinstance(value, str) or not value.strip() for value in cameras):
            raise LiberoConfigurationError("camera_names must contain non-empty names")
        required_cameras = 2 if self.observation_profile is LiberoObservationProfile.DUAL_RGB else 1
        if len(cameras) < required_cameras:
            raise LiberoConfigurationError(f"{self.observation_profile.value} requires {required_cameras} cameras")
        for name in ("camera_width", "camera_height", "maximum_episode_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise LiberoConfigurationError(f"{name} must be a positive integer")
        for name in ("initialization_settling_steps", "base_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LiberoConfigurationError(f"{name} must be a non-negative integer")
        indices = None if self.task_indices is None else tuple(self.task_indices)
        if indices is not None:
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices):
                raise LiberoConfigurationError("task_indices must be non-negative integers")
            if len(indices) != len(set(indices)):
                raise LiberoConfigurationError("task_indices must be unique")
        if self.render_gpu_device_id is not None and not isinstance(self.render_gpu_device_id, int):
            raise LiberoConfigurationError("render_gpu_device_id must be an integer or None")
        override = None
        if self.controller_configuration_override is not None:
            override = normalize_metadata(self.controller_configuration_override, type(self).__name__)
        object.__setattr__(self, "suite_names", suites)
        object.__setattr__(self, "task_indices", indices)
        object.__setattr__(self, "camera_names", cameras)
        object.__setattr__(self, "controller_configuration_override", override)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata, type(self).__name__))
