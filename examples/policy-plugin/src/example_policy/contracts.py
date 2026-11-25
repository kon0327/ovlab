"""The constant policy's own planar joint-delta command boundary."""
def planar_action_spec():
    import numpy as np
    from ovlab_core.contracts import ActionRepresentation, ActionSpec
    return ActionSpec(2, ActionRepresentation.JOINT_DELTA, units=('rad', 'rad'),
        minimum=np.full(2, -1.0, dtype=np.float32), maximum=np.full(2, 1.0, dtype=np.float32),
        dtype='float32', control_frequency_hz=5.0)
