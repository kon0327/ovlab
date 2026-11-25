"""Native benchmark interfaces; independent from the selected policy project."""
def action_spec():
    import numpy as np
    from ovlab_core.contracts import ActionRepresentation, ActionSpec
    return ActionSpec(2, ActionRepresentation.JOINT_DELTA, units=('rad', 'rad'),
        minimum=np.full(2, -1.0, dtype=np.float32), maximum=np.full(2, 1.0, dtype=np.float32),
        dtype='float32', control_frequency_hz=5.0)


def capabilities():
    from ovlab_core.contracts import (
        BenchmarkCapabilities, ObservationSpec, ProprioceptiveObservationSpec,
        OVLAB_CONTRACT_VERSION, SignalAccess, SignalRegistry, SignalSpec,
    )
    return BenchmarkCapabilities('example-planar', '1.0.0', OVLAB_CONTRACT_VERSION,
        ObservationSpec(proprioception=(ProprioceptiveObservationSpec(
            'fixture.joint_position', ((2,),), 'float32', ('rad', 'rad')),)), action_spec(),
        SignalRegistry((SignalSpec('benchmark.task_success', 'bool', (), '',
            SignalAccess.EVALUATION_ONLY, 'Authoritative fixture task success'),)),
        True, False, False, ('Planar fixture',), {'execution_backend': 'cpu'})
