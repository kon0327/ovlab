"""Public exports loaded on demand, keeping plugin discovery dependency-light."""
from importlib import import_module

_EXPORTS = {'LiberoActionCodec': ('.actions', 'LiberoActionCodec'),
 'LiberoActionChunkCodec': ('.actions', 'LiberoActionChunkCodec'),
 'LiberoActionCodecConfig': ('.actions', 'LiberoActionCodecConfig'),
 'OpenVlaDecodedAction': ('.actions', 'OpenVlaDecodedAction'),
 'OpenVlaDecodedActionChunk': ('.actions', 'OpenVlaDecodedActionChunk'),
 'action_specs_match': ('.actions', 'action_specs_match'),
 'libero_target_action_spec': ('.actions', 'libero_target_action_spec'),
 'CheckpointFileIdentity': ('.artifact', 'CheckpointFileIdentity'),
 'OpenVlaRuntimeArtifact': ('.artifact', 'OpenVlaRuntimeArtifact'),
 'OpenVlaActionCodecError': ('.errors', 'OpenVlaActionCodecError'),
 'OpenVlaCommonError': ('.errors', 'OpenVlaCommonError'),
 'OpenVlaObservationError': ('.errors', 'OpenVlaObservationError'),
 'OpenVlaCheckpointIdentity': ('.metadata', 'OpenVlaCheckpointIdentity'),
 'OpenVlaArtifactForm': ('.method', 'OpenVlaArtifactForm'),
 'OpenVlaMergeStatus': ('.method', 'OpenVlaMergeStatus'),
 'OpenVlaMethodDescriptor': ('.method', 'OpenVlaMethodDescriptor'),
 'OpenVlaMethodFamily': ('.method', 'OpenVlaMethodFamily'),
 'method_descriptor_from_registry': ('.method', 'method_descriptor_from_registry'),
 'vanilla_base_method_descriptor': ('.method', 'vanilla_base_method_descriptor'),
 'select_canonical_rgb': ('.observations', 'select_canonical_rgb'),
 'OpenVlaPromptFormatter': ('.prompt', 'OpenVlaPromptFormatter'),
 'OpenVlaPromptTemplate': ('.prompt', 'OpenVlaPromptTemplate'),
 'ModelQuantization': ('.quantization', 'ModelQuantization'),
 'CUDA_ALLOCATOR_SOURCE': ('.performance', 'CUDA_ALLOCATOR_SOURCE'),
 'INFERENCE_COMPUTE_ESTIMATOR': ('.performance', 'INFERENCE_COMPUTE_ESTIMATOR'),
 'PERFORMANCE_TELEMETRY_SCHEMA': ('.performance', 'PERFORMANCE_TELEMETRY_SCHEMA'),
 'TRAINING_COMPUTE_ESTIMATOR': ('.performance', 'TRAINING_COMPUTE_ESTIMATOR'),
 'cuda_allocator_snapshot': ('.performance', 'cuda_allocator_snapshot'),
 'estimated_inference_compute': ('.performance', 'estimated_inference_compute'),
 'estimated_training_compute': ('.performance', 'estimated_training_compute'),
 'parameter_inventory': ('.performance', 'parameter_inventory'),
 'performance_sample': ('.performance', 'performance_sample'),
 'reset_cuda_peak': ('.performance', 'reset_cuda_peak'),
 'OpenVlaModelSource': ('.settings', 'OpenVlaModelSource')}

__all__ = [
    "CheckpointFileIdentity", "LiberoActionCodec", "LiberoActionChunkCodec", "LiberoActionCodecConfig", "OpenVlaActionCodecError",
    "OpenVlaArtifactForm", "OpenVlaMergeStatus", "OpenVlaMethodDescriptor", "OpenVlaMethodFamily",
    "OpenVlaCheckpointIdentity", "OpenVlaCommonError", "OpenVlaDecodedAction", "OpenVlaDecodedActionChunk",
    "ModelQuantization", "OpenVlaModelSource", "OpenVlaObservationError", "OpenVlaPromptFormatter", "OpenVlaRuntimeArtifact",
    "OpenVlaPromptTemplate", "action_specs_match", "libero_target_action_spec",
    "method_descriptor_from_registry", "select_canonical_rgb", "vanilla_base_method_descriptor",
    "CUDA_ALLOCATOR_SOURCE", "INFERENCE_COMPUTE_ESTIMATOR", "PERFORMANCE_TELEMETRY_SCHEMA",
    "TRAINING_COMPUTE_ESTIMATOR", "cuda_allocator_snapshot", "estimated_inference_compute",
    "estimated_training_compute", "parameter_inventory", "performance_sample", "reset_cuda_peak",
]


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
