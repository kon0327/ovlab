"""Minimal third-party integration: no OVLAB orchestration or model imports."""
from dataclasses import dataclass
from ovlab_policy_sdk.plugin import PolicyPlugin, PolicyPluginCompatibilityError, PolicyPluginReferenceError
from ovlab_policy_sdk.validation import mapping, non_empty_string


@dataclass(frozen=True)
class Settings:
    action_spec: object
    checkpoint: dict
    record_raw_output: bool = False


class ExamplePlugin(PolicyPlugin):
    def validate_policy(self, doc, path):
        settings = mapping(doc['settings'], f'{path}.settings', required=('checkpoint_id', 'input', 'runtime', 'action', 'raw_output'))
        non_empty_string(settings['checkpoint_id'], f'{path}.settings.checkpoint_id')
        mapping(settings['input'], f'{path}.settings.input')
        runtime = mapping(settings['runtime'], f'{path}.settings.runtime', required=('quantization',))
        if runtime['quantization'] != 'none':
            from ovlab_policy_sdk.plugin import PolicyPluginValidationError
            raise PolicyPluginValidationError('example policy requires quantization none')
        action = mapping(settings['action'], f'{path}.settings.action', required=('action_interface',))
        non_empty_string(action['action_interface'], f'{path}.settings.action.action_interface')
        raw = mapping(settings['raw_output'], f'{path}.settings.raw_output', required=('enabled',))
        if raw['enabled'] is not False:
            from ovlab_policy_sdk.plugin import PolicyPluginValidationError
            raise PolicyPluginValidationError('example policy does not expose raw output')

    def resolve_settings(self, doc, benchmark_context, action_spec, checkpoints, devices, *, repository_root, environment):
        from .contracts import planar_action_spec
        from ovlab_policy_sdk.adapter import _action_specs_equal
        if not _action_specs_equal(action_spec, planar_action_spec()) or action_spec.control_frequency_hz != 5.0:
            raise PolicyPluginCompatibilityError('example policy requires its native two-joint action interface')
        try:
            checkpoint = checkpoints[doc['settings']['checkpoint_id']]
        except KeyError as exc:
            raise PolicyPluginReferenceError('example policy checkpoint is not registered') from exc
        return Settings(action_spec, dict(checkpoint))

    def describe_capabilities(self, settings):
        from ovlab_core.contracts import PolicyCapabilities, ObservationRequirements, OVLAB_CONTRACT_VERSION
        return PolicyCapabilities('example-policy', '2.0.0', OVLAB_CONTRACT_VERSION,
            ObservationRequirements(), settings.action_spec, True, False, 1, 1,
            False, True, False, {'checkpoint': settings.checkpoint})

    def create_adapter(self, settings):
        from .runtime import ConstantPolicy
        return ConstantPolicy(settings)

    def identity(self, capabilities):
        return {'model_identity': dict(capabilities.metadata['checkpoint']),
                'normalization_identity': {'kind': 'identity', 'units': 'rad'},
                'prompt_template_identity': {'kind': 'none'},
                'action_codec_identity': {'identifier': 'example-planar-joint-delta-v1',
                                          'conversion_owner': 'example-policy', 'application_count': 0},
                'method_descriptor': {'id': 'example-constant', 'version': '2.0.0', 'family': 'constant'},
                'runtime_versions': {'example-policy': '2.0.0'}}


plugin = ExamplePlugin()
