"""Minimal third-party integration: no OVLAB orchestration or model imports."""
from dataclasses import dataclass
from ovlab_policy_sdk.plugin import PolicyPlugin
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

    def resolve_settings(self, doc, benchmark_doc, action_spec, checkpoints, devices, *, repository_root, environment):
        return Settings(action_spec, dict(checkpoints[doc['settings']['checkpoint_id']]))

    def create_adapter(self, settings):
        from .runtime import ConstantPolicy
        return ConstantPolicy(settings)

    def identity(self, capabilities):
        return {'model_identity': dict(capabilities.metadata['checkpoint']),
                'method_identity': {'id': 'example-constant', 'version': '1.0.0'},
                'runtime_versions': {'example-policy': '1.0.0'}}


plugin = ExamplePlugin()
