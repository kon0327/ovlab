"""Runner-specific error hierarchy."""


class RunnerError(Exception): pass
class RunnerLifecycleError(RunnerError): pass
class ConnectionError(RunnerError): pass
class ExperimentExecutionError(RunnerError): pass
class ArtifactError(RunnerError): pass
class RecorderError(RunnerError): pass
