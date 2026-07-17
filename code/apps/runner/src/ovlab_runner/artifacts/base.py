"""Artifact store interface and in-memory implementation."""

from abc import ABC, abstractmethod

from ..errors import ArtifactError


class RunArtifactStore(ABC):
    @abstractmethod
    def create_run(self, run_id, started_manifest): ...
    def write_configuration(self, run_id, snapshot):
        """Persist configuration evidence when supported by a concrete store."""
        raise ArtifactError("artifact store does not support configuration snapshots")
    @abstractmethod
    def write_plan(self, run_id, plan): ...
    @abstractmethod
    def write_connection_report(self, run_id, report): ...
    @abstractmethod
    def write_episode_trace(self, run_id, trace): ...
    @abstractmethod
    def write_episode_metric_results(self, run_id, task_id, episode_id, results): ...
    @abstractmethod
    def write_task_metric_results(self, run_id, task_id, results): ...
    @abstractmethod
    def finalize_run(self, run_id, manifest): ...
    @abstractmethod
    def mark_run_failed(self, run_id, manifest): ...
    @abstractmethod
    def read_episode_trace(self, run_id, task_id, episode_id): ...
    @abstractmethod
    def read_metric_results(self, run_id, task_id, episode_id=None): ...


class InMemoryRunArtifactStore(RunArtifactStore):
    def __init__(self):
        self.runs = {}
        self.write_order = []

    def create_run(self, run_id, started_manifest):
        key = str(run_id)
        if key in self.runs:
            raise ArtifactError("run already exists")
        self.runs[key] = {"started": started_manifest, "episodes": {}, "task_metrics": {}}
        self.write_order.append("manifest.started")

    def write_configuration(self, run_id, snapshot):
        run = self._run(run_id)
        if "configuration" in run: raise ArtifactError("run configuration already exists")
        run["configuration"] = snapshot
        self.write_order.append("configuration")

    def _run(self, run_id):
        try: return self.runs[str(run_id)]
        except KeyError as exc: raise ArtifactError("run does not exist") from exc

    def write_plan(self, run_id, plan):
        run = self._run(run_id)
        if "plan" in run: raise ArtifactError("plan already exists")
        run["plan"] = plan
        self.write_order.append("plan")

    def write_connection_report(self, run_id, report):
        run = self._run(run_id)
        if "connection" in run: raise ArtifactError("connection report already exists")
        run["connection"] = report
        self.write_order.append("connection")

    def write_episode_trace(self, run_id, trace):
        run = self._run(run_id)
        key = (str(trace.episode_context.task_id), str(trace.episode_context.episode_id))
        if key in run["episodes"]: raise ArtifactError("finalized episode already exists")
        run["episodes"][key] = {"trace": trace}
        self.write_order.append(f"trace:{key[1]}")

    def write_episode_metric_results(self, run_id, task_id, episode_id, results):
        episode = self._run(run_id)["episodes"].get((str(task_id), str(episode_id)))
        if episode is None: raise ArtifactError("raw trace must be written before episode metrics")
        if "metrics" in episode: raise ArtifactError("episode metrics already exist")
        episode["metrics"] = tuple(results)
        self.write_order.append(f"episode-metrics:{episode_id}")

    def write_task_metric_results(self, run_id, task_id, results):
        run = self._run(run_id)
        key = str(task_id)
        if key in run["task_metrics"]: raise ArtifactError("task metrics already exist")
        run["task_metrics"][key] = tuple(results)
        self.write_order.append(f"task-metrics:{key}")

    def finalize_run(self, run_id, manifest):
        run = self._run(run_id)
        if "completed" in run or "failed" in run: raise ArtifactError("run already finalized")
        run["completed"] = manifest
        self.write_order.append("manifest.completed")

    def mark_run_failed(self, run_id, manifest):
        run = self._run(run_id)
        if "completed" in run or "failed" in run: raise ArtifactError("run already finalized")
        run["failed"] = manifest
        self.write_order.append("manifest.failed")

    def read_episode_trace(self, run_id, task_id, episode_id):
        try: return self._run(run_id)["episodes"][(str(task_id), str(episode_id))]["trace"]
        except KeyError as exc: raise ArtifactError("episode trace not found") from exc

    def read_metric_results(self, run_id, task_id, episode_id=None):
        run = self._run(run_id)
        if episode_id is None: return run["task_metrics"].get(str(task_id), ())
        return run["episodes"].get((str(task_id), str(episode_id)), {}).get("metrics", ())
