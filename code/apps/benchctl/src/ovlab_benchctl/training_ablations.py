"""Versioned ablation suites built from ordinary OVLAB training profiles.

Suite planning is dependency-light. Materialization writes reviewable resolved
profiles; execution continues through the existing isolated training deployment.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Mapping

from .strict_yaml import dumps
from .training_composition import load_training_document, load_training_profile
from .training_errors import TrainingProfileError, TrainingRuntimeError
from .training_identity import atomic_json, canonical_json, identity

ABLATION_SCHEMA = "ovlab.training-ablation/v1"
ABLATION_PLAN_SCHEMA = "ovlab.training-ablation-plan/v1"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_FACTORS = {"training.peft.adapter", "training.peft.placement", "training.peft.targets", "training.peft.max_trainable_parameters"}


def _differences(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) | set(second)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in first or key not in second:
                yield path
            else:
                yield from _differences(first[key], second[key], path)
    elif first != second:
        yield prefix


def _drop(document, dotted):
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(document, dict) or part not in document:
            return
        document = document[part]
    if isinstance(document, dict):
        document.pop(parts[-1], None)


class TrainingAblation:
    def __init__(self, path: Path, *, config_root: Path, fit_action_normalization: bool = False):
        self.path, self.config_root = Path(path).resolve(), Path(config_root).resolve()
        self.fit_action_normalization = fit_action_normalization
        if not self.path.is_relative_to(self.config_root):
            raise TrainingProfileError("ablation suite must be inside the configuration root")
        raw = load_training_document(self.path, config_root=self.config_root)
        allowed = {"schema_version", "id", "base_profile", "factors", "variants", "evaluation"}
        missing = allowed - {"evaluation"} - set(raw)
        if set(raw) - allowed or missing:
            raise TrainingProfileError(f"invalid ablation fields; unknown={sorted(set(raw) - allowed)}, missing={sorted(missing)}")
        if raw["schema_version"] != ABLATION_SCHEMA:
            raise TrainingProfileError(f"ablation schema_version must be {ABLATION_SCHEMA}")
        if not isinstance(raw["id"], str) or _SAFE.fullmatch(raw["id"]) is None:
            raise TrainingProfileError("ablation id must be a portable identifier")
        factors = raw["factors"]
        if not isinstance(factors, list) or not factors or any(not isinstance(factor, str) or not any(factor == allowed or factor.startswith(allowed + ".") for allowed in _ALLOWED_FACTORS) for factor in factors) or len(factors) != len(set(factors)):
            raise TrainingProfileError("ablation factors must be unique explicit training.peft adapter/placement/targets/budget paths; host, split, optimizer, seed and hardware remain shared")
        if any(left != right and right.startswith(left + ".") for left in factors for right in factors):
            raise TrainingProfileError("ablation factors must not overlap")
        variants = raw["variants"]
        if not isinstance(variants, dict) or len(variants) < 2 or any(not isinstance(key, str) or _SAFE.fullmatch(key) is None for key in variants):
            raise TrainingProfileError("ablation variants require at least two portable variant IDs")
        self.document = raw
        self.base_path, self.base = self._profile(raw["base_profile"])
        self.variants = {variant: self._profile(reference) for variant, reference in variants.items()}
        dataset = self.base.document["dataset"]
        if "partition" not in dataset:
            raise TrainingProfileError("ablation base profile requires explicit shared dataset.partition")
        from .training_data import validate_partition
        validate_partition(dataset["partition"])
        common = deepcopy(dict(self.base.document))
        common.pop("id", None)
        for factor in factors:
            _drop(common, factor)
        self.common = common
        seen = set()
        for variant, (_, profile) in self.variants.items():
            differences = set(_differences(self.base.document, profile.document)) - {"id"}
            undeclared = sorted(path for path in differences if not any(path == factor or path.startswith(factor + ".") for factor in factors))
            if undeclared:
                raise TrainingProfileError(f"ablation variant {variant!r} changes undeclared/shared fields: {', '.join(undeclared)}")
            scientific = dict(profile.document)
            scientific.pop("id", None)
            variant_hash = identity("ablation-variant", scientific, 32)
            if variant_hash in seen:
                raise TrainingProfileError("ablation suite contains scientifically identical variants")
            seen.add(variant_hash)
        self.evaluation = None
        if "evaluation" in raw:
            evaluation = self._reference(raw["evaluation"])
            if not evaluation.is_file():
                raise TrainingProfileError(f"shared evaluation experiment does not exist: {evaluation}")
            self.evaluation = evaluation

    def _reference(self, reference):
        if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
            raise TrainingProfileError("ablation profile/evaluation references must be nonempty relative paths")
        path = (self.path.parent / reference).resolve()
        if not path.is_relative_to(self.config_root):
            raise TrainingProfileError("ablation reference escapes configuration root")
        return path

    def _profile(self, reference):
        path = self._reference(reference)
        from .training_profiles import with_fitted_action_normalization
        return path, with_fitted_action_normalization(load_training_profile(path, config_root=self.config_root), self.fit_action_normalization)

    def plan(self, planner=None, **planner_kwargs) -> dict:
        scientific = {
            "schema_version": ABLATION_PLAN_SCHEMA,
            "suite_id": self.document["id"],
            "factors": self.document["factors"],
            "common": self.common,
            "partition_contract_id": identity("partition-contract", {"dataset": self.base.document["dataset"], "normalization": self.base.document["training"].get("normalization")}, 32),
            "variants": {variant: dict(profile.document) for variant, (_, profile) in self.variants.items()},
        }
        resolved = {}
        if planner is not None:
            for variant, (_, profile) in self.variants.items():
                resolved[variant] = planner.plan(profile, **planner_kwargs)
            dataset_ids = {canonical_json(plan["scientific"]["dataset"]) for plan in resolved.values()}
            checkpoints = {canonical_json(plan["scientific"]["base_checkpoint"]) for plan in resolved.values()}
            if len(dataset_ids) != 1 or len(checkpoints) != 1:
                raise TrainingProfileError("ablation resolution drifted across host checkpoints or immutable data")
            scientific["resolved_data"] = next(iter(resolved.values()))["scientific"]["dataset"]
            scientific["resolved_host"] = next(iter(resolved.values()))["scientific"]["base_checkpoint"]
        if self.evaluation is not None:
            scientific["evaluation"] = load_training_document(self.evaluation, config_root=self.config_root)
        return {
            "schema_version": ABLATION_PLAN_SCHEMA,
            "ablation_id": identity("training-ablation", scientific, 32),
            "suite_id": self.document["id"],
            "valid": True,
            "variant_count": len(self.variants),
            "scientific": scientific,
            "resolved_plans": resolved,
            "source": str(self.path),
            "model_initialized": False,
            "network_used": False,
            "data_preparation": "each isolated trainer resolves the same immutable source, hash-ranked whole-episode split, frame selection and frozen normalization; compare data-partition.json manifest_id across variants",
        }

    def materialize(self, output: Path, *, plan: Mapping | None = None) -> dict:
        output = Path(output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        profiles = output / "profiles"
        profiles.mkdir()
        for variant, (_, profile) in self.variants.items():
            target = profiles / f"{variant}.yaml"
            target.write_text(dumps(profile.document), encoding="utf-8")
            target.chmod(0o444)
        document = dict(plan or self.plan())
        atomic_json(output / "plan.json", document)
        atomic_json(output / "commands.json", {"schema_version": "ovlab.training-ablation-commands/v1", "commands": {variant: ["ovlab", "train", "run", str(profiles / f"{variant}.yaml")] for variant in self.variants}})
        return dict(document, output=str(output), profiles={variant: str(profiles / f"{variant}.yaml") for variant in self.variants})

    def run(self, output: Path, deployment, *, plan: Mapping) -> dict:
        """Run sequentially through the established container/finalizer owner.

        The first child publishes the shared immutable split manifest. Every
        later child's manifest must match, including frozen normalization.
        Failure evidence and completed child runs remain available.
        """
        materialized = self.materialize(output, plan=plan)
        output = Path(materialized["output"])
        result = {"schema_version": "ovlab.training-ablation-result/v1", "ablation_id": materialized["ablation_id"], "suite_id": materialized["suite_id"], "status": "running", "variants": {}, "shared_manifest_id": None, "output": str(output)}
        shared = None
        try:
            for variant, profile in materialized["profiles"].items():
                result["active_variant"] = variant
                atomic_json(output / "result.json", result, exclusive=False, mode=0o644)
                child = deployment.run(profile)
                manifest_path = Path(child["host_path"]) / "data-partition.json"
                if not manifest_path.is_file():
                    raise TrainingRuntimeError(f"ablation child {variant!r} did not publish data-partition.json")
                manifest = json.loads(manifest_path.read_text())
                from .training_data import verify_episode_manifest
                verify_episode_manifest(manifest)
                if shared is None:
                    shared = manifest
                    atomic_json(output / "shared-data-partition.json", shared)
                    result["shared_manifest_id"] = manifest["manifest_id"]
                elif manifest != shared:
                    raise TrainingRuntimeError(f"ablation child {variant!r} changed the shared episode split or action normalization")
                result["variants"][variant] = {"run_id": child["run_id"], "host_path": child["host_path"], "result": child["result"], "manifest_id": manifest["manifest_id"]}
            result["status"] = "completed"
            result.pop("active_variant", None)
        except BaseException as exc:
            result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            atomic_json(output / "result.json", result, exclusive=False, mode=0o644)
            raise
        atomic_json(output / "result.json", result, exclusive=False, mode=0o444)
        return result
