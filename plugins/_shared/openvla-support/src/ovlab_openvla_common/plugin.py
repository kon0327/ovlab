"""OpenVLA integration hooks. Model semantics stay in the integration package."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from ovlab_policy_sdk.plugin import PolicyPlugin, PolicyPluginValidationError as ConfigSchemaError, PolicyPluginCompatibilityError as ConfigCompatibilityError, PolicyPluginReferenceError as ConfigReferenceError
from ovlab_policy_sdk.validation import mapping, exact_type, enum, non_empty_string, number


class OpenVlaPlugin(PolicyPlugin):
    def __init__(self, config_type, component_name):
        self.config_type, self.component_name = config_type, component_name

    def deployment_context(self, policy, deployment):
        return {"settings": {
            "suite": policy["settings"].get("unnorm_key"),
            "observation": {"height": deployment.get("input", {}).get("height", 224),
                            "width": deployment.get("input", {}).get("width", 224),
                            "profile": "native_oft" if policy["type"] == "openvla_oft" else "primary_rgb"},
        }}

    def validate_checkpoint_template(self, doc, path):
        """Validate a QuIC policy whose adapter will come from training.

        Runtime validation remains strict. A template may omit adapter identity,
        while any explicitly supplied identity is still validated in full.
        """
        return self.validate_policy(doc, path, _checkpoint_template=True)

    def validate_policy(self, doc, path, *, _checkpoint_template=False):
        if doc["type"] not in ("openvla_vanilla", "openvla_lora_merged", "openvla_oft", "openvla_quic"):
            raise ConfigSchemaError(
                f"{path}.type supports openvla_vanilla, openvla_lora_merged, openvla_oft, openvla_quic or mock"
            )
        settings = mapping(doc["settings"], f"{path}.settings", required=(
            "checkpoint_id", "processor_id", "unnorm_key", "input", "runtime", "action", "raw_output"),
            optional=("quic",) if doc["type"] == "openvla_quic" else ())
        for key in ("checkpoint_id", "processor_id", "unnorm_key"):
            exact_type(settings[key], str, f"{path}.settings.{key}")
        if doc["type"] == "openvla_oft":
            input_ = mapping(settings["input"], f"{path}.settings.input", required=("cameras", "proprioception"))
            cameras = mapping(input_["cameras"], f"{path}.settings.input.cameras", required=("primary", "wrist"))
            for key, value in cameras.items(): non_empty_string(value, f"{path}.settings.input.cameras.{key}")
            non_empty_string(input_["proprioception"], f"{path}.settings.input.proprioception")
        else:
            input_ = mapping(settings["input"], f"{path}.settings.input", required=("camera",))
            exact_type(input_["camera"], str, f"{path}.settings.input.camera")
        runtime_keys = (
            "device_resource", "dtype", "attention_implementation", "local_files_only",
            "trust_remote_code", "deterministic", "synchronize_inference",
        )
        runtime_keys += ("quantization",)
        runtime = mapping(
            settings["runtime"], f"{path}.settings.runtime", required=runtime_keys
        )
        exact_type(runtime["device_resource"], str, f"{path}.settings.runtime.device_resource")
        enum(runtime["dtype"], ("bfloat16", "float16", "float32"), f"{path}.settings.runtime.dtype")
        enum(runtime["quantization"], ("none", "8bit", "4bit"), f"{path}.settings.runtime.quantization")
        if runtime["attention_implementation"] is not None: exact_type(runtime["attention_implementation"], str, f"{path}.settings.runtime.attention_implementation")
        for key in ("local_files_only", "trust_remote_code", "deterministic", "synchronize_inference"):
            exact_type(runtime[key], bool, f"{path}.settings.runtime.{key}")
        action = mapping(settings["action"], f"{path}.settings.action", required=("codec", "action_interface"))
        codecs = ("openvla-to-libero-v1", "openvla-decoded-v1") if doc["type"] == "openvla_quic" else ("openvla-to-libero-v1",)
        enum(action["codec"], codecs, f"{path}.settings.action.codec")
        non_empty_string(action["action_interface"], f"{path}.settings.action.action_interface")
        raw = mapping(settings["raw_output"], f"{path}.settings.raw_output", required=("enabled",))
        exact_type(raw["enabled"], bool, f"{path}.settings.raw_output.enabled")
        if doc["type"] == "openvla_quic":
            if "quic" not in settings:
                raise ConfigSchemaError(f"{path}.settings.quic is required")
            identity_keys = ("adapter_path", "adapter_manifest_sha256", "adapter_tensor_sha256")
            required = ("prompt_style", "center_crop", "adaptation_state")
            quic = mapping(settings["quic"], f"{path}.settings.quic",
                           required=required if _checkpoint_template else (*identity_keys, *required),
                           optional=identity_keys if _checkpoint_template else ())
            supplied_identity = set(identity_keys).intersection(quic)
            if supplied_identity and supplied_identity != set(identity_keys):
                raise ConfigSchemaError(f"{path}.settings.quic adapter identity must include path and both hashes")
            if "adapter_path" in quic:
                non_empty_string(quic["adapter_path"], f"{path}.settings.quic.adapter_path")
            for name in ("adapter_manifest_sha256", "adapter_tensor_sha256"):
                if name not in quic:
                    continue
                digest = quic[name]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ConfigSchemaError(f"{path}.settings.quic.{name} must be SHA256")
            enum(quic["prompt_style"], ("modern", "v01"), f"{path}.settings.quic.prompt_style")
            exact_type(quic["center_crop"], bool, f"{path}.settings.quic.center_crop")
            enum(quic["adaptation_state"], ("none", "merged"), f"{path}.settings.quic.adaptation_state")
            if runtime["quantization"] != "none":
                raise ConfigSchemaError("QuIC supports dense hosts only; quantization must be none")
            if raw["enabled"]:
                raise ConfigSchemaError("QuIC provider does not expose raw policy output")

    def resolve_settings(self, doc, benchmark_doc, action_spec, checkpoints, devices, *, repository_root, environment):
        from ovlab_openvla_common import LiberoActionCodecConfig, OpenVlaModelSource, OpenVlaRuntimeArtifact, method_descriptor_from_registry, vanilla_base_method_descriptor
        from ovlab_openvla_common import ModelQuantization
        settings = doc["settings"]
        quic_adapter_path = self.runtime_resources(doc, repository_root, environment).get("quic_adapter_path")
        runtime = settings["runtime"]
        if not runtime["local_files_only"]:
            raise ConfigCompatibilityError("OpenVLA production configuration requires local_files_only: true")
        try: model, processor = checkpoints[settings["checkpoint_id"]], checkpoints[settings["processor_id"]]
        except KeyError as exc: raise ConfigReferenceError(f"unknown checkpoint resource: {exc.args[0]}") from exc
        try: device = devices[runtime["device_resource"]]
        except KeyError as exc: raise ConfigReferenceError(f"unknown device resource: {runtime['device_resource']}") from exc
        obs = benchmark_doc["settings"]["observation"]
        source = lambda item: OpenVlaModelSource(
            item["source"], item["revision"], item["expected_sha256"], item.get("local_path")
        )
        if doc["type"] == "openvla_quic":
            from ovlab_openvla_quic import OpenVlaQuICSettings
            if settings["checkpoint_id"] != settings["processor_id"]:
                raise ConfigCompatibilityError("QuIC model and processor resources must be identical")
            quic = settings["quic"]
            return OpenVlaQuICSettings(
                model=source(model), unnorm_key=settings["unnorm_key"],
                adapter_path=quic_adapter_path or quic["adapter_path"], target_action_spec=action_spec,
                adapter_manifest_sha256=quic["adapter_manifest_sha256"],
                adapter_tensor_sha256=quic["adapter_tensor_sha256"],
                canonical_camera_name=settings["input"]["camera"],
                input_image_shape=(obs["height"], obs["width"], 3), device=device,
                model_dtype=runtime["dtype"], attention_implementation=runtime["attention_implementation"] or "eager",
                prompt_style=quic["prompt_style"], center_crop=quic["center_crop"],
                adaptation_state=quic["adaptation_state"], action_codec=settings["action"]["codec"],
                synchronize_inference=runtime["synchronize_inference"],
            )
        if doc["type"] == "openvla_oft":
            from ovlab_openvla_oft import OpenVlaOftArtifact, OpenVlaOftSettings
            if settings["checkpoint_id"] != settings["processor_id"]:
                raise ConfigCompatibilityError("OFT model and processor resources must be identical")
            if benchmark_doc["settings"]["suite"] != "libero_10" or obs["profile"] != "native_oft":
                raise ConfigCompatibilityError("native Gate E OFT requires LIBERO-10 and the native_oft profile")
            registry_entry = {
                "repo_id": model["source"], "revision": model["revision"],
                "expected_sha256": model["expected_sha256"], "artifact": model["artifact"],
                "method": model["method"],
            }
            artifact = OpenVlaOftArtifact.from_registry_entry(settings["checkpoint_id"], registry_entry)
            cameras = settings["input"]["cameras"]
            quantization = ModelQuantization(runtime["quantization"])
            return OpenVlaOftSettings(
                source(model), artifact, unnorm_key=settings["unnorm_key"],
                primary_camera_name=cameras["primary"], wrist_camera_name=cameras["wrist"],
                proprioception_name=settings["input"]["proprioception"],
                input_image_shape=(obs["height"], obs["width"], 3),
                device=device, quantization=quantization,
                attention_implementation=runtime["attention_implementation"],
                target_action_spec=action_spec, record_raw_output=settings["raw_output"]["enabled"],
                metadata={"execution_strategy": "open_loop_chunk"},
            )
        from ovlab_openvla_vanilla import InferenceSynchronization, ModelDType, OpenVlaVanillaSettings
        quantization = ModelQuantization(runtime["quantization"])
        method_descriptor = vanilla_base_method_descriptor()
        runtime_artifact = None
        if doc["type"] == "openvla_lora_merged":
            if settings["checkpoint_id"] != settings["processor_id"]:
                raise ConfigCompatibilityError("merged LoRA model and processor resources must be identical")
            if "artifact" not in model or "method" not in model:
                raise ConfigCompatibilityError("merged LoRA policy requires registered artifact and method identity")
            registry_entry = {
                "repo_id": model["source"],
                "revision": model["revision"],
                "expected_sha256": model["expected_sha256"],
                "artifact": model["artifact"],
                "method": model["method"],
            }
            method_descriptor = method_descriptor_from_registry(registry_entry)
            runtime_artifact = OpenVlaRuntimeArtifact.from_registry_entry(
                settings["checkpoint_id"], registry_entry
            )
            if settings["unnorm_key"] != "libero_10":
                raise ConfigCompatibilityError("official merged LoRA reference requires unnorm_key=libero_10")
            if benchmark_doc["settings"]["suite"] != "libero_10":
                raise ConfigCompatibilityError("merged LIBERO-10 LoRA is not a cross-suite acceptance policy")
        elif "artifact" in model or "method" in model:
            raise ConfigCompatibilityError("merged LoRA full weights cannot be classified as OpenVLA Vanilla")
        method_descriptor = replace(
            method_descriptor,
            quantization=quantization.value,
            declared_base_revision=(
                method_descriptor.declared_base_revision or model["revision"]
            ),
        )
        return OpenVlaVanillaSettings(
            source(model), settings["unnorm_key"], processor=source(processor),
            canonical_camera_name=settings["input"]["camera"], input_image_shape=(obs["height"], obs["width"], 3),
            device=device, model_dtype=ModelDType(runtime["dtype"]),
            quantization=quantization,
            attention_implementation=runtime["attention_implementation"], local_files_only=runtime["local_files_only"],
            trust_remote_code=runtime["trust_remote_code"], deterministic_inference=runtime["deterministic"],
            target_action_spec=action_spec, action_codec=LiberoActionCodecConfig(),
            synchronization=InferenceSynchronization.IF_CUDA if runtime["synchronize_inference"] else InferenceSynchronization.NONE,
            record_raw_output=settings["raw_output"]["enabled"],
            method_descriptor=method_descriptor, runtime_artifact=runtime_artifact,
        )

    def validate_observation(self, doc, benchmark_doc):
        policy_input = doc["settings"]["input"]
        observation = benchmark_doc["settings"]["observation"]
        benchmark_camera = (
            observation["cameras"]["primary"]["canonical_name"]
            if benchmark_doc["type"] == "libero"
            else observation["camera"]
        )
        if doc["type"] == "openvla_oft":
            cameras = policy_input["cameras"]
            supplied = {item["canonical_name"] for item in observation["cameras"].values()}
            if set(cameras.values()) != supplied:
                raise ConfigCompatibilityError("OFT image inputs differ from the benchmark observation interface")
            if policy_input["proprioception"] != "robot.proprioception" or observation["profile"] != "native_oft":
                raise ConfigCompatibilityError("OFT requires the native LIBERO proprioception observation profile")
        elif policy_input["camera"] != benchmark_camera:
            raise ConfigCompatibilityError("policy input camera is not supplied by the benchmark observation interface")
        if benchmark_doc["type"] == "mock":
            expected_proprio = doc["settings"]["input"].get("proprioception")
            if expected_proprio is not None and expected_proprio != observation["proprioception"]:
                raise ConfigCompatibilityError("policy proprioception is not supplied by the benchmark observation interface")

    def scientific_policy(self, doc):
        result = deepcopy(doc)
        if self.config_type == "openvla_quic":
            result["settings"]["quic"].pop("adapter_path", None)
        return result

    def runtime_resources(self, doc, repository_root, environment):
        if self.config_type != "openvla_quic": return {}
        value = environment.get("OVLAB_QUIC_ADAPTER_PATH", doc["settings"]["quic"]["adapter_path"])
        path = Path(value).expanduser()
        return {"quic_adapter_path": str((path if path.is_absolute() else Path(repository_root) / path).resolve())}

    def runtime_mounts(self, doc, repository_root, environment):
        if self.config_type != "openvla_quic": return []
        value = environment.get("OVLAB_QUIC_ADAPTER_HOST_PATH", doc["settings"]["quic"]["adapter_path"])
        path = Path(value).expanduser()
        return [{"source": str((path if path.is_absolute() else Path(repository_root) / path).resolve()), "target": "/checkpoints/quic-adapter", "environment": "OVLAB_QUIC_ADAPTER_PATH", "required_files": ["adapter.json", "adapter.safetensors"]}]

    def accepts_capabilities(self, capabilities):
        return capabilities.component_name == self.component_name

    def accepts_settings(self, settings):
        if self.config_type == "openvla_oft":
            from ovlab_openvla_oft import OpenVlaOftSettings
            return isinstance(settings, OpenVlaOftSettings)
        if self.config_type == "openvla_quic":
            from ovlab_openvla_quic import OpenVlaQuICSettings
            return isinstance(settings, OpenVlaQuICSettings)
        from ovlab_openvla_vanilla import OpenVlaVanillaSettings
        return isinstance(settings, OpenVlaVanillaSettings) and ((settings.method_descriptor.family.value == "lora") == (self.config_type == "openvla_lora_merged"))

    def create_adapter(self, settings):
        if self.config_type == "openvla_quic":
            from ovlab_openvla_quic import create_policy_adapter
            return create_policy_adapter(settings)
        if self.config_type == "openvla_oft":
            from ovlab_openvla_oft import OpenVlaOftAdapter
            return OpenVlaOftAdapter(settings)
        if self.config_type == "openvla_lora_merged":
            from ovlab_openvla_lora_merged import OpenVlaMergedLoraAdapter
            return OpenVlaMergedLoraAdapter(settings)
        from ovlab_openvla_vanilla import OpenVlaVanillaAdapter
        return OpenVlaVanillaAdapter(settings)

    def identity(self, capabilities):
        if self.config_type == "openvla_quic":
            from ovlab_openvla_quic.service import _identity_provider
        elif self.config_type == "openvla_oft":
            from ovlab_openvla_oft.service import _identity as _identity_provider
        elif self.config_type == "openvla_lora_merged":
            from ovlab_openvla_lora_merged.service import _identity_provider
        else:
            from ovlab_remote_policy.openvla_service import _identity_provider
        return _identity_provider(capabilities)

    def validate_checkpoint(self, entry, entry_path):
        if entry["method"].get("family") == "openvla_oft":
            artifact = mapping(
                entry["artifact"], f"{entry_path}.artifact",
                required=(
                    "form", "merge_status", "active_peft_adapter", "runtime_peft_modules",
                    "published_unmerged_adapter", "files", "parameter_counts", "byte_counts",
                ),
            )
            enum(artifact["form"], ("merged_backbone_with_auxiliary_components",), f"{entry_path}.artifact.form")
            enum(artifact["merge_status"], ("merged",), f"{entry_path}.artifact.merge_status")
            for key in ("active_peft_adapter", "runtime_peft_modules"):
                if artifact[key] is not False:
                    raise ConfigSchemaError(f"{entry_path}.artifact.{key} must be false")
            if artifact["published_unmerged_adapter"] is not True:
                raise ConfigSchemaError(f"{entry_path}.artifact.published_unmerged_adapter must be true")
            files = artifact["files"]
            exact_type(files, dict, f"{entry_path}.artifact.files")
            if not files:
                raise ConfigSchemaError(f"{entry_path}.artifact.files must not be empty")
            for name, item in files.items():
                non_empty_string(name, f"{entry_path}.artifact.files.name")
                mapping(item, f"{entry_path}.artifact.files.{name}", required=("size", "sha256"))
                exact_type(item["size"], int, f"{entry_path}.artifact.files.{name}.size")
                digest = item["sha256"]
                if item["size"] <= 0 or not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ConfigSchemaError(f"{entry_path}.artifact.files.{name} has invalid identity")
            counts = mapping(artifact["parameter_counts"], f"{entry_path}.artifact.parameter_counts", required=(
                "merged_backbone", "lora_trainable", "lora_file_tensors", "action_head",
                "proprio_projector", "auxiliary", "complete_adaptation", "total_runtime",
            ))
            byte_counts = mapping(artifact["byte_counts"], f"{entry_path}.artifact.byte_counts", required=(
                "merged_backbone_weights", "lora_file", "auxiliary", "complete_adaptation",
            ))
            if any(type(value) is not int or value <= 0 for value in (*counts.values(), *byte_counts.values())):
                raise ConfigSchemaError(f"{entry_path}.artifact parameter and byte counts must be positive integers")
            method = mapping(entry["method"], f"{entry_path}.method", required=(
                "id", "version", "family", "acronym_expansion", "backbone_adaptation",
                "declared_base_model", "declared_base_revision", "artifact_form", "backbone_merge_status",
                "runtime_active_adapter", "parallel_decoding", "action_representation", "objective",
                "action_chunk_size", "action_dimension", "normalization", "image_inputs",
                "proprioception_dimension", "film", "diffusion", "quantization",
                "adaptation_suite", "dataset_identity", "training_step", "lora", "training_provenance",
            ))
            enum(method["acronym_expansion"], ("optimized_fine_tuning",), f"{entry_path}.method.acronym_expansion")
            enum(method["quantization"], ("none",), f"{entry_path}.method.quantization")
            if method["film"] is not False or method["diffusion"] is not False:
                raise ConfigSchemaError(f"{entry_path}.method must not enable FiLM or diffusion")
            exact_type(method["lora"], dict, f"{entry_path}.method.lora")
            exact_type(method["training_provenance"], dict, f"{entry_path}.method.training_provenance")
            return
        artifact = mapping(
            entry["artifact"],
            f"{entry_path}.artifact",
            required=(
                "form", "merge_status", "active_peft_adapter", "runtime_peft_modules",
                "adapter_config", "adapter_recoverability", "files",
            ),
        )
        enum(artifact["form"], ("merged_full_weights",), f"{entry_path}.artifact.form")
        enum(artifact["merge_status"], ("merged",), f"{entry_path}.artifact.merge_status")
        for key in ("active_peft_adapter", "runtime_peft_modules"):
            exact_type(artifact[key], bool, f"{entry_path}.artifact.{key}")
            if artifact[key]:
                raise ConfigSchemaError(f"{entry_path}.artifact.{key} must be false for merged weights")
        enum(
            artifact["adapter_config"],
            ("not_present_in_published_artifact",),
            f"{entry_path}.artifact.adapter_config",
        )
        enum(
            artifact["adapter_recoverability"],
            ("not_recoverable_from_published_artifact",),
            f"{entry_path}.artifact.adapter_recoverability",
        )
        exact_type(artifact["files"], dict, f"{entry_path}.artifact.files")
        if not artifact["files"]:
            raise ConfigSchemaError(f"{entry_path}.artifact.files must not be empty")
        for name, item in artifact["files"].items():
            file_path = f"{entry_path}.artifact.files.{name}"
            non_empty_string(name, f"{file_path}.name")
            mapping(item, file_path, required=("size", "sha256"))
            exact_type(item["size"], int, f"{file_path}.size")
            if item["size"] <= 0:
                raise ConfigSchemaError(f"{file_path}.size must be positive")
            non_empty_string(item["sha256"], f"{file_path}.sha256")
            if len(item["sha256"]) != 64 or any(
                character not in "0123456789abcdef" for character in item["sha256"]
            ):
                raise ConfigSchemaError(f"{file_path}.sha256 must be a SHA-256 digest")
        method = mapping(
            entry["method"],
            f"{entry_path}.method",
            required=(
                "id", "version", "family", "declared_base_model", "declared_base_revision",
                "adaptation_suite", "quantization", "lora", "training_provenance",
            ),
        )
        for key in ("id", "version", "declared_base_model"):
            non_empty_string(method[key], f"{entry_path}.method.{key}")
        if method["declared_base_revision"] is not None:
            non_empty_string(method["declared_base_revision"], f"{entry_path}.method.declared_base_revision")
        enum(method["family"], ("lora",), f"{entry_path}.method.family")
        enum(method["adaptation_suite"], ("LIBERO-10",), f"{entry_path}.method.adaptation_suite")
        enum(method["quantization"], ("none",), f"{entry_path}.method.quantization")
        lora = mapping(
            method["lora"],
            f"{entry_path}.method.lora",
            required=(
                "rank", "alpha", "scaling", "dropout", "bias", "target_policy",
                "modules_to_save", "merge_procedure",
            ),
        )
        for key in ("rank", "alpha"):
            exact_type(lora[key], int, f"{entry_path}.method.lora.{key}")
        for key in ("scaling", "dropout"):
            number(lora[key], f"{entry_path}.method.lora.{key}")
        enum(lora["bias"], ("none",), f"{entry_path}.method.lora.bias")
        enum(lora["target_policy"], ("all-linear",), f"{entry_path}.method.lora.target_policy")
        if lora["modules_to_save"] is not None:
            raise ConfigSchemaError(f"{entry_path}.method.lora.modules_to_save must be null")
        enum(
            lora["merge_procedure"],
            ("merge_and_unload()+save_pretrained()",),
            f"{entry_path}.method.lora.merge_procedure",
        )
        exact_type(method["training_provenance"], dict, f"{entry_path}.method.training_provenance")
