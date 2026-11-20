"""OVLAB-owned named QuIC profiles, placement rules, and experiment selectors."""
import copy
import re


PROFILES = {
    "QP1": dict(orders=[1], block_count=8, share_across_blocks=False, base_size=None),
    "QP2": dict(orders=[1, 2], block_count=8, share_across_blocks=False, base_size=None),
    "QP3": dict(orders=[1, 2, 3], block_count=16, share_across_blocks=False, base_size=None),
    # n=3 is an explicit stress candidate, not a demonstrated stability threshold.
    "QP4": dict(orders=[1, 2, 3], block_count=32, share_across_blocks=True, base_size=3),
}
PLACEMENTS = {
    "A1": {"vision": ["QP2", "QP3", "QP4"]},
    "A2": {"projector": ["QP1", "QP2", "QP3"]},
    "A3": {"language": ["QP2", "QP3", "QP4"]},
    "A4": {"action": ["QP1"]},
    "A5": {"projector": ["QP2", "QP3"], "language": ["QP2", "QP3"]},
    "A6": {"projector": ["QP2", "QP3"], "language": ["QP2", "QP3"], "action": ["QP1"]},
    "A7": {"vision": ["QP2", "QP3", "QP4"], "projector": ["QP2", "QP3"],
           "language": ["QP2", "QP3", "QP4"], "action": ["QP1"]},
    "A8": {"vision": ["QP2", "QP3"], "action": ["QP1"]},
    "A9": {"vision": ["QP2", "QP3", "QP4"],
           "language": ["QP2", "QP3", "QP4"]},
}



def region(name, num_layers, protected_last_n):
    if name.startswith("vision_backbone."):
        return "vision"
    if name.startswith("projector."):
        return "projector"
    if name == "language_model.lm_head":
        return "action"
    m = re.match(r"language_model\.model\.layers\.(\d+)\.", name)
    if m:
        return "action" if int(m[1]) >= num_layers - protected_last_n else "language"
    return "other"


def adapter_config(spec):
    from quic_petf_vla import AdapterConfig
    profile = spec.get("profile", "custom")
    if profile not in {*PROFILES, "custom"}:
        raise ValueError(f"Unknown profile {profile}")
    values = copy.deepcopy(PROFILES.get(profile, {}))
    if spec.get("operation", "determinant") != "determinant":
        raise ValueError("QuIC uses determinant compounds; other minor operations are not this method")
    values.update({k: v for k, v in spec.items() if k not in {"profile", "block_size", "operation"}})
    values.setdefault("orthogonal", True)
    if values.get("base_size") == "largest_feasible":
        values["base_size"] = None
    if profile in PROFILES:
        if list(values["orders"]) != PROFILES[profile]["orders"] or not values["orthogonal"]:
            raise ValueError(f"{profile} fixes its compound set and orthogonality; use profile: custom for ablations")
        if profile in {"QP1", "QP2"} and values["share_across_blocks"]:
            raise ValueError(f"{profile} does not share blocks; use profile: custom")
        if profile == "QP4" and not values["share_across_blocks"]:
            raise ValueError("QP4 requires shared blocks")
    return AdapterConfig(**values)


def _selector_matches(lookup, selector, num_layers, last_n):
    if not isinstance(selector, dict):
        raise ValueError("Every target must be a selector object")
    if ("names" in selector) == ("pattern" in selector):
        raise ValueError("Specify exactly one of names or pattern per selector")
    if "names" in selector:
        matches = selector["names"]
        if not isinstance(matches, list) or not matches:
            raise ValueError("names must be a nonempty list")
        missing = set(matches) - lookup.keys()
        if missing:
            raise ValueError(f"Unknown target module(s): {sorted(missing)}")
    else:
        rx = re.compile(selector["pattern"])
        matches = [name for name in lookup if rx.fullmatch(name)]
    if selector.get("region"):
        matches = [name for name in matches if region(name, num_layers, last_n) == selector["region"]]
    if "layers" in selector:
        if not isinstance(selector["layers"], list) or any(type(i) is not int or i < 0 for i in selector["layers"]):
            raise ValueError("layers must contain nonnegative integer indices")
        layer_ids = set(selector["layers"])
        matches = [name for name in matches if (match := re.search(r"(?:layers|blocks)\.(\d+)\.", name))
                   and int(match[1]) in layer_ids]
    if not matches:
        raise ValueError(f"Selector matches no layers: {selector}")
    return matches


def target_coverage(inventory, config, targets, num_layers):
    """Record every matched path, including explicitly excluded inactive modules."""
    lookup = {entry["name"]: entry for entry in inventory}
    last_n = config.get("placement", {}).get("action_proximal_last_n", 4)
    matched = set()
    for selector in config["targets"]:
        matched.update(_selector_matches(lookup, selector, num_layers, last_n))
    active = {name for name in matched if lookup[name].get("used_in_policy", True)}
    selected = set(targets)
    excluded_inactive = matched - active - selected
    return {"selectors": copy.deepcopy(config["targets"]),
            "matched_module_count": len(matched), "selected_module_count": len(selected),
            "active_matched_module_count": len(active),
            "uncovered_active_module_count": len(active - selected),
            "active_coverage_fraction": len(active & selected) / len(active) if active else None,
            "matched_coverage_fraction": len(matched & selected) / len(matched) if matched else None,
            "matched_modules": [copy.deepcopy(lookup[name]) for name in sorted(matched)],
            "selected_modules": sorted(selected),
            "uncovered_active_modules": sorted(active - selected),
            "excluded_inactive_count": len(excluded_inactive),
            "excluded_inactive_fraction": len(excluded_inactive) / len(matched) if matched else None,
            "excluded_inactive_modules": [copy.deepcopy(lookup[name]) for name in sorted(excluded_inactive)]}


def resolve_targets(inventory, config, num_layers):
    """Select once by full path; reject missing matches, overlaps and invalid placements."""
    place = config.get("placement", {})
    placement_id = place.get("id", "custom")
    if placement_id not in {*PLACEMENTS, "custom"}:
        raise ValueError(f"Unknown placement {placement_id}")
    last_n = place.get("action_proximal_last_n", 4)
    if type(last_n) is not int or type(place.get("allow_action_proximal", False)) is not bool:
        raise ValueError("action_proximal_last_n must be integer and allow_action_proximal must be boolean")
    if not 0 <= last_n <= num_layers:
        raise ValueError("action_proximal_last_n is outside host layer range")
    lookup = {entry["name"]: entry for entry in inventory}
    selected = {}
    manifest = []
    for selector in config["targets"]:
        matches = _selector_matches(lookup, selector, num_layers, last_n)
        if type(selector.get("policy_active_only", False)) is not bool:
            raise ValueError("policy_active_only must be boolean")
        if selector.get("policy_active_only", False):
            matches = [name for name in matches if lookup[name].get("used_in_policy", True)]
        if not matches:
            raise ValueError(f"Selector matches no policy-active layers: {selector}")
        spec = dict(config.get("adapter", {}))
        profile_override = selector.get("profile", selector.get("adapter", {}).get("profile"))
        if profile_override is not None and profile_override != spec.get("profile", "custom"):
            # A named profile switch resets profile defaults; dtype/orientation
            # remain global and explicit per-selector adapter values apply last.
            for key in ("orders", "block_count", "block_size", "base_size", "share_across_blocks"):
                spec.pop(key, None)
            spec["profile"] = profile_override
        spec.update(selector.get("adapter", {}))
        for name in sorted(matches):
            if name in selected:
                raise ValueError(f"Overlapping selectors for {name}")
            if not lookup[name].get("used_in_policy", True):
                raise ValueError(f"Layer does not contribute to policy outputs: {name}")
            if not lookup[name].get("supported", True):
                raise ValueError(f"Unsupported QuIC target type {lookup[name]['type']}: {name}")
            reg = region(name, num_layers, last_n)
            if reg == "action" and not place.get("allow_action_proximal", False):
                raise ValueError(f"Protected action-proximal layer {name}; declare allow_action_proximal: true")
            profile = spec.get("profile", "custom")
            if placement_id in PLACEMENTS:
                allowed = PLACEMENTS[placement_id]
                if reg not in allowed or profile not in allowed[reg]:
                    raise ValueError(f"{placement_id} does not allow {reg}/{profile} at {name}")
            layer_spec = dict(spec)
            if "block_size" in layer_spec:
                size = layer_spec.pop("block_size")
                if type(size) is not int:
                    raise ValueError("block_size must be an integer")
                dim = lookup[name]["out_features"]
                if size <= 0 or dim % size:
                    raise ValueError(f"block_size {size} does not divide output {dim}: {name}")
                count = dim // size
                if "block_count" in layer_spec and layer_spec["block_count"] != count:
                    raise ValueError(f"Contradictory block_size/block_count at {name}")
                layer_spec["block_count"] = count
            cfg = adapter_config(layer_spec)
            if lookup[name].get("module_type") == "Conv2d" and cfg.block_count % lookup[name]["groups"]:
                raise ValueError(f"Conv2d block_count must be divisible by groups: {name}")
            resolved = cfg.resolve(lookup[name]["out_features"])
            selected[name] = cfg
            manifest.append(dict(lookup[name], region=reg, profile=profile, resolved=resolved.to_dict()))
    if placement_id in PLACEMENTS:
        seen = {entry["region"] for entry in manifest}
        required = set(PLACEMENTS[placement_id])
        if seen != required:
            raise ValueError(f"{placement_id} must select every declared region {sorted(required)}; got {sorted(seen)}")
    return selected, manifest
