from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .types import (
    Budget,
    Hardware,
    Layout,
    ModelProfile,
    Quote,
    RequestGroup,
    RuntimeProfile,
    Target,
    WORKLOAD_SCHEMA,
)


class ValidationError(ValueError):
    """Raised when an input document is not a strict v1 workload."""


def load_input(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_float=Decimal,
            parse_constant=lambda value: (_ for _ in ()).throw(ValidationError(f"non-finite JSON number is not allowed: {value}")),
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("top-level input must be an object")
    return data


def reject_unknown_keys(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ValidationError(f"{path} contains unknown keys: {', '.join(unknown)}")


def require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    return value


def require_array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be an array")
    return value


def require_str(obj: dict[str, Any], key: str, path: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{path}.{key} must be a non-empty string")
    return value


def optional_str(obj: dict[str, Any], key: str, path: str) -> str | None:
    if key not in obj:
        return None
    value = obj[key]
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{path}.{key} must be a non-empty string when provided")
    return value


def require_int(obj: dict[str, Any], key: str, path: str, minimum: int = 1) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValidationError(f"{path}.{key} must be an integer >= {minimum}")
    return value


def require_decimal(obj: dict[str, Any], key: str, path: str, minimum: Decimal) -> Decimal:
    value = obj.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"{path}.{key} must be a decimal string")
    try:
        dec = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValidationError(f"{path}.{key} must be decimal-compatible") from exc
    if not dec.is_finite() or dec < minimum:
        raise ValidationError(f"{path}.{key} must be >= {minimum}")
    return dec


def parse_model(obj: dict[str, Any], path: str) -> ModelProfile:
    reject_unknown_keys(obj, {"id", "name", "architecture", "weightGroupsBytes", "attention", "provenance", "notes"}, path)
    architecture = require_str(obj, "architecture", path)
    if architecture not in {"dense_gqa", "mla_moe"}:
        raise ValidationError(f"{path}.architecture unsupported: {architecture}")
    weights = require_object(obj.get("weightGroupsBytes"), f"{path}.weightGroupsBytes")
    reject_unknown_keys(weights, {"shared", "routed", "replicated"}, f"{path}.weightGroupsBytes")
    parsed_weights: dict[str, int] = {}
    for group in ("shared", "routed", "replicated"):
        parsed_weights[group] = require_int(weights, group, f"{path}.weightGroupsBytes", 0)
    if architecture == "dense_gqa" and parsed_weights["routed"]:
        raise ValidationError(f"{path}.weightGroupsBytes.routed must be 0 for dense_gqa")
    attention = require_object(obj.get("attention"), f"{path}.attention")
    allowed_attention = {"layers", "attentionHeads", "kvCacheDtype"}
    if architecture == "dense_gqa":
        allowed_attention |= {"kvHeads", "headDim"}
    if architecture == "mla_moe":
        allowed_attention |= {"kvLatentDim", "ropeDim", "routedExperts", "cacheBytesPerToken"}
    reject_unknown_keys(attention, allowed_attention, f"{path}.attention")
    common_keys = ("layers", "attentionHeads", "kvCacheDtype")
    for key in common_keys:
        if key == "kvCacheDtype":
            require_str(attention, key, f"{path}.attention")
        else:
            require_int(attention, key, f"{path}.attention")
    if attention["kvCacheDtype"] not in {"bf16", "fp8"}:
        raise ValidationError(f"{path}.attention.kvCacheDtype must be bf16 or fp8")
    if architecture == "dense_gqa":
        kv_heads = require_int(attention, "kvHeads", f"{path}.attention")
        require_int(attention, "headDim", f"{path}.attention")
        attention_heads = int(attention["attentionHeads"])
        if kv_heads > attention_heads or attention_heads % kv_heads != 0:
            raise ValidationError(f"{path}.attention requires kvHeads <= attentionHeads and attentionHeads % kvHeads == 0")
    if architecture == "mla_moe":
        latent = require_int(attention, "kvLatentDim", f"{path}.attention")
        rope = require_int(attention, "ropeDim", f"{path}.attention", 0)
        if "cacheBytesPerToken" in attention:
            floor = latent + rope if attention["kvCacheDtype"] == "fp8" else (latent + rope) * 2
            cache_bytes = require_int(attention, "cacheBytesPerToken", f"{path}.attention")
            if cache_bytes < floor:
                raise ValidationError(f"{path}.attention.cacheBytesPerToken must be >= the logical KV layout floor")
        if parsed_weights["routed"]:
            require_int(attention, "routedExperts", f"{path}.attention")
    provenance = require_array(obj.get("provenance"), f"{path}.provenance")
    for i, item in enumerate(provenance):
        pobj = require_object(item, f"{path}.provenance[{i}]")
        reject_unknown_keys(pobj, {"kind", "reference", "license", "notes"}, f"{path}.provenance[{i}]")
        require_str(pobj, "kind", f"{path}.provenance[{i}]")
        optional_str(pobj, "reference", f"{path}.provenance[{i}]")
        optional_str(pobj, "license", f"{path}.provenance[{i}]")
        optional_str(pobj, "notes", f"{path}.provenance[{i}]")
    return ModelProfile(
        id=require_str(obj, "id", path),
        name=require_str(obj, "name", path),
        architecture=architecture,
        weight_groups_bytes=parsed_weights,
        attention=attention,
        provenance=provenance,
        notes=optional_str(obj, "notes", path),
    )


def parse_runtime(obj: dict[str, Any], path: str) -> RuntimeProfile:
    reject_unknown_keys(obj, {"id", "name", "supportedArchitectures", "kvCacheDtypes", "dcpSupported", "measured", "observations", "notes"}, path)
    architectures_raw = require_array(obj.get("supportedArchitectures"), f"{path}.supportedArchitectures")
    dtypes_raw = require_array(obj.get("kvCacheDtypes"), f"{path}.kvCacheDtypes")
    architectures = tuple(x for x in architectures_raw if isinstance(x, str))
    dtypes = tuple(x for x in dtypes_raw if isinstance(x, str))
    if len(architectures) != len(architectures_raw):
        raise ValidationError(f"{path}.supportedArchitectures entries must be strings")
    if len(dtypes) != len(dtypes_raw):
        raise ValidationError(f"{path}.kvCacheDtypes entries must be strings")
    if any(x not in {"dense_gqa", "mla_moe"} for x in architectures):
        raise ValidationError(f"{path}.supportedArchitectures contains an unsupported architecture")
    if any(x not in {"bf16", "fp8"} for x in dtypes):
        raise ValidationError(f"{path}.kvCacheDtypes contains an unsupported dtype")
    measured = obj.get("measured", False)
    dcp_supported = obj.get("dcpSupported", False)
    if not isinstance(measured, bool) or not isinstance(dcp_supported, bool):
        raise ValidationError(f"{path}.measured and {path}.dcpSupported must be booleans")
    observations = []
    for i, item in enumerate(require_array(obj.get("observations", []), f"{path}.observations")):
        observation = require_object(item, f"{path}.observations[{i}]")
        reject_unknown_keys(
            observation,
            {"id", "asOf", "hardware", "runtimeVersion", "source", "outcome", "notes"},
            f"{path}.observations[{i}]",
        )
        require_str(observation, "id", f"{path}.observations[{i}]")
        require_str(observation, "asOf", f"{path}.observations[{i}]")
        require_str(observation, "hardware", f"{path}.observations[{i}]")
        require_str(observation, "runtimeVersion", f"{path}.observations[{i}]")
        require_str(observation, "source", f"{path}.observations[{i}]")
        require_str(observation, "outcome", f"{path}.observations[{i}]")
        observations.append(observation)
    if measured and not observations:
        raise ValidationError(f"{path}.measured=true requires at least one structured observation")
    return RuntimeProfile(
        id=require_str(obj, "id", path),
        name=require_str(obj, "name", path),
        supported_architectures=architectures,
        kv_cache_dtypes=dtypes,
        dcp_supported=dcp_supported,
        measured=measured and bool(observations),
        observations=tuple(observations),
        notes=optional_str(obj, "notes", path),
    )


def parse_quote(obj: dict[str, Any], path: str, hardware_id: str) -> Quote:
    reject_unknown_keys(obj, {"id", "provider", "asOf", "unit", "amountUsd", "purchaseTerm", "region", "source", "condition", "missingReason"}, path)
    unit = require_str(obj, "unit", path)
    if unit not in {"per_gpu_hour", "per_node_hour"}:
        raise ValidationError(f"{path}.unit must be per_gpu_hour or per_node_hour")
    amount_value = obj.get("amountUsd")
    amount = None
    if amount_value is not None:
        amount = require_decimal(obj, "amountUsd", path, Decimal("0"))
    return Quote(
        id=require_str(obj, "id", path),
        provider=require_str(obj, "provider", path),
        hardware_id=hardware_id,
        as_of=require_str(obj, "asOf", path),
        currency="USD",
        unit=unit,
        amount=amount,
        purchase_term=require_str(obj, "purchaseTerm", path),
        region=optional_str(obj, "region", path),
        source=optional_str(obj, "source", path),
        condition=optional_str(obj, "condition", path),
        missing_reason=optional_str(obj, "missingReason", path),
    )


def parse_target(obj: dict[str, Any], path: str) -> Target:
    reject_unknown_keys(obj, {"id", "name", "hardware", "runtime", "layout", "budget", "quotes"}, path)
    hardware_obj = require_object(obj.get("hardware"), f"{path}.hardware")
    runtime_obj = require_object(obj.get("runtime"), f"{path}.runtime")
    layout_obj = require_object(obj.get("layout"), f"{path}.layout")
    budget_obj = require_object(obj.get("budget"), f"{path}.budget")
    reject_unknown_keys(hardware_obj, {"id", "name", "gpuMemoryBytes", "gpusPerNode"}, f"{path}.hardware")
    reject_unknown_keys(layout_obj, {"tensorParallel", "dataParallel", "pipelineParallel", "decodeContextParallel"}, f"{path}.layout")
    reject_unknown_keys(budget_obj, {"runtimeReserveBytes", "deviceReserveFraction"}, f"{path}.budget")
    hardware = Hardware(
        id=require_str(hardware_obj, "id", f"{path}.hardware"),
        name=require_str(hardware_obj, "name", f"{path}.hardware"),
        gpu_memory_bytes=require_int(hardware_obj, "gpuMemoryBytes", f"{path}.hardware"),
        gpus_per_node=require_int(hardware_obj, "gpusPerNode", f"{path}.hardware"),
    )
    runtime = parse_runtime(runtime_obj, f"{path}.runtime")
    layout = Layout(
        tensor_parallel=require_int(layout_obj, "tensorParallel", f"{path}.layout"),
        data_parallel=require_int(layout_obj, "dataParallel", f"{path}.layout"),
        pipeline_parallel=require_int(layout_obj, "pipelineParallel", f"{path}.layout"),
        decode_context_parallel=require_int(layout_obj, "decodeContextParallel", f"{path}.layout"),
    )
    budget = Budget(
        runtime_reserve_bytes=require_int(budget_obj, "runtimeReserveBytes", f"{path}.budget", 0),
        device_reserve_fraction=require_decimal(budget_obj, "deviceReserveFraction", f"{path}.budget", Decimal("0")),
    )
    if budget.device_reserve_fraction >= Decimal("1"):
        raise ValidationError(f"{path}.budget.deviceReserveFraction must be below 1")
    quotes = tuple(
        parse_quote(require_object(item, f"{path}.quotes[{i}]"), f"{path}.quotes[{i}]", hardware.id)
        for i, item in enumerate(require_array(obj.get("quotes", []), f"{path}.quotes"))
    )
    return Target(
        id=require_str(obj, "id", path),
        name=require_str(obj, "name", path),
        hardware=hardware,
        runtime=runtime,
        layout=layout,
        budget=budget,
        quotes=quotes,
    )


def parse_workload(data: dict[str, Any]) -> tuple[list[RequestGroup], list[ModelProfile], list[Target]]:
    reject_unknown_keys(data, {"schemaVersion", "title", "requestGroups", "models", "targets"}, "input")
    if data.get("schemaVersion") != WORKLOAD_SCHEMA:
        raise ValidationError(f"schemaVersion must be {WORKLOAD_SCHEMA}")
    optional_str(data, "title", "input")
    groups = []
    for i, item in enumerate(require_array(data.get("requestGroups"), "requestGroups")):
        obj = require_object(item, f"requestGroups[{i}]")
        reject_unknown_keys(obj, {"name", "count", "tokens"}, f"requestGroups[{i}]")
        groups.append(
            RequestGroup(
                name=require_str(obj, "name", f"requestGroups[{i}]"),
                count=require_int(obj, "count", f"requestGroups[{i}]"),
                tokens=require_int(obj, "tokens", f"requestGroups[{i}]"),
            )
        )
    models = [
        parse_model(require_object(item, f"models[{i}]"), f"models[{i}]")
        for i, item in enumerate(require_array(data.get("models"), "models"))
    ]
    targets = [
        parse_target(require_object(item, f"targets[{i}]"), f"targets[{i}]")
        for i, item in enumerate(require_array(data.get("targets"), "targets"))
    ]
    model_ids = [model.id for model in models]
    target_ids = [target.id for target in targets]
    if len(model_ids) != len(set(model_ids)):
        raise ValidationError("models must have unique ids")
    if len(target_ids) != len(set(target_ids)):
        raise ValidationError("targets must have unique ids")
    return groups, models, targets
