from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .schema import parse_workload
from .types import ModelProfile, RequestGroup, RESULT_SCHEMA, Target


def ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def bins_for(dp: int, token_counts: list[int]) -> list[list[int]]:
    bins: list[list[int]] = [[] for _ in range(dp)]
    for tokens in sorted(token_counts, reverse=True):
        index = min(range(dp), key=lambda i: (sum(bins[i]), len(bins[i]), i))
        bins[index].append(tokens)
    return bins


def bytes_for_dtype(dtype: str) -> int:
    if dtype == "bf16":
        return 2
    if dtype == "fp8":
        return 1
    raise ValueError(f"unsupported KV dtype: {dtype}")


def fraction_from_decimal(value: Decimal) -> Fraction:
    return Fraction(value)


def fraction_to_decimal_string(value: Fraction) -> str:
    sign = "-" if value < 0 else ""
    numerator = abs(value.numerator)
    denominator = value.denominator
    power_two = power_five = 0
    while denominator % 2 == 0:
        denominator //= 2
        power_two += 1
    while denominator % 5 == 0:
        denominator //= 5
        power_five += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    scale = max(power_two, power_five)
    scaled = numerator * (2 ** (scale - power_two)) * (5 ** (scale - power_five))
    if scale == 0:
        return sign + str(scaled)
    text = str(scaled).rjust(scale + 1, "0")
    whole = text[:-scale]
    frac = text[-scale:].rstrip("0")
    return sign + (whole if not frac else f"{whole}.{frac}")


def compatibility(model: ModelProfile, target: Target) -> tuple[str, list[str]]:
    reasons: list[str] = []
    layout = target.layout
    attention = model.attention
    if layout.pipeline_parallel != 1:
        reasons.append("pipeline parallelism is not implemented in this alpha")
    if layout.physical_gpus < 1:
        reasons.append("physical GPU count must be positive")
    if attention["attentionHeads"] % layout.tensor_parallel != 0:
        reasons.append("tensorParallel must divide attentionHeads")
    if model.architecture == "dense_gqa":
        kv_heads = int(attention["kvHeads"])
        if layout.tensor_parallel <= kv_heads and kv_heads % layout.tensor_parallel != 0:
            reasons.append("kvHeads must divide tensorParallel or tensorParallel must divide kvHeads")
        if layout.tensor_parallel > kv_heads and layout.tensor_parallel % kv_heads != 0:
            reasons.append("tensorParallel above kvHeads must be an integer replication of KV-head groups")
    if model.architecture == "mla_moe" and model.weight_groups_bytes["routed"]:
        ep = layout.tensor_parallel * layout.data_parallel
        experts = int(attention["routedExperts"])
        if experts % ep != 0:
            reasons.append("expertParallel must divide routedExperts for uniform routed shards")
    if model.architecture == "mla_moe" and attention["kvCacheDtype"] == "fp8" and "cacheBytesPerToken" not in attention:
        reasons.append("FP8 MLA requires explicit cacheBytesPerToken from a reviewed runtime layout")
    if model.architecture not in target.runtime.supported_architectures:
        reasons.append("runtime does not declare support for model architecture")
    if attention["kvCacheDtype"] not in target.runtime.kv_cache_dtypes:
        reasons.append("runtime does not declare support for the requested KV cache dtype")
    if layout.decode_context_parallel > 1:
        if not target.runtime.dcp_supported:
            reasons.append("decode context parallelism requested but runtime does not declare support")
        if layout.tensor_parallel % layout.decode_context_parallel != 0:
            reasons.append("decodeContextParallel must divide tensorParallel")
        if model.architecture == "dense_gqa":
            kv_heads = int(attention["kvHeads"])
            max_dcp = max(1, layout.tensor_parallel // kv_heads)
            if layout.decode_context_parallel > max_dcp:
                reasons.append("decodeContextParallel exceeds available replicated KV-head rank groups")
    return ("fail" if reasons else "pass"), reasons


def evidence_status(model: ModelProfile, target: Target) -> tuple[str, list[str]]:
    notes: list[str] = []
    if not model.provenance:
        notes.append("model profile has no provenance records")
    elif any(not item.get("reference") for item in model.provenance):
        notes.append("one or more model provenance records omit a source reference")
    if target.runtime.observations:
        notes.append("runtime observations are attached as context but are not matched to this exact prediction in the MVP")
    else:
        notes.append("runtime profile has no measured GPU observation")
    return "analytical_alpha", notes


def weight_breakdown(model: ModelProfile, target: Target) -> dict[str, int]:
    tp = target.layout.tensor_parallel
    dp = target.layout.data_parallel
    groups = model.weight_groups_bytes
    return {
        "shared": ceildiv(groups["shared"], tp),
        "routed": ceildiv(groups["routed"], tp * dp),
        "replicated": groups["replicated"],
    }


def cache_breakdown_for_bin(model: ModelProfile, target: Target, seqs: list[int], block_tokens: int = 256) -> dict[str, int]:
    attention = model.attention
    tp = target.layout.tensor_parallel
    dcp = target.layout.decode_context_parallel
    dtype_bytes = bytes_for_dtype(attention["kvCacheDtype"])
    units = sum(ceildiv(tokens, block_tokens * dcp) * block_tokens for tokens in seqs)
    if model.architecture == "dense_gqa":
        shard = min(tp, int(attention["kvHeads"]))
        bytes_per_token = 2 * int(attention["kvHeads"]) * int(attention["headDim"]) * dtype_bytes
        return {"dense_gqa_kv": ceildiv(int(attention["layers"]) * bytes_per_token * units, shard)}
    if model.architecture == "mla_moe":
        width = int(attention.get("cacheBytesPerToken") or ((int(attention["kvLatentDim"]) + int(attention["ropeDim"])) * dtype_bytes))
        return {"mla_latent_kv": int(attention["layers"]) * width * units}
    raise ValueError(f"unsupported architecture: {model.architecture}")


def normalize_quote(quote: Any, target: Target) -> dict[str, Any]:
    nodes = ceildiv(target.layout.physical_gpus, target.hardware.gpus_per_node)
    if quote.amount is None:
        return {
            "id": quote.id,
            "provider": quote.provider,
            "hardwareId": quote.hardware_id,
            "asOf": quote.as_of,
            "status": "missing",
            "missingReason": quote.missing_reason or "amountUsd not provided",
            "condition": quote.condition,
            "source": quote.source,
        }
    amount = fraction_from_decimal(quote.amount)
    per_node = amount * target.hardware.gpus_per_node if quote.unit == "per_gpu_hour" else amount
    total_hour = per_node * nodes
    status = "conditional" if quote.condition else "priced"
    return {
        "id": quote.id,
        "provider": quote.provider,
        "hardwareId": quote.hardware_id,
        "asOf": quote.as_of,
        "status": status,
        "currency": quote.currency,
        "sourceUnit": quote.unit,
        "amountUsd": str(quote.amount),
        "usdPerNodeHour": fraction_to_decimal_string(per_node),
        "nodes": nodes,
        "usdPerHour": fraction_to_decimal_string(total_hour),
        "usdPer730Hours": fraction_to_decimal_string(total_hour * 730),
        "purchaseTerm": quote.purchase_term,
        "region": quote.region,
        "condition": quote.condition,
        "source": quote.source,
    }


def evaluate_pair(model: ModelProfile, target: Target, groups: list[RequestGroup]) -> dict[str, Any]:
    compat_status, compat_reasons = compatibility(model, target)
    evidence, evidence_notes = evidence_status(model, target)
    request_tokens = [tokens for group in groups for tokens in group.expand()]
    request_bins = bins_for(target.layout.data_parallel, request_tokens)
    reserve_fraction = fraction_from_decimal(target.budget.device_reserve_fraction)
    budget_fraction = target.hardware.gpu_memory_bytes * (Fraction(1) - reserve_fraction)
    budget_bytes = budget_fraction.numerator // budget_fraction.denominator
    weight_parts: dict[str, int] = {}
    cache_parts: dict[str, int] = {}
    weight_bytes = cache_bytes = payload_bytes = planned_bytes = 0
    worst_bin = 0
    capacity_status = "unknown"
    capacity_reasons = list(compat_reasons)
    if compat_status == "pass":
        weight_parts = weight_breakdown(model, target)
        per_bin_cache = [cache_breakdown_for_bin(model, target, seqs) for seqs in request_bins]
        worst_bin = max(range(len(per_bin_cache)), key=lambda i: sum(per_bin_cache[i].values()))
        cache_parts = per_bin_cache[worst_bin]
        weight_bytes = sum(weight_parts.values())
        cache_bytes = sum(cache_parts.values())
        payload_bytes = weight_bytes + cache_bytes
        planned_bytes = payload_bytes + target.budget.runtime_reserve_bytes
        if planned_bytes <= budget_fraction:
            capacity_status = "fits"
            capacity_reasons.append("planned bytes are within the reserved per-GPU budget")
        elif planned_bytes <= target.hardware.gpu_memory_bytes:
            capacity_status = "tight"
            capacity_reasons.append("planned bytes fit nominal memory but exceed the configured reserve")
        else:
            capacity_status = "exceeds"
            if weight_bytes <= target.hardware.gpu_memory_bytes and cache_bytes > 0:
                capacity_reasons.append("weights fit nominal memory but workload KV cache pushes the plan over budget")
            else:
                capacity_reasons.append("planned bytes exceed nominal per-GPU memory")
    ep = target.layout.tensor_parallel * target.layout.data_parallel if model.weight_groups_bytes["routed"] else 1
    quote_rows = [normalize_quote(quote, target) for quote in target.quotes]
    priced = [row for row in quote_rows if row["status"] in {"priced", "conditional"} and capacity_status == "fits"]
    priced.sort(key=lambda row: (Decimal(row["usdPerHour"]), row["status"] == "conditional", row["provider"]))
    if priced:
        cost_status = "priced" if priced[0]["status"] == "priced" else "conditional"
        selected_quote = priced[0]
    elif any(row["status"] == "missing" for row in quote_rows):
        cost_status = "quote_missing"
        selected_quote = None
    else:
        cost_status = "no_qualifying_quote"
        selected_quote = None
    return {
        "model": {
            "id": model.id,
            "name": model.name,
            "architecture": model.architecture,
            "weightGroupsBytes": model.weight_groups_bytes,
            "attention": model.attention,
        },
        "target": {"id": target.id, "name": target.name},
        "layout": {
            "tensorParallel": target.layout.tensor_parallel,
            "dataParallel": target.layout.data_parallel,
            "pipelineParallel": target.layout.pipeline_parallel,
            "decodeContextParallel": target.layout.decode_context_parallel,
            "physicalGPUs": target.layout.physical_gpus,
            "expertParallel": ep,
        },
        "hardware": {
            "id": target.hardware.id,
            "name": target.hardware.name,
            "gpuMemoryBytes": target.hardware.gpu_memory_bytes,
            "gpusPerNode": target.hardware.gpus_per_node,
        },
        "runtime": {
            "id": target.runtime.id,
            "name": target.runtime.name,
            "measured": target.runtime.measured,
            "supportedArchitectures": list(target.runtime.supported_architectures),
            "kvCacheDtypes": list(target.runtime.kv_cache_dtypes),
            "dcpSupported": target.runtime.dcp_supported,
            "observations": list(target.runtime.observations),
        },
        "assumptions": {
            "cacheAccounting": {
                "blockTokens": 256,
                "dcpDividesTokenBlocksForSupportedLayouts": True,
                "pipelineParallelImplemented": False,
                "routedWeightsShardedByExpertParallelOnce": True,
            },
            "budget": {
                "runtimeReserveBytes": target.budget.runtime_reserve_bytes,
                "deviceReserveFraction": str(target.budget.device_reserve_fraction),
            },
            "modelProvenance": model.provenance,
            "modelNotes": model.notes,
        },
        "statuses": {
            "compatibility": compat_status,
            "capacity": capacity_status,
            "evidence": evidence,
            "cost": cost_status,
        },
        "reasons": {
            "compatibility": compat_reasons,
            "capacity": capacity_reasons,
            "evidence": evidence_notes,
        },
        "bytes": {
            "weights": weight_bytes,
            "cache": cache_bytes,
            "payload": payload_bytes,
            "runtimeReserve": target.budget.runtime_reserve_bytes,
            "planned": planned_bytes,
            "perGpuBudget": budget_bytes,
            "nominalPerGpu": target.hardware.gpu_memory_bytes,
        },
        "weightBreakdown": weight_parts,
        "cacheBreakdown": cache_parts,
        "requestBins": request_bins,
        "worstBin": worst_bin,
        "quotes": quote_rows,
        "selectedQuote": selected_quote,
    }


def evaluate_workload(data: dict[str, Any]) -> dict[str, Any]:
    groups, models, targets = parse_workload(data)
    normalized_input = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    evaluations = [evaluate_pair(model, target, groups) for model in models for target in targets]
    cheapest_by_model: dict[str, Any] = {}
    for item in evaluations:
        if item["statuses"]["capacity"] != "fits" or item["selectedQuote"] is None:
            continue
        current = cheapest_by_model.get(item["model"]["id"])
        if current is None or Decimal(item["selectedQuote"]["usdPerHour"]) < Decimal(current["selectedQuote"]["usdPerHour"]):
            cheapest_by_model[item["model"]["id"]] = item
    return {
        "schemaVersion": RESULT_SCHEMA,
        "inputSchemaVersion": data["schemaVersion"],
        "metadata": {
            "title": data.get("title", "Hosting budget report"),
            "offline": True,
            "analyticalAlpha": True,
            "inputSha256": hashlib.sha256(normalized_input.encode("utf-8")).hexdigest(),
        },
        "requestGroups": [group.__dict__ for group in groups],
        "evaluations": evaluations,
        "cheapestByModel": cheapest_by_model,
    }
