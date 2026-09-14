from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


WORKLOAD_SCHEMA = "hosting-budget.workload.v1"
RESULT_SCHEMA = "hosting-budget.result.v1"


@dataclass(frozen=True)
class RequestGroup:
    name: str
    count: int
    tokens: int

    def expand(self) -> list[int]:
        return [self.tokens] * self.count


@dataclass(frozen=True)
class ModelProfile:
    id: str
    name: str
    architecture: str
    weight_groups_bytes: dict[str, int]
    attention: dict[str, Any]
    provenance: list[dict[str, Any]]
    notes: str | None = None


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    name: str
    supported_architectures: tuple[str, ...]
    kv_cache_dtypes: tuple[str, ...]
    dcp_supported: bool
    measured: bool
    observations: tuple[dict[str, Any], ...] = ()
    notes: str | None = None


@dataclass(frozen=True)
class Hardware:
    id: str
    name: str
    gpu_memory_bytes: int
    gpus_per_node: int = 8


@dataclass(frozen=True)
class Layout:
    tensor_parallel: int
    data_parallel: int
    pipeline_parallel: int = 1
    decode_context_parallel: int = 1

    @property
    def physical_gpus(self) -> int:
        return self.tensor_parallel * self.data_parallel * self.pipeline_parallel


@dataclass(frozen=True)
class Budget:
    runtime_reserve_bytes: int
    device_reserve_fraction: Decimal


@dataclass(frozen=True)
class Quote:
    id: str
    provider: str
    hardware_id: str
    as_of: str
    currency: str
    unit: str
    amount: Decimal | None
    purchase_term: str
    region: str | None = None
    source: str | None = None
    condition: str | None = None
    missing_reason: str | None = None


@dataclass(frozen=True)
class Target:
    id: str
    name: str
    hardware: Hardware
    runtime: RuntimeProfile
    layout: Layout
    budget: Budget
    quotes: tuple[Quote, ...]
