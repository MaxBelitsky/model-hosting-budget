"""Offline analytical-alpha GPU hosting budget evaluator."""

from .engine import evaluate_workload
from .schema import ValidationError

__all__ = ["ValidationError", "evaluate_workload"]
__version__ = "0.1.0"
