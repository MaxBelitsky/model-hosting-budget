from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .engine import evaluate_workload
from .reports import dumps_markdown, write_report
from .schema import ValidationError, load_input
from .types import RESULT_SCHEMA


def require_result_object(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    return value


def require_result_array(value: object, path: str) -> list:
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be an array")
    return value


def require_result_keys(obj: dict, keys: set[str], path: str) -> None:
    missing = sorted(keys - set(obj))
    if missing:
        raise ValidationError(f"{path} missing keys: {', '.join(missing)}")


def require_result_str(obj: dict, key: str, path: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{path}.{key} must be a non-empty string")
    return value


def require_result_int(obj: dict, key: str, path: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{path}.{key} must be an integer")
    return value


def require_result_decimal_string(obj: dict, key: str, path: str) -> Decimal:
    value = obj.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"{path}.{key} must be a decimal string")
    try:
        dec = Decimal(value)
    except InvalidOperation as exc:
        raise ValidationError(f"{path}.{key} must be decimal-compatible") from exc
    if not dec.is_finite():
        raise ValidationError(f"{path}.{key} must be finite")
    return dec


def validate_result(result: dict) -> None:
    metadata = require_result_object(result.get("metadata"), "result.metadata")
    require_result_str(metadata, "title", "result.metadata")
    evaluations = require_result_array(result.get("evaluations"), "result.evaluations")
    for i, item_raw in enumerate(evaluations):
        item = require_result_object(item_raw, f"result.evaluations[{i}]")
        require_result_keys(
            item,
            {"model", "target", "layout", "statuses", "bytes", "reasons", "quotes", "selectedQuote"},
            f"result.evaluations[{i}]",
        )
        model = require_result_object(item["model"], f"result.evaluations[{i}].model")
        target = require_result_object(item["target"], f"result.evaluations[{i}].target")
        layout = require_result_object(item["layout"], f"result.evaluations[{i}].layout")
        statuses = require_result_object(item["statuses"], f"result.evaluations[{i}].statuses")
        byte_counts = require_result_object(item["bytes"], f"result.evaluations[{i}].bytes")
        reasons = require_result_object(item["reasons"], f"result.evaluations[{i}].reasons")
        extra_reason_keys = sorted(set(reasons) - {"compatibility", "capacity", "evidence"})
        if extra_reason_keys:
            raise ValidationError(f"result.evaluations[{i}].reasons contains unknown keys: {', '.join(extra_reason_keys)}")
        require_result_keys(model, {"id", "name"}, f"result.evaluations[{i}].model")
        require_result_keys(target, {"id", "name"}, f"result.evaluations[{i}].target")
        require_result_keys(layout, {"tensorParallel", "dataParallel", "expertParallel", "decodeContextParallel"}, f"result.evaluations[{i}].layout")
        require_result_keys(statuses, {"compatibility", "capacity", "evidence", "cost"}, f"result.evaluations[{i}].statuses")
        require_result_keys(byte_counts, {"planned", "perGpuBudget"}, f"result.evaluations[{i}].bytes")
        require_result_str(model, "id", f"result.evaluations[{i}].model")
        require_result_str(model, "name", f"result.evaluations[{i}].model")
        require_result_str(target, "id", f"result.evaluations[{i}].target")
        require_result_str(target, "name", f"result.evaluations[{i}].target")
        for key in ("tensorParallel", "dataParallel", "expertParallel", "decodeContextParallel"):
            require_result_int(layout, key, f"result.evaluations[{i}].layout")
        for key in ("compatibility", "capacity", "evidence", "cost"):
            require_result_str(statuses, key, f"result.evaluations[{i}].statuses")
        require_result_int(byte_counts, "planned", f"result.evaluations[{i}].bytes")
        require_result_int(byte_counts, "perGpuBudget", f"result.evaluations[{i}].bytes")
        for key in ("compatibility", "capacity", "evidence"):
            values = require_result_array(reasons.get(key), f"result.evaluations[{i}].reasons.{key}")
            if any(not isinstance(value, str) for value in values):
                raise ValidationError(f"result.evaluations[{i}].reasons.{key} entries must be strings")
        require_result_array(item["quotes"], f"result.evaluations[{i}].quotes")
        if item["selectedQuote"] is not None:
            selected = require_result_object(item["selectedQuote"], f"result.evaluations[{i}].selectedQuote")
            require_result_keys(selected, {"provider", "usdPerHour", "status"}, f"result.evaluations[{i}].selectedQuote")
            require_result_str(selected, "provider", f"result.evaluations[{i}].selectedQuote")
            require_result_decimal_string(selected, "usdPerHour", f"result.evaluations[{i}].selectedQuote")
            status = require_result_str(selected, "status", f"result.evaluations[{i}].selectedQuote")
            if status not in {"priced", "conditional"}:
                raise ValidationError(f"result.evaluations[{i}].selectedQuote.status must be priced or conditional")


def load_result(path: str) -> dict:
    try:
        result = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValidationError(f"non-finite result JSON number is not allowed: {value}")),
        )
    except FileNotFoundError as exc:
        raise ValidationError(f"result file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid result JSON: {exc}") from exc
    if not isinstance(result, dict) or result.get("schemaVersion") != RESULT_SCHEMA:
        raise ValidationError(f"result schemaVersion must be {RESULT_SCHEMA}")
    if not isinstance(result.get("metadata"), dict):
        raise ValidationError("result.metadata must be an object")
    if not isinstance(result.get("evaluations"), list):
        raise ValidationError("result.evaluations must be an array")
    validate_result(result)
    return result


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not args.offline:
        raise ValidationError("only --offline mode is supported in this MVP")
    result = evaluate_workload(load_input(args.workload))
    write_report(result, args.out)
    print(Path(args.out, "result.json"))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    result = load_result(args.result)
    model_id = args.model
    matches = [
        item
        for item in result.get("evaluations", [])
        if item["model"]["id"] == model_id or item["model"]["name"] == model_id
    ]
    if not matches:
        raise ValidationError(f"model not found in result: {model_id}")
    print(dumps_markdown({"metadata": result["metadata"], "evaluations": matches, "cheapestByModel": {}}))
    for item in matches:
        print(f"Reasons for {item['target']['name']}:")
        for group, reasons in item["reasons"].items():
            if reasons:
                print(f"  {group}:")
                for reason in reasons:
                    print(f"    - {reason}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    before = load_result(args.before)
    after = load_result(args.after)
    before_rows = {(x["model"]["id"], x["target"]["id"]): x for x in before.get("evaluations", [])}
    after_rows = {(x["model"]["id"], x["target"]["id"]): x for x in after.get("evaluations", [])}
    for key in sorted(set(before_rows) | set(after_rows)):
        b = before_rows.get(key)
        a = after_rows.get(key)
        if b is None:
            print(f"added {key[0]} on {key[1]}")
            continue
        if a is None:
            print(f"removed {key[0]} on {key[1]}")
            continue
        delta = a["bytes"]["planned"] - b["bytes"]["planned"]
        status_changed = b["statuses"] != a["statuses"]
        before_price = b["selectedQuote"]["usdPerHour"] if b.get("selectedQuote") else None
        after_price = a["selectedQuote"]["usdPerHour"] if a.get("selectedQuote") else None
        price_delta = ""
        if before_price is not None and after_price is not None:
            price_delta = f", price delta ${Decimal(after_price) - Decimal(before_price):,.2f}/hour"
        if delta or status_changed or before_price != after_price:
            print(f"{key[0]} on {key[1]}: planned delta {delta / 1_000_000_000:.3f} GB{price_delta}")
            if status_changed:
                print(f"  statuses: {b['statuses']} -> {a['statuses']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hosting-budget")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="evaluate a v1 workload JSON document")
    evaluate.add_argument("workload")
    evaluate.add_argument("--offline", action="store_true", help="required; no network access is performed")
    evaluate.add_argument("--out", default="report")
    evaluate.set_defaults(func=cmd_evaluate)
    explain = sub.add_parser("explain", help="explain one model in a result JSON file")
    explain.add_argument("result")
    explain.add_argument("--model", required=True)
    explain.set_defaults(func=cmd_explain)
    compare = sub.add_parser("compare", help="compare two result JSON files")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        return args.func(args)
    except (OSError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
