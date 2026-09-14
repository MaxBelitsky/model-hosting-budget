from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from hosting_budget.cli import load_result
from hosting_budget.engine import evaluate_workload
from hosting_budget.reports import dumps_csv, dumps_html, dumps_markdown
from hosting_budget.schema import ValidationError, load_input


ROOT = Path(__file__).resolve().parents[1]


def read_example(name: str) -> dict:
    return load_input(ROOT / "examples" / name)


def base_gqa() -> dict:
    return {
        "schemaVersion": "hosting-budget.workload.v1",
        "title": "unit",
        "requestGroups": [{"name": "r", "count": 1, "tokens": 256}],
        "models": [
            {
                "id": "gqa",
                "name": "GQA",
                "architecture": "dense_gqa",
                "weightGroupsBytes": {"shared": 1024, "routed": 0, "replicated": 0},
                "attention": {
                    "layers": 1,
                    "attentionHeads": 8,
                    "kvHeads": 2,
                    "headDim": 1,
                    "kvCacheDtype": "bf16",
                },
                "provenance": [{"kind": "synthetic", "reference": "tests/test_engine.py"}],
            }
        ],
        "targets": [
            {
                "id": "t",
                "name": "T",
                "hardware": {"id": "h", "name": "H", "gpuMemoryBytes": 10240, "gpusPerNode": 8},
                "runtime": {
                    "id": "rt",
                    "name": "RT",
                    "supportedArchitectures": ["dense_gqa"],
                    "kvCacheDtypes": ["bf16"],
                    "dcpSupported": True,
                    "measured": False,
                },
                "layout": {
                    "tensorParallel": 2,
                    "dataParallel": 1,
                    "pipelineParallel": 1,
                    "decodeContextParallel": 1,
                },
                "budget": {"runtimeReserveBytes": 0, "deviceReserveFraction": "0"},
                "quotes": [],
            }
        ],
    }


def first_eval(doc: dict) -> dict:
    return evaluate_workload(doc)["evaluations"][0]


def test_shipped_examples_match_published_workload_schema():
    schema = json.loads((ROOT / "schemas" / "workload-v1.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for path in sorted((ROOT / "examples").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(data), key=lambda error: list(error.path))
        assert errors == [], f"{path.name}: {[error.message for error in errors]}"


def test_kv_failure_example_weights_fit_but_kv_exceeds():
    row = first_eval(read_example("kv_failure.json"))
    assert row["statuses"]["compatibility"] == "pass"
    assert row["statuses"]["capacity"] == "exceeds"
    assert row["bytes"]["weights"] == 10_000_000_000
    assert row["bytes"]["weights"] < row["bytes"]["nominalPerGpu"]
    assert row["bytes"]["planned"] > row["bytes"]["nominalPerGpu"]
    assert "KV cache" in " ".join(row["reasons"]["capacity"])


def test_tp_dp_dcp_changes_weight_and_cache_residency():
    result = evaluate_workload(read_example("tp_dp_dcp.json"))
    rows = {row["target"]["id"]: row for row in result["evaluations"]}
    assert rows["tp4-dp1-dcp1"]["weightBreakdown"]["routed"] == 40_000_000_000
    assert rows["tp2-dp2-dcp1"]["weightBreakdown"]["routed"] == 40_000_000_000
    assert rows["tp4-dp1-dcp1"]["weightBreakdown"]["shared"] == 20_000_000_000
    assert rows["tp2-dp2-dcp1"]["weightBreakdown"]["shared"] == 40_000_000_000
    assert rows["tp4-dp1-dcp4"]["bytes"]["cache"] < rows["tp4-dp1-dcp1"]["bytes"]["cache"] / 3
    assert rows["tp4-dp1-dcp4"]["bytes"]["weights"] == rows["tp4-dp1-dcp1"]["bytes"]["weights"]
    assert {row["statuses"]["evidence"] for row in rows.values()} == {"analytical_alpha"}


def test_quote_units_use_whole_node_billing_and_preserve_missing_conditional():
    result = evaluate_workload(read_example("quote_units.json"))
    rows = {row["target"]["id"]: row for row in result["evaluations"]}
    per_gpu = rows["per-gpu-8gpu-node"]["selectedQuote"]
    per_node = rows["per-node-8gpu-node"]["selectedQuote"]
    assert per_gpu["usdPerNodeHour"] == "16"
    assert per_gpu["usdPerHour"] == "16"
    assert per_gpu["nodes"] == 1
    assert per_node["provider"] == "Conditional Cloud"
    assert rows["per-node-8gpu-node"]["statuses"]["cost"] == "conditional"
    assert {quote["status"] for quote in rows["per-node-8gpu-node"]["quotes"]} == {"priced", "conditional", "missing"}


def test_dense_gqa_kv_head_topology_rules():
    doc = base_gqa()
    doc["models"][0]["attention"]["attentionHeads"] = 24
    doc["models"][0]["attention"]["kvHeads"] = 6
    doc["targets"][0]["layout"]["tensorParallel"] = 4
    row = first_eval(doc)
    assert row["statuses"]["compatibility"] == "fail"
    assert any("KV-head" in reason or "kvHeads" in reason for reason in row["reasons"]["compatibility"])

    doc["models"][0]["attention"]["attentionHeads"] = 8
    doc["models"][0]["attention"]["kvHeads"] = 2
    doc["targets"][0]["layout"]["tensorParallel"] = 8
    row = first_eval(doc)
    assert row["statuses"]["compatibility"] == "pass"
    assert row["cacheBreakdown"]["dense_gqa_kv"] == 1024


def test_complete_requests_are_binned_without_splitting():
    doc = base_gqa()
    doc["requestGroups"] = [
        {"name": "large", "count": 1, "tokens": 1000},
        {"name": "medium", "count": 2, "tokens": 700},
        {"name": "small", "count": 1, "tokens": 100},
    ]
    doc["targets"][0]["layout"]["dataParallel"] = 2
    row = first_eval(doc)
    assert sorted(token for bin_ in row["requestBins"] for token in bin_) == [100, 700, 700, 1000]
    assert row["requestBins"] == [[1000, 100], [700, 700]]


def test_mla_requires_cache_width_and_uniform_expert_shards():
    doc = read_example("tp_dp_dcp.json")
    del doc["models"][0]["attention"]["cacheBytesPerToken"]
    row = first_eval(doc)
    assert row["statuses"]["compatibility"] == "fail"
    assert any("cacheBytesPerToken" in reason for reason in row["reasons"]["compatibility"])

    doc = read_example("tp_dp_dcp.json")
    doc["models"][0]["attention"]["routedExperts"] = 10
    row = first_eval(doc)
    assert row["statuses"]["compatibility"] == "fail"
    assert any("expertParallel must divide routedExperts" in reason for reason in row["reasons"]["compatibility"])


def test_strict_schema_rejects_unknown_bool_negative_and_float_decimal():
    doc = base_gqa()
    doc["unexpected"] = True
    with pytest.raises(ValidationError):
        evaluate_workload(doc)

    doc = base_gqa()
    doc["title"] = None
    with pytest.raises(ValidationError):
        evaluate_workload(doc)

    doc = base_gqa()
    doc["requestGroups"][0]["count"] = True
    with pytest.raises(ValidationError):
        evaluate_workload(doc)

    doc = base_gqa()
    doc["targets"][0]["budget"]["runtimeReserveBytes"] = -1
    with pytest.raises(ValidationError):
        evaluate_workload(doc)

    doc = base_gqa()
    doc["targets"][0]["budget"]["deviceReserveFraction"] = 0.1
    with pytest.raises(ValidationError):
        evaluate_workload(doc)


def test_exact_budget_boundary_uses_fraction_not_decimal_context():
    doc = base_gqa()
    doc["models"][0]["weightGroupsBytes"]["shared"] = 1024
    doc["models"][0]["attention"]["layers"] = 0
    with pytest.raises(ValidationError):
        evaluate_workload(doc)

    doc = base_gqa()
    doc["requestGroups"][0]["tokens"] = 1
    doc["models"][0]["weightGroupsBytes"]["shared"] = 0
    doc["models"][0]["attention"]["layers"] = 1
    doc["models"][0]["attention"]["headDim"] = 1
    doc["targets"][0]["layout"]["tensorParallel"] = 2
    doc["targets"][0]["budget"]["runtimeReserveBytes"] = 0
    doc["targets"][0]["budget"]["deviceReserveFraction"] = "0.90000000000000000000000000000000000001"
    row = first_eval(doc)
    assert row["bytes"]["planned"] == 1024
    assert row["bytes"]["perGpuBudget"] == 1023
    assert row["statuses"]["capacity"] == "tight"


def test_reports_escape_html_markdown_and_csv_formula_prefixes():
    doc = base_gqa()
    doc["title"] = "<b>|demo</b>"
    doc["models"][0]["id"] = "=model"
    doc["models"][0]["name"] = "<script>|name"
    doc["targets"][0]["id"] = "+target"
    result = evaluate_workload(doc)
    csv_text = dumps_csv(result)
    md_text = dumps_markdown(result)
    html_text = dumps_html(result)
    assert "'=model" in csv_text
    assert "'+target" in csv_text
    assert "&lt;script&gt;" in md_text
    assert "\\|" in md_text
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_result_loader_rejects_malformed_nested_rows(tmp_path):
    malformed = {"schemaVersion": "hosting-budget.result.v1", "metadata": {}, "evaluations": [{}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))

    malformed = evaluate_workload(base_gqa())
    del malformed["evaluations"][0]["selectedQuote"]
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))

    malformed = evaluate_workload(base_gqa())
    del malformed["metadata"]["title"]
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))

    malformed = evaluate_workload(base_gqa())
    malformed["evaluations"][0]["bytes"]["perGpuBudget"] = "10240"
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))

    malformed = evaluate_workload(base_gqa())
    malformed["evaluations"][0]["reasons"]["capacity"] = [1]
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))

    malformed = evaluate_workload(base_gqa())
    malformed["evaluations"][0]["reasons"]["extra"] = 1
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))

    priced = read_example("quote_units.json")
    malformed = evaluate_workload(priced)
    del malformed["evaluations"][0]["selectedQuote"]["status"]
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))

    path.write_text('{"schemaVersion":"hosting-budget.result.v1","metadata":{"title":"x"},"evaluations":[NaN]}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_result(str(path))


def test_duplicate_model_and_target_ids_are_rejected():
    doc = base_gqa()
    doc["models"].append(copy.deepcopy(doc["models"][0]))
    with pytest.raises(ValidationError):
        evaluate_workload(doc)

    doc = base_gqa()
    doc["targets"].append(copy.deepcopy(doc["targets"][0]))
    with pytest.raises(ValidationError):
        evaluate_workload(doc)


def test_pp_is_rejected_until_implemented():
    doc = base_gqa()
    doc["targets"][0]["layout"]["pipelineParallel"] = 2
    row = first_eval(doc)
    assert row["statuses"]["compatibility"] == "fail"
    assert row["layout"]["physicalGPUs"] == 4


def test_result_preserves_audit_assumptions_and_input_hash():
    result = evaluate_workload(read_example("quote_units.json"))
    row = result["evaluations"][0]
    assert result["metadata"]["inputSha256"]
    assert row["assumptions"]["cacheAccounting"]["routedWeightsShardedByExpertParallelOnce"] is True
    assert row["assumptions"]["modelProvenance"]
    assert row["runtime"]["supportedArchitectures"] == ["dense_gqa"]


def test_attached_observations_do_not_promote_prediction_evidence():
    doc = base_gqa()
    doc["targets"][0]["runtime"]["measured"] = True
    doc["targets"][0]["runtime"]["observations"] = [
        {
            "id": "obs-1",
            "asOf": "2026-09-10",
            "hardware": "synthetic",
            "runtimeVersion": "0.0.0",
            "source": "tests",
            "outcome": "failure",
        }
    ]
    row = first_eval(doc)
    assert row["runtime"]["measured"] is True
    assert row["statuses"]["evidence"] == "analytical_alpha"
    assert "not matched to this exact prediction" in " ".join(row["reasons"]["evidence"])
