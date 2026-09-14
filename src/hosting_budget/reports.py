from __future__ import annotations

import csv
import html
import json
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any


def safe_csv(value: str) -> str:
    if value and value[0] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def dumps_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def fmt_gb(value: int) -> str:
    return f"{value / 1_000_000_000:.3f}"


def fmt_money(value: str | None) -> str:
    if value is None:
        return ""
    return f"${Decimal(value):,.2f}"


def md_cell(value: object) -> str:
    text = html.escape(str(value), quote=False).replace("\n", "<br>")
    for char in "\\`*_{}[]()#+-.!|":
        text = text.replace(char, "\\" + char)
    return text


def rows(result: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in result["evaluations"]:
        quote = item["selectedQuote"] or {}
        out.append(
            {
                "model": item["model"]["id"],
                "target": item["target"]["id"],
                "compatibility": item["statuses"]["compatibility"],
                "capacity": item["statuses"]["capacity"],
                "evidence": item["statuses"]["evidence"],
                "cost": item["statuses"]["cost"],
                "physical_gpus": str(item["layout"]["physicalGPUs"]),
                "tp": str(item["layout"]["tensorParallel"]),
                "dp": str(item["layout"]["dataParallel"]),
                "ep": str(item["layout"]["expertParallel"]),
                "dcp": str(item["layout"]["decodeContextParallel"]),
                "planned_gb": fmt_gb(item["bytes"]["planned"]),
                "budget_gb": fmt_gb(item["bytes"]["perGpuBudget"]),
                "provider": quote.get("provider", ""),
                "usd_per_hour": quote.get("usdPerHour", ""),
                "usd_per_730h": quote.get("usdPer730Hours", ""),
            }
        )
    return out


def dumps_csv(result: dict[str, Any]) -> str:
    fields = [
        "model",
        "target",
        "compatibility",
        "capacity",
        "evidence",
        "cost",
        "physical_gpus",
        "tp",
        "dp",
        "ep",
        "dcp",
        "planned_gb",
        "budget_gb",
        "provider",
        "usd_per_hour",
        "usd_per_730h",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: safe_csv(value) for key, value in row.items()} for row in rows(result))
    return buffer.getvalue()


def dumps_markdown(result: dict[str, Any]) -> str:
    lines = [f"# {md_cell(result['metadata']['title'])}", ""]
    lines.append("Analytical alpha, offline calculation. No throughput or real-model accuracy claims are made.")
    lines.append("")
    cheapest = result.get("cheapestByModel") or {}
    if cheapest:
        lines.append("Cheapest qualifying priced configuration by model:")
        for item in cheapest.values():
            quote = item["selectedQuote"]
            lines.append(
                f"- **{md_cell(item['model']['name'])}**: **{md_cell(item['target']['name'])}**, "
                f"{md_cell(quote['provider'])} {fmt_money(quote['usdPerHour'])}/hour"
            )
        lines.append("")
    lines.append("| Model | Target | Statuses | Layout | Planned GB | Budget GB | Selected quote |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for item in result["evaluations"]:
        statuses = item["statuses"]
        layout = item["layout"]
        quote = item["selectedQuote"]
        quote_text = "none"
        if quote:
            quote_text = f"{quote['provider']} {fmt_money(quote['usdPerHour'])}/hour ({quote['status']})"
        lines.append(
            "| {model} | {target} | compat={compat}; capacity={capacity}; evidence={evidence}; cost={cost} "
            "| TP{tp}/DP{dp}/EP{ep}/DCP{dcp} | {planned} | {budget} | {quote} |".format(
                model=md_cell(item["model"]["name"]),
                target=md_cell(item["target"]["name"]),
                compat=statuses["compatibility"],
                capacity=statuses["capacity"],
                evidence=statuses["evidence"],
                cost=statuses["cost"],
                tp=layout["tensorParallel"],
                dp=layout["dataParallel"],
                ep=layout["expertParallel"],
                dcp=layout["decodeContextParallel"],
                planned=fmt_gb(item["bytes"]["planned"]),
                budget=fmt_gb(item["bytes"]["perGpuBudget"]),
                quote=md_cell(quote_text),
            )
        )
    lines.append("")
    return "\n".join(lines)


def dumps_html(result: dict[str, Any]) -> str:
    table_rows = []
    for item in result["evaluations"]:
        quote = item["selectedQuote"]
        quote_text = "none"
        if quote:
            quote_text = f"{quote['provider']} {fmt_money(quote['usdPerHour'])}/hour ({quote['status']})"
        cells = [
            item["model"]["name"],
            item["target"]["name"],
            item["statuses"]["compatibility"],
            item["statuses"]["capacity"],
            item["statuses"]["evidence"],
            item["statuses"]["cost"],
            f"TP{item['layout']['tensorParallel']} / DP{item['layout']['dataParallel']} / EP{item['layout']['expertParallel']} / DCP{item['layout']['decodeContextParallel']}",
            fmt_gb(item["bytes"]["planned"]),
            fmt_gb(item["bytes"]["perGpuBudget"]),
            quote_text,
        ]
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>")
    title = html.escape(result["metadata"]["title"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 0.55rem; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    .note {{ color: #57606a; max-width: 70rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="note">Offline analytical-alpha report. Capacity, compatibility, evidence, and cost statuses are intentionally separate.</p>
  <table>
    <thead><tr><th>Model</th><th>Target</th><th>Compatibility</th><th>Capacity</th><th>Evidence</th><th>Cost</th><th>Layout</th><th>Planned GB</th><th>Budget GB</th><th>Quote</th></tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
</body>
</html>
"""


def write_report(result: dict[str, Any], out_dir: str | Path) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(dumps_json(result), encoding="utf-8")
    (path / "result.csv").write_text(dumps_csv(result), encoding="utf-8")
    (path / "report.md").write_text(dumps_markdown(result), encoding="utf-8")
    (path / "index.html").write_text(dumps_html(result), encoding="utf-8")
