from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


HYBRID_RE = re.compile(r"^hybrid_layers_(\d+)_(\d+)_ratio_")
DROIDSPEAK_RE = re.compile(r"^droidspeak_layers_(\d+)_(\d+)$")


def _ratio_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, default)
    return float(value) if value is not None else default


def _span_count(start: int, end: int) -> int:
    return end - start + 1


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float, digits: int = 2) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{value * 100:.{digits}f}%"


def collect_rows(results_root: Path, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(results_root.glob("ratio_*/summary.json")):
        summary = _read_json(summary_path)
        dataset_results = summary.get("datasets", {}).get(dataset)
        if not dataset_results:
            continue

        droidspeak_by_span: dict[tuple[int, int], tuple[str, dict[str, Any]]] = {}
        for name, result in dataset_results.items():
            match = DROIDSPEAK_RE.match(name)
            if match:
                span = (int(match.group(1)), int(match.group(2)))
                droidspeak_by_span[span] = (name, result)

        for name, result in dataset_results.items():
            match = HYBRID_RE.match(name)
            if not match:
                continue

            start, end = int(match.group(1)), int(match.group(2))
            span = (start, end)
            baseline_name, baseline = droidspeak_by_span.get(span, ("", {}))
            params = result.get("parameters", {})
            token_ratio = float(params.get("token_recompute_ratio", math.nan))
            layer_count = _span_count(start, end)

            f1 = _metric(result, "f1")
            baseline_f1 = _metric(baseline, "f1")
            p50 = _metric(result, "prefill_latency_s_p50")
            baseline_p50 = _metric(baseline, "prefill_latency_s_p50")
            mean = _metric(result, "prefill_latency_s_mean")
            baseline_mean = _metric(baseline, "prefill_latency_s_mean")
            p95 = _metric(result, "prefill_latency_s_p95")
            baseline_p95 = _metric(baseline, "prefill_latency_s_p95")
            f1_delta = f1 - baseline_f1
            p50_delta = p50 - baseline_p50
            mean_delta = mean - baseline_mean
            p95_delta = p95 - baseline_p95

            rows.append(
                {
                    "ratio_dir": summary_path.parent.name,
                    "method": name,
                    "baseline_method": baseline_name,
                    "token_recompute_ratio": token_ratio,
                    "token_ratio_label": _ratio_label(token_ratio),
                    "span_start": start,
                    "span_end": end,
                    "span": f"{start}-{end}",
                    "recompute_layer_count": layer_count,
                    "f1": f1,
                    "droidspeak_f1": baseline_f1,
                    "f1_delta": f1_delta,
                    "f1_delta_pct_point": f1_delta * 100,
                    "prefill_latency_s_mean": mean,
                    "droidspeak_latency_s_mean": baseline_mean,
                    "latency_mean_delta_s": mean_delta,
                    "latency_mean_ratio": mean / baseline_mean if baseline_mean > 0 else math.nan,
                    "prefill_latency_s_p50": p50,
                    "droidspeak_latency_s_p50": baseline_p50,
                    "latency_p50_delta_s": p50_delta,
                    "latency_p50_ratio": p50 / baseline_p50 if baseline_p50 > 0 else math.nan,
                    "prefill_latency_s_p95": p95,
                    "droidspeak_latency_s_p95": baseline_p95,
                    "latency_p95_delta_s": p95_delta,
                    "latency_p95_ratio": p95 / baseline_p95 if baseline_p95 > 0 else math.nan,
                    "recompute_work_ratio": float(params.get("recompute_work_ratio", math.nan)),
                    "full_recompute_layer_count_recorded": params.get("full_recompute_layer_count", ""),
                    "selected_count": params.get("selected_count", ""),
                    "candidate_count": params.get("candidate_count", ""),
                    "selected_fraction": params.get("selected_fraction", ""),
                    "samples": result.get("samples", ""),
                }
            )

    return rows


def build_report(
    all_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    output_dir: Path,
    dataset: str,
    f1_epsilon: float,
) -> None:
    def table(rows: list[dict[str, Any]], max_rows: int = 12) -> list[str]:
        lines = [
            "|ratio|span|layers|F1|DroidSpeak F1|Delta F1|p50 latency|DroidSpeak p50|Delta p50|work|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows[:max_rows]:
            lines.append(
                "|"
                + "|".join(
                    [
                        _fmt(row["token_recompute_ratio"], 2),
                        row["span"],
                        str(row["recompute_layer_count"]),
                        _fmt(row["f1"]),
                        _fmt(row["droidspeak_f1"]),
                        _fmt(row["f1_delta"]),
                        _fmt(row["prefill_latency_s_p50"]),
                        _fmt(row["droidspeak_latency_s_p50"]),
                        _fmt(row["latency_p50_delta_s"]),
                        _fmt_pct(row["recompute_work_ratio"]),
                    ]
                )
                + "|"
            )
        return lines

    by_ratio: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_ratio[row["token_recompute_ratio"]].append(row)

    best_by_ratio = []
    for ratio, rows in sorted(by_ratio.items()):
        best_by_ratio.append(sorted(rows, key=lambda r: (r["prefill_latency_s_p50"], r["recompute_work_ratio"]))[0])

    best_quality_by_ratio = []
    for ratio, rows in sorted(by_ratio.items()):
        best_quality_by_ratio.append(sorted(rows, key=lambda r: (-r["f1_delta"], r["prefill_latency_s_p50"]))[0])

    overall_fast = sorted(selected_rows, key=lambda r: (r["prefill_latency_s_p50"], r["recompute_work_ratio"]))
    overall_work = sorted(selected_rows, key=lambda r: (r["recompute_work_ratio"], r["prefill_latency_s_p50"]))
    overall_quality = sorted(selected_rows, key=lambda r: (-r["f1_delta"], r["prefill_latency_s_p50"]))
    faster_than_droidspeak = [r for r in selected_rows if r["latency_p50_delta_s"] <= 0]

    lines: list[str] = []
    lines.append("# DroidBlend candidate configuration analysis")
    lines.append("")
    lines.append(f"- Dataset: `{dataset}`")
    lines.append("- Selection rule: keep `hybrid_layers_*` only when `DroidBlend F1 >= same-span DroidSpeak F1`.")
    lines.append(f"- Numerical tolerance: `{f1_epsilon}`.")
    lines.append("- Ranking latency: `prefill_latency_s_p50` is used first; mean and p95 are kept in CSV.")
    lines.append("")
    lines.append("## Overall counts")
    lines.append("")
    lines.append(f"- Hybrid configurations checked: {len(all_rows)}")
    lines.append(f"- Configurations with no F1 drop: {len(selected_rows)}")
    lines.append(f"- Configurations with no F1 drop and p50 latency not higher than DroidSpeak: {len(faster_than_droidspeak)}")
    lines.append("")

    if best_by_ratio:
        lines.append("## Fastest no-F1-drop configuration per token ratio")
        lines.append("")
        lines.extend(table(best_by_ratio, max_rows=len(best_by_ratio)))
        lines.append("")

    if overall_fast:
        lines.append("## Fastest selected configurations")
        lines.append("")
        lines.extend(table(overall_fast, max_rows=15))
        lines.append("")

    if overall_work:
        lines.append("## Lowest recompute-work selected configurations")
        lines.append("")
        lines.extend(table(overall_work, max_rows=15))
        lines.append("")

    if overall_quality:
        lines.append("## Largest F1 gain among selected configurations")
        lines.append("")
        lines.extend(table(overall_quality, max_rows=15))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The primary selected set is suitable for the next-stage experiment because every row preserves or improves F1 "
        "relative to the DroidSpeak baseline with the same full recompute span. The fastest rows are best for latency-oriented "
        "follow-up tests; the lowest-work rows are best for exploring how far recomputation can be reduced without quality loss."
    )
    lines.append(
        "If no selected row has lower p50 latency than its DroidSpeak baseline, the current implementation has not yet shown a "
        "direct latency win over DroidSpeak under this timing protocol. In that case, the next step should be timing decomposition "
        "inside DroidBlend: layer-0 selection, full-span recompute, sparse-token recompute, cache merge, and first-token finalization."
    )
    lines.append("")

    (output_dir / "candidate_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")

    _write_csv(
        output_dir / "best_by_ratio.csv",
        best_by_ratio,
        [
            "token_recompute_ratio",
            "span",
            "recompute_layer_count",
            "f1",
            "droidspeak_f1",
            "f1_delta",
            "prefill_latency_s_p50",
            "droidspeak_latency_s_p50",
            "latency_p50_delta_s",
            "latency_p50_ratio",
            "recompute_work_ratio",
            "selected_count",
            "candidate_count",
            "selected_fraction",
            "method",
            "baseline_method",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select DroidBlend configurations that do not reduce F1.")
    parser.add_argument("--results-root", default="results/hybrid_by_ratio", type=Path)
    parser.add_argument("--dataset", default="hotpotqa_50")
    parser.add_argument("--output-dir", default="results/hybrid_by_ratio/analysis", type=Path)
    parser.add_argument("--f1-epsilon", default=1e-12, type=float)
    args = parser.parse_args()

    all_rows = collect_rows(args.results_root, args.dataset)
    selected_rows = [row for row in all_rows if row["f1_delta"] >= -args.f1_epsilon]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ratio_dir",
        "token_recompute_ratio",
        "span",
        "span_start",
        "span_end",
        "recompute_layer_count",
        "f1",
        "droidspeak_f1",
        "f1_delta",
        "f1_delta_pct_point",
        "prefill_latency_s_mean",
        "droidspeak_latency_s_mean",
        "latency_mean_delta_s",
        "latency_mean_ratio",
        "prefill_latency_s_p50",
        "droidspeak_latency_s_p50",
        "latency_p50_delta_s",
        "latency_p50_ratio",
        "prefill_latency_s_p95",
        "droidspeak_latency_s_p95",
        "latency_p95_delta_s",
        "latency_p95_ratio",
        "recompute_work_ratio",
        "selected_count",
        "candidate_count",
        "selected_fraction",
        "samples",
        "method",
        "baseline_method",
        "full_recompute_layer_count_recorded",
    ]
    _write_csv(args.output_dir / "all_hybrid_configs.csv", all_rows, fieldnames)
    _write_csv(args.output_dir / "selected_no_f1_drop_configs.csv", selected_rows, fieldnames)
    build_report(all_rows, selected_rows, args.output_dir, args.dataset, args.f1_epsilon)

    print(
        json.dumps(
            {
                "all_hybrid_configs": str(args.output_dir / "all_hybrid_configs.csv"),
                "selected_no_f1_drop_configs": str(args.output_dir / "selected_no_f1_drop_configs.csv"),
                "best_by_ratio": str(args.output_dir / "best_by_ratio.csv"),
                "report": str(args.output_dir / "candidate_analysis_report.md"),
                "checked": len(all_rows),
                "selected": len(selected_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
