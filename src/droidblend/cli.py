from __future__ import annotations

import argparse
import json
from typing import Sequence

from .baseline import evaluate_native_pair
from .capture import capture_sender_caches
from .compat import inspect_model, validate_pair
from .config import load_config
from .data import split_jsonl
from .diagnostics import direct_reuse_diagnostic, targeted_sensitivity_stability
from .evaluate import evaluate_selected_profile
from .hotpotqa import convert_hotpotqa_dataset
from .input_audit import audit_inputs
from .profile import profile_droidspeak
from .reproducibility import initialize_run
from .reports import render_reports
from .selection import select_profile
from .sensitivity import layer_swap_sensitivity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="droidblend", description="Independent DroidSpeak reproduction pipeline")
    parser.add_argument("--config", required=True, help="Experiment YAML")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-pair", help="Verify exact architecture/tokenizer compatibility")
    baseline = commands.add_parser("baseline", help="比较 A/B 原生质量和预填充时延，确认 Droid 迁移方向")
    baseline.add_argument("--split", choices=("calibration", "evaluation"), default="calibration")
    prepare = commands.add_parser("prepare-hotpotqa", help="Convert local HotpotQA dataset to DroidBlend JSONL")
    prepare.add_argument("--input", required=True, help="Directory passed to datasets.load_from_disk")
    prepare.add_argument("--output", required=True, help="Destination JSONL")
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--limit", type=int, default=None)
    split = commands.add_parser("split-jsonl", help="按固定随机种子划分标定集与保留评测集")
    split.add_argument("--input", required=True, help="待划分的完整 JSONL")
    split.add_argument("--calibration-output", required=True, help="标定集 JSONL 输出路径")
    split.add_argument("--evaluation-output", required=True, help="保留评测集 JSONL 输出路径")
    split.add_argument("--calibration-size", type=int, default=30, help="写入标定集的样本数")
    split.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    audit = commands.add_parser("audit-inputs", help="用 B tokenizer 检查 chat template 后的真实长度；不截断")
    audit.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    capture = commands.add_parser("capture-sender", help="Capture A's independent-prefix KV and E caches")
    capture.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    commands.add_parser("sensitivity", help="Swap one A KV layer into native B cache per trial")
    direct = commands.add_parser("direct-reuse", help="Figure-3 diagnostic: B decodes from all sender A KV layers")
    direct.add_argument("--split", choices=("calibration", "evaluation"), default="calibration")
    stability = commands.add_parser("sensitivity-stability", help="Paired per-example uncertainty for selected swap layers")
    stability.add_argument("--layers", required=True, help="Comma-separated layer indices, e.g. 14,15,16,19,26")
    stability.add_argument("--bootstrap-samples", type=int, default=2000)
    commands.add_parser("profile", help="Profile every configured continuous B-layer group")
    commands.add_parser("select-pareto", help="Select fastest calibration-feasible layer group")
    commands.add_parser("report", help="Render sensitivity and Pareto figures from calibration results")
    evaluate = commands.add_parser("evaluate", help="Evaluate selected group on held-out prompts")
    evaluate.add_argument("--profile", default=None, help="Override profile, e.g. layers_008_015")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    initialize_run(config, args.command)
    if args.command == "prepare-hotpotqa":
        result = convert_hotpotqa_dataset(args.input, args.output, args.split, args.limit)
    elif args.command == "split-jsonl":
        result = split_jsonl(
            args.input,
            args.calibration_output,
            args.evaluation_output,
            args.calibration_size,
            config.experiment.seed if args.seed is None else args.seed,
        )
    elif args.command == "audit-inputs":
        result = audit_inputs(config, args.split)
    elif args.command == "baseline":
        result = evaluate_native_pair(config, args.split)
    elif args.command == "validate-pair":
        config.assert_model_paths_configured()
        sender = inspect_model(config.models.sender_path, config.models.tokenizer_path, config.models.trust_remote_code)
        receiver = inspect_model(config.models.receiver_path, config.models.tokenizer_path, config.models.trust_remote_code)
        report = validate_pair(sender, receiver)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        report.require()
        return 0
    elif args.command == "capture-sender":
        result = capture_sender_caches(config, args.split)
    elif args.command == "sensitivity":
        result = layer_swap_sensitivity(config)
    elif args.command == "direct-reuse":
        result = direct_reuse_diagnostic(config, args.split)
    elif args.command == "sensitivity-stability":
        layers = [int(value.strip()) for value in args.layers.split(",") if value.strip()]
        result = targeted_sensitivity_stability(config, layers, args.bootstrap_samples)
    elif args.command == "profile":
        result = profile_droidspeak(config)
    elif args.command == "select-pareto":
        result = select_profile(config)
    elif args.command == "report":
        result = render_reports(config)
    elif args.command == "evaluate":
        result = evaluate_selected_profile(config, args.profile)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
