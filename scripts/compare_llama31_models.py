#!/usr/bin/env python
"""比对本地 ModelScope/Hugging Face checkpoint 是否满足 DroidBlend 的硬兼容条件。

无需加载模型权重，不联网；只读取 config、tokenizer 和词表，因此下载未完成时也能
明确指出缺失的目录或文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from droidblend.compat import inspect_model, validate_pair  # noqa: E402
from droidblend.errors import DroidBlendError  # noqa: E402


DEFAULT_SENDER = "/opt/hhy/models/Llama-3.1-8B"
DEFAULT_RECEIVER = "/opt/hhy/models/Llama-3.1-8B-Instruct"


def vocabulary_digest(tokenizer) -> str:
    """对完整 token->id 映射做哈希，避免只比较 vocab_size 的假阳性。"""
    rows = sorted(tokenizer.get_vocab().items())
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chat_template_digest(tokenizer) -> str:
    """Return a stable digest of the renderer used to form shared input_ids."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def tokenizer_probe(sender_path: str, receiver_path: str, trust_remote_code: bool) -> dict:
    from transformers import AutoTokenizer

    sender = AutoTokenizer.from_pretrained(sender_path, local_files_only=True, trust_remote_code=trust_remote_code, use_fast=True)
    receiver = AutoTokenizer.from_pretrained(receiver_path, local_files_only=True, trust_remote_code=trust_remote_code, use_fast=True)
    probes = [
        "A short plain-text knowledge chunk.",
        "Context: Paris is the capital of France. Question: What is the capital of France?",
        "中文知识片段：北京是中国的首都。",
    ]
    probe_results = []
    for text in probes:
        left = sender(text, add_special_tokens=True)["input_ids"]
        right = receiver(text, add_special_tokens=True)["input_ids"]
        probe_results.append({"text": text, "same_input_ids": left == right, "token_count": len(left)})
    special_ids = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    special = {key: [getattr(sender, key, None), getattr(receiver, key, None)] for key in special_ids}
    # A 从不用于生成；实验统一用 B tokenizer 编码并由 B 终止生成。因此 EOS 差异
    # 必须记录，但不能被错误当作 KV 复用的结构性不兼容。
    shared_input_specials = ("bos_token_id", "pad_token_id", "unk_token_id")
    return {
        "sender_class": sender.__class__.__name__,
        "receiver_class": receiver.__class__.__name__,
        "sender_vocab_digest": vocabulary_digest(sender),
        "receiver_vocab_digest": vocabulary_digest(receiver),
        "sender_chat_template_sha256": chat_template_digest(sender),
        "receiver_chat_template_sha256": chat_template_digest(receiver),
        "same_vocab_mapping": vocabulary_digest(sender) == vocabulary_digest(receiver),
        "special_token_ids": special,
        "same_special_token_ids": all(left == right for left, right in special.values()),
        "same_shared_input_special_token_ids": all(special[key][0] == special[key][1] for key in shared_input_specials),
        "eos_difference": special["eos_token_id"][0] != special["eos_token_id"][1],
        # chat template 可以不同；Droid 的硬要求是实验中实际传入的 input_ids 相同。
        "same_chat_template": getattr(sender, "chat_template", None) == getattr(receiver, "chat_template", None),
        "input_id_probes": probe_results,
        "all_probes_equal": all(row["same_input_ids"] for row in probe_results),
    }


def safetensor_shape_probe(sender_path: str, receiver_path: str) -> dict:
    """只读取 safetensors 文件头，比较全部权重名和形状，不加载 8B 权重。"""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        return {"available": False, "compatible": None, "reason": f"缺少 safetensors：{exc}"}

    def read_shapes(model_path: str) -> dict[str, tuple[int, ...]]:
        files = sorted(Path(model_path).glob("*.safetensors"))
        if not files:
            raise ValueError(f"{model_path} 中没有 safetensors 权重文件")
        output: dict[str, tuple[int, ...]] = {}
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in output:
                        raise ValueError(f"重复的权重名：{key}")
                    output[key] = tuple(handle.get_slice(key).get_shape())
        return output

    try:
        sender = read_shapes(sender_path)
        receiver = read_shapes(receiver_path)
    except (OSError, ValueError) as exc:
        return {"available": False, "compatible": None, "reason": str(exc)}
    missing_in_receiver = sorted(set(sender) - set(receiver))
    missing_in_sender = sorted(set(receiver) - set(sender))
    shape_mismatches = [
        {"name": name, "sender": list(sender[name]), "receiver": list(receiver[name])}
        for name in sorted(set(sender) & set(receiver))
        if sender[name] != receiver[name]
    ]
    return {
        "available": True,
        "sender_tensor_count": len(sender),
        "receiver_tensor_count": len(receiver),
        "missing_in_receiver": missing_in_receiver,
        "missing_in_sender": missing_in_sender,
        "shape_mismatches": shape_mismatches,
        "compatible": not missing_in_receiver and not missing_in_sender and not shape_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="比对两个本地 Llama checkpoint 的 DroidBlend 兼容性")
    parser.add_argument("--sender", default=DEFAULT_SENDER, help="模型 A 路径")
    parser.add_argument("--receiver", default=DEFAULT_RECEIVER, help="模型 B 路径")
    parser.add_argument("--tokenizer", default=DEFAULT_RECEIVER, help="DroidBlend 实验实际使用的统一 tokenizer 路径")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-weight-shape-check", action="store_true", help="跳过 safetensors 参数名/形状检查")
    parser.add_argument("--output", default=None, help="可选：保存 JSON 报告的位置")
    args = parser.parse_args()
    try:
        sender = inspect_model(args.sender, args.tokenizer, args.trust_remote_code)
        receiver = inspect_model(args.receiver, args.tokenizer, args.trust_remote_code)
        architecture = validate_pair(sender, receiver)
        tokenizers = tokenizer_probe(args.sender, args.receiver, args.trust_remote_code)
        weights = {"available": False, "compatible": None, "reason": "用户指定跳过"} if args.skip_weight_shape_check else safetensor_shape_probe(args.sender, args.receiver)
    except (DroidBlendError, OSError, ValueError) as exc:
        print(f"比对失败：{exc}", file=sys.stderr)
        return 2
    hard_compatible = architecture.compatible and tokenizers["same_vocab_mapping"] and tokenizers["same_shared_input_special_token_ids"] and tokenizers["all_probes_equal"]
    ready_for_experiment = hard_compatible and weights["compatible"] is True
    report = {
        "sender_path": args.sender,
        "receiver_path": args.receiver,
        "architecture": architecture.to_dict(),
        "tokenizer": tokenizers,
        "weight_shapes": weights,
        "hard_compatible_for_droidblend": hard_compatible,
        "ready_for_experiment": ready_for_experiment,
        "conclusion": "可进行 DroidBlend 实验" if ready_for_experiment else "不可直接进行实验；请先处理上面的不一致项",
        "note": "chat_template 与 eos_token_id 不相同本身不是硬错误；实际实验始终使用 config 中 B tokenizer 生成一套相同 input_ids，且只由 B 生成并判断 EOS。",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ready_for_experiment else 1


if __name__ == "__main__":
    raise SystemExit(main())
