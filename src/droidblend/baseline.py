"""模型 A/B 的原生质量与预填充时延基线。"""

from __future__ import annotations

from statistics import mean
from typing import Any, Literal

from .artifacts import atomic_json_write
from .config import ExperimentConfig
from .data import load_jsonl
from .inputs import tokenize_example
from .metrics import aggregate_scores
from .runner import native_generate, native_prefix_kv
from .runtime import load_causal_lm, load_tokenizer


def _evaluate_native_model(model: Any, tokenizer: Any, examples: list, tensors: list, config: ExperimentConfig, device: str) -> dict[str, Any]:
    predictions: list[str] = []
    prefill_ms: list[float] = []
    generated_tokens: list[int] = []
    prompt_tokens: list[int] = []
    for example, tensor in zip(examples, tensors, strict=True):
        generated, _ = native_generate(model, tensor, config.data.max_new_tokens, device)
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True))
        generated_tokens.append(len(generated))
        prompt_tokens.append(int(tensor.full_attention_mask.sum().item()))
        for _ in range(config.profiling.warmup_repetitions):
            native_prefix_kv(model, tensor, device)
        repeats = [
            native_prefix_kv(model, tensor, device)[1]
            for _ in range(config.profiling.measure_latency_repetitions)
        ]
        prefill_ms.extend(repeats)
    quality = aggregate_scores(predictions, [item.answers for item in examples])
    return {
        "quality": quality,
        "mean_prefill_ms": mean(prefill_ms),
        "mean_prompt_tokens": mean(prompt_tokens),
        "mean_generated_tokens": mean(generated_tokens),
        "prefill_samples_ms": prefill_ms,
        "predictions": [
            {"id": item.identifier, "prediction": prediction, "answers": list(item.answers)}
            for item, prediction in zip(examples, predictions, strict=True)
        ],
    }


def evaluate_native_pair(config: ExperimentConfig, split: Literal["calibration", "evaluation"] = "calibration") -> dict[str, Any]:
    """在同一输入 token 序列上比较 A/B 原生模型，决定 Droid 迁移方向。"""
    config.assert_model_paths_configured()
    data_path = config.resolve(config.data.calibration_jsonl if split == "calibration" else config.data.evaluation_jsonl)
    examples = load_jsonl(data_path)
    # 统一使用 B tokenizer：这是 A/B KV 可比和后续缓存复用的先决条件。
    tokenizer = load_tokenizer(config.models.tokenizer_path, config.models.trust_remote_code)
    tensors = [
        tokenize_example(tokenizer, example, config.data.max_prompt_tokens, config.data.prompt_suffix_tokens, config.data.input_format)
        for example in examples
    ]
    sender = load_causal_lm(
        config.models.sender_path,
        config.models.torch_dtype,
        config.models.device_sender,
        config.models.trust_remote_code,
        config.models.attention_implementation,
    )
    sender_result = _evaluate_native_model(sender, tokenizer, examples, tensors, config, config.models.device_sender)
    del sender
    receiver = load_causal_lm(
        config.models.receiver_path,
        config.models.torch_dtype,
        config.models.device_receiver,
        config.models.trust_remote_code,
        config.models.attention_implementation,
    )
    receiver_result = _evaluate_native_model(receiver, tokenizer, examples, tensors, config, config.models.device_receiver)
    del receiver
    # DroidSpeak 的方向选择条件：接收端应当在目标任务上优于发送端。
    qa_delta = receiver_result["quality"]["qa_f1"] - sender_result["quality"]["qa_f1"]
    em_delta = receiver_result["quality"]["exact_match"] - sender_result["quality"]["exact_match"]
    result = {
        "split": split,
        "samples": len(examples),
        "shared_tokenizer_path": config.models.tokenizer_path,
        "sender_A_native": sender_result,
        "receiver_B_native": receiver_result,
        "receiver_minus_sender": {"qa_f1": qa_delta, "exact_match": em_delta},
        "droid_direction_gate_passed": qa_delta >= 0.0,
        "recommendation": (
            "可继续使用 A→B 做 DroidSpeak：B 在该数据上的 QA-F1 不低于 A。"
            if qa_delta >= 0.0
            else "暂不应固定 A→B 方向：B 的 QA-F1 低于 A；请检查 prompt/数据，或改选模型对和任务。"
        ),
    }
    output = config.output_dir / "baseline" / f"native_pair_{split}.json"
    atomic_json_write(output, result)
    return {"baseline_path": str(output), **result}
