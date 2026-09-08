from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import PromptExample
from .errors import ConfigurationError


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the question based strictly on the given context. "
    "Provide a very short, concise answer containing ONLY the precise entity name or phrase, "
    "without writing full sentences or repeating the question."
)


@dataclass
class PromptTensors:
    full_input_ids: Any
    full_attention_mask: Any
    prefix_input_ids: Any
    prefix_attention_mask: Any
    suffix_input_ids: Any


@dataclass(frozen=True)
class RenderedInput:
    rendered_prompt: str
    input_ids: Any
    attention_mask: Any

    @property
    def token_count(self) -> int:
        return int(self.attention_mask.sum().item())


def hotpotqa_messages(example: PromptExample) -> list[dict[str, str]]:
    """构造与原 DroidSpeak 一致的 system/user 消息，而非平铺纯文本。"""
    context = example.metadata.get("context")
    question = example.metadata.get("question")
    if not isinstance(context, str) or not context.strip() or not isinstance(question, str) or not question.strip():
        raise ConfigurationError(
            f"Example {example.identifier} lacks string metadata.context / metadata.question. "
            "Re-run prepare-hotpotqa and split-jsonl with the chat16k protocol."
        )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Context:\n{context.strip()}\n\nQuestion: {question.strip()}"},
    ]


def render_example(tokenizer: Any, example: PromptExample, input_format: str) -> str:
    if input_format == "plain":
        return example.prompt
    if input_format != "receiver_chat":
        raise ConfigurationError(f"Unsupported data.input_format: {input_format!r}")
    if not getattr(tokenizer, "chat_template", None):
        raise ConfigurationError("data.input_format=receiver_chat requires a tokenizer with chat_template")
    return tokenizer.apply_chat_template(hotpotqa_messages(example), tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer: Any, example: PromptExample, input_format: str) -> RenderedInput:
    """编码完整输入，不截断；调用方必须显式决定容量是否足够。"""
    rendered = render_example(tokenizer, example, input_format)
    # The rendered chat template already contains BOS/role markers. Adding
    # special tokens again would break exact sender/receiver token alignment.
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False, truncation=False)
    return RenderedInput(rendered, encoded["input_ids"], encoded["attention_mask"])


def tokenize_example(
    tokenizer: Any,
    example: PromptExample,
    max_prompt_tokens: int,
    prompt_suffix_tokens: int,
    input_format: str,
) -> PromptTensors:
    """Create one shared A/B input sequence and reject, never hide, truncation."""
    if prompt_suffix_tokens < 1:
        raise ConfigurationError("data.prompt_suffix_tokens must be at least one")
    rendered = encode_example(tokenizer, example, input_format)
    token_count = rendered.token_count
    if token_count > max_prompt_tokens:
        raise ConfigurationError(
            f"Example {example.identifier} is {token_count} tokens, above max_prompt_tokens={max_prompt_tokens}. "
            "The chat16k protocol never silently truncates context; increase the configured limit or select a GPU-capacity-safe dataset."
        )
    if token_count <= prompt_suffix_tokens:
        raise ConfigurationError(
            f"Prompt contains only {token_count} token(s), insufficient for prefix plus {prompt_suffix_tokens}-token suffix"
        )
    split = token_count - prompt_suffix_tokens
    return PromptTensors(
        full_input_ids=rendered.input_ids,
        full_attention_mask=rendered.attention_mask,
        prefix_input_ids=rendered.input_ids[:, :split],
        prefix_attention_mask=rendered.attention_mask[:, :split],
        suffix_input_ids=rendered.input_ids[:, split:token_count],
    )
