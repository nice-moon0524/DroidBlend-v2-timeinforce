"""Shared model and prompt utilities for experiment entry points."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SENDER_PATH = "/opt/hhy/models/Mistral-7B-v0.1"
DEFAULT_RECEIVER_PATH = "/opt/hhy/models/mistrallite"


def resolve_model_path(env_var: str, fallback: str) -> str:
    value = os.getenv(env_var, "").strip()
    return value or fallback


def resolve_torch_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if not hasattr(torch, dtype):
        raise ValueError(f"Unknown torch dtype: {dtype}")
    resolved = getattr(torch, dtype)
    if device == "cpu" and resolved in {torch.float16, torch.bfloat16}:
        return torch.float32
    return resolved


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
    if not model_path:
        raise ValueError("model_path must be provided explicitly")
    torch_dtype = resolve_torch_dtype(dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    ).to(device).eval()
    return model, tokenizer


def make_prompt(record: dict) -> str:
    return f"{record['context']}\n\nQuestion: {record['question']}\nAnswer:"


def tokenize_prompt(
    tokenizer,
    record: dict,
    device: str,
    max_prompt_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    kwargs = {"return_tensors": "pt", "truncation": False}
    if max_prompt_tokens is not None:
        if max_prompt_tokens < 2:
            raise ValueError("max_prompt_tokens must be at least 2")
        kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": max_prompt_tokens,
        }
    return tokenizer(make_prompt(record), **kwargs).to(device)

