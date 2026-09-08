"""Shared model and prompt utilities for experiment entry points."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SENDER_PATH = "/opt/hhy/models/Mistral-7B-v0.1"
DEFAULT_RECEIVER_PATH = "/opt/hhy/models/mistrallite"


def resolve_model_path(env_var: str, fallback: str) -> str:
    """Resolve a model directory from an env var, then the supplied fallback."""
    value = os.getenv(env_var, "").strip()
    return value or fallback


def load_model_and_tokenizer(model_path: str, device: str = "cuda", dtype: str = "float16"):
    if not model_path:
        raise ValueError("model_path must be provided explicitly")
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype).to(device).eval()
    return model, tokenizer


def make_prompt(record: dict) -> str:
    return f"{record['context']}\n\nQuestion: {record['question']}\nAnswer:"


def tokenize_prompt(tokenizer, record: dict, device: str) -> dict[str, torch.Tensor]:
    return tokenizer(make_prompt(record), return_tensors="pt", truncation=False).to(device)
