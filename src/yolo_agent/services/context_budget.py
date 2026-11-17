"""统一估算请求 token；优先离线 tokenizer，始终保留供应商协议误差余量。"""

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path

from langchain_core.messages.utils import count_tokens_approximately

from yolo_agent.settings import Settings


class ContextBudgetError(ValueError):
    """必要上下文无法装入窗口时明确停止，不截断指令或工具配对。"""


def context_profile(settings):
    value = f"{settings.llm_base_url}|{settings.llm_model}|{settings.llm_tokenizer_path}"
    return hashlib.sha256(value.encode()).hexdigest()


def budget_for_run(settings, state):
    matched = state.get("context_profile") == context_profile(settings)
    return TokenBudget(settings, scale=state.get("token_estimation_scale", 1) if matched else 1,
        window_factor=state.get("context_window_factor", 1) if matched else 1)


@lru_cache(maxsize=8)
def _encoding(local_path: str | None):
    if local_path:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(local_path)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer, "local_tokenizer"
    root = Path(__file__).resolve().parents[1] / "token_assets"
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        cache_root = Path(os.environ.get("TIKTOKEN_CACHE_DIR", root))
        path = cache_root / manifest["cache_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            return None, "conservative_estimate"
        # 仅在已核对的缓存存在时调用，线上不触发 tokenizer 下载。
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(root))
        import tiktoken
        return tiktoken.get_encoding(manifest["encoding"]), "o200k_base_proxy"
    except (ImportError, OSError, ValueError, KeyError):
        return None, "conservative_estimate"


class TokenBudget:
    def __init__(self, settings: Settings, *, scale: float = 1.0, window_factor: float = 1.0):
        self.settings = settings
        self.scale = max(1.0, float(scale))
        self.window_factor = max(0.1, min(1.0, float(window_factor)))
        path = str(settings.llm_tokenizer_path.resolve()) if settings.llm_tokenizer_path else None
        self.encoding, self.method = _encoding(path)
        self.output_reserve = max(settings.llm_output_tokens, settings.agent_summary_tokens)
        self.input_limit = math.floor(settings.llm_context_window * (1 - settings.llm_context_safety_ratio) * self.window_factor) - self.output_reserve

    def text(self, content: str) -> int:
        if self.method == "local_tokenizer":
            return len(self.encoding.encode(content, add_special_tokens=False).ids)
        if self.encoding is not None:
            return len(self.encoding.encode(content, disallowed_special=()))
        return count_tokens_approximately([{"role": "user", "content": content}], chars_per_token=1.0)

    def message(self, message: dict) -> int:
        public = {key: value for key, value in message.items() if not key.startswith("_")}
        images = 0
        if isinstance(public.get("content"), list):
            blocks = []
            for block in public["content"]:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    images += 1
                    blocks.append({"type": "image_url", "image_url": {"url": "[image]"}})
                else:
                    blocks.append(block)
            public["content"] = blocks
        return self.text(json.dumps(public, ensure_ascii=False, default=str, separators=(",", ":"))) + 8 + images * self.settings.agent_image_token_reserve

    def messages(self, messages: list[dict]) -> int:
        return sum(self.message(message) for message in messages)

    def request(self, messages: list[dict], tools: list[dict] = ()) -> int:
        raw = self.messages(messages) + (self.text(json.dumps(tools, ensure_ascii=False, separators=(",", ":"))) if tools else 0) + 32
        return math.ceil(raw * self.scale)

    def check(self, messages: list[dict], tools: list[dict] = ()) -> int:
        tokens = self.request(messages, tools)
        if tokens > self.input_limit:
            raise ContextBudgetError(f"请求预计需要 {tokens} 个输入 tokens，当前安全预算为 {self.input_limit}；请压缩历史或减少常驻信息，不能截断操作者指令。")
        return tokens

    def info(self) -> dict:
        return {"context_window": self.settings.llm_context_window, "input_limit": self.input_limit,
            "output_reserve": self.output_reserve, "decision_output_limit": self.settings.llm_output_tokens,
            "summary_output_limit": self.settings.agent_summary_tokens, "safety_ratio": self.settings.llm_context_safety_ratio,
            "tokenizer": self.method, "estimated": True, "calibration_scale": self.scale, "window_factor": self.window_factor,
            "image_token_reserve": self.settings.agent_image_token_reserve}


def context_length_rejected(error) -> bool:
    payload = str(getattr(error, "body", "") or error).lower()
    return any(term in payload for term in ("context_length_exceeded", "maximum context length", "context window", "input is too long", "max_position_embeddings", "上下文长度"))


def images_rejected(error) -> bool:
    payload = str(getattr(error, "body", "") or error).lower()
    return any(term in payload for term in ("image_url", "image input", "vision", "image content")) and any(term in payload for term in ("unsupported", "not support", "not allowed", "not permitted"))
