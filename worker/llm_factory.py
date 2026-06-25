"""阶段 28: LLM 工厂。

env `LLM_PROVIDER` 切换三家：
  - gemini    : google.LLM（livekit-agents 自带包装；最新可用 gemini-3.5-flash）
  - deepseek  : DeepSeek Chat（OpenAI 兼容协议，自己包一个流式包装）
  - minimax   : MiniMax M2.7-highspeed（OpenAI 兼容协议，自己包一个流式包装）

主公对齐决策（2026-06-08）：
  - STT 已经换 Cartesia，Gemini 不再承担 STT 音频输入，3.5 flash 那个
    "audio hang 30s" bug 不影响 LLM-only 场景。
  - 不限制 LLM 输出 token（兜底 max_tokens=2048）。

设计原则：
  - 三家都暴露同样的最小 chat 接口：流式返回 token 字符串片段
  - 不在 LiveKit 框架内重写 LLM 协议；这里是轻包装，**只给 agent.py 用**
  - 失败抛出清晰异常（不 fallback，因为主公说 "STT 失败：直接报 error，
    不 fallback"——LLM 同理）
"""

from __future__ import annotations

import json
import os
import time
from typing import AsyncIterator

import httpx

# 阶段 28→30: 超时从 8s 提高到 30s。长对话上下文需要更多处理时间，
# 8s 对 MiniMax/DeepSeek 的长 context 请求太紧，会静默 timeout。
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_MAX_TOKENS = 2048

OPENAI_COMPAT_DEEPSEEK = "https://api.deepseek.com"
OPENAI_COMPAT_MINIMAX = "https://api.minimaxi.com"

MODEL_DEEPSEEK_CHAT = "deepseek-chat"
# 阶段 29: 主公拍板 MiniMax-M2.7-highspeed（platform.minimaxi.com 官方 OpenAI 兼容 schema 确认）
MODEL_MINIMAX_HS = "MiniMax-M2.7-highspeed"


# ── DeepSeek 流式包装 ───────────────────────────────────────
class DeepSeekChatStream:
    """DeepSeek Chat (OpenAI 兼容) 流式包装。"""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = MODEL_DEEPSEEK_CHAT,
        base_url: str = OPENAI_COMPAT_DEEPSEEK,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
                trust_env=False,
                http2=False,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        return self._stream(messages, temperature)

    async def _stream(
        self, messages: list[dict], temperature: float
    ) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        client = self._get_client()
        t0 = time.time()
        first_token = True
        async with client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"DeepSeek HTTP {r.status_code}: {body[:200]}"
                )
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if first_token:
                        print(f"[timing] llm_ttfb={int((time.time()-t0)*1000)}ms", flush=True)
                        first_token = False
                    yield piece


# ── MiniMax 流式包装 ─────────────────────────────────────────
class MiniMaxChatStream:
    """MiniMax M2.7-highspeed (OpenAI 兼容) 流式包装。"""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = MODEL_MINIMAX_HS,
        base_url: str = OPENAI_COMPAT_MINIMAX,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
                trust_env=False,
                http2=False,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        return self._stream(messages, temperature)

    async def _stream(
        self, messages: list[dict], temperature: float
    ) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        client = self._get_client()
        t0 = time.time()
        first_token = True
        async with client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"MiniMax HTTP {r.status_code}: {body[:200]}"
                )
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    if first_token:
                        print(f"[timing] llm_ttfb={int((time.time()-t0)*1000)}ms", flush=True)
                        first_token = False
                    yield piece


# ── 工厂 ────────────────────────────────────────────────
def build_llm_provider(provider: str | None = None):
    """根据 env LLM_PROVIDER 返回对应的实例。

    返回值是 dict { 'kind': 'gemini'|'deepseek'|'minimax'|'google_live',
                     'obj': 实例, 'model': 模型名 }

    注意：Gemini 走的是 livekit 框架的 google.LLM（不是这个文件包的流式）。
    DeepSeek / MiniMax 返回的是上文的 ChatStream 类（caller 用 .chat()）。
    """
    p = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    if p == "gemini" or p == "google":
        # 走 livekit-agents 自带的 google.LLM
        from livekit.plugins import google
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        api_key = os.getenv("GOOGLE_API_KEY", "")
        return {
            "kind": "google_live",
            "obj": google.LLM(model=model, api_key=api_key, temperature=0.7),
            "model": model,
        }
    if p == "deepseek":
        return {
            "kind": "deepseek",
            "obj": DeepSeekChatStream(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                model=os.getenv("DEEPSEEK_MODEL", MODEL_DEEPSEEK_CHAT),
            ),
            "model": os.getenv("DEEPSEEK_MODEL", MODEL_DEEPSEEK_CHAT),
        }
    if p in ("minimax", "minimax_m27", "minimax-hs"):
        return {
            "kind": "minimax",
            "obj": MiniMaxChatStream(
                api_key=os.getenv("MINIMAX_API_KEY", ""),
                model=os.getenv("MINIMAX_MODEL_NAME", MODEL_MINIMAX_HS),
            ),
            "model": os.getenv("MINIMAX_MODEL_NAME", MODEL_MINIMAX_HS),
        }
    raise ValueError(f"Unknown LLM_PROVIDER: {p!r}")


if __name__ == "__main__":
    import asyncio

    async def _self_test():
        # 跑通路径测试：哪个 LLM_PROVIDER 设了就测哪家
        prov = build_llm_provider()
        print(f"[llm_factory] kind={prov['kind']} model={prov['model']}")
        if prov["kind"] in ("deepseek", "minimax"):
            t0 = time.time()
            chunks = []
            async for c in prov["obj"].chat(
                [{"role": "user", "content": "用一句话介绍你自己"}],
                temperature=0.7,
            ):
                chunks.append(c)
            dt = time.time() - t0
            print(f"[llm_factory] {prov['kind']} elapsed={dt:.2f}s chunks={len(chunks)}")
            print("output:", "".join(chunks)[:200])
        else:
            print("[llm_factory] google.LLM skip self-test (livekit wrapped)")

    asyncio.run(_self_test())
