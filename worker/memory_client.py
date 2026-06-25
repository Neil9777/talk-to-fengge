"""OpenViking HTTP 客户端：session 生命周期 + search。

使用 vendor/openviking-ov.gemini.conf 中配置的地址。
默认连接 127.0.0.1:1933。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class MemoryConfig:
    base_url: str = "http://127.0.0.1:1933"
    account: str = "yehuiyu"
    user: str = "yehuiyu"
    timeout: float = 30.0


@dataclass
class MemoryClient:
    config: MemoryConfig = field(default_factory=MemoryConfig)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                trust_env=False,
            )
        return self._client

    async def health(self) -> bool:
        """检查 OpenViking 是否在线。"""
        try:
            resp = await self.client.get("/health")
            data = resp.json()
            return data.get("healthy", False) or data.get("status") == "ok"
        except Exception:
            return False

    @staticmethod
    def _unwrap_result(data: Any) -> Any:
        if isinstance(data, dict) and "result" in data:
            return data.get("result")
        return data

    async def create_session(self, label: str = "") -> str:
        """创建新 session，返回 session_id。"""
        payload: dict[str, Any] = {
            "account": self.config.account,
            "user": self.config.user,
        }
        if label:
            payload["label"] = label
        resp = await self.client.post("/api/v1/sessions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        result = self._unwrap_result(data)
        if isinstance(result, dict):
            session_id = result.get("session_id") or result.get("id")
            if isinstance(session_id, str) and session_id:
                return session_id
        raise KeyError(f"session_id not found in response: {data}")

    async def list_sessions(self) -> list[dict[str, Any]]:
        """列出当前 account 下的所有 session。"""
        resp = await self.client.get(
            "/api/v1/sessions",
            params={"account": self.config.account, "user": self.config.user},
        )
        resp.raise_for_status()
        data = resp.json()
        result = self._unwrap_result(data)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("sessions", [])
        return []

    async def add_messages(
        self, session_id: str, messages: list[dict[str, str]]
    ) -> None:
        """向 session 批量添加消息。

        Args:
            session_id: session ID
            messages: 消息列表，每条消息包含 role 和 content
                      例如 [{"role": "user", "content": "你好"}]
        """
        # OpenViking 的 /messages/batch 接口
        payload = {
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages
            ]
        }
        resp = await self.client.post(
            f"/api/v1/sessions/{session_id}/messages/batch",
            json=payload,
            params={"account": self.config.account, "user": self.config.user},
        )
        resp.raise_for_status()

    async def mark_used(self, session_id: str, context_ids: list[str]) -> None:
        """标记 session 中使用的上下文。"""
        resp = await self.client.post(
            f"/api/v1/sessions/{session_id}/used",
            json={"contexts": context_ids},
            params={"account": self.config.account, "user": self.config.user},
        )
        resp.raise_for_status()

    async def commit(self, session_id: str) -> dict[str, Any]:
        """提交 session，触发长期记忆提炼。"""
        resp = await self.client.post(
            f"/api/v1/sessions/{session_id}/commit",
            params={"account": self.config.account, "user": self.config.user},
        )
        resp.raise_for_status()
        return self._unwrap_result(resp.json())

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """搜索记忆中的相关内容。

        Returns:
            搜索结果列表，每项包含 content、score、source 等字段。
        """
        try:
            resp = await self.client.post(
                "/api/v1/search/find",
                json={
                    "query": query,
                    "account": self.config.account,
                    "user": self.config.user,
                    "top_k": top_k,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            result = self._unwrap_result(data)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("results", result.get("memories", []))
            return []
        except Exception:
            return []

    async def find(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """使用 find 接口搜索（可能返回不同格式）。

        保留此接口以兼容 OpenViking 的不同搜索端点。
        """
        try:
            resp = await self.client.get(
                "/api/v1/search/find",
                params={
                    "query": query,
                    "account": self.config.account,
                    "user": self.config.user,
                    "top_k": str(top_k),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            result = self._unwrap_result(data)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("results", result.get("memories", []))
            return []
        except Exception:
            return []

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

def _memory_title(uri: str) -> str:
    if not uri:
        return "未命名记忆"
    title = uri.rstrip("/").rsplit("/", 1)[-1]
    if title.endswith(".md"):
        title = title[:-3]
    return title or "未命名记忆"


def _memory_content(item: dict[str, Any]) -> str:
    content = (
        item.get("content")
        or item.get("text")
        or item.get("abstract")
        or item.get("overview")
        or item.get("memory", "")
    )
    return str(content or "").strip()


def build_memory_context(results: list[dict[str, Any]], max_items: int = 10) -> str:
    """将 search 结果拼成可注入 system prompt 的上下文片段。

    Args:
        results: search() 返回的结果列表
        max_items: 最多取几条

    Returns:
        Markdown 格式的记忆上下文，如果无结果则返回空字符串。
    """
    if not results:
        return ""

    lines = [
        "## 相关记忆（来自 OpenViking 长期记忆）",
        "",
        "使用要求：当用户问“上次聊了什么”“下一步计划是什么”“你记不记得”时，优先依据下面的记忆标题、摘要和用户原话回答；如果相关记忆没有出现，就直接说没检索到，不要编造。",
        "",
    ]
    item_index = 0
    for item in results:
        uri = str(item.get("uri", "") or "")
        abstract = str(item.get("abstract", "") or "")
        if uri.endswith("/agent/default/memories/identity.md") and (
            "她的数字镜像" in abstract or "无生理性别" in abstract
        ):
            continue
        content = _memory_content(item)
        if not content:
            continue
        # 截断过长内容
        if len(content) > 450:
            content = content[:450] + "..."
        item_index += 1
        title = _memory_title(uri)
        lines.append(f"{item_index}. [{title}] {content}")
        if item_index >= max_items:
            break

    if item_index == 0:
        return ""
    return "\n".join(lines)
