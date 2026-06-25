from __future__ import annotations

import os
from contextlib import contextmanager

_PROXY_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)
_NO_PROXY_KEYS = ("NO_PROXY", "no_proxy")
_LOCAL_NO_PROXY_HOSTS = ("127.0.0.1", "localhost", "::1")

# 默认出口代理（主公 ShadowsocksX-NG privoxy 1087）。
# 阶段 20 修复：worker 启动时没有这个 env，导致 google.genai / httpx 等无法联外网
# （macOS launchd 域进程不会自动带 zsh shell 里的代理 env）。
# 设了之后所有走 genai/httpx 的调用都走 1087，同时配合 NO_PROXY 把 localhost 排除。
_DEFAULT_EGRESS_PROXY = os.getenv("EGRESS_PROXY_URL", "http://127.0.0.1:1087")


def _merge_no_proxy(existing: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for item in [*(existing or "").split(","), *_LOCAL_NO_PROXY_HOSTS]:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return ",".join(values)


def configure_local_no_proxy() -> None:
    """把 localhost/127.0.0.1 加入 NO_PROXY，**不**改 http_proxy。"""
    merged = _merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "")
    for key in _NO_PROXY_KEYS:
        os.environ[key] = merged


def configure_egress_proxy(proxy_url: str | None = _DEFAULT_EGRESS_PROXY) -> None:
    """把 1087 写到 http_proxy/https_proxy env。

    必须在 import google.genai / httpx / aiohttp 之前调用。
    如果主公显式给了别的 URL，用主公的；否则用默认 1087。
    """
    if not proxy_url:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = proxy_url
    # all_proxy 也设上（部分库会读）
    os.environ.setdefault("all_proxy", proxy_url)
    os.environ.setdefault("ALL_PROXY", proxy_url)
    # NO_PROXY 排除 localhost
    configure_local_no_proxy()


@contextmanager
def local_service_env():
    """块内强制 localhost 直连（清掉所有代理 env + 设 NO_PROXY）。"""
    saved = {key: os.environ.get(key) for key in (*_PROXY_KEYS, *_NO_PROXY_KEYS)}
    try:
        configure_local_no_proxy()
        for key in _PROXY_KEYS:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
