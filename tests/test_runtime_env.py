"""沙箱内单测 — 验证 runtime_env 的 localhost 代理隔离行为。

不启服务、不改真实代理。用 monkeypatch 操作 os.environ。

对应阶段 8 修复：把 localhost 流量从本地代理污染中隔离开。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from worker.runtime_env import (  # noqa: E402
    _LOCAL_NO_PROXY_HOSTS,
    configure_local_no_proxy,
    local_service_env,
)


def _all_proxy_keys():
    return (
        "http_proxy", "https_proxy", "all_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "NO_PROXY", "no_proxy",
    )


def _clear_all():
    for k in _all_proxy_keys():
        os.environ.pop(k, None)


class RuntimeEnvTest(unittest.TestCase):
    """验证 runtime_env 在 4 种情形下的行为。"""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _all_proxy_keys()}
        _clear_all()

    def tearDown(self):
        _clear_all()
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_configure_adds_local_hosts_to_no_proxy_when_missing(self):
        """没设 NO_PROXY 时，configure 后必须包含 127.0.0.1/localhost/::1。"""
        configure_local_no_proxy()
        merged = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        self.assertIsNotNone(merged, "configure 后 NO_PROXY 应被设置")
        for host in _LOCAL_NO_PROXY_HOSTS:
            self.assertIn(host, merged, f"NO_PROXY 应包含 {host}，实际 {merged}")

    def test_configure_preserves_existing_no_proxy(self):
        """已设 NO_PROXY 时，configure 后必须保留原有 hosts。"""
        os.environ["NO_PROXY"] = "internal.example.com,*.corp"
        configure_local_no_proxy()
        merged = os.environ.get("NO_PROXY")
        self.assertIn("internal.example.com", merged)
        self.assertIn("*.corp", merged)
        for host in _LOCAL_NO_PROXY_HOSTS:
            self.assertIn(host, merged)

    def test_local_service_env_strips_proxy_inside_block(self):
        """local_service_env 块内必须把所有代理 env 临时清空。"""
        os.environ["http_proxy"] = "http://127.0.0.1:1087"
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:1087"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1087"
        os.environ["https_proxy"] = "http://127.0.0.1:1087"
        with local_service_env():
            for key in ("http_proxy", "https_proxy", "all_proxy",
                        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                self.assertIsNone(
                    os.environ.get(key),
                    f"块内 {key} 应被清空，实际 {os.environ.get(key)!r}"
                )
            # NO_PROXY 块内仍要有 localhost
            merged = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
            self.assertIn("127.0.0.1", merged or "")

    def test_local_service_env_restores_outside_block(self):
        """块退出后，代理 env 必须恢复原值。"""
        os.environ["http_proxy"] = "http://127.0.0.1:1087"
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:1087"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1087"
        os.environ["https_proxy"] = "http://127.0.0.1:1087"
        with local_service_env():
            pass
        self.assertEqual(os.environ.get("http_proxy"), "http://127.0.0.1:1087")
        self.assertEqual(os.environ.get("HTTP_PROXY"), "http://127.0.0.1:1087")
        self.assertEqual(os.environ.get("HTTPS_PROXY"), "http://127.0.0.1:1087")
        self.assertEqual(os.environ.get("https_proxy"), "http://127.0.0.1:1087")

    def test_local_service_env_restores_when_unset_before(self):
        """进入块前没设代理，块退出后仍应该是没设状态（不能凭空多出 key）。"""
        # 保险：显式 clear
        _clear_all()
        with local_service_env():
            pass
        for key in ("http_proxy", "https_proxy", "all_proxy",
                    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            self.assertIsNone(
                os.environ.get(key),
                f"原本未设的 {key} 不应在块退出后出现"
            )

    def test_local_service_env_isolates_nested_exception(self):
        """块内抛异常，代理 env 仍必须恢复。"""
        os.environ["http_proxy"] = "http://127.0.0.1:1087"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1087"
        try:
            with local_service_env():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertEqual(os.environ.get("http_proxy"), "http://127.0.0.1:1087")
        self.assertEqual(os.environ.get("HTTPS_PROXY"), "http://127.0.0.1:1087")


if __name__ == "__main__":
    unittest.main()
