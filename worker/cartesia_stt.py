"""阶段 28: STT 切到 Cartesia Ink 2 (ink-whisper, 中文)。

主公决策（2026-06-08）：统一用 Cartesia 一家账号；中文必须用 ink-whisper
（ink-2 只支持 en）。STT_PROVIDER=cartesia 即走这里。

行为：
- 直接包装 livekit-plugins-cartesia 的 STT 类（不用自己实现 ws）
- 关闭 turn_detection（让 EnergyVAD 决定 endpointing）
- 中文 + 22050 Hz（与 mic 默认一致）
"""

from __future__ import annotations

import os

# 关键：必须主线程 import（livekit-plugins-cartesia 的 @Plugin 装饰器限制）
from livekit.plugins import cartesia  # noqa: F401


def build_cartesia_stt(*, sample_rate: int | None = None):
    """构造一个 Cartesia STT 实例。

    阶段 28: 走 ink-whisper，因为中文只它支持。
    """
    api_key = os.getenv("CARTESIA_API_KEY", "")
    if not api_key:
        raise ValueError("CARTESIA_API_KEY is required for STT_PROVIDER=cartesia")

    return cartesia.STT(
        model=os.getenv("CARTESIA_STT_MODEL", "ink-whisper"),
        sample_rate=sample_rate or int(os.getenv("DEEPGRAM_SAMPLE_RATE", "22050")),
        api_key=api_key,
        language="zh",
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    env = Path(__file__).resolve().parent.parent / ".env.local"
    if env.exists():
        load_dotenv(env)
    import asyncio

    async def _smoke():
        stt = build_cartesia_stt()
        print(f"[cartesia_stt] model={stt.model}")
        print("[cartesia_stt] constructed ok")

    asyncio.run(_smoke())
