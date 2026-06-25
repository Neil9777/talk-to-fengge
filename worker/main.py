"""入口 — 直接启动 agent。

阶段 23：TTS provider 插件必须在主线程注册（livekit 1.5+ Plugin 装饰器限制），
所以这里顶层 import livekit.plugins.cartesia，让 plugin 注册发生在 main thread。
子线程（job runner）再调 build_tts(...) 时不会再走 plugin 装饰器。
"""
from worker.agent import main

# 阶段 23：主线程预注册 TTS 插件（cartesia / minimax 都列上，未来切换不用重启）
from livekit.plugins import cartesia  # noqa: F401  (side-effect: registers plugin)
try:
    from livekit.plugins import minimax  # noqa: F401  (阶段 24 才有此模块)
except Exception:
    pass

main()
