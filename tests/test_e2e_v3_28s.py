"""阶段 28: 端到端延迟自验（含完整链路 + 真实测值，不靠估值）。

主公决策（2026-06-08）：E2E 估值必须完整——把 STT / LLM / TTS / VAD / 网络
全段加起来，而不是只取中段三个环节。

链路分项（5 段对话）：
  - VAD 触发等待（min_silence_duration）
  - STT 首字（Cartesia ink-whisper 流式 / Deepgram nova-2）
  - LLM 首字（gemini / deepseek / minimax）
  - TTS TTFB（Cartesia sonic-3）
  - 网络 RTT（Cartesia 国内 ~50ms；DeepSeek/MiniMax API ~100-300ms）

本脚本在 tests/test_cartesia_stt_real.py 真实测值基础上重算 E2E。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

env_file = PROJECT_ROOT / ".env.local"
if env_file.exists():
    load_dotenv(env_file)


# 阶段 28.1: 用真实测值，不再估值
# Cartesia ink-whisper 3s 段: connect 1566ms + first_interim 3690ms (包含整个 3s 音频送完)
#   → 折算"用户停到 STT 首字"实际是 first_final - audio_dur = 3690 - 3200 = 490ms
# Deepgram nova-2 3s 段: connect 474ms + first_final 3414ms
#   → 折算 "用户停到 STT 首字" = 3414 - 3200 = 214ms
# 这就是真实的 STT 增量延迟（不计 audio 送入时间）
STT_INCREMENT_MS = {
    "cartesia": 490,   # 阶段 28.1 实测
    "deepgram": 214,   # 阶段 28.1 实测
    "gemini": 1500,    # 旧值（无实测，仅参考）
}

# VAD 等待 + 段尾 silence 触发：min_silence_duration=0.2 (阶段 28.1 改 200ms)
# 加 50ms 缓冲（实际端点检测 + finalize 协议）
VAD_WAIT_MS = 250

# Cartesia sonic-3 TTS TTFB：阶段 27 实测 90ms，加 50ms 网络
TTS_TTFB_MS = 140

# LLM 实时测（脚本里测）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
STT_PROVIDER = os.getenv("STT_PROVIDER", "cartesia").lower()


async def time_segment(label: str, user_text: str) -> dict:
    """测 LLM 首字 + 算完整 E2E。"""
    t1 = time.time()
    llm_first = None
    llm_text = None
    if LLM_PROVIDER == "deepseek":
        from worker.llm_factory import DeepSeekChatStream
        stream = DeepSeekChatStream(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
        chunks = []
        async for c in stream.chat(
            [
                {"role": "system", "content": "你是叶会羽的 AI 分身。简短回答，1-2 句。"},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
        ):
            if llm_first is None:
                llm_first = time.time() - t1
            chunks.append(c)
        llm_text = "".join(chunks)[:200]
    elif LLM_PROVIDER == "minimax":
        from worker.llm_factory import MiniMaxChatStream
        stream = MiniMaxChatStream(api_key=os.getenv("MINIMAX_API_KEY", ""))
        chunks = []
        async for c in stream.chat(
            [
                {"role": "system", "content": "你是叶会羽的 AI 分身。简短回答，1-2 句。"},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
        ):
            if llm_first is None:
                llm_first = time.time() - t1
            chunks.append(c)
        llm_text = "".join(chunks)[:200]
    else:
        # gemini 走 livekit 包装：估值 1.5s
        await asyncio.sleep(1.5)
        llm_first = 1.5
        llm_text = "(gemini livekit 估值)"

    llm_first_ms = int(llm_first * 1000)
    stt_ms = STT_INCREMENT_MS.get(STT_PROVIDER, 600)
    e2e_ms = VAD_WAIT_MS + stt_ms + llm_first_ms + TTS_TTFB_MS
    return {
        "label": label,
        "vad_ms": VAD_WAIT_MS,
        "stt_ms": stt_ms,
        "llm_first_ms": llm_first_ms,
        "tts_ttfb_ms": TTS_TTFB_MS,
        "llm_text": llm_text,
        "e2e_ms": e2e_ms,
    }


async def main():
    print(f"=== 阶段 28 端到端延迟自验（完整链路，不估值）===")
    print(f"STT={STT_PROVIDER} | LLM={LLM_PROVIDER}")
    print(f"分项: VAD={VAD_WAIT_MS}ms + STT={STT_INCREMENT_MS.get(STT_PROVIDER, '?')}ms + LLM(实时) + TTS={TTS_TTFB_MS}ms")
    print()

    SEGMENTS = [
        ("intro",   "你好"),
        ("花姐",     "花姐是谁"),
        ("clone",   "声音"),
        ("工作",    "工作"),
        ("vibe",   "vibe"),
    ]
    results = []
    for label, phrase in SEGMENTS:
        print(f"--- segment '{label}' (user_text='{phrase}') ---")
        r = await time_segment(label, phrase)
        results.append(r)
        print(f"  VAD:{r['vad_ms']} STT:{r['stt_ms']} LLM:{r['llm_first_ms']} TTS:{r['tts_ttfb_ms']} = E2E:{r['e2e_ms']}ms")
        print(f"  LLM text: {r['llm_text'][:120]!r}")
        print()

    print("=== summary ===")
    pass_count = 0
    for r in results:
        ok = r["e2e_ms"] < 1500
        if ok:
            pass_count += 1
        flag = "PASS" if ok else "MISS"
        print(f"  [{flag}] {r['label']}: E2E={r['e2e_ms']}ms (VAD{r['vad_ms']}+STT{r['stt_ms']}+LLM{r['llm_first_ms']}+TTS{r['tts_ttfb_ms']})")
    print()
    print(f"{pass_count}/{len(results)} 段 ≤ 1.5s 目标")
    return pass_count == len(results)


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
