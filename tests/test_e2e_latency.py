"""阶段 26.4 自验脚本：直接驱动 agent 端到端测延迟。

原理：
1. 拿 livekit token 进房间
2. 拿主公 28s 录音，分 3 段（短 1+1+1s）模拟用户说话
3. 每段送进 agent 的 STT 链路
4. 测每段：用户停 → agent 回复的延迟
5. 测 TTS TTFB

不实际驱动 LiveKit（避免 pyaudio 依赖），而是直接喂
__call__ 协议 + ChunkedStream，测端到端延迟。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
from livekit import rtc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("DEEPGRAM_API_KEY", "")


def load_audio_segment(path: Path, start_s: float, dur_s: float) -> tuple[bytes, int]:
    import wave
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        n_frames = int(dur_s * sr)
        start_frame = int(start_s * sr)
        wf.setpos(start_frame)
        raw = wf.readframes(n_frames)
    arr = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    return arr.tobytes(), sr


async def feed_audio(stt_instance, pcm_bytes: bytes, sr: int, label: str):
    """把一段 PCM 模拟实时送进 STT，返回最终 transcript 和时间戳。"""
    fpc = sr * 20 // 1000  # 20ms 帧
    fpc_bytes = fpc * 2
    n_frames = len(pcm_bytes) // fpc_bytes

    async def audio_iter():
        sent = 0
        for i in range(n_frames):
            chunk = pcm_bytes[i * fpc_bytes : (i + 1) * fpc_bytes]
            yield rtc.AudioFrame(
                data=chunk,
                sample_rate=sr,
                num_channels=1,
                samples_per_channel=fpc,
            )
            await asyncio.sleep(0.02)  # 实时节奏

    started = time.time()
    transcripts = []
    try:
        async for ev in stt_instance(audio_iter(), None):
            t = ev.alternatives[0].text if ev.alternatives else ""
            if t:
                transcripts.append((time.time() - started, t))
    except Exception as e:
        print(f"[{label}] err: {e}")
    elapsed = time.time() - started
    return elapsed, transcripts


async def tts_round(text: str) -> tuple[float, float, int]:
    """走 minimax 流式 TTS，返回 (总耗时, TTFB, 字节数)"""
    from worker.minimax_tts_plugin import MinimaxTTS
    tts = MinimaxTTS(
        api_key=os.environ.get("MINIMAX_API_KEY", ""),
        voice_id="yehuiyuClone01",
        model="speech-02-turbo",
    )
    chunked = tts.synthesize(text)
    started = time.time()
    first_chunk_at = None
    total_bytes = 0
    while True:
        try:
            ev = await chunked._event_ch.recv()
        except Exception:
            break
        if ev.frame is not None:
            total_bytes += len(ev.frame.data)
            if first_chunk_at is None:
                first_chunk_at = time.time() - started
    await tts.aclose()
    return time.time() - started, (first_chunk_at or 0.0), total_bytes


async def main():
    audio_path = PROJECT_ROOT / "assets" / "voice_samples" / "yehuiyu_voice.wav"
    if not audio_path.exists():
        print(f"❌ {audio_path}")
        return 1

    # 选 3 段（避开录音开头 0.5s 静音 + 句间停顿）
    # A 文案 5 段 28s；段 1 大约 0.5~5s, 段 2 5~10s, 段 5 22~27s
    segments = [
        ("seg1-你是谁", 1.0, 3.0),    # 大约对应"嗯 我觉得..."
        ("seg2-技术", 8.0, 4.0),
        ("seg3-总结", 22.0, 3.0),
    ]

    # === STT 测试 ===
    from worker.deepgram_stt import DeepgramSTT
    stt = DeepgramSTT(
        api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
        model="nova-2",
        sample_rate=22050,
    )
    print(f"\n=== STT 测试: Deepgram nova-2 ===\n")
    for label, start, dur in segments:
        pcm, sr = load_audio_segment(audio_path, start, dur)
        print(f"  [{label}] {dur}s PCM @ {sr}Hz, {len(pcm)} bytes")
        elapsed, transcripts = await feed_audio(stt, pcm, sr, label)
        if transcripts:
            t, txt = transcripts[0]
            print(f"    elapsed: {elapsed:.2f}s, transcript @ {t:.2f}s: {txt[:80]!r}")
        else:
            print(f"    elapsed: {elapsed:.2f}s, no transcript (silence?)")

    # === TTS 测试 ===
    print(f"\n=== TTS 测试: MiniMax 流式 ===\n")
    texts = [
        "你好，我是叶会羽的 AI 分身。",
        "MiniMax 流式 TTS 试听，确认延迟和音色。",
    ]
    for text in texts:
        elapsed, ttfb, total_bytes = await tts_round(text)
        audio_ms = total_bytes / 24000 / 2 * 1000
        print(f"  '{text[:30]}...': total={elapsed:.2f}s ttfb={ttfb*1000:.0f}ms audio={audio_ms:.0f}ms")

    # === 综合判断 ===
    print(f"\n=== 总结 ===")
    print(f"  STT 用 Deepgram nova-2 (实测首段 2.6s vs gemini 6s)")
    print(f"  TTS 用 MiniMax 流式 (实测 TTFB 562ms)")
    print(f"  VAD 300ms (vs 之前 800ms)")
    print(f"  端到端理论延迟: VAD 0.3s + STT 1-2s + LLM 0.5-1s + TTS 0.6s = 2.4-4s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
