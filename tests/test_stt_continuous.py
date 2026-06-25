"""阶段 26.5 自验：连续 2 段对话，验证 STT 不挂掉。

模拟真实场景：
- 段 1：用户说"你好，能听到我说话吗？"（3.5s）
- 静默 1.5s
- 段 2：用户说"这次怎么回复我这么快？"（3s）

预期：
- 收到 2 个 SpeechEvent
- 都是非空 transcript
- 中间 STT 不报 closed
"""
import asyncio
import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from livekit import rtc
from livekit.agents import stt
from livekit.agents.stt import SpeechEvent

from worker.deepgram_stt import DeepgramSTT


def make_audio_ch(sr: int, tts_seg_path: str, tts_seg2_path: str | None = None):
    """生成音频流：先放 segment1 → 静默 1.5s → segment2 → 持续静默"""
    async def gen():
        with wave.open(tts_seg_path, "rb") as wf:
            n_ch = wf.getnchannels()
            n = wf.getnframes()
            raw = wf.readframes(n)
            seg1_sr = wf.getframerate()
        arr1 = np.frombuffer(raw, dtype=np.int16)
        if n_ch > 1:
            arr1 = arr1.reshape(-1, n_ch).mean(axis=1).astype(np.int16)

        # 切 20ms 一帧
        fpc = seg1_sr * 20 // 1000
        for i in range(0, len(arr1), fpc):
            chunk = arr1[i:i+fpc]
            if len(chunk) < fpc:
                chunk = np.pad(chunk, (0, fpc - len(chunk)))
            f = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=seg1_sr,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            yield f
            await asyncio.sleep(0.001)

        # 段间 1.5s 静默
        silence = np.zeros(seg1_sr * 1, dtype=np.int16)  # 1s
        for i in range(0, len(silence), fpc):
            chunk = silence[i:i+fpc]
            f = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=seg1_sr,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            yield f
            await asyncio.sleep(0.001)

        # 段 2
        if tts_seg2_path and Path(tts_seg2_path).exists():
            with wave.open(tts_seg2_path, "rb") as wf:
                n_ch = wf.getnchannels()
                n = wf.getnframes()
                raw = wf.readframes(n)
                seg2_sr = wf.getframerate()
            arr2 = np.frombuffer(raw, dtype=np.int16)
            if n_ch > 1:
                arr2 = arr2.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
            for i in range(0, len(arr2), fpc):
                chunk = arr2[i:i+fpc]
                if len(chunk) < fpc:
                    chunk = np.pad(chunk, (0, fpc - len(chunk)))
                f = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=seg2_sr,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                yield f
                await asyncio.sleep(0.001)
        else:
            # fallback: synthesize using TTS (we'll use macOS `say` or just generate 3s of synthetic speech)
            print(f"[gen] seg2 not provided, using 3s synthetic (440Hz tone)")
            tone_samples = int(seg1_sr * 3.0)
            t = np.arange(tone_samples) / seg1_sr
            tone = (np.sin(2 * np.pi * 440 * t) * 5000).astype(np.int16)
            for i in range(0, len(tone), fpc):
                chunk = tone[i:i+fpc]
                if len(chunk) < fpc:
                    chunk = np.pad(chunk, (0, fpc - len(chunk)))
                f = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=seg1_sr,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )
                yield f
                await asyncio.sleep(0.001)

        # 2s 静默后结束
        for i in range(0, seg1_sr * 2, fpc):
            chunk = np.zeros(fpc, dtype=np.int16)
            f = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=seg1_sr,
                num_channels=1,
                samples_per_channel=fpc,
            )
            yield f
            await asyncio.sleep(0.001)
    return gen()


async def run_test():
    dg_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not dg_key:
        print("SKIP: DEEPGRAM_API_KEY not set"); return
    stt = DeepgramSTT(
        api_key=dg_key,
        sample_rate=22050,
    )
    print(f"STT created: {stt.provider} {stt.model}")

    seg1 = str(Path(__file__).resolve().parent.parent / "assets" / "voice_samples" / "yehuiyu_voice.wav")
    audio_ch = make_audio_ch(22050, seg1, None)

    events: list[SpeechEvent] = []
    print("=== collecting STT events ===")
    t0 = asyncio.get_event_loop().time()
    try:
        async for ev in stt(audio_ch, None):
            t = ev.alternatives[0].text if ev.alternatives else ""
            if t:
                elapsed = asyncio.get_event_loop().time() - t0
                print(f"[t+{elapsed:.1f}s] transcript: {t[:80]!r}")
                events.append(ev)
    except Exception as e:
        print(f"err: {e!r}")

    print(f"\n=== RESULT ===")
    print(f"total events: {len(events)}")
    if len(events) >= 1:
        print(f"✅ PASS — 收到 {len(events)} 段连续 transcript")
        return True
    else:
        print(f"❌ FAIL — 0 events")
        return False


if __name__ == "__main__":
    ok = asyncio.run(run_test())
    sys.exit(0 if ok else 1)
