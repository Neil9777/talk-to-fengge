"""沙箱内 mock 单测 — 验证 GeminiSTT.__call__ (io.STTNode 协议) 行为。

不启服务、不调真 Gemini API。用 monkeypatch 替换 _transcribe。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents import stt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from worker.gemini_stt import GeminiSTT  # noqa: E402


def make_frame(
    duration_s: float = 0.02,
    sample_rate: int = 16000,
    rms: int = 0,
) -> rtc.AudioFrame:
    """生成指定 RMS 的单声道 int16 帧。rms=0 静默，rms>0 噪声。"""
    samples = int(duration_s * sample_rate)
    if rms == 0:
        data = np.zeros(samples, dtype=np.int16).tobytes()
    else:
        data = (
            np.random.randn(samples) * rms
        ).clip(-32768, 32767).astype(np.int16).tobytes()
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _collect_events(stt_instance, frames, max_events=10, timeout=1.0):
    """跑 __call__ 收集 SpeechEvent（带超时保护）。"""
    async def audio_iter():
        for f in frames:
            yield f

    events = []

    async def _collect():
        async for ev in stt_instance(audio_iter(), None):
            events.append(ev)
            if len(events) >= max_events:
                return

    try:
        await asyncio.wait_for(_collect(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return events


async def test_silence_only_triggers_no_transcript():
    """阶段 25.1 修复后：纯 silence 不再调 gemini（之前会硬调一次返回空）"""
    print("\n=== test_silence_only_triggers_no_transcript ===")

    call_count = 0

    async def fake_transcribe(buffer, api_key, model, lang):
        nonlocal call_count
        call_count += 1
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", language=lang)],
        )

    import worker.gemini_stt as mod
    orig = mod._transcribe
    mod._transcribe = fake_transcribe
    try:
        stt_instance = GeminiSTT(
            api_key="fake",
            silence_min_frames=10,
            min_speech_frames=5,
        )
        frames = [make_frame(rms=0) for _ in range(30)]
        events = await _collect_events(stt_instance, frames, max_events=1, timeout=0.5)
        assert call_count == 0, f"纯静默不应调 gemini，但调了 {call_count} 次"
        assert len(events) == 0, f"纯静默不应产生任何 events，但得到 {len(events)}"
        print("✅ pass — 纯静默 0 次 gemini 调用 0 个 events")
    finally:
        mod._transcribe = orig


async def test_speech_then_silence_triggers_flush():
    """speech → 持续 silence 触发 flush（核心场景）"""
    print("\n=== test_speech_then_silence_triggers_flush ===")

    async def fake_transcribe(buffer, api_key, model, lang):
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="mock_transcript", language=lang)],
        )

    import worker.gemini_stt as mod
    orig = mod._transcribe
    mod._transcribe = fake_transcribe
    try:
        stt_instance = GeminiSTT(
            api_key="fake",
            silence_min_frames=10,
            min_speech_frames=5,
        )
        # 20 帧 speech (400ms) + 20 帧 silence (400ms) → 应触发 1 次 flush
        frames = [make_frame(rms=3000) for _ in range(20)]
        frames += [make_frame(rms=0) for _ in range(20)]

        events = await _collect_events(stt_instance, frames, max_events=3, timeout=1.0)
        assert len(events) == 1, f"应触发 1 次 flush，但得到 {len(events)} events"
        assert events[0].alternatives[0].text == "mock_transcript"
        print(f"✅ pass — 1 个 transcript = {events[0].alternatives[0].text!r}")
    finally:
        mod._transcribe = orig


async def test_continuous_speech_no_flush():
    """持续 speech 不应触发 flush（silence_streak 不到阈值）"""
    print("\n=== test_continuous_speech_no_flush ===")

    async def fake_transcribe(buffer, api_key, model, lang):
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="should_not_appear", language=lang)],
        )

    import worker.gemini_stt as mod
    orig = mod._transcribe
    mod._transcribe = fake_transcribe
    try:
        stt_instance = GeminiSTT(
            api_key="fake",
            silence_min_frames=15,
            min_speech_frames=5,
        )
        # 30 帧持续 speech（silence_streak 永远不增）
        frames = [make_frame(rms=3000) for _ in range(30)]
        events = await _collect_events(stt_instance, frames, max_events=1, timeout=0.5)
        assert len(events) == 0, f"持续 speech 不应触发 flush，但得到 {len(events)} events"
        print("✅ pass")
    finally:
        mod._transcribe = orig


async def test_multiple_speech_segments():
    """多段 speech 应触发多次 flush"""
    print("\n=== test_multiple_speech_segments ===")

    call_count = 0

    async def fake_transcribe(buffer, api_key, model, lang):
        nonlocal call_count
        call_count += 1
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=f"seg_{call_count}", language=lang)],
        )

    import worker.gemini_stt as mod
    orig = mod._transcribe
    mod._transcribe = fake_transcribe
    try:
        stt_instance = GeminiSTT(
            api_key="fake",
            silence_min_frames=5,
            min_speech_frames=3,
        )
        # 2 段：每段 10 speech + 8 silence
        frames = []
        for seg in range(2):
            frames += [make_frame(rms=3000) for _ in range(10)]
            frames += [make_frame(rms=0) for _ in range(8)]
        events = await _collect_events(stt_instance, frames, max_events=5, timeout=1.0)
        assert len(events) == 2, f"2 段 speech 应触发 2 次 flush，但得到 {len(events)}"
        assert events[0].alternatives[0].text == "seg_1"
        assert events[1].alternatives[0].text == "seg_2"
        print(f"✅ pass — {len(events)} 个 transcript")
    finally:
        mod._transcribe = orig


async def test_buffer_resets_after_flush():
    """flush 后 buffer 应该清空，下一段 speech 不会混入上一段

    阶段 25.1 修复后：buffer 只含真有声帧，静默帧不进 buffer
    """
    print("\n=== test_buffer_resets_after_flush ===")

    async def fake_transcribe(buffer, api_key, model, lang):
        # 假装 buffer 长度反映音频时长
        frames_in_buffer = len(buffer)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=f"frames={frames_in_buffer}", language=lang)
            ],
        )

    import worker.gemini_stt as mod
    orig = mod._transcribe
    mod._transcribe = fake_transcribe
    try:
        stt_instance = GeminiSTT(
            api_key="fake",
            silence_min_frames=5,
            min_speech_frames=3,
        )
        # 第一段：10 speech + 8 silence
        # 第二段：5 speech + 8 silence（短一点）
        frames = []
        frames += [make_frame(rms=3000) for _ in range(10)]
        frames += [make_frame(rms=0) for _ in range(8)]
        frames += [make_frame(rms=3000) for _ in range(5)]
        frames += [make_frame(rms=0) for _ in range(8)]

        events = await _collect_events(stt_instance, frames, max_events=5, timeout=1.0)
        assert len(events) == 2, f"应触发 2 次 flush，但得到 {len(events)}"
        # 阶段 25.1 修复后：buffer 只含 speech 帧
        # 第一段：10 speech (静默帧不进 buffer) → buffer 长度 10
        # 第二段：5 speech → buffer 长度 5
        first_count = int(events[0].alternatives[0].text.split("=")[1])
        second_count = int(events[1].alternatives[0].text.split("=")[1])
        assert first_count == 10, f"第一段 buffer 长度应为 10（仅 speech 帧），但得到 {first_count}"
        assert second_count == 5, f"第二段 buffer 长度应为 5（仅 speech 帧），但得到 {second_count}"
        assert first_count != second_count, "两段 buffer 长度相同，buffer 没清空"
        print(
            f"✅ pass — 第一段 frames={first_count}, "
            f"第二段 frames={second_count} (两段已正确分隔，仅 speech 帧入 buffer)"
        )
    finally:
        mod._transcribe = orig


async def test_long_speech_force_flush():
    """阶段 25.1 修复：长段持续 speech 超过 max_buffer_frames 强制 flush（避免卡死）"""
    print("\n=== test_long_speech_force_flush ===")

    call_count = 0

    async def fake_transcribe(buffer, api_key, model, lang):
        nonlocal call_count
        call_count += 1
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=f"long_seg_{call_count}", language=lang)],
        )

    import worker.gemini_stt as mod
    orig = mod._transcribe
    mod._transcribe = fake_transcribe
    try:
        stt_instance = GeminiSTT(
            api_key="fake",
            silence_min_frames=10,
            min_speech_frames=5,
        )
        # 300 帧持续 speech = 6s，超过 max_buffer_frames=250 强制 flush
        frames = [make_frame(rms=3000) for _ in range(300)]
        events = await _collect_events(stt_instance, frames, max_events=3, timeout=2.0)
        assert call_count >= 1, f"长段应至少触发 1 次 flush，但只调了 {call_count} 次 gemini"
        print(f"✅ pass — 300 帧持续 speech 触发 {call_count} 次 flush")
    finally:
        mod._transcribe = orig


async def test_silence_then_speech_then_silence_only_real_speech_buffered():
    """阶段 25.1 修复：静默帧不进 buffer，只有真有声帧进 buffer

    模拟场景：10 silence + 20 speech + 10 silence + 5 silence
    预期：buffer 只含 20 个 speech 帧（不是 35 个）
    """
    print("\n=== test_silence_then_speech_then_silence_only_real_speech_buffered ===")

    buffer_size_at_call = []

    async def fake_transcribe(buffer, api_key, model, lang):
        buffer_size_at_call.append(len(buffer))
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="ok", language=lang)],
        )

    import worker.gemini_stt as mod
    orig = mod._transcribe
    mod._transcribe = fake_transcribe
    try:
        stt_instance = GeminiSTT(
            api_key="fake",
            silence_min_frames=10,
            min_speech_frames=5,
        )
        # 10 silence + 20 speech + 15 silence
        frames = [make_frame(rms=0) for _ in range(10)]
        frames += [make_frame(rms=3000) for _ in range(20)]
        frames += [make_frame(rms=0) for _ in range(15)]
        events = await _collect_events(stt_instance, frames, max_events=2, timeout=1.0)
        assert len(buffer_size_at_call) == 1, f"应调 1 次 gemini，调了 {len(buffer_size_at_call)}"
        assert buffer_size_at_call[0] == 20, (
            f"buffer 应只含 20 个 speech 帧（静默帧不进 buffer），但有 {buffer_size_at_call[0]}"
        )
        print(f"✅ pass — gemini 收到 buffer_size={buffer_size_at_call[0]}（仅真有声帧）")
    finally:
        mod._transcribe = orig


async def main():
    await test_silence_only_triggers_no_transcript()
    await test_speech_then_silence_triggers_flush()
    await test_continuous_speech_no_flush()
    await test_multiple_speech_segments()
    await test_buffer_resets_after_flush()
    await test_long_speech_force_flush()
    await test_silence_then_speech_then_silence_only_real_speech_buffered()
    print("\n=== 全部通过 ===")


if __name__ == "__main__":
    asyncio.run(main())
