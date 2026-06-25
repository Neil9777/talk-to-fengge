"""EnergyVAD 单元测试 — 不启服务、不发网络请求。"""

import asyncio
import sys
import unittest
from pathlib import Path

import numpy as np
from livekit.rtc import AudioFrame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from worker.energy_vad import EnergyVAD


def make_frame(*, samples: int = 480, sample_rate: int = 16000, channels: int = 1, rms: int = 0) -> AudioFrame:
    """构造指定 RMS 的 frame（int16 PCM）。"""
    amplitude = rms  # int16 满量程线性
    if amplitude <= 0:
        raw = np.zeros(samples * channels, dtype=np.int16)
    else:
        # 用稳定的正弦信号模拟有声帧。
        t = np.arange(samples * channels) / sample_rate
        raw = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    return AudioFrame(
        data=raw.tobytes(),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=samples,
    )


async def collect_events(stream, *, max_events: int = 10, timeout: float = 1.0):
    out = []
    try:
        for _ in range(max_events):
            ev = await asyncio.wait_for(stream.__anext__(), timeout=timeout)
            out.append(ev)
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass
    return out


class EnergyVADTest(unittest.TestCase):
    def test_low_amplitude_noise_does_not_trigger_speech(self):
        async def go():
            vad = EnergyVAD(speech_threshold=200, silence_threshold=100,
                            activity_amplitude_gate=96,
                            min_speech_duration=0.05, min_silence_duration=0.1,
                            update_interval=0.05)
            stream = vad.stream()
            # 所有 sample 都非零，但幅度远低于 gate，不应被算成"持续说话"
            for _ in range(8):
                stream.push_frame(make_frame(rms=8))
            await asyncio.sleep(0.3)
            events = await collect_events(stream, max_events=8, timeout=0.3)
            types = [ev.type.value for ev in events]
            self.assertNotIn("start_of_speech", types, f"unexpected speech from low noise: {types}")
        asyncio.run(go())

    def test_below_speech_threshold_frames_do_not_trigger_speech(self):
        async def go():
            vad = EnergyVAD(speech_threshold=200, silence_threshold=100,
                            activity_amplitude_gate=96,
                            min_speech_duration=0.05, min_silence_duration=0.1,
                            update_interval=0.05)
            stream = vad.stream()
            # 模拟真人验收里出现过的低幅活动帧：不是纯静音，但不足以算用户重新开口。
            for _ in range(20):
                stream.push_frame(make_frame(rms=40))
            await asyncio.sleep(0.3)
            events = await collect_events(stream, max_events=10, timeout=0.3)
            types = [ev.type.value for ev in events]
            self.assertNotIn("start_of_speech", types, f"unexpected speech from sub-threshold frames: {types}")
        asyncio.run(go())

    def test_end_of_speech_clears_activity_window(self):
        async def go():
            vad = EnergyVAD(speech_threshold=200, silence_threshold=100,
                            activity_amplitude_gate=96,
                            min_speech_duration=0.05, min_silence_duration=0.1,
                            update_interval=0.05)
            stream = vad.stream()
            for _ in range(5):
                stream.push_frame(make_frame(rms=5000))
            for _ in range(5):
                stream.push_frame(make_frame(rms=0))
            # 结束后继续推低幅底噪，旧窗口不应把它误判成第二次开口。
            for _ in range(10):
                stream.push_frame(make_frame(rms=40))
            await asyncio.sleep(0.3)
            events = await collect_events(stream, max_events=20, timeout=0.4)
            types = [ev.type.value for ev in events]
            self.assertEqual(types.count("start_of_speech"), 1, f"unexpected repeated starts: {types}")
            self.assertIn("end_of_speech", types)
        asyncio.run(go())

    def test_silence_does_not_trigger_speech(self):
        async def go():
            vad = EnergyVAD(speech_threshold=500, silence_threshold=300,
                            min_speech_duration=0.05, min_silence_duration=0.1,
                            update_interval=0.05)
            stream = vad.stream()
            # 5 帧全静音
            for _ in range(5):
                stream.push_frame(make_frame(rms=0))
            await asyncio.sleep(0.3)
            events = await collect_events(stream, max_events=5, timeout=0.3)
            types = [ev.type.value for ev in events]
            self.assertNotIn("start_of_speech", types)
        asyncio.run(go())

    def test_loud_speech_triggers_start_then_end(self):
        async def go():
            vad = EnergyVAD(speech_threshold=500, silence_threshold=300,
                            min_speech_duration=0.05, min_silence_duration=0.1,
                            update_interval=0.05)
            stream = vad.stream()
            # 5 帧高能量（30ms × 5 = 150ms > min_speech_duration 50ms）
            for _ in range(5):
                stream.push_frame(make_frame(rms=5000))
            # 5 帧静音（150ms > min_silence_duration 100ms）
            for _ in range(5):
                stream.push_frame(make_frame(rms=0))
            await asyncio.sleep(0.2)
            events = await collect_events(stream, max_events=20, timeout=0.4)
            types = [ev.type.value for ev in events]
            self.assertIn("start_of_speech", types, f"missing start_of_speech in {types}")
            self.assertIn("end_of_speech", types, f"missing end_of_speech in {types}")
            # start 必须在 end 之前
            self.assertLess(types.index("start_of_speech"), types.index("end_of_speech"))
        asyncio.run(go())

    def test_flush_resets_state(self):
        async def go():
            vad = EnergyVAD(speech_threshold=500, silence_threshold=300,
                            min_speech_duration=0.05, min_silence_duration=0.1,
                            update_interval=0.05)
            stream = vad.stream()
            for _ in range(3):
                stream.push_frame(make_frame(rms=5000))
            # 此时应该已 speaking（但 events 可能还没到 queue 头）
            stream.flush()
            await asyncio.sleep(0.1)
            # flush 后再喂一帧高能量，应该能再 start
            for _ in range(3):
                stream.push_frame(make_frame(rms=5000))
            for _ in range(3):
                stream.push_frame(make_frame(rms=0))
            await asyncio.sleep(0.2)
            events = await collect_events(stream, max_events=20, timeout=0.4)
            types = [ev.type.value for ev in events]
            self.assertGreaterEqual(types.count("start_of_speech"), 1, f"expected at least 1 start, got {types}")
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
