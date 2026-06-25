"""轻量能量 VAD — 不依赖外部模型，按 RMS 阈值判断 start / end of speech。

适用场景：单声道 16-bit PCM（LiveKit 默认麦克风输入）。
算法：
- 逐帧计算"有效样本占比"而不是简单 non-zero 比例。
- 只有绝对值超过一个小门限的样本才算"有效"，避免底噪/量化抖动把整帧都误判成有声。
- 累积连续"超过 speech_threshold"的样本时长，到达 min_speech_duration → emit START_OF_SPEECH。
- 处于 speaking 状态时，累积连续"低于 silence_threshold"的样本时长，
  到达 min_silence_duration → emit END_OF_SPEECH。
- flush() 重置累积状态，让下一个 frame 从 idle 重新开始。
"""

from __future__ import annotations

import time

from livekit import rtc
from livekit.agents.vad import VAD, VADCapabilities, VADEvent, VADEventType, VADStream


class EnergyVAD(VAD):
    """基于 RMS 能量阈值的 VAD。

    Args:
        speech_threshold: 判定"有声"的 RMS 阈值（int16 满量程线性值，0~32767）。
        silence_threshold: 判定"无声"的 RMS 阈值。
        min_speech_duration: 连续有声多久才确认 start_of_speech（秒）。
        min_silence_duration: 连续无声多久才确认 end_of_speech（秒）。
        update_interval: 多久 emit 一次 INFERENCE_DONE（秒）。
    """

    def __init__(
        self,
        *,
        speech_threshold: int = 50,
        silence_threshold: int = 10,
        speech_ratio_threshold: float = 0.3,
        activity_amplitude_gate: int = 96,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.6,
        update_interval: float = 0.1,
    ) -> None:
        super().__init__(
            capabilities=VADCapabilities(update_interval=update_interval),
        )
        self._opts = {
            "speech_threshold": int(speech_threshold),
            "silence_threshold": int(silence_threshold),
            "speech_ratio_threshold": float(speech_ratio_threshold),
            "activity_amplitude_gate": int(activity_amplitude_gate),
            "min_speech_duration": float(min_speech_duration),
            "min_silence_duration": float(min_silence_duration),
            "update_interval": float(update_interval),
        }

    @property
    def model(self) -> str:
        return "energy-v1"

    @property
    def provider(self) -> str:
        return "local"

    def stream(self) -> VADStream:
        return _EnergyVADStream(vad=self, opts=self._opts)


class _EnergyVADStream(VADStream):
    def __init__(self, *, vad: EnergyVAD, opts: dict) -> None:
        super().__init__(vad=vad)
        self._opts = opts
        # 状态机
        self._speaking: bool = False
        self._speech_accum: float = 0.0
        self._silence_accum: float = 0.0
        self._frames: list[rtc.AudioFrame] = []
        self._last_update: float = 0.0
        self._frame_counter: int = 0
        # 滑动窗口
        self._window_size: int = 20
        self._window: list[int] = []
        # 防抖：允许连续 N 帧低于阈值而不重置 speech_accum
        self._below_threshold_accum: float = 0.0
        self._below_threshold_forgiveness: float = opts.get("speech_forgiveness", 0.15)

    async def _main_task(self) -> None:
        try:
            while True:
                item = await self._input_ch.recv()
                if isinstance(item, VADStream._FlushSentinel):
                    self._reset_state(emit_end=self._speaking)
                    continue
                if not isinstance(item, rtc.AudioFrame):
                    continue

                rms = self._frame_rms(item)
                frame_duration = self._frame_duration(item)
                self._frames.append(item)
                if self._frame_counter <= 5 or self._frame_counter % 20 == 1:
                    data_len = len(item.data) if hasattr(item.data, '__len__') else 0
                    first_bytes = list(bytes(item.data)[:8]) if data_len else []
                    print(f"[vad] frame #{self._frame_counter} rms={rms} sr={item.sample_rate} ch={item.num_channels} samples={item.samples_per_channel} data_len={data_len} first_bytes={first_bytes}", flush=True)
                self._frame_counter += 1
                if len(self._frames) > 256:
                    self._frames = self._frames[-128:]

                if self._speaking:
                    if rms >= self._opts["silence_threshold"]:
                        self._silence_accum = 0.0
                    else:
                        self._silence_accum += frame_duration
                    if self._silence_accum >= self._opts["min_silence_duration"]:
                        self._emit_event(VADEventType.END_OF_SPEECH, item)
                        self._speaking = False
                        self._speech_accum = 0.0
                        self._silence_accum = 0.0
                        self._frames = []
                        self._window = []
                else:
                    # 滑动窗口：过去 20 帧内达到 speech 阈值的比例。
                    # 低幅底噪可能持续非零，但不应该反复触发用户打断。
                    self._window.append(rms)
                    if len(self._window) > self._window_size:
                        self._window.pop(0)
                    active_ratio = self._active_ratio()
                    if active_ratio >= self._opts["speech_ratio_threshold"]:
                        self._speech_accum += frame_duration
                        self._below_threshold_accum = 0.0
                        if self._speech_accum >= self._opts["min_speech_duration"]:
                            self._emit_event(VADEventType.START_OF_SPEECH, item)
                            self._speaking = True
                            self._silence_accum = 0.0
                    else:
                        # 防抖：允许短暂低于阈值而不立即重置 speech_accum，
                        # 避免 prob 在阈值附近振荡导致 start_of_speech 永远触发不了。
                        if self._speech_accum > 0:
                            self._below_threshold_accum += frame_duration
                            if self._below_threshold_accum >= self._below_threshold_forgiveness:
                                self._speech_accum = 0.0
                                self._below_threshold_accum = 0.0
                        else:
                            self._below_threshold_accum = 0.0

                now = time.time()
                if now - self._last_update >= self._opts["update_interval"]:
                    active_ratio = self._active_ratio()
                    self._emit_event(
                        VADEventType.INFERENCE_DONE,
                        item,
                        probability=min(1.0, active_ratio),
                    )
                    self._last_update = now
        except Exception:
            return

    def _active_ratio(self) -> float:
        if not self._window:
            return 0.0
        threshold = int(self._opts["speech_threshold"])
        return sum(1 for r in self._window if r >= threshold) / len(self._window)

    def _frame_rms(self, frame: rtc.AudioFrame) -> int:
        # 用"超过小幅度门限的 sample 占比 × 1000"作为能量指标。
        # 这样仍然保留对稀疏脉冲的敏感性，但不会把几乎全非零的低幅底噪误判成持续说话。
        import numpy as _np
        raw = frame.data
        try:
            arr = _np.asarray(raw, dtype=_np.int16)
        except Exception:
            return 0
        if arr.size == 0:
            return 0
        if frame.num_channels > 1:
            arr = arr.reshape(-1, frame.num_channels).mean(axis=1).astype(_np.int16)
        gate = max(int(self._opts["activity_amplitude_gate"]), 1)
        active_ratio = float(_np.count_nonzero(_np.abs(arr) >= gate)) / float(arr.size)
        return int(round(active_ratio * 1000))

    def _frame_duration(self, frame: rtc.AudioFrame) -> float:
        samples = getattr(frame, "samples_per_channel", 0) or 0
        sr = frame.sample_rate or 0
        if samples <= 0 or sr <= 0:
            return 0.0
        return samples / sr

    def _rms_to_prob(self, rms: int) -> float:
        hi = max(self._opts["speech_threshold"] * 4, 1)
        lo = max(self._opts["speech_threshold"], 1)
        if rms <= lo:
            return 0.0
        if rms >= hi:
            return 1.0
        return min(1.0, (rms - lo) / (hi - lo))

    def _emit_event(
        self,
        event_type: VADEventType,
        frame: rtc.AudioFrame,
        probability: float = 0.0,
    ) -> None:
        ev = VADEvent(
            type=event_type,
            samples_index=0,
            timestamp=time.time(),
            speech_duration=self._speech_accum if event_type == VADEventType.END_OF_SPEECH else 0.0,
            silence_duration=self._silence_accum if event_type == VADEventType.END_OF_SPEECH else 0.0,
            frames=list(self._frames),
            probability=probability,
            inference_duration=0.0,
        )
        print(f"[vad] emit {event_type.value} prob={probability:.2f} speech_dur={ev.speech_duration:.2f}s silence_dur={ev.silence_duration:.2f}s t={time.time():.3f}", flush=True)
        self._event_ch.send_nowait(ev)

    def _reset_state(self, *, emit_end: bool) -> None:
        if emit_end:
            ev = VADEvent(
                type=VADEventType.END_OF_SPEECH,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=0.0,
                silence_duration=0.0,
                frames=[],
            )
            self._event_ch.send_nowait(ev)
        self._speaking = False
        self._speech_accum = 0.0
        self._silence_accum = 0.0
        self._frames = []
        self._window = []


__all__ = ["EnergyVAD"]
