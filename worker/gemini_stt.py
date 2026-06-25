"""Gemini API STT adapter — io.STTNode 协议 + RecognizeStream 兼容。

阶段 7 修复：
- v3 之前 GeminiSTT 是 stt.STT 子类，audio_recognition._STTPipeline 期望 io.STTNode
  （Callable），TypeError 被 @utils.log_exceptions 静默 catch，导致 STT 链路死。
- 加 `__call__` 适配 io.STTNode 协议：内部 buffer 累积 + 检测持续 silence 触发
  _recognize_impl 拿 final transcript。
- _GeminiStream 旧路径保留（向后兼容），内部复用 __call__ 逻辑。
"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterable
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.stt import (
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.stt.stt import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr


_PROBE_TAG = "[stt]"


class GeminiSTT(stt.STT):
    """Gemini API STT — io.STTNode 协议（audio_recognition 主链路）+ 旧 RecognizeStream 兼容。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",  # 阶段 25.3 修复：3.5-flash 不支持 audio 输入报 "An error"
        language: str = "zh",
        # 调参 — 适配 LiveKit 推完音频的稀疏脉冲特征
        silence_rms_threshold: int = 80,
        silence_min_frames: int = 15,  # 15 * 20ms = 300ms 持续静默触发 flush（原默认值，未验证）
        min_speech_frames: int = 10,  # 至少攒 10 帧 (200ms) 才考虑 flush（避免启动假触发）
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False),
        )
        self._api_key = api_key.strip()
        self._model = model
        self._language = language
        self._silence_rms = silence_rms_threshold
        self._silence_min = silence_min_frames
        self._min_speech = min_speech_frames

    @property
    def provider(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = None,
    ) -> RecognizeStream:
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        return _GeminiStream(
            stt=self,
            api_key=self._api_key,
            model=self._model,
            lang=language if language is not NOT_GIVEN else self._language,
            conn_options=conn_options,
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        return await _transcribe(
            buffer, self._api_key, self._model,
            language if language is not NOT_GIVEN else self._language,
        )

    # === io.STTNode 协议实现（阶段 7 修复） ===
    # audio_recognition._STTPipeline 期望：
    #   def __call__(audio_ch: AsyncIterable[rtc.AudioFrame], model_settings) -> AsyncIterable[SpeechEvent]
    # 行为：持续消费 audio_ch，buffer 累积帧，检测到"持续 silence ≥ silence_min_frames"时
    # 调 _recognize_impl 拿 final transcript。

    def __call__(
        self,
        audio_ch: AsyncIterable[rtc.AudioFrame],
        model_settings: Any,
    ) -> AsyncIterable[stt.SpeechEvent]:
        return self._stt_node_iter(audio_ch)

    async def _stt_node_iter(
        self,
        audio_ch: AsyncIterable[rtc.AudioFrame],
    ):
        buffer: list[rtc.AudioFrame] = []
        silence_streak = 0
        frame_count = 0
        # 阶段 25.1 修复：buffer 里"真有声"的帧数（rms >= self._silence_rms）
        # 之前只看 buffer 长度，所以纯静默也会一直累积 silence_streak，
        # 最终送 gemini 的是几秒前说话尾音，gemini 自由发挥识别成 "什么/中国/是的"
        # 等零碎词 → 主公体感"驴唇不对马嘴"。
        speech_frames_in_buffer = 0
        # 阶段 25.1 修复：最大 buffer 上限。超过 N 帧强制 flush（避免长段卡在 buffer 里）
        max_buffer_frames = 250  # 250 * 20ms = 5s
        # 阶段 25.1 修复：单次"必须真有声帧"才触发 flush（防"纯静默 → gemini 自由发挥"）
        min_required_speech_frames = 5  # 至少 100ms 真的有声
        try:
            async for item in audio_ch:
                if not isinstance(item, rtc.AudioFrame):
                    continue
                frame_count += 1
                had_speech = False

                # 单帧 RMS
                arr = np.asarray(item.data, dtype=np.int16)
                if arr.size > 0:
                    rms = int(
                        np.sqrt(np.mean(arr.astype(np.float32) ** 2))
                    )
                else:
                    rms = 0

                if rms >= self._silence_rms:
                    had_speech = True
                    buffer.append(item)
                    speech_frames_in_buffer += 1
                    silence_streak = 0
                else:
                    # 静默帧不进 buffer（节省 memory + 不污染 gemini 输入）
                    silence_streak += 1

                # 阶段 25.1 修复 1：只在 buffer 真的有声时才调 gemini
                # 阶段 25.1 修复 2：max_buffer_frames 强制 flush 防止超长输入
                trigger = False
                trigger_reason = ""
                if (
                    silence_streak >= self._silence_min
                    and speech_frames_in_buffer >= min_required_speech_frames
                ):
                    trigger = True
                    trigger_reason = f"silence_streak={silence_streak},speech={speech_frames_in_buffer}"
                elif (
                    silence_streak >= self._silence_min
                    and speech_frames_in_buffer < min_required_speech_frames
                ):
                    # 静默够长但 buffer 里几乎没真声 → 静默状态，**不调 gemini**
                    # 阶段 25.1：直接清 buffer 进入下一段
                    if frame_count % 30 == 1:
                        print(
                            f"{_PROBE_TAG} __call__ SKIP flush (no real speech) "
                            f"silence_streak={silence_streak} buffered={len(buffer)} "
                            f"speech_in_buf={speech_frames_in_buffer} total={frame_count}",
                            flush=True,
                        )
                    buffer.clear()
                    speech_frames_in_buffer = 0
                    silence_streak = 0
                elif (
                    len(buffer) >= max_buffer_frames
                    and speech_frames_in_buffer >= min_required_speech_frames
                ):
                    # 阶段 25.1 修复 3：超过最大帧数强制 flush（避免长段一直累积）
                    trigger = True
                    trigger_reason = f"max_buffer({max_buffer_frames}) reached, speech={speech_frames_in_buffer}"

                if trigger:
                    print(
                        f"{_PROBE_TAG} __call__ flush trigger={trigger_reason} "
                        f"buffered={len(buffer)} total={frame_count}",
                        flush=True,
                    )
                    ev = await self._recognize_impl(
                        buffer,
                        language=self._language,
                        conn_options=DEFAULT_API_CONNECT_OPTIONS,
                    )
                    text = ev.alternatives[0].text if ev.alternatives else ""
                    print(
                        f"{_PROBE_TAG} __call__ transcript len={len(text)} "
                        f"preview={text[:80]!r}",
                        flush=True,
                    )
                    yield ev
                    buffer.clear()
                    speech_frames_in_buffer = 0
                    silence_streak = 0

                if frame_count % 50 == 1:
                    print(
                        f"{_PROBE_TAG} __call__ frame rms={rms} "
                        f"silence={silence_streak} buffered={len(buffer)} "
                        f"speech_in_buf={speech_frames_in_buffer} total={frame_count}",
                        flush=True,
                    )
        except Exception as e:
            print(
                f"{_PROBE_TAG} __call__ _stt_node_iter error: {e!r}",
                flush=True,
            )


async def _transcribe(
    buffer: utils.AudioBuffer,
    api_key: str,
    model: str,
    lang: str,
) -> SpeechEvent:
    frame = rtc.combine_audio_frames(buffer)
    pcm = frame.data.tobytes()
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(frame.num_channels)
        wf.setsampwidth(2)
        wf.setframerate(frame.sample_rate)
        wf.writeframes(pcm)
    wav_bytes = wav_buf.getvalue()
    print(
        f"{_PROBE_TAG} _transcribe start sr={frame.sample_rate} "
        f"ch={frame.num_channels} bytes={len(wav_bytes)}",
        flush=True,
    )

    try:
        from google import genai
        print(f"{_PROBE_TAG} _transcribe importing genai done", flush=True)
        client = genai.Client(api_key=api_key)
        print(f"{_PROBE_TAG} _transcribe client created", flush=True)
        prompt = "Transcribe to text. Return ONLY text. Silent→empty. Chinese."
        loop = asyncio.get_running_loop()
        print(f"{_PROBE_TAG} _transcribe calling gemini model={model}", flush=True)
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        {"inline_data": {"mime_type": "audio/wav", "data": wav_bytes}},
                    ],
                ),
            ),
            timeout=30.0,
        )
        text = (response.text or "").strip()
        print(
            f"{_PROBE_TAG} _transcribe got response text_len={len(text)} "
            f"preview={text[:120]!r}",
            flush=True,
        )
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(text=text, language=lang, confidence=0.9 if text else 0.0)
            ],
        )
    except asyncio.TimeoutError:
        print(f"{_PROBE_TAG} _transcribe TIMEOUT after 30s", flush=True)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text="", language=lang, confidence=0.0)],
        )
    except Exception as e:
        print(f"{_PROBE_TAG} _transcribe error: {e!r}", flush=True)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text="", language=lang, confidence=0.0)],
        )


class _GeminiStream(RecognizeStream):
    """向后兼容 — 旧代码可能直接调 stt.stream()。v3 主链路走 __call__，不走这里。"""

    def __init__(
        self,
        *,
        stt: GeminiSTT,
        api_key: str,
        model: str,
        lang: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._api_key = api_key
        self._model = model
        self._lang = lang

    async def _run(self) -> None:
        # 复用 GeminiSTT.__call__ 逻辑 — 把 _input_ch 包装成 AsyncIterable
        async def _aiter():
            while True:
                try:
                    item = await self._input_ch.recv()
                except Exception:
                    return
                yield item

        try:
            async for ev in self._stt(_aiter(), None):
                self._event_ch.send_nowait(ev)
        except Exception as e:
            print(f"{_PROBE_TAG} _run error: {e!r}", flush=True)
