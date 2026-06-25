"""阶段 26: Deepgram Live STT — livekit.agents.stt.STT 协议适配。

Deepgram nova-2 实测：
- 28s 主公录音首段 final transcript 4.3s 出来（vs gemini-2.5-flash 6s）
- 识别质量 95%+ 字符匹配
- WebSocket 协议走 aiohttp

阶段 26.5 重写：长连 ws 模式
- 之前每次 flush 都开新 ws 关闭 → LiveKit 那边 _check_not_closed() raise
- 现在：开 1 个 ws，整个 session 一直开
- Deepgram 服务端 endpointing=300ms 自动帮我们切段（is_final + speech_final）
- 边收边 yield SpeechEvent
- 借鉴 gemini_stt.py 的 25.1 修复：只把真有声帧入 buffer
- audio_ch 结束 + buffer 非空 → flush (走 ws 端 CloseStream)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterable
from typing import Any

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt
from livekit.agents.stt import (
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.stt.stt import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr

DEEPGRAM_WSS_URL = "wss://api.deepgram.com/v1/listen"

_PROBE_TAG = "[stt-dg]"


class DeepgramSTT(stt.STT):
    """Deepgram nova-2 Live STT — 走 WebSocket 持续流式识别。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "nova-2",
        language: str = "zh",
        sample_rate: int = 22050,  # 默认匹配主公录音 22kHz
        # 25.1 修复同样的 buffer 策略
        silence_rms_threshold: int = 80,
        silence_min_frames: int = 15,  # 300ms 持续静默 → 关闭 ws
        min_speech_frames: int = 5,  # 100ms 真的有声才发
        max_buffer_frames: int = 250,  # 5s 长段强制 flush
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False),
        )
        self._api_key = api_key.strip()
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._silence_rms = silence_rms_threshold
        self._silence_min = silence_min_frames
        self._min_speech = min_speech_frames
        self._max_buffer = max_buffer_frames

    @property
    def provider(self) -> str:
        return "deepgram"

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
        return _DeepgramStream(
            stt=self,
            api_key=self._api_key,
            model=self._model,
            lang=language if language is not NOT_GIVEN else self._language,
            sample_rate=self._sample_rate,
            conn_options=conn_options,
        )

    # === io.STTNode 协议（阶段 25.1 同样的 buffer 策略） ===

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
        """
        阶段 26.5 重写：**长连 ws 模式**。

        关键修复：之前每次 flush 都开新 ws 关闭，导致 LiveKit RecognizeStream 报 `_check_not_closed()` raise，
        之后所有 audio frame 都丢，永久听不到用户的第二句话。

        现在：
        1. **开 1 个 ws 持续连着**（不重开）
        2. audio frame 持续推给 ws
        3. Deepgram 服务端 endpointing=300ms 自动切段
        4. 收 is_final=True 且 speech_final=True 的事件 → yield SpeechEvent
        5. ws 异常 → yield SpeechEvent(空) + 退出（让 LiveKit 知道 STT 完结）

        简化：VAD 触发逻辑（silence/max_buffer/audio_ch_end）不再调 ws 关闭，
        只在 audio_ch 真正结束 + buffer 非空时，发 CloseStream + 等最后 final。
        """
        url = (
            f"{DEEPGRAM_WSS_URL}"
            f"?language={self._language}"
            f"&model={self._model}"
            f"&smart_format=true"
            f"&encoding=linear16"
            f"&sample_rate={self._sample_rate}"
            f"&channels=1"
            f"&interim_results=false"
            f"&endpointing=300"
            f"&vad_events=true"
        )
        headers = {"Authorization": f"Token {self._api_key}"}

        session: aiohttp.ClientSession | None = None
        ws = None
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=15.0)
            )
            ws = await session.ws_connect(url, headers=headers, autoping=True, autoclose=False)
            print(f"{_PROBE_TAG} ws connected", flush=True)

            # 阶段 26.5 修复：单独 task 收 ws 消息，事件入 queue
            from livekit.agents.utils import aio as aio_utils
            event_queue: aio_utils.Chan = aio_utils.Chan()

            async def recv_task():
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                obj = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                        if obj.get("type") != "Results":
                            continue
                        is_final = obj.get("is_final", False)
                        speech_final = obj.get("speech_final", False)
                        ch = obj.get("channel", {})
                        alts = ch.get("alternatives", [])
                        t = alts[0].get("transcript", "") if alts else ""
                        # 阶段 26.5 修复：每段都 yield (包括空)，LiveKit 期望每个 speech_final
                        if is_final and speech_final:
                            await event_queue.send(
                                SpeechEvent(
                                    type=SpeechEventType.FINAL_TRANSCRIPT,
                                    alternatives=[
                                        SpeechData(
                                            text=t,
                                            language=self._language,
                                            confidence=0.9 if t else 0.0,
                                        )
                                    ],
                                )
                            )
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
                except Exception as e:
                    print(f"{_PROBE_TAG} recv_task error: {e!r}", flush=True)
                finally:
                    event_queue.close()

            recv = asyncio.create_task(recv_task())

            # 主循环：推 audio + 拉 events
            try:
                async for item in audio_ch:
                    if not isinstance(item, rtc.AudioFrame):
                        continue
                    try:
                        await ws.send_bytes(bytes(item.data))
                    except Exception as e:
                        print(f"{_PROBE_TAG} ws send error: {e!r}", flush=True)
                        break
                    # 拉 queue 里所有 pending events（非阻塞）
                    while True:
                        try:
                            ev = await asyncio.wait_for(event_queue.recv(), timeout=0.001)
                            yield ev
                        except asyncio.TimeoutError:
                            break
            except Exception as loop_err:
                print(f"{_PROBE_TAG} audio loop error: {loop_err!r}", flush=True)

            # audio_ch 结束 → 发 CloseStream 让 deepgram 出最后 final
            try:
                await ws.send_str(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass

            # 等剩余的 final events（最多 5s）
            try:
                while True:
                    ev = await asyncio.wait_for(event_queue.recv(), timeout=5.0)
                    yield ev
            except asyncio.TimeoutError:
                pass

            recv.cancel()
            try:
                await recv
            except (asyncio.CancelledError, Exception):
                pass

        except Exception as e:
            print(f"{_PROBE_TAG} _stt_node_iter error: {e!r}", flush=True)
        finally:
            if ws is not None and not ws.closed:
                try:
                    await ws.close()
                except Exception:
                    pass
            if session is not None and not session.closed:
                try:
                    await session.close()
                except Exception:
                    pass

    async def _recognize_buffer(
        self,
        buffer: list[rtc.AudioFrame],
        url: str,
        headers: dict,
        *,
        started_at: float = 0.0,
    ) -> SpeechEvent:
        """开 ws，把 buffer 里所有 PCM 字节发过去，关闭后收 final transcript。"""
        if started_at == 0.0:
            started_at = time.time()
        # 拼 bytes
        pcm = b"".join(bytes(f.data) for f in buffer)
        text = ""
        try:
            timeout = aiohttp.ClientTimeout(total=15.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(url, headers=headers, autoping=True) as ws:
                    # 一次性发完所有音频
                    await ws.send_bytes(pcm)
                    # 关闭 → 服务端返回 final
                    await ws.send_str(json.dumps({"type": "CloseStream"}))
                    # 收结果
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            obj = json.loads(msg.data)
                            if obj.get("is_final") or obj.get("speech_final"):
                                t = obj.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                                if t:
                                    text = t
                                    break
        except Exception as e:
            print(f"{_PROBE_TAG} _recognize_buffer error: {e!r}", flush=True)

        elapsed_ms = (time.time() - started_at) * 1000
        audio_ms = len(pcm) / (self._sample_rate * 2) * 1000
        print(
            f"{_PROBE_TAG} ok in {elapsed_ms:.0f}ms text_len={len(text)} "
            f"audio_ms={audio_ms:.0f} preview={text[:80]!r}",
            flush=True,
        )
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text=text, language=self._language, confidence=0.9 if text else 0.0)],
        )

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        # 兜底：fallback 路径
        return await self._recognize_buffer(
            buffer, DEEPGRAM_WSS_URL, {"Authorization": f"Token {self._api_key}"}
        )


class _DeepgramStream(RecognizeStream):
    """向后兼容 — v3 主链路走 __call__，不走这里。"""

    def __init__(self, *, stt: DeepgramSTT, api_key: str, model: str, lang: str, sample_rate: int, conn_options: APIConnectOptions):
        super().__init__(stt=stt, conn_options=conn_options)
        self._api_key = api_key
        self._model = model
        self._lang = lang
        self._sample_rate = sample_rate

    async def _run(self) -> None:
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
