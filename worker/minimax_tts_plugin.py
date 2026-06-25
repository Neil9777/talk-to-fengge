"""阶段 26: MiniMax TTS LiveKit 插件 — 流式 (SSE) 版本。

对比非流式（HTTP 一次性）：
- 非流式：发完请求 → 等所有 hex 一次性回来 → 一次性推 emitter
- 流式：发请求 → 服务端边合成边推 SSE 事件 → 边收边推 emitter
  → **TTFB 从 ~770ms 降到 ~600ms**（实测），体感"边说边出"

API 端点：仍是 `https://api.minimaxi.com/v1/t2a_v2`，只是
- body 加 `stream: True` + `stream_options: {exclude_aggregated_audio: True}`
- response 变成 SSE (`text/event-stream`)，多行 `data: {json}`
- 每行 JSON 的 `data.audio` 是 hex PCM 片段
- 最后一行的 `data.status == 2` 表示合成完成

阶段 26.4 补丁：实现 stream() 方法。
LiveKit agents LLM 走 stream() 协议（持续接收文字流 → 边收边合成），
MiniMax 不支持真正的 token-level 流式输入，所以我们在 stream() 里
**收齐一整段输入**（input 关闭 / FlushSentinel 触发），再走 synthesize() 走流式 SSE。
折中：边收齐 + 边合成（一次整段 → SSE 流式输出到 emitter）。
"""

from __future__ import annotations

import asyncio
import binascii
import json
import os
import time
from dataclasses import dataclass

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, tts
from livekit.agents.types import APIConnectOptions
from livekit.agents.tts import TTS, ChunkedStream, SynthesizeStream

MINIMAX_T2A_URL = "https://api.minimaxi.com/v1/t2a_v2"


@dataclass
class _Options:
    api_key: str
    voice_id: str
    model: str
    sample_rate: int
    language_boost: str
    speed: float
    vol: float
    pitch: int
    channel: int


class MinimaxTTS(TTS):
    """MiniMax T2A v2 LiveKit 插件 — 流式 SSE 版本。"""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "speech-02-turbo",
        sample_rate: int = 24000,
        language_boost: str = "Chinese",
        speed: float = 1.0,
        vol: float = 1.0,
        pitch: int = 0,
        channel: int = 1,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=channel,
        )
        self._opts = _Options(
            api_key=api_key,
            voice_id=voice_id,
            model=model,
            sample_rate=sample_rate,
            language_boost=language_boost,
            speed=speed,
            vol=vol,
            pitch=pitch,
            channel=channel,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "MiniMax"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get_session(self) -> aiohttp.ClientSession:
        """异步版 _ensure_session（保留 aclose 路径兼容）"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    @property
    def _http_url(self) -> str:
        return "https://api.minimaxi.com/v1/t2a_v2"

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()
    ) -> ChunkedStream:
        return _MinimaxChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = APIConnectOptions()) -> SynthesizeStream:
        # LiveKit agents LLM streaming 走 stream()；我们用 SynthesizeStream 包装，
        # 在输入流末端拿到完整文本后调 synthesize()（一次完整合成）
        return _MinimaxSynthesizeStream(tts=self, conn_options=conn_options)


class _MinimaxChunkedStream(ChunkedStream):
    """流式合成：SSE 边收边推 emitter。"""

    def __init__(self, *, tts: MinimaxTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: MinimaxTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        started = time.time()
        first_chunk_at: float | None = None
        body = {
            "model": self._tts._opts.model,
            "text": self._input_text,
            # 阶段 26：开 SSE 流式
            "stream": True,
            "stream_options": {"exclude_aggregated_audio": True},
            "voice_setting": {
                "voice_id": self._tts._opts.voice_id,
                "speed": self._tts._opts.speed,
                "vol": self._tts._opts.vol,
                "pitch": self._tts._opts.pitch,
            },
            "audio_setting": {
                "sample_rate": self._tts._opts.sample_rate,
                "bitrate": 128000,
                "format": "pcm",
                "channel": self._tts._opts.channel,
            },
            "language_boost": self._tts._opts.language_boost,
        }
        headers = {
            "Authorization": f"Bearer {self._tts._opts.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            async with self._tts._ensure_session().post(
                MINIMAX_T2A_URL,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise APIStatusError(
                        message=f"MiniMax HTTP {resp.status}: {text[:200]}",
                        status_code=resp.status,
                        request_id=None,
                        body=None,
                    )

                output_emitter.initialize(
                    request_id=f"minimax-stream-{int(time.time()*1000)}",
                    sample_rate=self._tts._opts.sample_rate,
                    num_channels=self._tts._opts.channel,
                    mime_type="audio/pcm",
                )

                # SSE 解析：每个事件以 \n\n 分隔，每行 "data: {json}"
                buffer = ""
                total_bytes = 0
                chunk_count = 0
                status_final_seen = False

                async for raw in resp.content.iter_any():
                    buffer += raw.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        event_block, _, buffer = buffer.partition("\n\n")
                        for line in event_block.split("\n"):
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if not payload:
                                continue
                            try:
                                obj = json.loads(payload)
                            except json.JSONDecodeError:
                                continue

                            base_resp = obj.get("base_resp", {})
                            if base_resp.get("status_code", 0) != 0:
                                raise APIConnectionError(
                                    f"MiniMax status_code={base_resp.get('status_code')} msg={base_resp.get('status_msg')}"
                                )

                            data = obj.get("data") or {}
                            audio_hex = data.get("audio", "")
                            if audio_hex:
                                pcm = binascii.unhexlify(audio_hex)
                                total_bytes += len(pcm)
                                chunk_count += 1
                                if first_chunk_at is None:
                                    first_chunk_at = time.time() - started
                                output_emitter.push(pcm)

                            status = data.get("status")
                            if status == 2:
                                status_final_seen = True

                # 处理 buffer 剩余（最后一个事件可能没 \n\n）
                if buffer.strip() and not status_final_seen:
                    for line in buffer.split("\n"):
                        line = line.strip()
                        if line.startswith("data:"):
                            payload = line[5:].strip()
                            if payload:
                                try:
                                    obj = json.loads(payload)
                                    data = obj.get("data") or {}
                                    audio_hex = data.get("audio", "")
                                    if audio_hex:
                                        pcm = binascii.unhexlify(audio_hex)
                                        total_bytes += len(pcm)
                                        if first_chunk_at is None:
                                            first_chunk_at = time.time() - started
                                        output_emitter.push(pcm)
                                except json.JSONDecodeError:
                                    pass

                output_emitter.flush()

                elapsed_ms = (time.time() - started) * 1000
                ttfb_ms = (first_chunk_at * 1000) if first_chunk_at else 0
                audio_ms = total_bytes / (self._tts._opts.sample_rate * 2) * 1000
                print(
                    f"[minimax_tts stream] ok total={elapsed_ms:.0f}ms ttfb={ttfb_ms:.0f}ms "
                    f"text_len={len(self._input_text)} audio_ms={audio_ms:.0f} chunks={chunk_count} "
                    f"bytes={total_bytes}",
                    flush=True,
                )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIConnectionError:
            raise
        except Exception as e:
            print(f"[minimax_tts stream] error: {e!r}", flush=True)
            raise APIConnectionError() from e


class _MinimaxSynthesizeStream(SynthesizeStream):
    """阶段 26.4 补丁：LLM streaming 兼容。

    LiveKit agents LLM 持续往 input_ch 推文字 + FlushSentinel。
    我们的策略：累积到 FlushSentinel 触发后，调一次 synthesize() 走 SSE 流式输出。
    这样既兼容 LLM streaming（不会因为 sync 阻塞让 LLM 卡住），又能用 MiniMax 流式输出。
    """

    def __init__(self, *, tts: "MinimaxTTS", conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: MinimaxTTS = tts
        # 阶段 26.4 修复：复制 tts 端的 _opts（Cartesia 模式：wrap 一下供子类用）
        self._opts = tts._opts
        self._segments_to_synthesize: list[str] = []  # 多个待合成段（按 FlushSentinel 切分）

    async def _run(self, output_emitter: AudioEmitter) -> None:
        """从 input_ch 收文字段（以 FlushSentinel 分段），每段直接走 HTTP SSE 推给 output_emitter。

        阶段 26.4 重写：直接调 HTTP，**不再嵌套 chunked stream**（嵌套会导致 ChanClosed 冲突）。
        流程：
        1. collector_task 后台跑，持续从 input_ch 读文字 + FlushSentinel 分段
        2. 主循环读 _segments_to_synthesize
        3. 每段文字 → 直接调 aiohttp 走 SSE 边收边推 output_emitter
        """
        request_id = f"minimax-synth-{int(time.time() * 1000)}"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        pending_text = ""

        async def collect_input() -> None:
            nonlocal pending_text
            try:
                async for data in self._input_ch:
                    if isinstance(data, SynthesizeStream._FlushSentinel):
                        if pending_text.strip():
                            self._segments_to_synthesize.append(pending_text)
                            pending_text = ""
                    else:
                        pending_text += data
            except asyncio.CancelledError:
                pass
            finally:
                if pending_text.strip():
                    self._segments_to_synthesize.append(pending_text)
                    pending_text = ""

        collector_task = asyncio.create_task(collect_input())

        # 获取 / 创建 aiohttp session
        session = await self._tts._get_session()

        try:
            idle_count = 0
            while True:
                if self._segments_to_synthesize:
                    text = self._segments_to_synthesize.pop(0)
                    if not text.strip():
                        continue
                    # 直接走 HTTP SSE，**不复用 chunked**（避免 chan 冲突）
                    body = {
                        "model": self._tts._opts.model,
                        "text": text,
                        "stream": True,
                        "stream_options": {"exclude_aggregated_audio": True},
                        "voice_setting": {
                            "voice_id": self._tts._opts.voice_id,
                            "speed": self._tts._opts.speed,
                            "vol": self._tts._opts.vol,
                            "pitch": self._tts._opts.pitch,
                        },
                        "audio_setting": {
                            "sample_rate": self._tts._opts.sample_rate,
                            "format": "pcm",
                            "channel": 1,
                        },
                        "language_boost": self._tts._opts.language_boost,
                    }
                    headers = {
                        "Authorization": f"Bearer {self._tts._opts.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    }
                    t_start = time.time()
                    ttfb_at = [0.0]   # 用 list 存 mutable（避免 holder=0）
                    total_bytes = [0]
                    try:
                        async with session.post(
                            self._tts._http_url,
                            json=body,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30.0),
                        ) as resp:
                            if resp.status != 200:
                                err_text = await resp.text()
                                raise APIStatusError(
                                    message=err_text[:200],
                                    status_code=resp.status,
                                    request_id=None,
                                    body=err_text,
                                )
                            buffer = ""
                            async for raw_chunk in resp.content.iter_any():
                                buffer += raw_chunk.decode("utf-8", errors="replace")
                                # 按双换行切 SSE event
                                while "\n\n" in buffer:
                                    event, _, buffer = buffer.partition("\n\n")
                                    await self._process_sse_event(
                                        event, output_emitter,
                                        t_start=t_start,
                                        ttfb_at_holder=ttfb_at,
                                        total_bytes_holder=total_bytes,
                                    )
                            # 处理剩余 buffer
                            if buffer.strip():
                                await self._process_sse_event(
                                    buffer, output_emitter,
                                    t_start=t_start,
                                    ttfb_at_holder=ttfb_at,
                                    total_bytes_holder=total_bytes,
                                )
                            # 打印一次 segment 完成日志
                            if total_bytes[0] > 0:
                                elapsed = time.time() - t_start
                                audio_ms = total_bytes[0] / (self._tts._opts.sample_rate * 2) * 1000
                                print(
                                    f"[minimax_tts stream] ok total={elapsed*1000:.0f}ms "
                                    f"ttfb={ttfb_at[0]*1000:.0f}ms text_len={len(text)} "
                                    f"audio_ms={audio_ms:.0f} bytes={total_bytes[0]}",
                                    flush=True,
                                )
                    except asyncio.TimeoutError:
                        raise APITimeoutError() from None
                    except aiohttp.ClientResponseError as e:
                        raise APIStatusError(
                            message=e.message,
                            status_code=e.status,
                            request_id=None,
                            body=None,
                        ) from None
                    except APIConnectionError:
                        raise
                    except APIStatusError:
                        raise
                    except Exception as e:
                        print(f"[minimax_tts stream] HTTP error: {e!r}", flush=True)
                        raise APIConnectionError() from e
                    idle_count = 0
                else:
                    await asyncio.sleep(0.03)
                    idle_count += 1
                    if (
                        collector_task.done()
                        and not pending_text.strip()
                        and not self._segments_to_synthesize
                    ):
                        break
                    if idle_count > 200:
                        print(
                            f"[minimax_tts stream] WARN idle > 6s, breaking",
                            flush=True,
                        )
                        break
        finally:
            if not collector_task.done():
                collector_task.cancel()
            # 不关 session（MinimaxTTS 持有）

    async def _process_sse_event(
        self, event: str, output_emitter: AudioEmitter,
        t_start: float, ttfb_at_holder: list, total_bytes_holder: list,
    ):
        """解析一行 SSE event，提取 audio hex 推给 output_emitter。"""
        for line in event.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                obj = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            audio_hex = (obj.get("data") or {}).get("audio", "")
            if audio_hex:
                pcm = binascii.unhexlify(audio_hex)
                output_emitter.push(pcm)
                total_bytes_holder[0] += len(pcm)
                if ttfb_at_holder[0] == 0.0:
                    ttfb_at_holder[0] = time.time() - t_start
