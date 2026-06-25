"""VoxCPM2 流式 TTS plugin for LiveKit agents.

streaming=True：LLM 边生成边按句子边界触发 TTS，首句延迟目标 ~1.5s。
关键：aiohttp 使用 force_close=True 禁止连接复用，防止 SSH 隧道空闲超时
导致下一句 ServerDisconnectedError（这是6/17卡顿的根因）。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

import aiohttp
from livekit.agents import APIConnectOptions, APIStatusError, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import SynthesizeStream, TTS, TTSCapabilities

_SENT_ENDS = frozenset('。！？；!?;')

TARGET_SAMPLE_RATE = 24000


def _split_with_remainder(text: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remainder). Splits on Chinese/English sentence ends."""
    parts = []
    last = 0
    for j, ch in enumerate(text):
        if ch in _SENT_ENDS:
            s = text[last:j + 1].strip()
            if s:
                parts.append(s)
            last = j + 1
    return parts, text[last:]


def _resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    import audioop
    if src_rate == dst_rate:
        return pcm
    result, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
    return result


@dataclass
class VoxCPMOptions:
    url: str
    voice: str
    style: str
    sample_rate: int


class VoxCPMHttpTTS(TTS):
    def __init__(
        self,
        *,
        url: str = "http://localhost:8000",
        voice: str = "fengge",
        style: str = "",
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = VoxCPMOptions(url=url, voice=voice, style=style, sample_rate=sample_rate)
        self._http_session: aiohttp.ClientSession | None = None
        self._server_sample_rate: int | None = None

    @property
    def provider(self) -> str:
        return "VoxCPM2-nano"

    @property
    def model(self) -> str:
        return "VoxCPM2"

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return _VoxCPMSynthesizeStream(tts=self, conn_options=conn_options)

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return self._synthesize_with_stream(text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            # force_close=True：每次请求用新 TCP 连接，不复用。
            # 必须设置，否则 SSH 隧道在句子间隙空闲超时后断连，
            # aiohttp 复用旧连接会触发 ServerDisconnectedError。
            connector = aiohttp.TCPConnector(force_close=True)
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=120, sock_read=60),
            )
        return self._http_session

    async def _get_server_sample_rate(self) -> int:
        return 48000


class _VoxCPMSynthesizeStream(SynthesizeStream):
    """LLM token 流 → 句子边界切割 → VoxCPM 流式生成，句子级 pipelining。"""

    def __init__(self, *, tts: VoxCPMHttpTTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._vox = tts

    async def _run(self, output_emitter) -> None:
        request_id = str(uuid.uuid4())
        initialized = False
        server_sr = await self._vox._get_server_sample_rate()
        buffer = ""
        t0 = time.time()

        async def send_sentence(text: str) -> None:
            nonlocal initialized
            text = text.strip()
            if not text:
                return
            styled = f"({self._vox._opts.style}){text}" if self._vox._opts.style else text
            payload = {"input": styled, "voice": self._vox._opts.voice}
            print(f"[tts] voxcpm→send t={time.time()-t0:.3f}s: {text[:30]!r}", flush=True)

            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    # 每次请求重新获取 session（force_close 保证新 TCP 连接）
                    sess = await self._vox._ensure_session()
                    async with sess.post(
                        f"{self._vox._opts.url}/v1/audio/speech/stream",
                        json=payload,
                    ) as resp:
                        if resp.status >= 400:
                            body = await resp.text()
                            raise APIStatusError(
                                f"VoxCPM {resp.status}: {body[:200]}",
                                status_code=resp.status,
                                request_id=request_id,
                                body=body,
                            )

                        if not initialized:
                            output_emitter.initialize(
                                request_id=request_id,
                                sample_rate=self._vox.sample_rate,
                                num_channels=1,
                                mime_type="audio/pcm",
                            )
                            initialized = True

                        async for chunk in resp.content.iter_chunked(4096):
                            if not chunk:
                                continue
                            if len(chunk) % 2 != 0:
                                chunk = chunk[:-1]
                            if not chunk:
                                continue
                            pcm = _resample_pcm(chunk, server_sr, self._vox.sample_rate)
                            output_emitter.push(pcm)
                    return  # 成功，退出重试循环
                except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError) as exc:
                    last_exc = exc
                    print(f"[tts] voxcpm disconnect attempt={attempt+1}: {exc!r}", flush=True)
                    await asyncio.sleep(0.3)
            raise RuntimeError(f"VoxCPM TTS failed after 3 attempts: {last_exc}")

        async for token_or_sentinel in self._input_ch:
            if isinstance(token_or_sentinel, SynthesizeStream._FlushSentinel):
                # LLM 结束或 flush() 调用 — 发送剩余 buffer
                if buffer.strip():
                    await send_sentence(buffer)
                    buffer = ""
            else:
                buffer += token_or_sentinel
                sentences, buffer = _split_with_remainder(buffer)
                for sentence in sentences:
                    await send_sentence(sentence)

        # input_ch 关闭后处理残留
        if buffer.strip():
            await send_sentence(buffer)

        if not initialized:
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._vox.sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
