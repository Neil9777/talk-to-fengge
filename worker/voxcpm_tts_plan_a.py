"""VoxCPM2 TTS plugin — 方案 A（非流式，稳定版）

streaming=False：LLM 全文生成完毕后，整段文本一次性发给 TTS。
每轮只有 1 次 HTTP 请求，不会触发 SSH 隧道空闲超时问题，稳定性最高。
代价：首包延迟 = LLM 全文生成时间（约 3-5s）+ TTS 首包（~0.6s）。
这是 v3.6-voxcpm-base tag 的行为。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import aiohttp
from livekit.agents import APIConnectOptions, APIStatusError, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import ChunkedStream, TTS, TTSCapabilities

TARGET_SAMPLE_RATE = 24000


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
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = VoxCPMOptions(url=url, voice=voice, style=style, sample_rate=sample_rate)
        self._http_session: aiohttp.ClientSession | None = None

    @property
    def provider(self) -> str:
        return "VoxCPM2-nano-A"

    @property
    def model(self) -> str:
        return "VoxCPM2"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        return _VoxCPMChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            connector = aiohttp.TCPConnector(force_close=True)
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=120, sock_read=60),
            )
        return self._http_session


class _VoxCPMChunkedStream(ChunkedStream):
    def __init__(self, *, tts: VoxCPMHttpTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._vox = tts

    async def _run(self, output_emitter) -> None:
        request_id = str(uuid.uuid4())
        text = self.input_text.strip()
        if not text:
            return
        if self._vox._opts.style:
            text = f"({self._vox._opts.style}){text}"

        t0 = time.time()
        print(f"[tts-A] voxcpm→send t={time.time()-t0:.3f}s: {text[:40]!r}", flush=True)

        session = await self._vox._ensure_session()
        async with session.post(
            f"{self._vox._opts.url}/v1/audio/speech/stream",
            json={"input": text, "voice": self._vox._opts.voice},
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise APIStatusError(
                    f"VoxCPM {resp.status}: {body[:200]}",
                    status_code=resp.status,
                    request_id=request_id,
                    body=body,
                )
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._vox.sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
            async for chunk in resp.content.iter_chunked(4096):
                if not chunk:
                    continue
                if len(chunk) % 2 != 0:
                    chunk = chunk[:-1]
                if not chunk:
                    continue
                pcm = _resample_pcm(chunk, 48000, self._vox.sample_rate)
                output_emitter.push(pcm)
