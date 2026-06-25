from __future__ import annotations

import asyncio
import audioop
import base64
import io
import json
import uuid
import wave
from dataclasses import dataclass

import aiohttp
from livekit.agents import APIConnectOptions, APIError, APIStatusError, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.tts import ChunkedStream, TTS, TTSCapabilities


@dataclass
class MossTTSOptions:
    url: str
    voice: str
    model: str = "moss-tts-nano"
    sample_rate: int = 24000
    num_channels: int = 1


class MossHttpTTS(TTS):
    def __init__(
        self,
        *,
        url: str,
        voice: str,
        model: str = "moss-tts-nano",
        sample_rate: int = 24000,
        num_channels: int = 1,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._opts = MossTTSOptions(
            url=url,
            voice=voice,
            model=model,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._http_session: aiohttp.ClientSession | None = None

    @property
    def provider(self) -> str:
        return "MOSS HTTP"

    @property
    def model(self) -> str:
        return self._opts.model

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return _MossChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def _request_audio(self, text: str) -> tuple[str, bytes]:
        session = await self._ensure_session()
        payload = {
            "model": self._opts.model,
            "input": text,
            "voice": self._opts.voice,
            "response_format": "wav",
        }
        async with session.post(self._opts.url, json=payload) as response:
            raw_text = await response.text()
            if response.status >= 400:
                raise APIStatusError(
                    f"MOSS request failed ({response.status}): {raw_text[:500]}",
                    status_code=response.status,
                    request_id="",
                    body=raw_text,
                )
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise APIError(f"MOSS returned invalid JSON: {raw_text[:500]}") from exc

        audio_base64 = str(data.get("audio_base64", "")).strip()
        if not audio_base64:
            raise APIError("MOSS returned JSON without audio_base64")

        request_id = str(data.get("request_id") or uuid.uuid4())
        try:
            wav_bytes = base64.b64decode(audio_base64)
        except Exception as exc:  # noqa: BLE001
            raise APIError("Failed to decode MOSS audio_base64") from exc
        return request_id, wav_bytes


class _MossChunkedStream(ChunkedStream):
    def __init__(self, *, tts: MossHttpTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        self._moss_tts = tts
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter) -> None:
        request_id, wav_bytes = await self._moss_tts._request_audio(self.input_text)
        pcm_bytes = await asyncio.to_thread(self._decode_wav_to_pcm, wav_bytes)
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._moss_tts.sample_rate,
            num_channels=self._moss_tts.num_channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm_bytes)

    def _decode_wav_to_pcm(self, wav_bytes: bytes) -> bytes:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            pcm_bytes = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise APIError(f"Unsupported sample width from MOSS: {sample_width}")

        if channels != self._moss_tts.num_channels:
            if self._moss_tts.num_channels == 1 and channels == 2:
                pcm_bytes = audioop.tomono(pcm_bytes, sample_width, 0.5, 0.5)
            else:
                raise APIError(f"Unsupported channel conversion from {channels} to {self._moss_tts.num_channels}")

        if sample_rate != self._moss_tts.sample_rate:
            pcm_bytes, _ = audioop.ratecv(
                pcm_bytes,
                sample_width,
                self._moss_tts.num_channels,
                sample_rate,
                self._moss_tts.sample_rate,
                None,
            )

        return pcm_bytes
