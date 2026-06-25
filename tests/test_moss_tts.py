"""沙箱内单测 — 验证 MossHttpTTS 行为。

不启 MOSS 服务、不发真实 HTTP。mock aiohttp.ClientSession。

覆盖：
- _request_audio 200 + audio_base64 → (request_id, wav_bytes)
- _request_audio 4xx → APIStatusError（带 500 字截断错误体）
- _request_audio 200 + 非 JSON → APIError
- _request_audio 200 + JSON 缺 audio_base64 → APIError
- _request_audio 200 + audio_base64 不是合法 base64 → APIError
- _MossChunkedStream._decode_wav_to_pcm 1ch 静默 wav → 等长 PCM
- _MossChunkedStream._decode_wav_to_pcm 2ch → 1ch 转码
- _MossChunkedStream._decode_wav_to_pcm 非 16bit → APIError
- MossHttpTTS.synthesize 返回 _MossChunkedStream 实例

对应真人日志里阶段 15 的"MOSS 收到请求但 BrokenPipe"现象：
本次只锁代码层行为，不验证 MOSS 服务连通性。
"""
from __future__ import annotations

import asyncio
import base64
import io
import sys
import unittest
import warnings
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 抑制 mock aiohttp session 残留的"Unclosed client session"警告
# （必须在 livekit/aiohttp import 之前设，否则 aiohttp 默认 filter 已生效）
warnings.simplefilter("ignore", ResourceWarning)

from livekit.agents import APIConnectOptions, APIError, APIStatusError, DEFAULT_API_CONNECT_OPTIONS  # noqa: E402
from worker.moss_tts import MossHttpTTS, _MossChunkedStream  # noqa: E402


def _make_wav_bytes(*, sr: int = 24000, ch: int = 1, seconds: float = 0.05,
                    sample_width: int = 2) -> bytes:
    """生成最小 wav。frames = sr*seconds，每帧 ch 个 sample，每个 sample sample_width 字节。"""
    frames = int(sr * seconds)
    pcm = b"\x00\x00" * frames * ch  # 字节数 = frames * ch * sample_width
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sample_width)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def _mock_aiohttp_response(*, status: int, body_text: str) -> MagicMock:
    """构造 aiohttp 响应 mock — 支持 async with 上下文管理。"""
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body_text)
    # aiohttp 用的是 __aenter__/__aexit__ 而不是 __enter__/__exit__
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=None)
    return resp


def _mock_aiohttp_session(resp: MagicMock) -> MagicMock:
    session = MagicMock()
    session.closed = False
    session.post = MagicMock(return_value=resp)
    return session


class MossHttpTTSRequestTest(unittest.TestCase):
    """_request_audio 的请求/响应行为。"""

    def setUp(self):
        self.tts = MossHttpTTS(
            url="http://127.0.0.1:18083/v1/audio/speech",
            voice="ye-local",
            model="moss-tts-nano",
        )

    def test_request_audio_200_with_audio_base64_returns_wav_bytes(self):
        """200 + 合法 audio_base64 → (request_id, wav_bytes)。"""
        wav = _make_wav_bytes()
        body = json_dumps({
            "audio_base64": base64.b64encode(wav).decode(),
            "request_id": "req-abc-123",
        })
        resp = _mock_aiohttp_response(status=200, body_text=body)
        session = _mock_aiohttp_session(resp)
        with patch.object(self.tts, "_ensure_session", AsyncMock(return_value=session)):
            rid, got = asyncio.run(self.tts._request_audio("你好"))

        self.assertEqual(rid, "req-abc-123")
        self.assertEqual(got, wav)
        # 验证 post 的 payload 包含关键字段
        called_args, called_kwargs = session.post.call_args
        self.assertEqual(called_args[0], "http://127.0.0.1:18083/v1/audio/speech")
        payload = called_kwargs["json"]
        self.assertEqual(payload["model"], "moss-tts-nano")
        self.assertEqual(payload["input"], "你好")
        self.assertEqual(payload["voice"], "ye-local")
        self.assertEqual(payload["response_format"], "wav")

    def test_request_audio_4xx_raises_api_status_error_with_truncated_body(self):
        """4xx → APIStatusError，message 截断到 500 字（body 字段保持完整便于调试）。"""
        long_body = "x" * 800
        resp = _mock_aiohttp_response(status=500, body_text=long_body)
        session = _mock_aiohttp_session(resp)
        with patch.object(self.tts, "_ensure_session", AsyncMock(return_value=session)):
            with self.assertRaises(APIStatusError) as ctx:
                asyncio.run(self.tts._request_audio("hi"))

        exc = ctx.exception
        self.assertEqual(exc.status_code, 500)
        msg = exc.args[0]  # message 字段，body 单独存
        self.assertIn("MOSS request failed (500)", msg)
        # message 截断到 500 字：args[0] 里的 x 数量 = 500
        self.assertEqual(msg.count("x"), 500)
        # body 字段保留完整 800 个 x（便于排查）
        self.assertEqual(exc.body.count("x"), 800)

    def test_request_audio_200_invalid_json_raises_api_error(self):
        """200 + 非 JSON → APIError。"""
        resp = _mock_aiohttp_response(status=200, body_text="not-json{")
        session = _mock_aiohttp_session(resp)
        with patch.object(self.tts, "_ensure_session", AsyncMock(return_value=session)):
            with self.assertRaises(APIError) as ctx:
                asyncio.run(self.tts._request_audio("hi"))
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_request_audio_200_missing_audio_base64_raises_api_error(self):
        """200 + JSON 但缺 audio_base64 → APIError。"""
        body = json_dumps({"request_id": "x"})
        resp = _mock_aiohttp_response(status=200, body_text=body)
        session = _mock_aiohttp_session(resp)
        with patch.object(self.tts, "_ensure_session", AsyncMock(return_value=session)):
            with self.assertRaises(APIError) as ctx:
                asyncio.run(self.tts._request_audio("hi"))
        self.assertIn("audio_base64", str(ctx.exception))

    def test_request_audio_200_empty_audio_base64_raises_api_error(self):
        """200 + audio_base64 为空串 → APIError。"""
        body = json_dumps({"audio_base64": ""})
        resp = _mock_aiohttp_response(status=200, body_text=body)
        session = _mock_aiohttp_session(resp)
        with patch.object(self.tts, "_ensure_session", AsyncMock(return_value=session)):
            with self.assertRaises(APIError) as ctx:
                asyncio.run(self.tts._request_audio("hi"))
        self.assertIn("audio_base64", str(ctx.exception))

    def test_request_audio_200_invalid_base64_raises_api_error(self):
        """200 + audio_base64 不是合法 base64 → APIError。"""
        body = json_dumps({"audio_base64": "!!!not-base64!!!"})
        resp = _mock_aiohttp_response(status=200, body_text=body)
        session = _mock_aiohttp_session(resp)
        with patch.object(self.tts, "_ensure_session", AsyncMock(return_value=session)):
            with self.assertRaises(APIError) as ctx:
                asyncio.run(self.tts._request_audio("hi"))
        self.assertIn("decode", str(ctx.exception).lower())


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


class MossHttpTTSDecodeTest(unittest.TestCase):
    """_MossChunkedStream._decode_wav_to_pcm 的 wav→pcm 转换。"""

    def setUp(self):
        self.tts = MossHttpTTS(
            url="http://127.0.0.1:18083/v1/audio/speech",
            voice="ye-local",
        )

    def _make_stream(self):
        """在 event loop 内创建 _MossChunkedStream（基类要 aio.Chan）。"""
        return _MossChunkedStream(
            tts=self.tts,
            input_text="test",
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )

    def test_decode_1ch_24k_silence_returns_equal_length_pcm(self):
        """1ch/24k 静默 wav → 等长 PCM。"""
        wav = _make_wav_bytes(sr=24000, ch=1, seconds=0.05)

        async def go():
            stream = self._make_stream()
            return stream._decode_wav_to_pcm(wav)

        pcm = asyncio.run(go())
        # 静默 wav 0.05s @ 24k = 1200 samples * 2 bytes = 2400 bytes
        self.assertEqual(len(pcm), 2400)

    def test_decode_2ch_to_1ch_conversion(self):
        """2ch → 1ch 转码：转后 PCM 帧数减半，字节数减半。"""
        wav = _make_wav_bytes(sr=24000, ch=2, seconds=0.05)

        async def go():
            stream = self._make_stream()
            return stream._decode_wav_to_pcm(wav)

        pcm2 = asyncio.run(go())
        # 2ch 0.05s @ 24k = 2400 samples = 4800 bytes
        # tomono 后 1ch 1200 samples = 2400 bytes
        self.assertEqual(len(pcm2), 2400)

    def test_decode_unsupported_sample_width_raises(self):
        """32-bit wav → APIError（实现只支持 16-bit）。"""
        wav = _make_wav_bytes(sr=24000, ch=1, seconds=0.02, sample_width=4)

        async def go():
            stream = self._make_stream()
            return stream._decode_wav_to_pcm(wav)

        with self.assertRaises(APIError) as ctx:
            asyncio.run(go())
        self.assertIn("Unsupported sample width", str(ctx.exception))

    def test_decode_sample_rate_mismatch_uses_ratecv(self):
        """非 24k 采样率 → 走 audioop.ratecv 转码（按帧对齐，最后 1-2 字节可能差）。"""
        wav = _make_wav_bytes(sr=16000, ch=1, seconds=0.05)

        async def go():
            stream = self._make_stream()
            return stream._decode_wav_to_pcm(wav)

        pcm = asyncio.run(go())
        # 16000 → 24000：0.05s 16k = 800 frames，转 24k ≈ 1200 frames = 2400 bytes
        # ratecv 按整帧对齐，允许 ±5 字节误差
        self.assertAlmostEqual(len(pcm), 2400, delta=5)


class MossHttpTTSSynthesizeTest(unittest.TestCase):
    """synthesize 返回的 stream 类型。"""

    def test_synthesize_returns_chunked_stream(self):
        tts = MossHttpTTS(
            url="http://127.0.0.1:18083/v1/audio/speech",
            voice="ye-local",
        )

        async def go():
            stream = tts.synthesize("hello", conn_options=DEFAULT_API_CONNECT_OPTIONS)
            return stream, stream.input_text, stream._moss_tts

        stream, input_text, moss_ref = asyncio.run(go())
        self.assertIsInstance(stream, _MossChunkedStream)
        self.assertEqual(input_text, "hello")
        self.assertIs(moss_ref, tts)


if __name__ == "__main__":
    unittest.main()
