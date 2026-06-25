"""阶段 28.1: Cartesia STT vs Deepgram STT 真实延迟自验（已用裸 ws 跑通，固定脚本）。

跑法：
    .venv/bin/python tests/test_cartesia_stt_real.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

env_file = PROJECT_ROOT / ".env.local"
if env_file.exists():
    load_dotenv(env_file)

import numpy as np
import websockets

VOICE_FILE = PROJECT_ROOT / "assets" / "voice_samples" / "yehuiyu_voice.wav"
SR = 16000
CHUNK_MS = 320
CHUNK_BYTES = SR * CHUNK_MS // 1000 * 2


def load_segment_pcm(start_s: float, dur_s: float) -> bytes:
    with wave.open(str(VOICE_FILE), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        wf.setpos(int(start_s * sr))
        raw = wf.readframes(int(dur_s * sr))
    arr = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    if sr != SR:
        n_target = int(len(arr) * SR / sr)
        idx = np.linspace(0, len(arr) - 1, n_target).astype(int)
        arr = arr[idx]
    return arr.astype(np.int16).tobytes()


async def test_cartesia(pcm: bytes) -> dict:
    api_key = os.getenv("CARTESIA_API_KEY", "")
    if not api_key:
        return {"ok": False, "err": "no key"}
    url = (
        f"wss://api.cartesia.ai/stt/websocket"
        f"?model=ink-whisper&language=zh&sample_rate={SR}&encoding=pcm_s16le"
    )
    headers = [
        ("Authorization", f"Bearer {api_key}"),
        ("Cartesia-Version", "2026-03-01"),
    ]
    t0 = time.time()
    ws = await asyncio.wait_for(
        websockets.connect(url, additional_headers=headers, proxy=None),
        timeout=8.0,
    )
    connect_ms = (time.time() - t0) * 1000
    t_send = time.time()
    for i in range(0, len(pcm), CHUNK_BYTES):
        await ws.send(pcm[i : i + CHUNK_BYTES])
        await asyncio.sleep(0.32)
    await ws.send("finalize")  # 阶段 28.1: livekit 插件源码 send_str('finalize')
    first_interim = first_final = None
    first_text = final_text = ""
    for _ in range(20):
        try:
            raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
        except asyncio.TimeoutError:
            break
        try:
            msg = json.loads(raw_msg)
        except Exception:
            continue
        t = (time.time() - t_send) * 1000
        if msg.get("type") == "transcript":
            txt = msg.get("text", "")
            is_final = msg.get("is_final", False)
            if first_interim is None and txt.strip():
                first_interim = t
                first_text = txt
            if is_final and first_final is None and txt.strip():
                first_final = t
                final_text = txt
                break
    await ws.close()
    return {
        "ok": True,
        "connect_ms": round(connect_ms, 1),
        "first_interim_ms": round(first_interim, 1) if first_interim else None,
        "first_final_ms": round(first_final, 1) if first_final else None,
        "first_text": first_text,
        "final_text": final_text,
    }


async def test_deepgram(pcm: bytes) -> dict:
    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        return {"ok": False, "err": "no key"}
    url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-2&language=zh&sample_rate={SR}"
        f"&encoding=linear16&endpointing=300&interim_results=true"
    )
    headers = [("Authorization", f"Token {api_key}")]
    t0 = time.time()
    ws = await asyncio.wait_for(
        websockets.connect(url, additional_headers=headers, proxy=None),
        timeout=8.0,
    )
    connect_ms = (time.time() - t0) * 1000
    t_send = time.time()
    for i in range(0, len(pcm), CHUNK_BYTES):
        await ws.send(pcm[i : i + CHUNK_BYTES])
        await asyncio.sleep(0.32)
    await ws.send(json.dumps({"type": "CloseStream"}))
    first_interim = first_final = None
    final_text = ""
    for _ in range(30):
        try:
            raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
        except asyncio.TimeoutError:
            break
        try:
            msg = json.loads(raw_msg)
        except Exception:
            continue
        t = (time.time() - t_send) * 1000
        channel = msg.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            continue
        transcript = alternatives[0].get("transcript", "").strip()
        is_final = msg.get("is_final", False)
        speech_final = msg.get("speech_final", False)
        if transcript and first_interim is None:
            first_interim = t
        if (is_final or speech_final) and transcript and first_final is None:
            first_final = t
            final_text = transcript
            break
    await ws.close()
    return {
        "ok": True,
        "connect_ms": round(connect_ms, 1),
        "first_interim_ms": round(first_interim, 1) if first_interim else None,
        "first_final_ms": round(first_final, 1) if first_final else None,
        "final_text": final_text,
    }


async def main():
    print("=== 阶段 28.1: Cartesia vs Deepgram 真实流式 STT 延迟自验 ===")
    print(f"voice: {VOICE_FILE.name} | chunk=320ms @ 16kHz")
    if not VOICE_FILE.exists():
        print("ERROR: 录音文件不存在")
        return
    SEG = (0.0, 3.0)
    pcm = load_segment_pcm(SEG[0], SEG[1])
    print(f"segment: {SEG[0]}s..{SEG[1]}s ({len(pcm)/SR:.1f}s audio)")
    print()

    cr = await test_cartesia(pcm)
    print(f"Cartesia: connect={cr.get('connect_ms')}ms | first_interim={cr.get('first_interim_ms')}ms | first_final={cr.get('first_final_ms')}ms")
    if cr.get("first_text"):
        print(f"  interim text: {cr['first_text'][:60]!r}")
    if cr.get("final_text"):
        print(f"  final text:   {cr['final_text'][:80]!r}")
    print()

    dr = await test_deepgram(pcm)
    print(f"Deepgram: connect={dr.get('connect_ms')}ms | first_interim={dr.get('first_interim_ms')}ms | first_final={dr.get('first_final_ms')}ms")
    if dr.get("final_text"):
        print(f"  final text:   {dr['final_text'][:80]!r}")
    print()

    if cr.get("ok") and dr.get("ok"):
        print("=== summary ===")
        for k in ("connect_ms", "first_interim_ms", "first_final_ms"):
            cv, dv = cr[k], dr[k]
            winner = "CARTESIA" if cv and dv and cv < dv else ("DEEPGRAM" if cv and dv and cv > dv else "?")
            print(f"  {k}: Cartesia={cv}ms | Deepgram={dv}ms | {winner}")


if __name__ == "__main__":
    asyncio.run(main())
