"""阶段 26.5 直接调 deepgram ws 验证协议。"""
import asyncio
import os
import sys
import wave
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp
import numpy as np


async def go():
    url = "wss://api.deepgram.com/v1/listen?language=zh&model=nova-2&smart_format=true&encoding=linear16&sample_rate=22050&channels=1&interim_results=false&endpointing=300&vad_events=true"
    dg_key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not dg_key:
        print("SKIP: DEEPGRAM_API_KEY not set"); return
    headers = {"Authorization": f"Token {dg_key}"}
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=15.0))
    ws = await session.ws_connect(url, headers=headers, autoping=True, autoclose=False)
    print("ws connected")

    with wave.open("assets/voice_samples/yehuiyu_voice.wav", "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        sr = wf.getframerate()
        nch = wf.getnchannels()
    arr = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)

    fpc = sr * 20 // 1000
    for i in range(0, len(arr), fpc):
        chunk = arr[i:i+fpc]
        if len(chunk) < fpc:
            chunk = np.pad(chunk, (0, fpc - len(chunk)))
        await ws.send_bytes(chunk.tobytes())
        await asyncio.sleep(0.001)
    print(f"audio sent ({len(arr)/sr:.1f}s), waiting for results...")

    transcripts = []
    try:
        async with asyncio.timeout(10):
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    obj = json.loads(msg.data)
                    if obj.get("type") != "Results":
                        print(f"  event: type={obj.get('type')}")
                        continue
                    is_final = obj.get("is_final")
                    speech_final = obj.get("speech_final")
                    ch = obj.get("channel", {})
                    alts = ch.get("alternatives", [])
                    t = alts[0].get("transcript", "") if alts else ""
                    print(f"  Results is_final={is_final} speech_final={speech_final} t={t[:60]!r}")
                    if t and (is_final or speech_final):
                        transcripts.append(t)
    except (asyncio.TimeoutError, TimeoutError):
        pass

    print(f"\ntotal transcripts: {len(transcripts)}")
    print(f"full: {' '.join(transcripts)[:500]}")
    await ws.close()
    await session.close()


if __name__ == "__main__":
    asyncio.run(go())
