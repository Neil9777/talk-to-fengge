"""
阶段 26.4 修复后的 self-test：
模拟 LiveKit 真实 stream() 协议 — 推 input_ch + FlushSentinel，看 output_emitter 能否收到 audio。
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from livekit.agents.tts import AudioEmitter, SynthesizedAudio
from livekit.agents.utils import aio

from worker.minimax_tts_plugin import MinimaxTTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_file = PROJECT_ROOT / ".env.local"
if _env_file.exists():
    load_dotenv(_env_file)


async def test_stream_protocol():
    """模拟 LLM 推文字 + FlushSentinel 流程，验证 stream() 协议。"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    voice_id = os.environ.get("MINIMAX_VOICE_ID", "yehuiyuClone01")
    if not api_key:
        print("SKIP: MINIMAX_API_KEY not set (see .env.local)")
        return True

    tts = MinimaxTTS(
        api_key=api_key,
        voice_id=voice_id,
    )

    # 用真实 SynthesizeStream
    stream = tts.stream()
    print(f"stream type: {type(stream).__name__}")

    # 把内部 input_ch 拿出来（基类 aio.Chan）
    input_ch = stream._input_ch
    event_ch = stream._event_ch

    # 1) 模拟 LLM 推一段文字（不通过 FlushSentinel）
    print("\n[1] pushing text '你好，我是叶会羽的' to input_ch...")
    await input_ch.send("你好，我是叶会羽的")
    await asyncio.sleep(0.2)

    # 2) 推 FlushSentinel 触发第一段合成
    print("[2] pushing FlushSentinel to trigger first synthesis...")
    await input_ch.send(stream._FlushSentinel())

    # 3) 后台推更多文字（模拟 LLM 持续 streaming）
    print("[3] pushing more text 'AI 分身。' to input_ch...")
    await input_ch.send("AI 分身。")
    await asyncio.sleep(0.2)
    await input_ch.send(stream._FlushSentinel())

    # 4) 关闭 input (模拟 LLM 结束)
    print("[4] closing input_ch...")
    input_ch.close()

    # 5) 收集 events
    print("\n[5] collecting events from event_ch...")
    events: list[SynthesizedAudio] = []
    total_bytes = 0
    chunk_count = 0
    t_start = asyncio.get_event_loop().time()
    while True:
        try:
            ev = await asyncio.wait_for(event_ch.recv(), timeout=15.0)
            if ev.frame is not None:
                total_bytes += len(ev.frame.data)
                chunk_count += 1
                if chunk_count == 1 or chunk_count % 5 == 0:
                    print(f"  chunk {chunk_count}: {len(ev.frame.data)} bytes")
            events.append(ev)
        except asyncio.TimeoutError:
            print("  TIMEOUT (15s)")
            break
        except aio.ChanClosed:
            print("  event_ch closed")
            break

    elapsed = asyncio.get_event_loop().time() - t_start
    print(f"\n=== RESULT ===")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"events: {len(events)}")
    print(f"chunks with audio: {chunk_count}")
    print(f"total audio bytes: {total_bytes}")
    if total_bytes > 0:
        audio_ms = total_bytes / (tts.sample_rate * 2) * 1000
        print(f"audio duration: {audio_ms:.0f}ms @ {tts.sample_rate}Hz mono16")
    print(f"stream._task done: {stream._task.done() if hasattr(stream, '_task') else 'n/a'}")
    if hasattr(stream, '_task') and stream._task.done():
        try:
            print(f"task exception: {stream._task.exception()}")
        except Exception:
            pass

    await tts.aclose()

    if chunk_count > 0 and total_bytes > 0:
        print("\n✅ PASS — stream() 协议工作正常")
        return True
    else:
        print("\n❌ FAIL — 没收到 audio")
        return False


if __name__ == "__main__":
    ok = asyncio.run(test_stream_protocol())
    sys.exit(0 if ok else 1)
