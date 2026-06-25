"""阶段 25.2 自测脚本：用主公录的 28s 音频，按 A 文案切句，验证 STT 输出与原文一致。

原理：
1. 把 `assets/voice_samples/yehuiyu_voice.wav` 按时间戳切分成 5 个短句
2. 每句送进 GeminiSTT.__call__（直接调 STT，不走 livekit 房间）
3. 实际调真 Gemini API（按阶段 24 的能力走）
4. 比对 STT 输出 vs 原文，记录匹配率

跑法：
    cd projects/talk-to-me-v3
    .venv/bin/python tests/test_stt_e2e_real.py

不需要 livekit server 在跑（不走 voice pipeline），但需要 1087 代理（调 Gemini API）。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
from livekit import rtc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from worker.gemini_stt import GeminiSTT  # noqa: E402

# A 文案（主公按这个录的 28s 连续录音，按换行分成 5 段比对）
EXPECTED_SENTENCES = [
    "嗯，我觉得这个事情其实是这样的。咱们不要着急，先想清楚到底要解决什么问题。",
    "很多时候方案本身没问题，但执行过程中容易跑偏，因为一开始就没定义清楚成功长什么样。",
    "所以我习惯先把问题拆开来看，一步一步推，推到最后再看这个方向对不对。",
    "如果不对，换方向也不丢人，对吧？",
    "反正做产品就是这样，你得允许自己犯错，但不能允许自己不知道错在哪。",
]

AUDIO_PATH = PROJECT_ROOT / "assets" / "voice_samples" / "yehuiyu_voice.wav"


def load_wav_frames(path: Path, sample_rate: int = 24000) -> list[rtc.AudioFrame]:
    """读 wav 文件，按 20ms 一帧切分。"""
    import wave
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        file_sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    print(f"[load] file={path.name} sr={file_sr} ch={n_channels} width={sample_width} frames={n_frames}")
    if file_sr != sample_rate:
        print(f"[load] WARN: file sr {file_sr} != target {sample_rate}, 暂时不重采样（按原始 sr 切分）")
        sample_rate = file_sr
    if n_channels != 1:
        print(f"[load] WARN: file is {n_channels}ch, 临时 mix down")

    # 24kHz, 16bit, mono: 1 帧 20ms = 480 samples = 960 bytes
    # 16kHz, 16bit, mono: 1 帧 20ms = 320 samples = 640 bytes
    samples_per_frame = sample_rate * 20 // 1000
    bytes_per_frame = samples_per_frame * n_channels * sample_width

    # 转 int16 数组
    arr = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        # mix down to mono
        arr = arr.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    frames = []
    n_total = len(arr) // samples_per_frame
    print(f"[load] total frames @ 20ms: {n_total} (={n_total * 20 / 1000:.1f}s)")
    for i in range(n_total):
        chunk = arr[i * samples_per_frame : (i + 1) * samples_per_frame]
        frames.append(rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(chunk),
        ))
    return frames


def cut_sentences_by_time(frames: list[rtc.AudioFrame], sentences: list[str], total_duration_s: float) -> list[tuple[str, list[rtc.AudioFrame]]]:
    """按 5 等分时间窗口切分（粗略：录音是 5 段连续说话）

    在段间插入 200ms 静音，帮助 VAD 边界检测
    """
    n = len(frames)
    seg_size = n // len(sentences)
    # 200ms 静音帧 (10 帧 @ 20ms)
    silence_pad = [make_silent_frame(20) for _ in range(10)]
    result = []
    for i, s in enumerate(sentences):
        start = i * seg_size
        end = (i + 1) * seg_size if i < len(sentences) - 1 else n
        seg = frames[start:end]
        # 段尾加 200ms 静音
        if i < len(sentences) - 1:
            seg = seg + silence_pad
        result.append((s, seg))
    return result


def make_silent_frame(duration_ms: int = 20, sample_rate: int = 24000) -> rtc.AudioFrame:
    samples = sample_rate * duration_ms // 1000
    data = np.zeros(samples, dtype=np.int16).tobytes()
    return rtc.AudioFrame(
        data=data, sample_rate=sample_rate, num_channels=1, samples_per_channel=samples
    )


async def transcribe_one(stt_instance: GeminiSTT, frames: list[rtc.AudioFrame], label: str) -> str:
    """跑 STT 收集 1 个 transcript。"""
    async def audio_iter():
        for f in frames:
            yield f

    transcripts = []
    async def collector():
        async for ev in stt_instance(audio_iter(), None):
            t = ev.alternatives[0].text if ev.alternatives else ""
            transcripts.append(t)
            return  # 第一个就退出
    try:
        await asyncio.wait_for(collector(), timeout=30.0)
    except asyncio.TimeoutError:
        print(f"[{label}] TIMEOUT")
    return transcripts[0] if transcripts else ""


def score(expected: str, actual: str) -> tuple[int, int, float]:
    """简单字符级匹配分：返回 (matched_chars, total_chars, ratio)。"""
    if not expected:
        return 0, 0, 0.0
    e, a = expected.replace(" ", ""), actual.replace(" ", "")
    matched = sum(1 for x, y in zip(e, a) if x == y)
    return matched, max(len(e), len(a)), matched / max(len(e), len(a), 1)


async def main():
    if not AUDIO_PATH.exists():
        print(f"❌ 音频文件不存在: {AUDIO_PATH}")
        sys.exit(1)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set")
        sys.exit(1)

    print(f"\n=== 阶段 25.2 自测：主公录音 + 真 Gemini STT ===\n")
    print(f"音频: {AUDIO_PATH.name}")
    print(f"API key: {api_key[:10]}...{api_key[-4:]}")

    frames = load_wav_frames(AUDIO_PATH)
    total_s = len(frames) * 20 / 1000
    print(f"总时长: {total_s:.1f}s ({len(frames)} 帧 @ 20ms)")

    # 按 5 等分切
    segments = cut_sentences_by_time(frames, EXPECTED_SENTENCES, total_s)

    # 阶段 25.1 修复后的 STT
    stt_instance = GeminiSTT(
        api_key=api_key,
        model="gemini-2.5-flash",  # 真模型
        silence_rms_threshold=80,
        silence_min_frames=15,
        min_speech_frames=10,
    )

    print(f"\n--- 跑 STT 5 段 ---\n")
    results = []
    started = time.time()
    for i, (expected, seg_frames) in enumerate(segments):
        t0 = time.time()
        actual = await transcribe_one(stt_instance, seg_frames, f"seg{i+1}")
        elapsed = time.time() - t0
        m, t, r = score(expected, actual)
        match = "✅" if r >= 0.7 else "⚠️" if r >= 0.4 else "❌"
        results.append((expected, actual, r))
        print(f"{match} 段 {i+1} ({elapsed:.1f}s, 匹配 {r:.0%}):")
        print(f"   期望: {expected}")
        print(f"   实际: {actual}")
        print()

    avg = sum(r for _, _, r in results) / len(results)
    total_elapsed = time.time() - started
    print(f"=== 总览: 平均匹配率 {avg:.0%}, 5 段耗时 {total_elapsed:.1f}s ===")
    if avg >= 0.7:
        print("✅ STT 阶段 25.1 修复有效 — 5 段中 70% 以上字符识别正确")
        return 0
    else:
        print("❌ 匹配率 < 70%, STT 还需调参")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
