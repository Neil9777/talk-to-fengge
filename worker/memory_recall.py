"""阶段 28: OpenViking 本地记忆快照——全量 list → 同类合并 → 压缩到 ≤800 字。

主公的简化思路（你这次原话）：
> 之前不是说有预热吗？看你是否还有必要……
> 你直接把 Viking 本地沉淀的一些记忆全部 list 出来，合并一下同类项，压缩整合一下不就行了吗？

实现策略（与本地 fs 真实结构对齐）：
- 读 `OPENVIKING_DATA_DIR/user/default/memories/<memory_type>/default/*.md`
- 解析每个 md 的 bullet（行首 "- "）和尾部 MEMORY_FIELDS memory_type
- 按 memory_type 分组 → 拼成 markdown → 按中英文标点分句 → 截到 800 字
- 完全不依赖 OpenViking server，0 网络抖动，单次 ~5-20ms
- 失败 → 返回 ""（绝不阻塞 agent 启动）
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 阶段 28: 默认走 .env.local 里的 OPENVIKING_DATA_DIR，与 agent.py 保持一致
_persona_for_memory = os.getenv("PERSONA_NAME", "fengge").strip().lower()
_project_root = Path(__file__).resolve().parent.parent
_default_data_dir = str(_project_root / "data" / "openviking" / "viking" / "default")
DEFAULT_DATA_DIR = Path(os.getenv("OPENVIKING_DATA_DIR", _default_data_dir))

# 阶段 28：主公拍板 800 字上限
DEFAULT_MAX_CHARS = 800

# type 排序：entities > events > preference > general
# entities 是"我是谁/花姐/射手座"等硬事实，最先保留；events 偏回顾；preferences 占最多
_TYPE_PRIORITY = [
    "identity", "identity_card",
    "entity", "entities",
    "relationship",
    "event", "events",
    "preference", "preferences",
    "plan",
    "general",
]

_SENT_END = re.compile(r"(?<=[。！？!?；;\n])")
_BULLET = re.compile(r"^[\s]*[-*]\s*(.+)$", re.MULTILINE)
_MEMORY_FIELDS = re.compile(r"<!--\s*MEMORY_FIELDS\s*(\{.*?\})\s*-->", re.DOTALL)
_SUMMARY_LINE = re.compile(r"^Summary:\s*(.+)$", re.MULTILINE)


@dataclass
class RecallConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    max_chars: int = DEFAULT_MAX_CHARS
    # 阶段 28：先做"全部 1 周内"，目前没时间过滤，先全量；将来用 mtime 筛
    freshness_days: int | None = None  # None = 全部


# ── 纯本地读取 ───────────────────────────────────────────────
def _parse_memory_file(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    if not text.strip():
        return None

    # 1. 提取 MEMORY_FIELDS 里的 memory_type
    mt = "general"
    summary_in_fields: str | None = None
    m = _MEMORY_FIELDS.search(text)
    if m:
        try:
            fields = json.loads(m.group(1))
            mt = str(fields.get("memory_type") or fields.get("category") or "general").strip() or "general"
            s = fields.get("summary")
            if isinstance(s, str) and s.strip():
                summary_in_fields = s.strip()
        except Exception:
            pass

    # 2. 优先提取 bullet（"- xxx"），长度 > 4 避免匹到 "->" 这种箭头
    bullets = [b.strip() for b in _BULLET.findall(text) if b.strip() and len(b.strip()) > 4]
    if not bullets and summary_in_fields:
        # event 文件: MEMORY_FIELDS.summary 就是核心
        bullets = [summary_in_fields]
    if not bullets:
        # 退化: 找行内 "Summary: xxx"（一些 entity/event 用这种格式）
        m2 = _SUMMARY_LINE.search(text)
        if m2:
            bullets = [m2.group(1).strip()]
    if not bullets:
        # 兜底: 把所有非空、非 heading、非 comment、非 bullet 的行当 summary
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("<!--") or s.startswith("-"):
                continue
            bullets.append(s)
    if not bullets:
        return None
    return {"memory_type": mt, "bullets": bullets, "path": str(path)}


def _scan_memories(data_dir: Path, freshness_days: int | None) -> list[dict[str, Any]]:
    memories_dir = data_dir / "user" / "default" / "memories"
    if not memories_dir.exists():
        return []
    import time
    cutoff = None
    if freshness_days is not None and freshness_days > 0:
        cutoff = time.time() - freshness_days * 86400

    out: list[dict[str, Any]] = []
    for md_path in memories_dir.rglob("*.md"):
        if md_path.name.startswith("."):
            continue
        if md_path.name.endswith(".ovlock.profile.md"):
            continue
        if cutoff is not None and md_path.stat().st_mtime < cutoff:
            continue
        rec = _parse_memory_file(md_path)
        if rec:
            out.append(rec)
    return out


def _group_by_type(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for r in records:
        for b in r["bullets"]:
            if b not in grouped[r["memory_type"]]:  # 同 bullet 去重
                grouped[r["memory_type"]].append(b)
    return grouped


def _merge_groups(grouped: dict[str, list[str]]) -> str:
    ordered_types = sorted(
        grouped.keys(),
        key=lambda t: (_TYPE_PRIORITY.index(t) if t in _TYPE_PRIORITY else len(_TYPE_PRIORITY), t),
    )
    parts: list[str] = []
    for t in ordered_types:
        parts.append(f"## {t}")
        for s in grouped[t]:
            parts.append(f"- {s}")
    return "\n".join(parts)


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # 按行/bullet 先切，再在行内按中英文句号切
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    for ln in lines:
        parts = _SENT_END.split(ln)
        out.extend(p.strip() for p in parts if p and p.strip())
    return out


def _trim_to_chars(sentences: list[str], max_chars: int) -> str:
    out: list[str] = []
    used = 0
    for idx, s in enumerate(sentences):
        # 首行不计换行；其余 +1
        add = len(s) + (0 if idx == 0 else 1)
        if used + add > max_chars:
            remain = max_chars - used
            if remain >= 8:
                out.append(s[:remain].rstrip() + "…")
            break
        out.append(s)
        used += add
    return "\n".join(out)


def build_memory_block(max_chars: int = DEFAULT_MAX_CHARS, data_dir: Path | None = None, freshness_days: int | None = None) -> str:
    """同步入口：拉全量本地 OpenViking 记忆 → 合并去重 → 压缩到 ≤max_chars 字。

    返回空字符串 = 没有任何 memory 或出错（绝不抛）
    """
    try:
        records = _scan_memories(data_dir or DEFAULT_DATA_DIR, freshness_days)
        if not records:
            return ""
        grouped = _group_by_type(records)
        merged = _merge_groups(grouped)
        if len(merged) <= max_chars:
            return merged
        # 字符级截断：先按行/句切，再按 char 上限拼，超出加省略号
        return _char_trim(merged, max_chars)
    except Exception as e:
        print(f"[memory_recall] build error: {e!r}", flush=True)
        return ""


def _char_trim(text: str, max_chars: int) -> str:
    """按行切，优先按整行塞入；最后一行做字符级截断（保证 ≤ max_chars）。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    used = 0
    for ln in lines:
        sep = 1 if out else 0  # 拼接换行
        # 整行能塞下（不含 "…" 截断），直接放
        if used + sep + len(ln) <= max_chars:
            out.append(ln)
            used += sep + len(ln)
            continue
        # 整行放不下，字符级截断（保留 ≤ max_chars - 1 字符给 "…"）
        remain = max_chars - used - sep
        if remain >= 2:
            out.append(ln[: max(0, remain - 1)].rstrip() + "…")
        return "\n".join(out)
    return "\n".join(out)


# ── 调试/测试便利函数 ───────────────────────────────────────
def stats(config: RecallConfig | None = None) -> dict[str, Any]:
    """统计：每个 type 多少条 bullet，总字长，方便评估 800 字上限是否合理。"""
    cfg = config or RecallConfig()
    records = _scan_memories(cfg.data_dir, cfg.freshness_days)
    grouped = _group_by_type(records)
    return {
        "total_files": len(records),
        "by_type": {t: len(bullets) for t, bullets in grouped.items()},
        "total_bullets": sum(len(b) for b in grouped.values()),
        "merged_chars": len(_merge_groups(grouped)),
    }


if __name__ == "__main__":
    s = stats()
    print("=== memory stats ===")
    print(f"total files: {s['total_files']}")
    print(f"total bullets: {s['total_bullets']}")
    print(f"by type: {s['by_type']}")
    print(f"merged chars: {s['merged_chars']}")
    print()
    block = build_memory_block()
    print(f"=== compressed block ({len(block)} chars) ===")
    print(block if block else "(empty)")
