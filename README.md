# Talk to 峰哥

**克隆任何人的声音和性格，实时语音对话，工程延迟 < 1 秒。**

这个项目的起点是「Talk to Me」——我想做一个能跟自己对话的 AI 分身。后来我克隆了 B 站百万粉博主「峰哥亡命天涯」的声音和性格，发了个 demo，五六万人围观。

**但核心不是峰哥。** 换成任何人——你自己、你的偶像、你想克隆的任何声音——只要有语音素材和人格描述，这套架构都能跑。峰哥只是第一个完整跑通的例子。

> [Demo 视频（5.6 万+ 围观）](https://x.com/leaf_sanren/status/2069342335268507976)

[English Version](#english)

---

## 这个项目有什么不同？

市面上有很多语音克隆项目，也有很多实时语音对话项目。但它们通常是割裂的：

- **能实时对话的**（如 GPT-4o Voice）→ 不支持自定义音色克隆
- **能克隆音色的**（如 Bark、XTTS）→ 只能文本转语音，不能实时对话

这个项目把三件事合在了一起：

1. **音色克隆**——用 15-45 秒的语音素材克隆任何人的声音
2. **人格注入**——说话风格、口头禅、思维方式，不只是声音像，性格也像
3. **实时对话**——像打电话一样聊天，工程链路延迟压到 1 秒以内（实际体感约 2 秒，含网络）

## 技术栈

```
用户说话 → STT（语音识别）→ LLM（大语言模型）→ TTS（音色克隆合成）→ 用户听到回复
                                  ↑
                          人格注入（system prompt）
                          记忆召回（OpenViking，可选）
```

| 模块 | 默认方案 | 说明 |
|------|---------|------|
| 实时音视频 | [LiveKit](https://livekit.io/) | WebRTC 框架，处理浏览器与 Agent 之间的音频流 |
| 语音识别 (STT) | [Cartesia ink-whisper](https://www.cartesia.ai/ink/)（推荐） | 免费层可用，中文效果好，延迟低 |
| 大语言模型 (LLM) | [MiniMax-Text-01](https://www.minimaxi.com/)（推荐） | 国产模型，TTFB 极低，无需 VPN |
| 语音合成 (TTS) | [VoxCPM](https://github.com/openbmb/VoxCPM)（推荐） | 开源音色克隆，需 GPU（云 GPU 或本地） |
| 人格系统 | 自研 persona 模块 | 说话风格 + 口头禅 + 思维方式，注入 system prompt |
| 记忆系统 | [OpenViking](https://github.com/nicepkg/openviking)（可选） | 对话记忆沉淀与召回 |

### 备选方案

通过 `.env.local` 一行切换：

- **STT**：Cartesia ink-whisper（推荐）/ Deepgram nova-2
- **LLM**：MiniMax（推荐，国产无需 VPN）/ DeepSeek / Gemini
- **TTS**：VoxCPM（推荐，开源）/ [MOSS-TTS](https://github.com/open-moss/moss-tts-nano)（CPU 可跑，兜底方案）/ Cartesia Sonic（云端，需 $5/月 Pro 订阅）/ MiniMax TTS

## 快速开始

### 最简单的方式：让 AI 编程助手帮你配

```bash
git clone https://github.com/YeJe-cpu/talk-to-fengge.git
cd talk-to-fengge
```

然后把这个项目扔给任何 AI 编程助手——[Claude Code](https://claude.ai/code)、[Cursor](https://cursor.com/)、[Codex](https://openai.com/codex)、[Windsurf](https://codeium.com/windsurf)，或者你用的其他工具都行。告诉它「帮我配置并启动这个项目」，Agent 会读取 `.env.example`，引导你填 API key、装依赖、启动服务。

### 手动配置

<details>
<summary>展开手动配置步骤</summary>

#### 1. 前置依赖

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- [LiveKit Server](https://docs.livekit.io/home/self-hosting/local/)

#### 2. 安装

```bash
# 安装 Python 依赖
uv sync

# 安装 LiveKit（macOS）
brew install livekit
```

#### 3. 配置

```bash
cp .env.example .env.local
# 编辑 .env.local，填入你的 API key
```

你至少需要：
- **Cartesia API Key**（STT，[免费注册](https://www.cartesia.ai/)）
- **MiniMax API Key**（LLM，[注册](https://www.minimaxi.com/)，国产，无需 VPN）
- **TTS 方案**（见下方）

#### 4. TTS 音色克隆方案

**方案 A：VoxCPM（推荐，开源免费）**

需要 GPU（NVIDIA，显存 >= 8GB）。可以用云 GPU（如 RunPod L4）或本地 GPU：

```bash
# 在 GPU 机器上
bash runpod_setup.sh  # 安装依赖 + 下载模型（首次约 10 分钟）
```

本机有 GPU 的话，直接本地跑效果最好，延迟最低。

**方案 B：MOSS-TTS（CPU 可跑，兜底方案）**

不需要 GPU，但音色克隆效果和速度不如 VoxCPM。在 `.env.local` 设置 `TTS_PROVIDER=moss`。

**方案 C：Cartesia Sonic（云端，无需 GPU）**

需要 Cartesia Pro 订阅（$5/月）才能克隆音色。在 `.env.local` 设置 `TTS_PROVIDER=cartesia`。

#### 5. 启动

```bash
# 方式一：双击启动脚本（macOS）
./Talk-to-Me-V3.6.command

# 方式二：手动启动各组件
livekit-server --dev --node-ip=127.0.0.1  # 终端 1
LLM_PROVIDER=minimax python -m worker.main start  # 终端 2
python -m worker.web_server  # 终端 3
```

打开 http://127.0.0.1:8766 开始聊天。

</details>

## 架构

```
┌──────────────────────────────────────────────────┐
│                  浏览器前端                         │
│         HTML + LiveKit Web SDK                    │
│         麦克风采集 → WebRTC → 播放回复               │
└─────────────────────┬────────────────────────────┘
                      │ WebRTC (audio)
                      ▼
┌──────────────────────────────────────────────────┐
│              LiveKit Server（本地）                 │
│         房间管理 / 音频流转发                        │
└──────┬──────────────────────────┬────────────────┘
       │                          │
       ▼                          ▼
┌──────────────────┐     ┌──────────────────┐
│  Agent Worker     │     │  Web Server       │
│  (Python)         │     │  (端口 8766)       │
│                   │     │  提供前端页面       │
│  ┌─────────────┐ │     └──────────────────┘
│  │ 人格注入      │ │
│  │ 记忆召回      │ │
│  └──────┬──────┘ │
│         ▼        │
│  音频 → STT      │
│         ↓        │
│        LLM       │
│  (人格+记忆已     │
│   注入 system    │
│   prompt)        │
│         ↓        │
│        TTS → 音频 │
└────────┬─────────┘
         │ (可选外部服务)
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│OpenViking│ │VoxCPM  │
│记忆服务  │ │GPU TTS │
│(可选)   │ │(可选)  │
└────────┘ └────────┘
```

**说明**：人格和记忆在对话开始前就注入到 LLM 的 system prompt 里，不是在 STT→LLM→TTS 管道中间插入的。Web Server 是一个独立的轻量 HTTP 服务，只负责提供前端 HTML 页面。

## 已知不足 & 后续方向

这个项目是个人开发阶段的产物，有不少值得改进的地方：

- **部署环节多**：STT / LLM / TTS 各需要不同的 API key 或服务，没有一键部署方案
- **记忆系统受限**：OpenViking 主要识别用户侧的事件和主体来沉淀记忆。但在峰哥场景下，峰哥话多用户话少，导致可沉淀的记忆有限
- **海外 API 需 VPN**：Cartesia（STT）等海外服务需要 VPN（MiniMax 是国产的，不需要）
- **前端简陋**：目前是星空粒子效果的单页，还没有灵动的数字人 / 虚拟形象

### 后续想做的

- [ ] 灵动数字人 / 虚拟形象接入
- [ ] 全链路国产低延迟模型替代（去 VPN 依赖）
- [ ] 一键部署 + 一键人格蒸馏（提供素材 → 自动生成 persona）
- [ ] Docker Compose 一键启动
- [ ] 更多克隆人格模板

**欢迎在 [Issues](https://github.com/YeJe-cpu/talk-to-fengge/issues) 告诉我你最想要哪个功能。**

## 项目结构

```
talk-to-fengge/
├── worker/              # Python Agent 核心
│   ├── agent.py         # 主 agent：STT → LLM → TTS 管道
│   ├── persona.py       # 人格注入模块（峰哥性格 + 说话风格）
│   ├── memory_recall.py # 记忆召回（OpenViking）
│   ├── llm_factory.py   # LLM 多 provider 工厂
│   ├── tts_factory.py   # TTS 多 provider 工厂
│   ├── cartesia_stt.py  # Cartesia STT 实现
│   ├── voxcpm_tts.py    # VoxCPM TTS 实现
│   └── ...
├── web/                 # 前端（星空粒子单页）
│   └── index.html
├── docs/                # 峰哥人格素材
│   ├── persona-speech-habits.md    # 口头禅与语气词
│   ├── persona-emotion-patterns.md # 情感回应模式
│   └── persona-fewshot-examples.md # 对话示例
├── tests/               # 测试（STT / TTS / E2E 延迟）
├── .env.example         # 配置模板（所有可配项）
└── Talk-to-Me-V3.6.command  # macOS 一键启动
```

## Star / Fork / PR

如果你觉得这个项目有意思，请给个 Star。

想贡献代码、新的人格模板、或者改进建议？欢迎 Fork + PR。

有问题？开 [Issue](https://github.com/YeJe-cpu/talk-to-fengge/issues) 聊。

## 致谢

这个项目站在很多优秀开源项目的肩膀上：

- [LiveKit](https://github.com/livekit/livekit) — 实时音视频框架
- [VoxCPM](https://github.com/openbmb/VoxCPM) — 开源音色克隆模型（OpenBMB）
- [MOSS-TTS-Nano](https://github.com/open-moss/moss-tts-nano) — 轻量 TTS（CPU 可跑）
- [OpenViking](https://github.com/nicepkg/openviking) — 本地记忆系统
- [Cartesia](https://www.cartesia.ai/) — ink-whisper STT + Sonic TTS
- [MiniMax](https://www.minimaxi.com/) — 高速国产 LLM
- [DeepSeek](https://www.deepseek.com/) — 备选 LLM

## 作者

**叶晖宇 (Leaf)**

- [作品集](https://www.uncleleaf.cc/)
- [X / Twitter](https://x.com/leaf_sanren)

## License

MIT

---

<a id="english"></a>

# Talk to Fengge (English)

**Clone anyone's voice and personality. Real-time voice conversation. Engineering latency < 1 second.**

This project started as "Talk to Me" — I wanted to build an AI clone I could talk to. Then I cloned the voice and personality of Fengge (峰哥亡命天涯), a Chinese content creator with 1M+ followers on Bilibili, and the demo went viral (56K+ views).

**But it's not about Fengge.** Swap in anyone — yourself, a celebrity, anyone — as long as you have voice samples and a personality description. Fengge is just the first fully working example.

> [Demo video (56K+ views)](https://x.com/leaf_sanren/status/2069342335268507976)

## What makes this different?

Most voice cloning projects only do text-to-speech. Most real-time voice AI (like GPT-4o Voice) doesn't support custom voice cloning. This project combines three things:

1. **Voice cloning** — clone any voice from 15-45 seconds of audio
2. **Persona injection** — speech patterns, catchphrases, thinking style
3. **Real-time conversation** — like a phone call, engineering latency under 1 second (actual experience ~2s with network)

## Tech Stack

| Module | Default | Notes |
|--------|---------|-------|
| Real-time audio | [LiveKit](https://livekit.io/) | WebRTC framework |
| STT | [Cartesia ink-whisper](https://www.cartesia.ai/ink/) | Free tier available, good Chinese support |
| LLM | [MiniMax-Text-01](https://www.minimaxi.com/) | Fastest TTFB, Chinese model (no VPN needed in China) |
| TTS | [VoxCPM](https://github.com/openbmb/VoxCPM) | Open-source voice cloning, needs GPU |
| Persona | Custom module | Speech style + catchphrases + thinking patterns |
| Memory | [OpenViking](https://github.com/nicepkg/openviking) (optional) | Conversation memory |

Alternatives supported: MOSS-TTS (CPU fallback), Cartesia Sonic (cloud TTS), DeepSeek / Gemini (LLM), Deepgram (STT).

## Quick Start

```bash
git clone https://github.com/YeJe-cpu/talk-to-fengge.git
cd talk-to-fengge
```

**Easiest way**: Open this project in any AI coding assistant — [Claude Code](https://claude.ai/code), [Cursor](https://cursor.com/), [Codex](https://openai.com/codex), [Windsurf](https://codeium.com/windsurf), or others — and ask it to help you set up and run the project.

**Manual setup**: See the [Chinese README above](#talk-to-峰哥) for detailed steps. Code and config are bilingual.

## Known Limitations & Roadmap

- Multi-step setup (separate API keys for STT / LLM / TTS)
- Memory system limited in scenarios where the AI talks more than the user
- Some APIs require VPN access from China (Cartesia)
- Frontend is minimal (particle effect, no digital avatar yet)

**Planned**: Digital avatar, all-Chinese model pipeline, one-click deploy + persona distillation, Docker Compose.

## Acknowledgments

Built on: [LiveKit](https://github.com/livekit/livekit), [VoxCPM](https://github.com/openbmb/VoxCPM) (OpenBMB), [MOSS-TTS-Nano](https://github.com/open-moss/moss-tts-nano), [OpenViking](https://github.com/nicepkg/openviking), [Cartesia](https://www.cartesia.ai/), [MiniMax](https://www.minimaxi.com/), [DeepSeek](https://www.deepseek.com/).

## Author

**Yehuiyu (Leaf)** — [Portfolio](https://www.uncleleaf.cc/) | [X](https://x.com/leaf_sanren)

## License

MIT
