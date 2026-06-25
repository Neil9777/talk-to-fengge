# Talk to 峰哥

**实时语音对话 + 音色克隆 + 人格注入，端到端延迟 < 1.5 秒。**

和 B 站百万粉丝博主「峰哥亡命天涯」的 AI 分身实时语音聊天。不是文字转语音——是真的像打电话一样聊天，而且声音和性格都是峰哥的。

> [Demo 视频（5.6 万+ 围观）](https://x.com/leaf_sanren/status/2069342335268507976)

---

## 这个项目有什么不同？

市面上有很多语音克隆项目，也有很多实时语音对话项目。但它们通常是割裂的：

- **能实时对话的**（如 GPT-4o Voice）→ 不支持自定义音色克隆
- **能克隆音色的**（如 Bark、XTTS）→ 只能文本转语音，不能实时对话

Talk to 峰哥把这两件事合在了一起：**用克隆的声音进行实时语音对话**，同时注入完整的人格特征（说话风格、口头禅、思维方式），端到端延迟压到 1.5 秒以内。

## 技术栈

```
用户说话 → STT（Cartesia ink-whisper）→ LLM（MiniMax-Text-01）→ TTS（VoxCPM 音色克隆）→ 用户听到回复
                                            ↑
                                    人格注入 + 记忆召回
                                    （OpenViking 可选）
```

| 模块 | 方案 | 说明 |
|------|------|------|
| 实时音视频 | [LiveKit](https://livekit.io/) | WebRTC 框架，处理音频流转发 |
| 语音识别 (STT) | [Cartesia ink-whisper](https://www.cartesia.ai/ink/) | 免费层可用，中文识别效果好 |
| 大语言模型 (LLM) | [MiniMax-Text-01](https://www.minimaxi.com/) | 响应速度最快，TTFB 极低 |
| 语音合成 (TTS) | [VoxCPM](https://github.com/openbmb/VoxCPM) | 开源音色克隆模型，需 GPU |
| 人格系统 | 自研 persona 模块 | 峰哥的说话风格、口头禅、思维方式 |
| 记忆系统 | [OpenViking](https://github.com/nicepkg/openviking)（可选） | 对话记忆沉淀与召回 |

### 备选方案

代码支持多种 STT / LLM / TTS 组合，通过 `.env.local` 切换：

- **STT**：Cartesia（推荐）/ Deepgram / Gemini
- **LLM**：MiniMax（推荐）/ DeepSeek / Gemini
- **TTS**：VoxCPM（推荐，需 GPU）/ Cartesia Sonic（云端，需 Pro 订阅克隆音色）/ MiniMax TTS

## 快速开始

### 最简单的方式：让 AI Agent 帮你配

```bash
git clone https://github.com/YeJe-cpu/talk-to-fengge.git
cd talk-to-fengge
```

然后把这个项目扔给你的 AI 编程助手：

- **[Claude Code](https://claude.ai/code)**：`claude` 打开项目，告诉它"帮我配置并启动这个项目"
- **[Cursor](https://cursor.com/)**：用 Cursor 打开项目，在 Chat 里说"帮我配置并启动"
- **[Codex](https://openai.com/codex)**：类似流程

Agent 会读取 `.env.example`，引导你填入 API key，安装依赖，启动服务。

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
- **MiniMax API Key**（LLM，[注册](https://www.minimaxi.com/)）
- **TTS 方案**（三选一，见下方说明）

#### 4. TTS 音色克隆方案

**方案 A：VoxCPM（推荐，开源免费但需 GPU）**

需要一台有 GPU 的服务器（推荐 RunPod L4 或本地 GPU）：

```bash
# 在 GPU 服务器上
bash runpod_setup.sh  # 安装依赖 + 下载模型（首次约 10 分钟）
```

如果你本机有 GPU（NVIDIA，显存 >= 8GB），也可以本地跑。

**方案 B：Cartesia Sonic（云端，无需 GPU）**

需要 Cartesia Pro 订阅（$5/月）才能克隆音色。在 `.env.local` 设置 `TTS_PROVIDER=cartesia`。

**方案 C：MiniMax TTS（云端，无需 GPU）**

在 `.env.local` 设置 `TTS_PROVIDER=minimax`。

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
┌─────────────┐          ┌──────────────────┐
│  Agent Worker │          │  Web Server       │
│  (Python)     │          │  (端口 8766)       │
│               │          │  提供前端页面       │
│  STT ──→ LLM │          └──────────────────┘
│    ↑     │    │
│    │     ▼    │
│  音频   TTS   │
│  输入   输出   │
│         │    │
│    人格注入    │
│    记忆召回    │
└─────────────┘
       │
       ▼ (可选)
┌─────────────┐     ┌─────────────┐
│  OpenViking   │     │  VoxCPM GPU  │
│  记忆服务      │     │  音色克隆服务  │
└─────────────┘     └─────────────┘
```

## 已知不足 & 后续方向

这个项目目前是个人开发阶段的产物，有很多值得改进的地方：

- **部署门槛高**：VoxCPM 需要 GPU 服务器，LiveKit / OpenViking 需要本地安装，没有一键部署方案
- **记忆系统初级**：OpenViking 的记忆召回效果有限，对话深度受影响
- **网络依赖**：部分组件需要 VPN 才能访问（Cartesia、MiniMax API）
- **单人格**：目前只内置了峰哥人格，还没有通用的"克隆任意人"流程
- **前端简陋**：纯 HTML 单页，没有移动端适配

### 后续想做的

- [ ] 一键部署脚本 / Docker Compose
- [ ] 更丰富的记忆源（对话历史自动沉淀）
- [ ] 数字人 / 虚拟形象接入
- [ ] 国产低延迟模型替代方案（去 VPN 依赖）
- [ ] 通用人格克隆流程（提供素材 → 自动生成 persona）

**欢迎大家在 Issues 里告诉我你最想要哪个功能。**

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
├── web/                 # 前端
│   └── index.html       # 单页应用
├── docs/                # 峰哥人格素材
│   ├── persona-speech-habits.md
│   ├── persona-emotion-patterns.md
│   └── persona-fewshot-examples.md
├── tests/               # 测试
├── .env.example         # 配置模板
└── Talk-to-Me-V3.6.command  # macOS 一键启动
```

## Star / Fork / PR

如果你觉得这个项目有意思，请给个 Star。

想要贡献代码或人格模板？欢迎 Fork + PR。

有问题或建议？开 Issue 聊。

## 作者

**叶晖宇 (Leaf)**

- [作品集](https://www.uncleleaf.cc/)
- [X / Twitter](https://x.com/leaf_sanren)

## License

MIT

---

# Talk to Fengge (English)

**Real-time voice conversation + voice cloning + persona injection. End-to-end latency < 1.5s.**

Have a real-time voice chat with the AI clone of Fengge (峰哥亡命天涯), a Chinese content creator with 1M+ followers on Bilibili. It's not text-to-speech — it's like a real phone call, with his cloned voice and personality.

> [Demo video (56K+ views)](https://x.com/leaf_sanren/status/2069342335268507976)

## What makes this different?

Most voice cloning projects only do text-to-speech. Most real-time voice AI (like GPT-4o Voice) doesn't support custom voice cloning. This project combines both: **real-time conversation with a cloned voice and injected personality**, with end-to-end latency under 1.5 seconds.

## Tech Stack

| Module | Solution | Notes |
|--------|----------|-------|
| Real-time audio | [LiveKit](https://livekit.io/) | WebRTC framework |
| STT | [Cartesia ink-whisper](https://www.cartesia.ai/ink/) | Free tier available |
| LLM | [MiniMax-Text-01](https://www.minimaxi.com/) | Fastest response time |
| TTS | [VoxCPM](https://github.com/openbmb/VoxCPM) | Open-source voice cloning, needs GPU |
| Persona | Custom persona module | Speech patterns, catchphrases, thinking style |
| Memory | [OpenViking](https://github.com/nicepkg/openviking) (optional) | Conversation memory |

## Quick Start

```bash
git clone https://github.com/YeJe-cpu/talk-to-fengge.git
cd talk-to-fengge
```

**Easiest way**: Open this project in [Claude Code](https://claude.ai/code), [Cursor](https://cursor.com/), or [Codex](https://openai.com/codex) and ask the AI to help you set it up.

**Manual setup**: See the Chinese README above for detailed steps (the code and config are all in English/bilingual).

## Author

**Yehuiyu (Leaf)** — [Portfolio](https://www.uncleleaf.cc/) | [X](https://x.com/leaf_sanren)

## License

MIT
