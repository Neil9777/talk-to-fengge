#!/bin/bash
# Talk to Me V3.6 — 双击启动（峰哥网红克隆版，VoxCPM2 声音克隆）
# 与其他 Talk-to-Me 版本互斥：启动前会杀掉所有 Talk-to-Me 服务

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PID_DIR="/tmp/talk-to-me-pids"
VERSION="V3.6"

mkdir -p "$PID_DIR"

echo "════════════════════════════════════════════"
echo "  Talk to Me $VERSION 启动中..."
echo "════════════════════════════════════════════"
echo ""

# ── RunPod 地址配置 ──────────────────────────────
# ⚠️ GPU 迁移后只需更新 runpod_config.env（同目录），不需要改这个文件
CONFIG_FILE="$PROJECT_DIR/runpod_config.env"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "  ❌ 缺少 runpod_config.env，请在 $PROJECT_DIR 目录创建该文件"
    exit 1
fi

# ── 通用函数 ────────────────────────────────────
kill_from_pidfile() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"
    if [ ! -f "$pidfile" ]; then return; fi
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "  停止 $name (pid=$pid)"
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
}

write_pidfile() { echo "$2" > "$PID_DIR/$1.pid"; }

wait_port() {
    local port="$1" label="$2" max="$3"
    for i in $(seq 1 "$max"); do
        if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    echo "  ❌ $label 端口 $port 未就绪"; return 1
}

# ── 清理所有 Talk-to-Me 服务 ──────────────────
echo "[0] 清理旧服务..."
kill_from_pidfile agent-minimax
kill_from_pidfile agent-deepseek
kill_from_pidfile agent-gemini
kill_from_pidfile agent
kill_from_pidfile web
# LiveKit 和 OpenViking 如果在跑就复用，不杀

# 强杀残留 worker/web 端口
for port in 8081 8082 8083 8766; do
    pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  释放端口 $port"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done

cleanup() {
    echo ""
    echo "正在关闭 $VERSION 服务..."
    kill_from_pidfile agent-minimax
    kill_from_pidfile web
    echo "已关闭。"
}
trap cleanup EXIT INT TERM

# ── [0/5] VoxCPM：自动启动远程服务 + SSH 隧道 ────────────────────
# 两个 SSH 地址，用途不同：
#   RUNPOD_PROXY = 代理 SSH，绑定 Pod ID，Stop→Start 不变，用于远端命令执行（expect）
#   RUNPOD_TCP_*  = 直连 SSH，支持端口转发，用于 -L 隧道。
#   ⚠️ TCP 地址在 pod Stop→Start 后会变化，迁移后需要在此更新！
#   更新方法：RunPod 控制台 → Connect → "SSH over exposed TCP" → 复制新地址
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "[0/5] VoxCPM 服务 + SSH 隧道..."

if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "  ✅ 已就绪（复用现有隧道）"
else
    # ── 步骤 A：用 expect 在远端执行 start.sh ──
    # 首次迁移后 setup.sh 需安装 pip 包 + 下载 4GB 模型，约 5-10 分钟
    echo "  连接 RunPod，启动远端服务..."
    expect -c "
        set timeout 30
        log_user 0
        spawn ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -i $SSH_KEY $RUNPOD_PROXY
        expect {
            -re {[#\\\$] } {}
            timeout { puts \"  [ERR] 无法连接 RunPod（是否已开机？）\"; exit 1 }
        }
        log_user 1
        send \"bash /workspace/start.sh 2>&1\r\"
        set timeout 700
        expect {
            -re {\\[start\\] pid=\\d+} { expect -re {[#\\\$] } }
            timeout { puts \"  [ERR] 远端启动超时\"; exit 1 }
        }
        log_user 0
        send \"exit\r\"
        expect eof
    " 2>/dev/null
    if [ $? -ne 0 ]; then
        echo ""
        echo "  ❌ 远端启动失败，请检查 RunPod 控制台"
        echo ""
        echo "按回车键退出..."
        read -r
        exit 1
    fi

    # ── 步骤 B：建立本地 SSH 隧道（走直连 TCP，代理 SSH 不支持端口转发）──
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "$SSH_KEY" \
        -p "$RUNPOD_TCP_PORT" -f -N -L 8000:localhost:8000 \
        "root@$RUNPOD_TCP_HOST" 2>/dev/null || true

    # ── 步骤 C：轮询等模型加载就绪（torch.compile 首次热身约 110s）──
    echo -n "  等待模型加载（首次启动约 2 分钟）"
    READY=false
    for i in $(seq 1 40); do
        sleep 5
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            READY=true
            break
        fi
        echo -n "."
    done
    echo ""

    if ! $READY; then
        echo ""
        echo "  ❌ 200s 内未就绪，请查看远端日志：/tmp/voxcpm.log"
        echo ""
        echo "按回车键退出..."
        read -r
        exit 1
    fi
    echo "  ✅ 就绪"
fi


# ── [1/5] LiveKit ─────────────────────────────
echo "[1/4] LiveKit..."
if lsof -nP -iTCP:7880 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  ✅ 已在运行"
else
    nohup livekit-server --dev --node-ip=127.0.0.1 > /tmp/livekit-v3.6.log 2>&1 < /dev/null &
    write_pidfile livekit "$!"
    wait_port 7880 "LiveKit" 15 || exit 1
    echo "  ✅ 已启动"
fi

# ── [2/4] OpenViking ──────────────────────────
echo "[2/5] OpenViking..."
if lsof -nP -iTCP:1933 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  ✅ 已在运行"
else
    O="${OPENVIKING_SERVER:-openviking-server}"
    C="${OPENVIKING_CONF:-$PROJECT_DIR/openviking.conf}"
    if command -v "$O" >/dev/null 2>&1 && [ -f "$C" ]; then
        set -a; source "$PROJECT_DIR/.env.local"; set +a
        export GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
        nohup "$O" --config "$C" > /tmp/openviking-v3.6.log 2>&1 < /dev/null &
        write_pidfile openviking "$!"
        wait_port 1933 "OpenViking" 20 || exit 1
        echo "  ✅ 已启动"
    else
        echo "  ⚠ 跳过（文件不存在）"
    fi
fi

# ── [3/4] Agent: MiniMax ──────────────────────
echo "[3/5] Agent — VoxCPM2 ($VERSION)..."
cd "$PROJECT_DIR"
set -a; source .env.local; set +a
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

LLM_PROVIDER=minimax LIVEKIT_WORKER_PORT=8081 \
    nohup .venv/bin/python -u -m worker.main start \
    > "/tmp/agent-${VERSION}-minimax.log" 2>&1 < /dev/null &
AGENT_PID=$!
write_pidfile agent-minimax "$AGENT_PID"
sleep 5
if kill -0 "$AGENT_PID" 2>/dev/null; then
    echo "  ✅ MiniMax (pid=$AGENT_PID)"
else
    echo "  ❌ 启动失败:"
    tail -10 "/tmp/agent-${VERSION}-minimax.log" 2>/dev/null || true
    echo ""
    echo "按回车键退出..."
    read -r
    exit 1
fi

# ── [4/4] 前端 ────────────────────────────────
echo "[4/5] 前端..."
nohup .venv/bin/python -m worker.web_server > "/tmp/web-${VERSION}.log" 2>&1 < /dev/null &
WEB_PID=$!
write_pidfile web "$WEB_PID"
sleep 2
if kill -0 "$WEB_PID" 2>/dev/null; then
    echo "  ✅ http://127.0.0.1:8766"
else
    echo "  ❌ 前端启动失败"
    tail -5 "/tmp/web-${VERSION}.log" 2>/dev/null || true
fi

echo ""
echo "════════════════════════════════════════════"
echo "  Talk to Me $VERSION 已就绪"
echo "  打开 http://127.0.0.1:8766"
echo "  TTS: VoxCPM2 声音克隆（RunPod L4）"
echo "════════════════════════════════════════════"
echo ""
echo "日志: /tmp/agent-${VERSION}-minimax.log"
echo ""
echo "按回车键关闭服务并退出..."
read -r
