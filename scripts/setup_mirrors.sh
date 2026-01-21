#!/bin/bash
# 设置镜像源配置脚本
# 支持 pip 和 HuggingFace 镜像源配置

set -e

echo "=========================================="
echo "镜像源配置工具"
echo "=========================================="
echo ""

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    OS="unknown"
fi

# 创建 pip 配置目录
PIP_CONFIG_DIR="$HOME/.pip"
mkdir -p "$PIP_CONFIG_DIR"

# 设置 pip 镜像源
setup_pip_mirror() {
    echo "配置 pip 镜像源..."
    
    # 常用的国内镜像源
    echo "可用的 pip 镜像源："
    echo "  1) 清华大学 (推荐)"
    echo "  2) 阿里云"
    echo "  3) 中科大"
    echo "  4) 豆瓣"
    echo "  5) 官方源 (PyPI)"
    echo ""
    read -p "请选择镜像源 [1-5] (默认: 1): " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
            PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
            MIRROR_NAME="清华大学"
            ;;
        2)
            PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
            PIP_TRUSTED_HOST="mirrors.aliyun.com"
            MIRROR_NAME="阿里云"
            ;;
        3)
            PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple/"
            PIP_TRUSTED_HOST="pypi.mirrors.ustc.edu.cn"
            MIRROR_NAME="中科大"
            ;;
        4)
            PIP_INDEX_URL="https://pypi.douban.com/simple/"
            PIP_TRUSTED_HOST="pypi.douban.com"
            MIRROR_NAME="豆瓣"
            ;;
        5)
            PIP_INDEX_URL="https://pypi.org/simple"
            PIP_TRUSTED_HOST=""
            MIRROR_NAME="官方源"
            ;;
        *)
            echo "无效选择，使用默认：清华大学"
            PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
            PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
            MIRROR_NAME="清华大学"
            ;;
    esac
    
    # 创建 pip.conf 文件
    PIP_CONF="$PIP_CONFIG_DIR/pip.conf"
    if [[ "$OS" == "macos" ]]; then
        PIP_CONF="$PIP_CONFIG_DIR/pip.conf"
    fi
    
    cat > "$PIP_CONF" << EOF
[global]
index-url = $PIP_INDEX_URL
EOF

    if [[ -n "$PIP_TRUSTED_HOST" ]]; then
        echo "trusted-host = $PIP_TRUSTED_HOST" >> "$PIP_CONF"
    fi
    
    echo "✓ pip 镜像源已设置为: $MIRROR_NAME"
    echo "  配置文件: $PIP_CONF"
    echo ""
}

# 设置 HuggingFace 镜像源
setup_huggingface_mirror() {
    echo "配置 HuggingFace 镜像源..."
    
    echo "可用的 HuggingFace 镜像源："
    echo "  1) HF-Mirror (推荐，国内)"
    echo "  2) 不使用镜像（直接连接 HuggingFace）"
    echo ""
    read -p "请选择镜像源 [1-2] (默认: 1): " choice
    choice=${choice:-1}
    
    case $choice in
        1)
            HF_ENDPOINT="https://hf-mirror.com"
            MIRROR_NAME="HF-Mirror"
            ;;
        2)
            HF_ENDPOINT="https://huggingface.co"
            MIRROR_NAME="官方源"
            ;;
        *)
            HF_ENDPOINT="https://hf-mirror.com"
            MIRROR_NAME="HF-Mirror"
            ;;
    esac
    
    # 检查是否存在 .env 文件
    ENV_FILE=".env"
    ENV_EXAMPLE="env.example"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "提示: .env 文件不存在，将使用 env.example 作为模板"
        if [[ -f "$ENV_EXAMPLE" ]]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            echo "✓ 已从 env.example 创建 .env 文件"
        fi
    fi
    
    # 更新或添加 HF_ENDPOINT 到 .env 文件
    if [[ -f "$ENV_FILE" ]]; then
        if grep -q "^HF_ENDPOINT=" "$ENV_FILE"; then
            # 如果存在，更新它
            if [[ "$OS" == "macos" ]]; then
                sed -i '' "s|^HF_ENDPOINT=.*|HF_ENDPOINT=$HF_ENDPOINT|" "$ENV_FILE"
            else
                sed -i "s|^HF_ENDPOINT=.*|HF_ENDPOINT=$HF_ENDPOINT|" "$ENV_FILE"
            fi
        else
            # 如果不存在，添加到文件末尾
            echo "" >> "$ENV_FILE"
            echo "# HuggingFace 镜像源配置" >> "$ENV_FILE"
            echo "HF_ENDPOINT=$HF_ENDPOINT" >> "$ENV_FILE"
        fi
        echo "✓ HuggingFace 镜像源已设置: $MIRROR_NAME"
        echo "  已更新 .env 文件: HF_ENDPOINT=$HF_ENDPOINT"
    else
        echo "⚠ 警告: .env 文件不存在，请手动添加："
        echo "  HF_ENDPOINT=$HF_ENDPOINT"
    fi
    
    # 设置环境变量（当前会话）
    export HF_ENDPOINT="$HF_ENDPOINT"
    echo ""
}

# 显示当前配置
show_current_config() {
    echo "当前配置："
    echo "----------------------------------------"
    
    # pip 配置
    PIP_CONF="$PIP_CONFIG_DIR/pip.conf"
    if [[ -f "$PIP_CONF" ]]; then
        echo "pip 镜像源:"
        grep "index-url" "$PIP_CONF" || echo "  未配置"
    else
        echo "pip 镜像源: 未配置"
    fi
    
    # HuggingFace 配置
    ENV_FILE=".env"
    if [[ -f "$ENV_FILE" ]]; then
        if grep -q "^HF_ENDPOINT=" "$ENV_FILE"; then
            HF_ENDPOINT=$(grep "^HF_ENDPOINT=" "$ENV_FILE" | cut -d'=' -f2)
            echo "HuggingFace 镜像源: $HF_ENDPOINT"
        else
            echo "HuggingFace 镜像源: 未配置"
        fi
    else
        echo "HuggingFace 镜像源: 未配置（.env 文件不存在）"
    fi
    
    echo "----------------------------------------"
    echo ""
}

# 主菜单
main() {
    cd "$(dirname "$0")/.." || exit 1
    
    echo "选择操作："
    echo "  1) 配置 pip 镜像源"
    echo "  2) 配置 HuggingFace 镜像源"
    echo "  3) 配置所有镜像源（推荐）"
    echo "  4) 查看当前配置"
    echo "  5) 退出"
    echo ""
    read -p "请选择 [1-5] (默认: 3): " action
    action=${action:-3}
    
    case $action in
        1)
            setup_pip_mirror
            ;;
        2)
            setup_huggingface_mirror
            ;;
        3)
            setup_pip_mirror
            setup_huggingface_mirror
            ;;
        4)
            show_current_config
            ;;
        5)
            echo "退出"
            exit 0
            ;;
        *)
            echo "无效选择"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "✓ 配置完成！"
    echo "=========================================="
    echo ""
    echo "提示："
    echo "  1. pip 镜像源配置已生效，下次安装包时将自动使用"
    echo "  2. HuggingFace 镜像源需要在项目启动前设置环境变量"
    echo "     可以运行: export HF_ENDPOINT=\$(grep HF_ENDPOINT .env | cut -d'=' -f2)"
    echo "     或者在代码启动前加载 .env 文件（使用 python-dotenv）"
    echo ""
}

main
