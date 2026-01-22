#!/usr/bin/env python3
"""
检查镜像源配置脚本
用于验证 pip 和 HuggingFace 镜像源是否正确配置
"""

import os
import subprocess
from pathlib import Path

def check_pip_mirror():
    """检查 pip 镜像源配置"""
    print("=" * 60)
    print("📦 pip 镜像源配置检查")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["pip", "config", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "index-url" in result.stdout:
            for line in result.stdout.split("\n"):
                if "index-url" in line:
                    print(f"✅ pip 镜像源: {line.split('=')[-1].strip()}")
                    break
        else:
            print("⚠️  pip 镜像源: 未配置（使用官方源）")
            print("   提示: 运行 bash scripts/setup_mirrors.sh 配置镜像源")
    except Exception as e:
        print(f"❌ 无法检查 pip 配置: {e}")
    
    print()

def check_huggingface_mirror():
    """检查 HuggingFace 镜像源配置"""
    print("=" * 60)
    print("🤗 HuggingFace 镜像源配置检查")
    print("=" * 60)
    
    # 检查 .env 文件
    env_file = Path(".env")
    found_in_env = False
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("HF_ENDPOINT="):
                    endpoint = line.split("=", 1)[1].strip()
                    print(f"✅ .env 文件中的 HF_ENDPOINT: {endpoint}")
                    found_in_env = True
                    break
        if not found_in_env:
            print("⚠️  .env 文件中未找到 HF_ENDPOINT")
    else:
        print("⚠️  .env 文件不存在")
    
    # 检查环境变量
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        print(f"✅ 环境变量 HF_ENDPOINT: {hf_endpoint}")
    else:
        print("⚠️  环境变量 HF_ENDPOINT: 未设置")
    
    # 尝试导入并检查 huggingface_hub
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # 检查是否使用了镜像
        if hasattr(api, 'endpoint'):
            print(f"✅ huggingface_hub 端点: {api.endpoint}")
        else:
            print("ℹ️  无法获取 huggingface_hub 端点信息")
    except ImportError:
        print("⚠️  huggingface_hub 未安装")
    except Exception as e:
        print(f"ℹ️  无法检查 huggingface_hub: {e}")
    
    print()

def check_python_dotenv():
    """检查 python-dotenv 是否安装"""
    print("=" * 60)
    print("📋 依赖检查")
    print("=" * 60)
    
    try:
        import dotenv
        print("✅ python-dotenv: 已安装")
    except ImportError:
        print("❌ python-dotenv: 未安装")
        print("   提示: pip install python-dotenv")
    
    print()

def main():
    """主函数"""
    print("\n🔍 镜像源配置检查工具\n")
    
    # 切换到项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    check_pip_mirror()
    check_huggingface_mirror()
    check_python_dotenv()
    
    print("=" * 60)
    print("📝 配置建议")
    print("=" * 60)
    print("1. 如果 pip 镜像未配置，运行: bash scripts/setup_mirrors.sh")
    print("2. 如果 HuggingFace 镜像未配置，在 .env 文件中添加:")
    print("   HF_ENDPOINT=https://hf-mirror.com")
    print("3. 确保在运行脚本前加载 .env 文件（使用 python-dotenv）")
    print()

if __name__ == "__main__":
    main()
