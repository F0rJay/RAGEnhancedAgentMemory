#!/usr/bin/env python3
"""
从 HuggingFace 下载模型到本地

使用方法：
    python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --output models/
    
支持 HuggingFace 镜像源（通过 HF_ENDPOINT 环境变量）
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

# 设置 HuggingFace 镜像源（如果配置了）
hf_endpoint = os.getenv("HF_ENDPOINT")
if hf_endpoint:
    os.environ["HF_ENDPOINT"] = hf_endpoint
    try:
        from huggingface_hub import set_endpoint
        set_endpoint(hf_endpoint)
        print(f"🌐 使用 HuggingFace 镜像源: {hf_endpoint}")
    except ImportError:
        pass

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("❌ huggingface_hub 未安装")
    print("   请运行: pip install huggingface_hub")
    sys.exit(1)


def download_model(model_id: str, output_dir: str, resume_download: bool = True):
    """
    下载模型到本地
    
    Args:
        model_id: HuggingFace 模型 ID（如 "Qwen/Qwen2.5-7B-Instruct"）
        output_dir: 输出目录
        resume_download: 是否支持断点续传
    """
    output_path = Path(output_dir) / model_id.replace("/", "--")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 开始下载模型: {model_id}")
    print(f"   输出目录: {output_path}")
    print()
    
    try:
        # 使用 snapshot_download 下载完整模型
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(output_path),
            resume_download=resume_download,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
        )
        
        print()
        print(f"✅ 模型下载完成！")
        print(f"   本地路径: {downloaded_path}")
        print()
        
        # 检查关键文件
        required_files = ["config.json", "tokenizer.json", "model.safetensors"]
        missing_files = []
        for file in required_files:
            # 检查是否有这些文件（可能在子目录中）
            found = False
            for root, dirs, files in os.walk(output_path):
                if file in files:
                    found = True
                    break
            if not found:
                # 检查是否有 .bin 文件（旧格式）
                if file == "model.safetensors":
                    for root, dirs, files in os.walk(output_path):
                        if any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
                            found = True
                            break
                if not found:
                    missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  警告: 以下文件可能缺失: {', '.join(missing_files)}")
            print("   但模型可能仍然可用（文件可能在子目录中）")
        else:
            print("✅ 关键文件检查通过")
        
        return downloaded_path
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="从 HuggingFace 下载模型到本地")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace 模型 ID（默认: Qwen/Qwen2.5-7B-Instruct）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="输出目录（默认: models）"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传"
    )
    
    args = parser.parse_args()
    
    # 检查输出目录
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 下载模型
    downloaded_path = download_model(
        model_id=args.model,
        output_dir=str(output_dir),
        resume_download=not args.no_resume
    )
    
    if downloaded_path:
        print()
        print("=" * 60)
        print("📝 下一步：更新配置文件")
        print("=" * 60)
        print()
        print(f"1. 编辑 .env 文件，设置模型路径：")
        print(f"   VLLM_MODEL={downloaded_path}")
        print()
        print(f"2. 或者使用相对路径：")
        rel_path = Path(downloaded_path).relative_to(project_root)
        print(f"   VLLM_MODEL={rel_path}")
        print()
        print("3. 启动 vLLM 服务：")
        print(f"   python scripts/launch_vllm_server.py --model {downloaded_path} --port 8000")
        print()
        sys.exit(0)
    else:
        print("❌ 下载失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
