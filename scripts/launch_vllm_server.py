#!/usr/bin/env python3
"""
vLLM 服务启动脚本（可选）

这个脚本可以帮助用户自动启动 vLLM 服务。
注意：推荐在生产环境中让用户手动启动 vLLM 服务（常驻），而不是每次插件启动都重启。

使用方法：
    python scripts/launch_vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

特性：
- 自动设置环境变量避免 duplicate template name 错误
- 默认禁用 CUDA graph（使用 --enforce-eager），确保系统稳定性
- 前缀缓存（Prefix Caching）：默认启用（可选，提升重复提示的性能）
- 健康检查和错误提示

性能优化：
- CUDA graph：默认禁用（使用 --enforce-eager），避免 duplicate template name 错误
- 前缀缓存（Prefix Caching）：默认启用（可选，提升重复提示的性能）

如需启用 CUDA graph 优化，可使用 --no-enforce-eager 参数启动服务（可能触发 duplicate template name 错误）。
"""

import subprocess
import time
import sys
import signal
import os
import threading
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌ requests 库未安装，请运行: pip install requests")
    sys.exit(1)

try:
    from src.config import get_settings
except ImportError:
    # 如果无法导入配置，使用默认值
    class DefaultSettings:
        vllm_model_path = None
        vllm_model = None
        vllm_gpu_memory_utilization = 0.4
        vllm_max_model_len = 1024  # 默认 1024
        hf_endpoint = None
    
    def get_settings():
        return DefaultSettings()


def check_vllm_installed() -> bool:
    """检查 vLLM 是否已安装"""
    # 方法1: 尝试直接导入（最快）
    try:
        import vllm
        return True
    except ImportError:
        pass
    
    # 方法2: 尝试运行 vllm 命令（可能因为 duplicate template name 失败，但不代表未安装）
    try:
        result = subprocess.run(
            ["python", "-m", "vllm", "--help"],
            capture_output=True,
            timeout=5
        )
        # 即使返回非0，也可能是其他错误（如 duplicate template name），不是未安装
        # 所以我们只检查是否能找到命令
        return True  # 如果能运行命令（即使报错），说明已安装
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    except Exception:
        # 其他异常（如 duplicate template name），说明 vLLM 已安装但有问题
        return True


def find_local_model(model_name: str) -> str:
    """
    查找本地模型路径
    
    优先级：
    1. 如果 model_name 是绝对路径或相对路径，直接返回
    2. 检查 HuggingFace 缓存目录
    3. 检查常见的模型存储位置
    4. 如果都找不到，返回原始 model_name（让 vLLM 处理）
    
    Args:
        model_name: 模型名称或路径
    
    Returns:
        模型路径（如果找到本地模型）或原始 model_name
    """
    from pathlib import Path
    
    # 如果是绝对路径或相对路径，直接返回
    if os.path.isabs(model_name) or model_name.startswith('./') or model_name.startswith('../'):
        if os.path.exists(model_name):
            return model_name
        return model_name  # 即使不存在也返回，让 vLLM 报错
    
    # 检查 HuggingFace 缓存
    hf_cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.environ.get("HF_HOME", ""),
        os.environ.get("TRANSFORMERS_CACHE", ""),
    ]
    
    for cache_dir in hf_cache_dirs:
        if not cache_dir:
            continue
        
        # HuggingFace 缓存路径格式：models--org--model_name
        # 例如：Qwen/Qwen2.5-7B-Instruct -> models--Qwen--Qwen2.5-7B-Instruct
        cache_name = model_name.replace("/", "--")
        model_cache_path = os.path.join(cache_dir, f"models--{cache_name}")
        
        if os.path.exists(model_cache_path):
            # 查找 snapshots 目录下的最新版本
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    # 使用最新的 snapshot
                    latest_snapshot = sorted(snapshots)[-1]
                    model_path = os.path.join(snapshots_dir, latest_snapshot)
                    if os.path.exists(os.path.join(model_path, "config.json")):
                        return model_path
    
    # 检查常见的模型存储位置
    common_paths = [
        os.path.expanduser(f"~/models/{model_name}"),
        os.path.expanduser(f"~/autodl-tmp/models/{model_name}"),
        f"/root/models/{model_name}",
        f"/root/autodl-tmp/models/{model_name}",
        f"./models/{model_name}",
    ]
    
    for path in common_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    
    # 如果都找不到，返回原始 model_name（让 vLLM 处理，可能会下载）
    return model_name


def start_vllm_server(
    model: str = None,
    port: int = 8000,
    gpu_memory_utilization: float = None,
    max_model_len: int = None,
    enforce_eager: bool = True,  # 默认禁用 CUDA graph（避免 duplicate template name 错误）
    enable_prefix_caching: bool = True,  # 默认启用前缀缓存（性能优化）
    **kwargs
) -> subprocess.Popen:
    """
    启动 vLLM 服务
    
    Args:
        model: 模型路径或 HuggingFace ID
        port: 服务端口
        gpu_memory_utilization: GPU 内存使用率
        max_model_len: 最大模型长度
        enforce_eager: 是否禁用 CUDA graph（默认 True，避免 duplicate template name 错误）
                      如果环境稳定，可设为 False 以启用 CUDA graph 提升性能
        enable_prefix_caching: 是否启用前缀缓存（默认 True，启用以提升性能）
        **kwargs: 其他 vLLM 启动参数
    
    Returns:
        subprocess.Popen: 服务进程对象
    """
    settings = get_settings()
    
    # 使用配置或参数
    model = model or settings.vllm_model_path or settings.vllm_model
    
    # 如果没有指定模型，提示用户
    if not model:
        print("❌ 未指定模型")
        print("   请使用 --model 参数指定模型路径或 HuggingFace ID")
        print("   例如：")
        print("     --model /path/to/local/model  # 本地模型路径")
        print("     --model Qwen/Qwen2.5-7B-Instruct  # HuggingFace ID")
        sys.exit(1)
    
    # 尝试查找本地模型
    original_model = model
    local_model = find_local_model(model)
    
    if local_model != original_model:
        print(f"   ✓ 找到本地模型: {local_model}")
        model = local_model
    elif os.path.exists(model) or os.path.isabs(model) or model.startswith('./'):
        print(f"   ✓ 使用指定的模型路径: {model}")
    else:
        print(f"   ⚠️  未找到本地模型，将使用 HuggingFace ID: {original_model}")
        print(f"      如果网络不可达，请：")
        print(f"      1. 下载模型到本地后指定路径")
        print(f"      2. 或配置 HuggingFace 镜像源（HF_ENDPOINT）")
        model = original_model
    gpu_memory_utilization = (
        gpu_memory_utilization 
        if gpu_memory_utilization is not None 
        else getattr(settings, 'vllm_gpu_memory_utilization', 0.7)  # 默认 0.7（70%）
    )
    max_model_len = (
        max_model_len 
        if max_model_len is not None 
        else getattr(settings, 'vllm_max_model_len', 1024)  # 默认 1024
    )
    
    # 检查 vLLM 是否已安装
    if not check_vllm_installed():
        print("❌ vLLM 未安装或无法导入")
        print("   请运行: pip install vllm>=0.6.0")
        print("   注意：vLLM 需要 CUDA >= 11.8 环境")
        print()
        print("   如果已安装但仍显示此错误，可能是导入时遇到问题")
        print("   请尝试：")
        print("   1. 检查 Python 环境是否正确")
        print("   2. 检查 vLLM 是否正确安装: pip show vllm")
        print("   3. 尝试手动导入: python -c 'import vllm'")
        sys.exit(1)
    
    # 设置环境变量，避免 duplicate template name 错误
    # 这些环境变量必须在启动 vLLM 之前设置
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    # 为 Triton 缓存设置唯一目录，避免冲突
    env["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_vllm_{os.getpid()}"
    
    # 配置 HuggingFace 镜像源（如果设置了）
    settings = get_settings()
    if settings.hf_endpoint:
        env["HF_ENDPOINT"] = settings.hf_endpoint
        print(f"   🌐 使用 HuggingFace 镜像源: {settings.hf_endpoint}")
    else:
        print(f"   ⚠️  未配置 HuggingFace 镜像源，如果网络不可达，请设置 HF_ENDPOINT")
        print(f"      例如: export HF_ENDPOINT=https://hf-mirror.com")
    
    # 构建启动命令
    # 使用 vllm serve 命令（推荐）
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ]
    
    # 添加 enforce-eager 参数（如果启用，会禁用 CUDA graph）
    if enforce_eager:
        cmd.append("--enforce-eager")
        print("   ⚠️  使用 --enforce-eager 模式（禁用 CUDA graph，可能影响性能）")
    else:
        print("   ✅ 启用 CUDA graph 优化（提升推理性能）")
    
    # 添加前缀缓存参数（性能优化）
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
        print("   ✅ 启用前缀缓存（Prefix Caching）优化（提升重复提示的性能）")
    
    # 添加其他参数
    if "trust_remote_code" in kwargs and kwargs["trust_remote_code"]:
        cmd.append("--trust-remote-code")
    
    print(f"🚀 正在启动 vLLM 服务...")
    print(f"   模型: {model}")
    print(f"   端口: {port}")
    print(f"   GPU 内存使用率: {gpu_memory_utilization}")
    print(f"   最大模型长度: {max_model_len}")
    print(f"   命令: {' '.join(cmd)}")
    print()
    
    # 启动服务（后台运行，实时显示日志）
    print("⏳ 正在启动 vLLM 服务（首次启动可能需要 1-3 分钟）...")
    print("   正在加载模型到 GPU，请耐心等待...")
    print()
    
    try:
        # 使用实时输出模式，让用户看到启动进度
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,  # 传递环境变量
        )
        
        # 等待服务就绪
        base_url = f"http://localhost:{port}"
        health_url = f"{base_url}/health"
        models_url = f"{base_url}/v1/models"
        
        max_wait = 600  # 最多等待 10 分钟（首次启动可能需要更长时间）
        wait_interval = 3  # 每 3 秒检查一次
        last_log_time = time.time()
        log_buffer = []
        
        print("📋 启动日志（实时显示）：")
        print("-" * 60)
        
        for attempt in range(max_wait // wait_interval):
            # 检查进程是否还在运行
            if process.poll() is not None:
                # 进程已退出，读取剩余输出
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output, end='')
                
                print("-" * 60)
                print(f"❌ vLLM 服务启动失败")
                print(f"   退出码: {process.returncode}")
                print()
                print("💡 可能的原因：")
                print("   1. GPU 内存不足（尝试降低 --gpu-memory-utilization）")
                print("   2. 模型路径错误或模型文件损坏")
                print("   3. CUDA 版本不兼容")
                print("   4. 端口被占用（尝试使用其他端口：--port 8001）")
                sys.exit(1)
            
            # 实时读取并显示日志（非阻塞）
            import select
            import fcntl
            
            # 设置非阻塞模式
            try:
                flags = fcntl.fcntl(process.stdout.fileno(), fcntl.F_GETFL)
                fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                # 读取可用输出
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    line = line.rstrip()
                    if line:
                        print(f"   {line}")
                        log_buffer.append(line)
                        last_log_time = time.time()
            except (IOError, OSError):
                # 没有更多输出，继续等待
                pass
            
            # 尝试连接健康检查端点
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    print("-" * 60)
                    print("✅ vLLM 服务已就绪！")
                    print(f"   服务地址: {base_url}")
                    print(f"   健康检查: {health_url}")
                    print()
                    print("💡 提示：服务将在后台运行，按 Ctrl+C 停止服务")
                    return process
            except requests.exceptions.RequestException:
                pass
            
            # 每 10 秒显示一次进度提示
            elapsed = attempt * wait_interval
            if elapsed > 0 and elapsed % 10 == 0:
                print(f"   ⏳ 仍在启动中... ({elapsed} 秒)")
                if elapsed > 60:
                    print(f"   💡 首次启动通常需要 1-3 分钟，请耐心等待...")
            
            time.sleep(wait_interval)
        
        # 超时
        print(f"❌ vLLM 服务启动超时（{max_wait} 秒）")
        print("   请检查：")
        print("   1. GPU 是否可用")
        print("   2. 模型路径是否正确")
        print("   3. 端口是否被占用")
        process.terminate()
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        if process.poll() is None:
            process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        if process.poll() is None:
            process.terminate()
        sys.exit(1)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="启动 vLLM 服务（OpenAI-compatible API）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径或 HuggingFace ID（默认从配置读取）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口（默认: 8000）"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,  # 默认 70%
        help="GPU 内存使用率（默认 0.7，即 70%）"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,  # 默认 1024
        help="最大模型长度（默认 1024）"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="信任远程代码"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,  # 默认禁用 CUDA graph（避免 duplicate template name 错误）
        help="禁用 CUDA graph（默认启用，避免 duplicate template name 错误）"
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="启用 CUDA graph（可能触发 duplicate template name 错误，但性能更好）"
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="禁用前缀缓存（Prefix Caching）。默认启用前缀缓存以提升性能"
    )
    
    args = parser.parse_args()
    
    # 启动服务
    process = start_vllm_server(
        model=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.disable_prefix_caching,  # 默认启用
        trust_remote_code=args.trust_remote_code,
    )
    
    # 注册信号处理，确保退出时清理
    def signal_handler(sig, frame):
        print("\n⚠️  收到退出信号，正在停止服务...")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 等待进程结束（或用户中断）
    try:
        # 实时输出日志
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    # 等待进程结束
    process.wait()


if __name__ == "__main__":
    main()
