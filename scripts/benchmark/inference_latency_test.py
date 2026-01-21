"""
推理延迟优化基准测试

对比基线推理（transformers）和 vLLM 优化推理的性能差异。
测量指标：
- TTFT (Time to First Token)
- 端到端延迟
- 吞吐量 (Tokens/Second)
- 内存使用

目标：验证 vLLM PagedAttention + Prefix Caching 带来的 25%+ 延迟降低。
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 在导入任何库之前设置 HuggingFace 镜像源（如果配置了）
# 这确保 huggingface_hub 和 transformers 使用正确的镜像
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")  # 加载 .env 文件
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        # 尝试使用 huggingface_hub 的 set_endpoint（如果可用）
        try:
            from huggingface_hub import set_endpoint
            set_endpoint(hf_endpoint)
        except ImportError:
            pass
except Exception:
    pass

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# 测试场景定义
# ============================================================================

def generate_test_prompts() -> List[Dict[str, str]]:
    """
    生成测试提示
    
    包含不同长度的提示，模拟真实使用场景
    """
    return [
        {
            "name": "短提示",
            "prompt": "介绍一下 Python 编程语言的特点。",
            "system_prompt": "你是一个专业的编程助手。",
        },
        {
            "name": "中等提示",
            "prompt": "请详细说明 RAG（检索增强生成）的工作原理，包括向量检索、上下文融合和生成过程。",
            "system_prompt": "你是一个 AI 技术专家，擅长解释复杂的技术概念。",
        },
        {
            "name": "长提示（带上下文）",
            "prompt": "基于以下对话历史，回答用户的问题。\n\n对话历史：\n用户：我喜欢用 Python 编程\n助手：Python 是一个很好的选择，语法简洁易懂。\n用户：我想学习机器学习\n助手：机器学习有很多优秀的库，如 scikit-learn、TensorFlow。\n\n当前问题：推荐一些适合初学者的机器学习资源。",
            "system_prompt": "你是一个专业的编程导师。",
        },
        {
            "name": "重复提示（测试 Prefix Caching）",
            "prompt": "介绍一下 Python 编程语言的特点。",
            "system_prompt": "你是一个专业的编程助手。",
        },
    ]


# ============================================================================
# 基线推理测试
# ============================================================================

def test_baseline_inference(
    test_prompts: List[Dict[str, str]],
    model_path: str,
) -> Dict[str, Any]:
    """
    测试基线推理性能
    
    Args:
        test_prompts: 测试提示列表
        model_path: 模型路径
    
    Returns:
        测试结果
    """
    try:
        from src.inference.baseline_inference import BaselineInference
        import torch
    except ImportError as e:
        logger.error(f"无法导入基线推理模块: {e}")
        return {"error": str(e)}
    
    logger.info("=" * 80)
    logger.info("基线推理测试（transformers）")
    logger.info("=" * 80)
    
    try:
        # 初始化推理引擎
        logger.info("初始化基线推理引擎...")
        start_init = time.time()
        engine = BaselineInference(model_path=model_path)
        init_time = time.time() - start_init
        logger.info(f"初始化耗时: {init_time:.2f} 秒")
        
        # 运行测试
        results = []
        total_ttft = 0.0
        total_latency = 0.0
        total_tokens = 0
        
        for i, test_case in enumerate(test_prompts, 1):
            logger.info(f"\n测试 {i}/{len(test_prompts)}: {test_case['name']}")
            logger.info(f"提示: {test_case['prompt'][:50]}...")
            
            try:
                # 生成
                start_time = time.time()
                generated_text, metrics = engine.generate(
                    prompt=test_case['prompt'],
                    system_prompt=test_case.get('system_prompt'),
                    max_tokens=256,
                )
                elapsed = time.time() - start_time
                
                total_ttft += metrics.get('ttft', 0)
                total_latency += metrics.get('total_time', elapsed)
                total_tokens += metrics.get('tokens_generated', 0)
                
                result = {
                    "name": test_case['name'],
                    "prompt_length": len(test_case['prompt']),
                    "generated_length": len(generated_text),
                    "ttft": metrics.get('ttft', 0),
                    "total_time": metrics.get('total_time', elapsed),
                    "tokens_generated": metrics.get('tokens_generated', 0),
                    "tokens_per_second": metrics.get('tokens_per_second', 0),
                }
                results.append(result)
                
                logger.info(f"  TTFT: {result['ttft']*1000:.2f} ms")
                logger.info(f"  总延迟: {result['total_time']*1000:.2f} ms")
                logger.info(f"  生成 tokens: {result['tokens_generated']}")
                logger.info(f"  吞吐量: {result['tokens_per_second']:.2f} tokens/s")
                
            except Exception as e:
                logger.error(f"测试失败: {e}")
                results.append({
                    "name": test_case['name'],
                    "error": str(e),
                })
        
        # 计算平均值
        n_successful = len([r for r in results if "error" not in r])
        avg_ttft = total_ttft / n_successful if n_successful > 0 else 0
        avg_latency = total_latency / n_successful if n_successful > 0 else 0
        avg_throughput = total_tokens / total_latency if total_latency > 0 else 0
        
        # 清理 GPU 内存（为 vLLM 测试释放空间）
        logger.info("清理基线推理引擎并释放 GPU 内存...")
        try:
            # 先删除模型和tokenizer
            if hasattr(engine, 'model'):
                del engine.model
            if hasattr(engine, 'tokenizer'):
                del engine.tokenizer
            del engine
            # 强制垃圾回收
            import gc
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # 尝试释放所有保留的内存
                torch.cuda.reset_peak_memory_stats()
                logger.info("✓ GPU 内存已清理")
        except Exception as e:
            logger.warning(f"清理 GPU 内存时出现警告: {e}")
        
        return {
            "engine_type": "baseline",
            "init_time": init_time,
            "test_results": results,
            "summary": {
                "avg_ttft": avg_ttft,
                "avg_latency": avg_latency,
                "avg_throughput": avg_throughput,
                "total_tokens": total_tokens,
                "successful_tests": n_successful,
                "total_tests": len(test_prompts),
            },
        }
    
    except Exception as e:
        logger.error(f"基线推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# vLLM 优化推理测试
# ============================================================================

def test_vllm_inference(
    test_prompts: List[Dict[str, str]],
    model_path: str,
) -> Dict[str, Any]:
    """
    测试 vLLM 优化推理性能
    
    Args:
        test_prompts: 测试提示列表
        model_path: 模型路径
    
    Returns:
        测试结果
    """
    try:
        from src.inference.vllm_inference import VLLMInference
    except ImportError as e:
        logger.error(f"无法导入 vLLM 推理模块: {e}")
        return {"error": str(e)}
    
    logger.info("=" * 80)
    logger.info("vLLM 优化推理测试（PagedAttention + Prefix Caching）")
    logger.info("=" * 80)
    
    try:
        # 检查 GPU 内存状态（初始化前）
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free_memory = total_memory - reserved
                logger.info("=" * 80)
                logger.info("GPU 内存状态检查（vLLM 初始化前）")
                logger.info("=" * 80)
                logger.info(f"  GPU 总内存: {total_memory:.2f} GB")
                logger.info(f"  已分配内存: {allocated:.2f} GB")
                logger.info(f"  已保留内存: {reserved:.2f} GB")
                logger.info(f"  可用内存: {free_memory:.2f} GB")
                logger.info("=" * 80)
                if reserved > 1.0:
                    logger.warning(f"⚠️  检测到 GPU 仍有 {reserved:.2f} GB 被保留")
                    logger.warning("   这可能会影响 vLLM 的内存分配")
                if free_memory < 18.0:
                    logger.warning(f"⚠️  可用内存 {free_memory:.2f} GB 可能不足以加载 7B 模型（需要 ~15GB）")
        except Exception as e:
            logger.warning(f"无法检查 GPU 内存状态: {e}")
        
        # 初始化推理引擎
        logger.info("\n初始化 vLLM 推理引擎...")
        start_init = time.time()
        
        # 尝试初始化，显式使用更低的配置以确保内存充足
        # 对于 31GB GPU，模型本身需要 ~14.25GB，需要非常保守的配置
        # 注意：gpu_memory_utilization 是 vLLM 尝试使用的总 GPU 内存比例
        # 但模型权重会先占用 ~14.25GB，剩下的才是 KV cache
        try:
            engine = VLLMInference(
                model_path=model_path,
                # 关键：gpu_memory_utilization 是 vLLM 可使用的总 GPU 内存比例
                # 模型权重会先占用 ~14.25GB，剩下的才是 KV cache
                # 31GB * 0.7 = 21.7GB，模型 14.25GB，剩下 ~7.45GB 给 KV cache
                gpu_memory_utilization=0.7,  # 提高到 0.7，确保有足够空间加载模型和 KV cache
                max_model_len=256,  # 降低最大长度以节省 KV cache
                enable_prefix_caching=True,  # 启用前缀缓存优化
                enforce_eager=False,  # 启用 CUDA graph 优化以提升性能  
            )
        except Exception as e:
            # 如果内存不足，尝试禁用 prefix caching 以节省内存
            if "memory" in str(e).lower() or "cache" in str(e).lower():
                logger.warning("标准配置失败（内存不足），尝试禁用 prefix caching 以节省内存...")
                try:
                    engine = VLLMInference(
                        model_path=model_path,
                        gpu_memory_utilization=0.65,  # 稍微降低，但仍需足够加载模型
                        max_model_len=128,  # 最小化 KV cache
                        enable_prefix_caching=False,  # 禁用前缀缓存以节省内存
                    )
                    logger.info("✓ 已禁用 prefix caching，成功初始化 vLLM")
                except Exception as e2:
                    logger.error(f"即使禁用 prefix caching 后仍然失败: {e2}")
                    raise RuntimeError(
                        "vLLM 初始化失败：GPU 内存不足。"
                        "建议：1) 确保基线测试后 GPU 内存已完全释放；"
                        "2) 降低 gpu_memory_utilization 到 0.3 或更低；"
                        "3) 降低 max_model_len 到 256 或更低"
                    ) from e2
            else:
                raise
        
        init_time = time.time() - start_init
        logger.info(f"初始化耗时: {init_time:.2f} 秒")
        
        # 运行测试
        results = []
        total_ttft = 0.0
        total_latency = 0.0
        total_tokens = 0
        
        for i, test_case in enumerate(test_prompts, 1):
            logger.info(f"\n测试 {i}/{len(test_prompts)}: {test_case['name']}")
            logger.info(f"提示: {test_case['prompt'][:50]}...")
            
            try:
                # 生成
                start_time = time.time()
                generated_text, metrics = engine.generate(
                    prompt=test_case['prompt'],
                    system_prompt=test_case.get('system_prompt'),
                    max_tokens=256,
                )
                elapsed = time.time() - start_time
                
                total_ttft += metrics.get('ttft', 0)
                total_latency += metrics.get('total_time', elapsed)
                total_tokens += metrics.get('tokens_generated', 0)
                
                result = {
                    "name": test_case['name'],
                    "prompt_length": len(test_case['prompt']),
                    "generated_length": len(generated_text),
                    "ttft": metrics.get('ttft', 0),
                    "total_time": metrics.get('total_time', elapsed),
                    "tokens_generated": metrics.get('tokens_generated', 0),
                    "tokens_per_second": metrics.get('tokens_per_second', 0),
                }
                results.append(result)
                
                logger.info(f"  TTFT: {result['ttft']*1000:.2f} ms")
                logger.info(f"  总延迟: {result['total_time']*1000:.2f} ms")
                logger.info(f"  生成 tokens: {result['tokens_generated']}")
                logger.info(f"  吞吐量: {result['tokens_per_second']:.2f} tokens/s")
                
            except Exception as e:
                logger.error(f"测试失败: {e}")
                results.append({
                    "name": test_case['name'],
                    "error": str(e),
                })
        
        # 计算平均值
        n_successful = len([r for r in results if "error" not in r])
        avg_ttft = total_ttft / n_successful if n_successful > 0 else 0
        avg_latency = total_latency / n_successful if n_successful > 0 else 0
        avg_throughput = total_tokens / total_latency if total_latency > 0 else 0
        
        return {
            "engine_type": "vllm",
            "init_time": init_time,
            "test_results": results,
            "summary": {
                "avg_ttft": avg_ttft,
                "avg_latency": avg_latency,
                "avg_throughput": avg_throughput,
                "total_tokens": total_tokens,
                "successful_tests": n_successful,
                "total_tests": len(test_prompts),
            },
        }
    
    except Exception as e:
        logger.error(f"vLLM 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# 并发压力测试
# ============================================================================

def test_baseline_concurrent(
    prompts: List[str],
    model_path: str,
    concurrency: int = 10,
) -> Dict[str, Any]:
    """
    测试基线推理的并发性能（串行处理）
    
    Args:
        prompts: 提示列表
        model_path: 模型路径
        concurrency: 并发数（实际为串行）
    
    Returns:
        测试结果
    """
    try:
        from src.inference.baseline_inference import BaselineInference
    except ImportError as e:
        logger.error(f"无法导入基线推理模块: {e}")
        return {"error": str(e)}
    
    logger.info("=" * 80)
    logger.info(f"基线推理并发测试（串行处理，模拟 {concurrency} 并发）")
    logger.info("=" * 80)
    
    try:
        # 初始化推理引擎
        logger.info("初始化基线推理引擎...")
        engine = BaselineInference(model_path=model_path)
        
        # 串行处理（baseline 不支持真正的并发）
        logger.info(f"\n开始串行处理 {len(prompts)} 个请求...")
        start_time = time.time()
        results = []
        total_tokens = 0
        
        for i, prompt in enumerate(prompts, 1):
            try:
                generated_text, metrics = engine.generate(
                    prompt=prompt,
                    max_tokens=256,
                )
                total_tokens += metrics.get('tokens_generated', 0)
                results.append({
                    "prompt_id": i,
                    "tokens_generated": metrics.get('tokens_generated', 0),
                    "latency": metrics.get('total_time', 0),
                })
                logger.info(f"  完成请求 {i}/{len(prompts)}: {metrics.get('total_time', 0)*1000:.2f} ms")
            except Exception as e:
                logger.error(f"请求 {i} 失败: {e}")
                results.append({
                    "prompt_id": i,
                    "error": str(e),
                })
        
        total_time = time.time() - start_time
        successful = len([r for r in results if "error" not in r])
        
        # 清理基线引擎以释放 GPU 内存
        if engine is not None:
            try:
                del engine.model
                del engine.tokenizer
                del engine
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"清理基线引擎时出现警告: {e}")
        
        return {
            "engine_type": "baseline",
            "concurrency": concurrency,
            "total_requests": len(prompts),
            "successful_requests": successful,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "throughput": total_tokens / total_time if total_time > 0 else 0,
            "avg_latency": sum(r.get('latency', 0) for r in results if "error" not in r) / successful if successful > 0 else 0,
            "results": results,
        }
    
    except Exception as e:
        logger.error(f"基线并发测试失败: {e}")
        return {"error": str(e)}


def test_vllm_concurrent(
    prompts: List[str],
    model_path: str,
    concurrency: int = 10,
) -> Dict[str, Any]:
    """
    测试 vLLM 的并发性能（真正并行处理）
    
    Args:
        prompts: 提示列表
        model_path: 模型路径
        concurrency: 并发数
    
    Returns:
        测试结果
    """
    try:
        from src.inference.vllm_inference import VLLMInference
    except ImportError as e:
        logger.error(f"无法导入 vLLM 推理模块: {e}")
        return {"error": str(e)}
    
    logger.info("=" * 80)
    logger.info(f"vLLM 并发测试（真正并行处理，并发数: {concurrency}）")
    logger.info("=" * 80)
    
    try:
        # 检查可用 GPU 内存（使用 CUDA 实际内存，而不是 PyTorch 统计）
        try:
            import torch
            if torch.cuda.is_available():
                # 获取 CUDA 设备实际的内存使用情况（更准确）
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                actual_free_gb = free_mem / 1024**3
                actual_total_gb = total_mem / 1024**3
                actual_used_gb = (total_mem - free_mem) / 1024**3
                
                # PyTorch 统计（可能不准确）
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                pytorch_free = total_memory - reserved
                
                logger.info(f"初始化前 GPU 内存状态:")
                logger.info(f"  CUDA 实际可用: {actual_free_gb:.2f} GB / {actual_total_gb:.2f} GB")
                logger.info(f"  CUDA 实际已用: {actual_used_gb:.2f} GB")
                logger.info(f"  PyTorch 统计可用: {pytorch_free:.2f} GB (可能不准确)")
                logger.info("")
                
                # 根据 CUDA 实际可用内存自动调整 gpu_memory_utilization
                # 模型需要约 14.25GB，需要确保有足够空间
                target_memory = 0.7 * actual_total_gb
                if actual_free_gb < target_memory:
                    # 降低内存使用率
                    adjusted_util = (actual_free_gb - 1.0) / actual_total_gb  # 保留 1GB 缓冲
                    adjusted_util = max(0.4, min(0.7, adjusted_util))  # 限制在 0.4-0.7 之间
                    logger.warning(f"CUDA 实际可用内存不足，降低 gpu_memory_utilization: 0.7 → {adjusted_util:.2f}")
                    logger.warning(f"  (目标需要 {target_memory:.2f} GB，实际可用 {actual_free_gb:.2f} GB)")
                    gpu_mem_util = adjusted_util
                else:
                    gpu_mem_util = 0.7
                    logger.info(f"✓ 使用标准配置: gpu_memory_utilization={gpu_mem_util}")
        except Exception as e:
            logger.warning(f"无法检查 GPU 内存，使用默认配置: {e}")
            gpu_mem_util = 0.7
        
        # 初始化推理引擎（尝试标准配置，失败则降级）
        logger.info("初始化 vLLM 推理引擎...")
        try:
            engine = VLLMInference(
                model_path=model_path,
                gpu_memory_utilization=gpu_mem_util,
                max_model_len=256,
                enable_prefix_caching=True,
                enforce_eager=False,
            )
        except Exception as e:
            if "memory" in str(e).lower():
                logger.warning("标准配置失败（内存不足），尝试降低配置...")
                engine = VLLMInference(
                    model_path=model_path,
                    gpu_memory_utilization=0.5,  # 降低到 0.5
                    max_model_len=128,  # 降低最大长度
                    enable_prefix_caching=False,  # 禁用前缀缓存
                    enforce_eager=True,  # 启用 enforce_eager 以节省内存
                )
                logger.info("✓ 使用降低的配置成功初始化 vLLM")
            else:
                raise
        
        # 使用 vLLM 的原生并发支持（一次性发送所有请求，vLLM 会自动并行处理）
        logger.info(f"\n开始并发处理 {len(prompts)} 个请求（vLLM 自动并行，并发数: {concurrency}）...")
        start_time = time.time()
        
        try:
            # vLLM 的 generate 方法可以接受列表，会自动并行处理
            # 使用单个 generate_batch 调用来利用 vLLM 的并发能力
            batch_results = engine.generate_batch(
                prompts=prompts,
                max_tokens=256,
            )
            
            # 计算总耗时
            total_time = time.time() - start_time
            
            all_results = []
            total_tokens = 0
            
            # batch_results 是 (generated_text, metrics) 元组的列表
            for i, (generated_text, metrics) in enumerate(batch_results):
                prompt_id = i + 1
                tokens = metrics.get('tokens_generated', 0)
                # 对于批量生成，每个请求的平均延迟
                # generate_batch 返回的 metrics 中的 total_time 已经是平均时间
                latency = metrics.get('total_time', total_time / len(prompts))
                total_tokens += tokens
                all_results.append({
                    "prompt_id": prompt_id,
                    "tokens_generated": tokens,
                    "latency": latency,
                })
                logger.info(f"  完成请求 {prompt_id}/{len(prompts)}: {latency*1000:.2f} ms ({tokens} tokens)")
        
        except Exception as e:
            logger.error(f"并发处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        successful = len([r for r in all_results if "error" not in r])
        
        return {
            "engine_type": "vllm",
            "concurrency": concurrency,
            "total_requests": len(prompts),
            "successful_requests": successful,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "throughput": total_tokens / total_time if total_time > 0 else 0,
            "avg_latency": sum(r.get('latency', 0) for r in all_results if "error" not in r) / successful if successful > 0 else 0,
            "results": all_results,
        }
    
    except Exception as e:
        logger.error(f"vLLM 并发测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_concurrency_test(
    model_path: str,
    concurrency_levels: List[int] = [1, 5, 10, 20],
    num_requests: int = 20,
    test_mode: str = "both",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    运行并发压力测试
    
    Args:
        model_path: 模型路径
        concurrency_levels: 并发级别列表
        num_requests: 总请求数
        test_mode: 测试模式 - "baseline"（仅基线）、"vllm"（仅vLLM）、"both"（两者，默认）
        output_dir: 结果保存目录（用于在不同进程间共享结果）
    
    Returns:
        测试结果
    """
    logger.info("\n" + "=" * 80)
    logger.info("并发压力测试")
    logger.info("=" * 80)
    logger.info(f"总请求数: {num_requests}")
    logger.info(f"并发级别: {concurrency_levels}")
    logger.info(f"测试模式: {test_mode}")
    logger.info("")
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "benchmark_results" / "concurrency"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成测试提示（使用简单提示以加快测试）
    test_prompts = [
        f"请用一句话介绍 Python 编程语言的特点。请求编号: {i}"
        for i in range(1, num_requests + 1)
    ]
    
    all_results = {}
    
    # 如果是仅运行 vLLM 测试，尝试加载之前的基线结果
    if test_mode == "vllm":
        baseline_results_file = output_dir / "baseline_concurrency_results.json"
        if baseline_results_file.exists():
            try:
                with open(baseline_results_file, "r", encoding="utf-8") as f:
                    loaded_baseline = json.load(f)
                    all_results.update(loaded_baseline)
                logger.info(f"✓ 已从 {baseline_results_file} 加载基线测试结果")
            except Exception as e:
                logger.warning(f"加载基线测试结果失败: {e}")
        else:
            logger.warning("⚠️  未找到基线测试结果文件，将无法进行完整对比")
            logger.warning(f"   请先运行: python scripts/benchmark/inference_latency_test.py --test concurrency --concurrency-mode baseline")
    
    # 测试不同并发级别
    for concurrency in concurrency_levels:
        logger.info("\n" + "=" * 80)
        logger.info(f"测试并发数: {concurrency}")
        logger.info("=" * 80)
        
        # 1. Baseline 测试（串行）
        if test_mode in ["both", "baseline"]:
            logger.info("\n--- Baseline (串行处理) ---")
            baseline_result = test_baseline_concurrent(
                prompts=test_prompts[:num_requests],
                model_path=model_path,
                concurrency=concurrency,
            )
            
            if "error" not in baseline_result:
                all_results[f"baseline_c{concurrency}"] = baseline_result
                # 保存基线结果（实时保存，以便后续加载）
                baseline_results_file = output_dir / "baseline_concurrency_results.json"
                try:
                    with open(baseline_results_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {k: v for k, v in all_results.items() if k.startswith("baseline_")},
                            f,
                            indent=2,
                            ensure_ascii=False
                        )
                    logger.info(f"✓ 基线结果已保存到 {baseline_results_file}")
                except Exception as e:
                    logger.warning(f"保存基线结果失败: {e}")
            
            if test_mode == "baseline":
                # 如果只运行基线测试，跳过 vLLM 测试
                continue
            
            # 如果是 both 模式，提示用户可以在新的进程中运行 vLLM
            logger.info("\n" + "=" * 80)
            logger.info("⚠️  重要提示")
            logger.info("=" * 80)
            logger.info("基线测试已完成，但为了避免内存干扰，建议在单独进程中运行 vLLM 测试：")
            logger.info(f"  python scripts/benchmark/inference_latency_test.py --test concurrency --concurrency-mode vllm --concurrency {concurrency}")
            logger.info("")
            logger.info("或者等待清理完成后再继续...")
            logger.info("=" * 80)
            logger.info("")
        
        # 2. vLLM 测试（并行）
        if test_mode in ["both", "vllm"]:
            # 如果是 both 模式且之前运行了基线测试，先清理 GPU 内存
            if test_mode == "both":
                logger.info("\n强制清理 GPU 内存...")
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        # PyTorch 的内存分配器会保留内存块，需要更彻底的清理
                        # 多次清理以确保内存完全释放
                        for i in range(10):
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            import time as time_module
                            time_module.sleep(0.5)
                        
                        # 尝试重置内存统计（虽然不会释放内存，但有助于诊断）
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except Exception:
                            pass
                        
                        # 显示当前内存状态（PyTorch 视角）
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        
                        # 获取 CUDA 设备实际的内存使用情况（更准确）
                        try:
                            free_mem, total_mem = torch.cuda.mem_get_info(0)
                            actual_free = free_mem / 1024**3
                            actual_total = total_mem / 1024**3
                            actual_used = (total_mem - free_mem) / 1024**3
                        except Exception:
                            actual_free = total_memory - reserved
                            actual_total = total_memory
                            actual_used = reserved
                        
                        logger.info(f"清理后 GPU 内存状态（PyTorch 统计）:")
                        logger.info(f"  已分配: {allocated:.2f} GB")
                        logger.info(f"  已保留: {reserved:.2f} GB")
                        logger.info(f"  PyTorch 计算可用: {total_memory - reserved:.2f} GB")
                        logger.info(f"")
                        logger.info(f"清理后 GPU 内存状态（CUDA 实际）:")
                        logger.info(f"  实际已使用: {actual_used:.2f} GB")
                        logger.info(f"  实际可用: {actual_free:.2f} GB / {actual_total:.2f} GB")
                        logger.info(f"")
                        logger.info(f"说明: PyTorch 可能保留内存块以提高分配效率，")
                        logger.info(f"      但实际 GPU 内存使用可能更低。vLLM 会检查实际可用内存。")
                except Exception as e:
                    logger.warning(f"清理 GPU 内存时出现警告: {e}")
            
            # vLLM 测试
        logger.info("\n--- vLLM (并行处理) ---")
        vllm_result = test_vllm_concurrent(
            prompts=test_prompts[:num_requests],
            model_path=model_path,
            concurrency=concurrency,
        )
        
        if "error" not in vllm_result:
            all_results[f"vllm_c{concurrency}"] = vllm_result
            
            # 计算性能提升
            if f"baseline_c{concurrency}" in all_results:
                baseline = all_results[f"baseline_c{concurrency}"]
                speedup = baseline["total_time"] / vllm_result["total_time"] if vllm_result["total_time"] > 0 else 0
                throughput_improvement = (vllm_result["throughput"] / baseline["throughput"] - 1) * 100 if baseline["throughput"] > 0 else 0
                
                logger.info("\n" + "-" * 80)
                logger.info(f"并发数 {concurrency} 性能对比:")
                logger.info(f"  基线总耗时: {baseline['total_time']:.2f} 秒")
                logger.info(f"  vLLM 总耗时: {vllm_result['total_time']:.2f} 秒")
                logger.info(f"  速度提升: {speedup:.2f}x")
                logger.info(f"  基线吞吐量: {baseline['throughput']:.2f} tokens/s")
                logger.info(f"  vLLM 吞吐量: {vllm_result['throughput']:.2f} tokens/s")
                logger.info(f"  吞吐量提升: {throughput_improvement:.1f}%")
                logger.info("-" * 80)
    
    return all_results


# ============================================================================
# 主测试流程
# ============================================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="推理延迟优化基准测试")
    parser.add_argument(
        "--test",
        choices=["baseline", "vllm", "both", "concurrency"],
        default="both",
        help="选择要运行的测试: baseline (仅基线), vllm (仅vLLM), both (两者，默认), concurrency (并发测试)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="并发测试的并发级别列表（默认: 1 5 10 20）"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="并发测试的总请求数（默认: 20）"
    )
    parser.add_argument(
        "--concurrency-mode",
        choices=["baseline", "vllm", "both"],
        default="both",
        help="并发测试模式: baseline (仅基线), vllm (仅vLLM), both (两者，默认)。建议分开运行以避免内存干扰。"
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("推理延迟优化基准测试")
    logger.info("=" * 80)
    logger.info("")
    
    # 获取模型路径
    from src.config import get_settings
    settings = get_settings()
    vllm_model_path = settings.vllm_model_path
    
    if not vllm_model_path:
        logger.error("未配置模型路径。请在 .env 文件中设置 VLLM_MODEL_PATH")
        return
    
    # 基线测试和 vLLM 使用相同的模型，确保公平对比
    baseline_model_path = vllm_model_path
    
    logger.info(f"测试配置:")
    logger.info(f"  - vLLM 使用模型: {vllm_model_path}")
    logger.info(f"  - 基线测试使用模型: {baseline_model_path} (相同模型，公平对比)")
    logger.info(f"  - 运行模式: {args.test}")
    logger.info("")
    
    # 生成测试提示
    test_prompts = generate_test_prompts()
    logger.info(f"测试提示数量: {len(test_prompts)}")
    logger.info("")
    
    # 输出目录
    output_dir = Path(__file__).parent.parent / "benchmark_results" / "inference_latency"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. 基线推理测试
    if args.test in ["baseline", "both"]:
        logger.info("\n" + "=" * 80)
        logger.info("阶段 1: 基线推理测试（transformers）")
        logger.info("=" * 80)
    
        baseline_results = test_baseline_inference(test_prompts, baseline_model_path)
        
        # 如果基线测试成功，保存结果
        if "error" not in baseline_results:
            all_results["baseline"] = baseline_results
            # 保存基线测试结果到文件
            baseline_output_file = output_dir / "baseline_results.json"
            with open(baseline_output_file, "w", encoding="utf-8") as f:
                json.dump(baseline_results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n✓ 基线测试结果已保存到: {baseline_output_file}")
        else:
            logger.warning("⚠️  基线测试失败")
            logger.warning(f"   错误信息: {baseline_results.get('error', '未知错误')}")
            if args.test == "both":
                logger.info("   继续运行 vLLM 测试...")
            logger.info("")
        
        # 如果只运行基线测试，直接返回
        if args.test == "baseline":
            logger.info("\n" + "=" * 80)
            logger.info("✅ 基线测试完成！")
            logger.info("=" * 80)
            return
        
        # 等待 GPU 内存完全释放
        logger.info("\n等待 GPU 内存完全释放...")
        try:
            import torch
            if torch.cuda.is_available():
                # 强制清理基线模型
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import time as time_module
                # 多次清理以确保内存释放
                for i in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    time_module.sleep(2)
                torch.cuda.synchronize()
                # 显示当前 GPU 内存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"✓ GPU 内存清理完成")
                    logger.info(f"  当前已分配: {allocated:.2f} GB")
                    logger.info(f"  当前已保留: {reserved:.2f} GB")
                    logger.info(f"  GPU 总内存: {total_memory:.2f} GB")
                    logger.info(f"  可用内存: {total_memory - reserved:.2f} GB")
                    if reserved > 1.0:
                        logger.warning(f"⚠️  GPU 仍有 {reserved:.2f} GB 被保留")
                        logger.warning("   建议：在不同的进程中分别运行基线测试和 vLLM 测试")
                        logger.warning("   命令：")
                        logger.warning("     python scripts/benchmark/inference_latency_test.py --test baseline")
                        logger.warning("     python scripts/benchmark/inference_latency_test.py --test vllm")
        except Exception as e:
            logger.warning(f"清理 GPU 内存时出现警告: {e}")
    
    # 2. vLLM 优化推理测试
    if args.test in ["vllm", "both"]:
        logger.info("\n" + "=" * 80)
        logger.info("阶段 2: vLLM 优化推理测试（PagedAttention + Prefix Caching）")
        logger.info("=" * 80)
    
        vllm_results = test_vllm_inference(test_prompts, vllm_model_path)
        if "error" not in vllm_results:
            all_results["vllm"] = vllm_results
            # 保存 vLLM 测试结果到文件
            vllm_output_file = output_dir / "vllm_results.json"
            with open(vllm_output_file, "w", encoding="utf-8") as f:
                json.dump(vllm_results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n✓ vLLM 测试结果已保存到: {vllm_output_file}")
        
        # 如果只运行 vLLM 测试，检查是否有基线测试结果文件
        if args.test == "vllm":
            baseline_results_file = output_dir / "baseline_results.json"
            if baseline_results_file.exists():
                logger.info(f"\n找到基线测试结果文件: {baseline_results_file}")
                try:
                    with open(baseline_results_file, "r", encoding="utf-8") as f:
                        all_results["baseline"] = json.load(f)
                    logger.info("✓ 已加载基线测试结果，将生成对比报告")
                except Exception as e:
                    logger.warning(f"加载基线测试结果失败: {e}")
    
    # 3. 并发压力测试
    if args.test == "concurrency":
        concurrency_results = run_concurrency_test(
            model_path=vllm_model_path,
            concurrency_levels=args.concurrency,
            num_requests=args.num_requests,
            test_mode=args.concurrency_mode,
            output_dir=output_dir,
        )
        all_results.update(concurrency_results)
        
        # 保存并发测试结果
        concurrency_output_file = output_dir / "concurrency_results.json"
        with open(concurrency_output_file, "w", encoding="utf-8") as f:
            json.dump(concurrency_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✓ 并发测试结果已保存到: {concurrency_output_file}")
        
        # 生成并发测试总结报告
        logger.info("\n" + "=" * 80)
        logger.info("并发测试总结报告")
        logger.info("=" * 80)
        logger.info("")
        
        for concurrency in args.concurrency:
            baseline_key = f"baseline_c{concurrency}"
            vllm_key = f"vllm_c{concurrency}"
            
            if baseline_key in concurrency_results and vllm_key in concurrency_results:
                baseline = concurrency_results[baseline_key]
                vllm = concurrency_results[vllm_key]
                
                speedup = baseline["total_time"] / vllm["total_time"] if vllm["total_time"] > 0 else 0
                throughput_improvement = (vllm["throughput"] / baseline["throughput"] - 1) * 100 if baseline["throughput"] > 0 else 0
                
                logger.info(f"并发数 {concurrency}:")
                logger.info(f"  基线: {baseline['total_time']:.2f}s, {baseline['throughput']:.2f} tokens/s")
                logger.info(f"  vLLM: {vllm['total_time']:.2f}s, {vllm['throughput']:.2f} tokens/s")
                logger.info(f"  速度提升: {speedup:.2f}x | 吞吐量提升: {throughput_improvement:.1f}%")
                logger.info("")
        
        logger.info("=" * 80)
        logger.info("✅ 并发测试完成！")
        logger.info("=" * 80)
        return
    
    # 3. 生成对比报告
    if "baseline" in all_results and "vllm" in all_results:
        generate_comparison_report(all_results, output_dir)
    elif "vllm" in all_results:
        # 仅 vLLM 测试成功，生成单独报告
        logger.info("\n" + "=" * 80)
        logger.info("vLLM 性能报告（基线测试因兼容性问题跳过）")
        logger.info("=" * 80)
        logger.info("")
        vllm_summary = all_results["vllm"]["summary"]
        logger.info("【vLLM 优化推理性能】")
        logger.info(f"  平均 TTFT: {vllm_summary['avg_ttft']*1000:.2f} ms")
        logger.info(f"  平均延迟: {vllm_summary['avg_latency']*1000:.2f} ms")
        logger.info(f"  平均吞吐量: {vllm_summary['avg_throughput']:.2f} tokens/s")
        logger.info("")
        logger.info("说明：")
        logger.info("  - 基线测试失败，仅显示 vLLM 性能结果")
        logger.info("  - 如需完整对比，请确保基线测试可正常运行")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 测试完成！")
    logger.info("=" * 80)


def generate_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
):
    """生成对比报告"""
    
    baseline = results["baseline"]
    vllm = results["vllm"]
    
    baseline_summary = baseline["summary"]
    vllm_summary = vllm["summary"]
    
    # 计算改进百分比
    ttft_improvement = (
        (baseline_summary["avg_ttft"] - vllm_summary["avg_ttft"]) 
        / baseline_summary["avg_ttft"] * 100
        if baseline_summary["avg_ttft"] > 0 else 0
    )
    
    latency_improvement = (
        (baseline_summary["avg_latency"] - vllm_summary["avg_latency"])
        / baseline_summary["avg_latency"] * 100
        if baseline_summary["avg_latency"] > 0 else 0
    )
    
    throughput_improvement = (
        (vllm_summary["avg_throughput"] - baseline_summary["avg_throughput"])
        / baseline_summary["avg_throughput"] * 100
        if baseline_summary["avg_throughput"] > 0 else 0
    )
    
    # 生成报告
    report = {
        "timestamp": time.time(),
        "test_type": "推理延迟优化基准测试",
        "model_path": vllm.get("model_path", ""),
        "baseline": baseline,
        "vllm": vllm,
        "improvements": {
            "ttft_improvement_pct": ttft_improvement,
            "latency_improvement_pct": latency_improvement,
            "throughput_improvement_pct": throughput_improvement,
        },
        "target_achievement": {
            "ttft_reduction_25pct": ttft_improvement >= 25.0,
            "latency_reduction_25pct": latency_improvement >= 25.0,
        },
    }
    
    # 保存报告
    report_path = output_dir / "inference_latency_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    logger.info("\n" + "=" * 80)
    logger.info("对比报告")
    logger.info("=" * 80)
    logger.info("")
    logger.info("【基线推理（transformers）】")
    logger.info(f"  平均 TTFT: {baseline_summary['avg_ttft']*1000:.2f} ms")
    logger.info(f"  平均延迟: {baseline_summary['avg_latency']*1000:.2f} ms")
    logger.info(f"  平均吞吐量: {baseline_summary['avg_throughput']:.2f} tokens/s")
    logger.info("")
    logger.info("【vLLM 优化推理】")
    logger.info(f"  平均 TTFT: {vllm_summary['avg_ttft']*1000:.2f} ms")
    logger.info(f"  平均延迟: {vllm_summary['avg_latency']*1000:.2f} ms")
    logger.info(f"  平均吞吐量: {vllm_summary['avg_throughput']:.2f} tokens/s")
    logger.info("")
    logger.info("【性能改进】")
    logger.info(f"  TTFT 降低: {ttft_improvement:.2f}% {'✅' if ttft_improvement >= 25 else '❌'} (目标: ≥25%)")
    logger.info(f"  延迟降低: {latency_improvement:.2f}% {'✅' if latency_improvement >= 25 else '❌'} (目标: ≥25%)")
    logger.info(f"  吞吐量提升: {throughput_improvement:.2f}%")
    logger.info("")
    logger.info(f"报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
