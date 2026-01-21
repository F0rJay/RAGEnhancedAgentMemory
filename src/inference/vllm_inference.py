"""
vLLM 推理模块

使用 vLLM 进行高性能 LLM 推理，支持：
- PagedAttention 优化
- Prefix Caching（前缀缓存）
- 并发请求处理
"""

import time
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 在导入 vllm 之前设置镜像源（如果配置了）
try:
    from ..config import get_settings
    settings = get_settings()
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint
        # 尝试使用 huggingface_hub 的 set_endpoint（如果可用）
        try:
            from huggingface_hub import set_endpoint
            set_endpoint(settings.hf_endpoint)
        except ImportError:
            pass
except Exception:
    pass

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings


class VLLMInference:
    """vLLM 推理引擎"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        quantization: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        enable_prefix_caching: bool = True,
        enforce_eager: Optional[bool] = None,
        **kwargs
    ):
        """
        初始化 vLLM 推理引擎
        
        Args:
            model_path: 模型路径（本地路径或 HuggingFace 模型 ID）
            quantization: 量化方法 ('awq' 或 'gptq')，如果使用量化模型
            gpu_memory_utilization: GPU 内存使用率 (0.0-1.0)
            max_model_len: 最大模型长度
            enable_prefix_caching: 是否启用前缀缓存
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM 未安装。请运行: pip install vllm>=0.6.0"
            )
        
        settings = get_settings()
        
        # 配置 HuggingFace 镜像源（如果设置了）
        # 注意：环境变量已在文件顶部设置，这里只是记录日志
        if settings.hf_endpoint:
            logger.info(f"使用 HuggingFace 镜像源: {settings.hf_endpoint}")
        
        # 使用配置或参数
        self.model_path = model_path or settings.vllm_model_path
        if not self.model_path:
            raise ValueError(
                "未指定模型路径。请在 .env 文件中设置 VLLM_MODEL_PATH 或传递 model_path 参数"
            )
        
        # 处理相对路径（仅对本地路径）
        if not self.model_path.startswith("/") and "/" not in self.model_path:
            # 可能是 HuggingFace 模型 ID，不处理
            pass
        elif self.model_path.startswith("./") or not Path(self.model_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            abs_path = (project_root / self.model_path).resolve()
            if abs_path.exists():
                self.model_path = str(abs_path)
        
        # 量化方法（仅通过参数或模型路径自动检测）
        # 如果模型路径包含 AWQ 或 GPTQ，自动检测
        if quantization:
            self.quantization = quantization
        elif "awq" in self.model_path.lower():
            self.quantization = "awq"
            logger.info("检测到模型路径包含 'awq'，自动启用 AWQ 量化")
        elif "gptq" in self.model_path.lower():
            self.quantization = "gptq"
            logger.info("检测到模型路径包含 'gptq'，自动启用 GPTQ 量化")
        else:
            self.quantization = None
        
        self.gpu_memory_utilization = (
            gpu_memory_utilization 
            if gpu_memory_utilization is not None 
            else settings.vllm_gpu_memory_utilization
        )
        self.max_model_len = (
            max_model_len 
            if max_model_len is not None 
            else settings.vllm_max_model_len
        )
        self.enable_prefix_caching = enable_prefix_caching
        
        logger.info(f"初始化 vLLM 推理引擎...")
        logger.info(f"  模型路径: {self.model_path}")
        if self.quantization:
            logger.info(f"  量化方法: {self.quantization.upper()}")
        logger.info(f"  GPU 内存使用率: {self.gpu_memory_utilization}")
        logger.info(f"  最大模型长度: {self.max_model_len}")
        logger.info(f"  前缀缓存: {'启用' if enable_prefix_caching else '禁用'}")
        
        # 处理 enforce_eager 参数
        # 默认禁用 enforce_eager（False）以启用 CUDA graph 优化，提升性能
        # 只有在内存不足时才需要设置为 True 以节省内存
        enforce_eager_value = enforce_eager if enforce_eager is not None else False
        logger.info(f"  CUDA graph 优化: {'禁用' if enforce_eager_value else '启用'}")
        
        try:
            # 初始化 vLLM
            # 注意：如果 GPU 内存不足，可以降低 gpu_memory_utilization 或 max_model_len
            # 也可以尝试使用 enforce_eager=True 来禁用 CUDA graph 以节省内存
            llm_kwargs = {
                "model": self.model_path,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "enable_prefix_caching": enable_prefix_caching,
                "trust_remote_code": True,  # Qwen 模型需要
                "disable_log_stats": True,  # 禁用日志统计以节省内存
                "enforce_eager": enforce_eager_value,  # False 启用 CUDA graph 优化，True 禁用以节省内存
            }
            
            # 如果指定了量化方法，添加到参数中
            if self.quantization:
                llm_kwargs["quantization"] = self.quantization
            
            self.llm = LLM(**llm_kwargs, **kwargs)
            
            # 采样参数
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
            )
            
            logger.info("✓ vLLM 推理引擎初始化成功")
        except Exception as e:
            logger.error(f"vLLM 初始化失败: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            **kwargs: 其他采样参数
        
        Returns:
            (生成的文本, 性能指标)
        """
        # 构建完整提示
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n用户: {prompt}\n助手: "
        else:
            full_prompt = prompt
        
        # 更新采样参数
        sampling_params = self.sampling_params
        if max_tokens is not None or temperature is not None or kwargs:
            sampling_params = SamplingParams(
                temperature=temperature if temperature is not None else self.sampling_params.temperature,
                top_p=kwargs.get("top_p", self.sampling_params.top_p),
                max_tokens=max_tokens if max_tokens is not None else self.sampling_params.max_tokens,
            )
        
        # 生成（测量延迟）
        start_time = time.time()
        first_token_time = None
        
        try:
            # vLLM 生成
            # 注意：首次生成可能需要较长时间进行 prefill，这是正常的
            logger.debug(f"开始生成，提示长度: {len(full_prompt)} 字符")
            outputs = self.llm.generate([full_prompt], sampling_params)
            generation_time = time.time() - start_time
            logger.debug(f"生成完成，耗时: {generation_time:.2f} 秒")
            
            # 提取结果
            if outputs and len(outputs) > 0:
                output = outputs[0]
                generated_text = output.outputs[0].text
                
                # 尝试从 vLLM 的 metadata 中获取时间信息
                # vLLM 的 RequestOutput 包含 metrics
                if hasattr(output, 'metrics'):
                    # 获取首次 token 时间（如果有）
                    if hasattr(output.metrics, 'time_to_first_token'):
                        ttft = output.metrics.time_to_first_token / 1000.0  # 转换为秒
                    elif hasattr(output.metrics, 'ttft'):
                        ttft = output.metrics.ttft / 1000.0
                    else:
                        # 使用首次输出时间作为近似
                        # 注意：这是不准确的，但比总时间好
                        ttft = generation_time * 0.1  # 粗略估计：首次 token 约占总时间的 10%
                else:
                    # 如果没有 metrics，使用粗略估计
                    # 对于 AWQ 模型，prefill 阶段通常占用较长时间
                    # 我们使用总时间的 15-20% 作为 TTFT 的估计
                    ttft = generation_time * 0.15  # 粗略估计
                
                # 计算吞吐量
                tokens_generated = len(output.outputs[0].token_ids)
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                metrics = {
                    "ttft": ttft,
                    "total_time": generation_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tokens_per_second,
                }
                
                return generated_text, metrics
            else:
                raise ValueError("生成结果为空")
        
        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        批量生成文本
        
        Args:
            prompts: 提示列表
            system_prompt: 系统提示（可选）
            **kwargs: 采样参数
        
        Returns:
            生成结果列表
        """
        # 构建完整提示
        if system_prompt:
            full_prompts = [f"{system_prompt}\n\n用户: {p}\n助手: " for p in prompts]
        else:
            full_prompts = prompts
        
        # 更新采样参数
        sampling_params = self.sampling_params
        if kwargs:
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", self.sampling_params.temperature),
                top_p=kwargs.get("top_p", self.sampling_params.top_p),
                max_tokens=kwargs.get("max_tokens", self.sampling_params.max_tokens),
            )
        
        start_time = time.time()
        
        try:
            outputs = self.llm.generate(full_prompts, sampling_params)
            total_time = time.time() - start_time
            
            results = []
            for i, output in enumerate(outputs):
                if output.outputs:
                    generated_text = output.outputs[0].text
                    token_count = len(output.outputs[0].token_ids)
                    
                    metrics = {
                        "ttft": total_time / len(prompts),  # 简化：平均时间
                        "total_time": total_time / len(prompts),
                        "tokens_generated": token_count,
                        "tokens_per_second": token_count / (total_time / len(prompts)) if total_time > 0 else 0,
                    }
                    
                    results.append((generated_text, metrics))
                else:
                    results.append(("", {}))
            
            return results
        
        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理引擎统计信息"""
        return {
            "model_path": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "enable_prefix_caching": self.enable_prefix_caching,
        }
