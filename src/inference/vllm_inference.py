"""
vLLM 推理模块（Client-Server 架构）

使用 vLLM 作为独立服务，插件通过 OpenAI-compatible API 调用。
这彻底解决了 duplicate template name 错误，因为插件进程不再导入 vLLM/torch。

使用方式：
1. 用户启动 vLLM 服务（独立进程）：
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

2. 插件作为客户端连接服务：
   from src.inference import VLLMInference
   engine = VLLMInference()
   response, metrics = engine.generate("Hello")
"""

import time
import os
from typing import List, Dict, Any, Optional, Tuple
import requests

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings


class VLLMInference:
    """
    vLLM 推理引擎（客户端模式）
    
    通过 OpenAI-compatible API 连接到独立的 vLLM 服务。
    插件进程不再导入 vLLM/torch，彻底避免 duplicate template name 错误。
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        model_path: Optional[str] = None,  # 向后兼容：支持旧参数名
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """
        初始化 vLLM 推理引擎（客户端）
        
        Args:
            base_url: vLLM 服务地址（默认从配置读取）
            model: 模型名称（默认从配置读取）
            model_path: [已废弃] 模型路径，等同于 model 参数（向后兼容）
            api_key: API 密钥（默认从配置读取，本地运行通常为 "EMPTY"）
            timeout: 请求超时时间（秒，默认从配置读取）
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai 库未安装。请运行: pip install openai>=1.0.0"
            )
        
        settings = get_settings()
        
        # 向后兼容：支持 model_path 参数（等同于 model）
        if model_path and not model:
            model = model_path
            logger.warning(
                "⚠️ [VLLM] model_path 参数已废弃，请使用 model 参数"
            )
        
        # 使用配置或参数
        self.base_url = base_url or settings.vllm_base_url
        self.model = model or settings.vllm_model
        self.api_key = api_key if api_key is not None else settings.vllm_api_key
        self.timeout = timeout if timeout is not None else settings.vllm_timeout
        
        # 兼容旧配置（向后兼容）
        if not self.model and settings.vllm_model_path:
            self.model = settings.vllm_model_path
            logger.warning(
                "⚠️ [VLLM] 检测到使用已废弃的 VLLM_MODEL_PATH 配置，"
                "请改用 VLLM_MODEL 和 VLLM_BASE_URL"
            )
        
        if not self.model:
            raise ValueError(
                "未指定模型名称。请在 .env 文件中设置 VLLM_MODEL 或传递 model 参数"
            )
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        
        logger.info(f"🚀 [VLLM Client] 已连接到推理服务: {self.base_url}")
        logger.info(f"   模型: {self.model}")
        
        # 健康检查
        self._health_check()
    
    def _health_check(self) -> None:
        """
        检查 vLLM 服务是否可用
        
        Raises:
            RuntimeError: 如果服务不可用
        """
        try:
            # 尝试调用 /v1/models 端点
            response = self.client.models.list()
            available_models = [model.id for model in response.data]
            
            if self.model not in available_models:
                logger.warning(
                    f"⚠️ [VLLM Client] 模型 '{self.model}' 不在可用模型列表中"
                )
                logger.warning(f"   可用模型: {available_models}")
                logger.warning(
                    "   请确保 vLLM server 启动时指定的模型名称与配置一致"
                )
            else:
                logger.info(f"✓ [VLLM Client] 服务健康检查通过")
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                logger.error("❌ [VLLM Client] 无法连接到 vLLM 服务")
                logger.error("   请确保已启动 vLLM 服务：")
                logger.error("   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000")
                logger.error("   或使用其他端口（需在 .env 中设置 VLLM_BASE_URL）")
            else:
                logger.warning(f"⚠️ [VLLM Client] 健康检查失败: {e}")
                logger.warning("   服务可能仍在启动中，将继续尝试...")
    
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
            **kwargs: 其他采样参数（top_p, top_k 等）
        
        Returns:
            (生成的文本, 性能指标)
        """
        # 构造消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 准备请求参数
        # 估算输入 tokens 数量（粗略估计：中文约 1.5 tokens/字符，英文约 0.5 tokens/字符）
        # 为了安全，使用更保守的估计：2 tokens/字符（混合文本）
        total_input_text = ""
        for msg in messages:
            total_input_text += msg.get("content", "")
        
        # 粗略估算输入 tokens（保守估计：2 tokens/字符）
        estimated_input_tokens = int(len(total_input_text) * 2)
        
        # vLLM 服务的 max_model_len 默认是 1024（从配置读取，但这里使用默认值）
        # 实际应该从服务端获取，但为了简化，使用 1024 作为默认值
        max_model_len = 1024  # 这是 vLLM 服务启动时的 max_model_len
        
        # 计算可用的 max_tokens
        available_tokens = max_model_len - estimated_input_tokens - 10  # 留 10 tokens 缓冲
        
        # 设置默认 max_tokens（max_model_len=1024 时，可以设置更大的默认值）
        default_max_tokens = min(512, max(50, available_tokens))  # 至少 50，最多 512，但不能超过可用值
        
        if max_tokens is None:
            max_tokens = default_max_tokens
        else:
            # 确保 max_tokens 不会超过可用值，但允许最大到 512
            max_tokens = min(max_tokens, min(512, available_tokens))
        
        # 如果计算出的 max_tokens 太小，至少设为 50
        max_tokens = max(50, max_tokens)
        
        logger.debug(f"输入 tokens 估算: {estimated_input_tokens}, 可用 tokens: {available_tokens}, 设置 max_tokens: {max_tokens}")
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.7,
            "max_tokens": max_tokens,
        }
        
        # 添加其他参数
        if "top_p" in kwargs:
            request_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            request_params["top_k"] = kwargs["top_k"]
        
        # 生成（测量延迟）
        start_time = time.time()
        first_token_time = None
        
        try:
            logger.debug(f"开始生成，提示长度: {len(prompt)} 字符")
            logger.debug(f"请求参数: model={self.model}, max_tokens={max_tokens}, timeout={self.timeout}")
            
            # 调用 OpenAI API（注意：首次推理可能需要较长时间预热）
            logger.info(f"⏳ [VLLM Client] 正在生成（首次推理可能需要预热，请耐心等待...）")
            response = self.client.chat.completions.create(**request_params)
            
            generation_time = time.time() - start_time
            logger.debug(f"生成完成，耗时: {generation_time:.2f} 秒")
            
            # 提取结果
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                
                # 计算 tokens（如果可用）
                tokens_generated = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_generated = response.usage.completion_tokens or 0
                else:
                    # 粗略估计：平均每个字符约 0.25 tokens（中文）或 0.5 tokens（英文）
                    tokens_generated = int(len(generated_text) * 0.4)
                
                # 计算吞吐量
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                # 估算 TTFT（首次 token 延迟）
                # OpenAI API 不直接提供 TTFT，我们使用总时间的 10-15% 作为估计
                ttft = generation_time * 0.12  # 粗略估计
                
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
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                logger.error("❌ [VLLM Client] 无法连接到 vLLM 服务")
                logger.error("   请确保已启动 vLLM 服务：")
                logger.error("   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000")
            elif "timeout" in error_msg:
                logger.error(f"❌ [VLLM Client] 请求超时（{self.timeout} 秒）")
                logger.error("   可能是模型响应太慢，尝试增加 VLLM_TIMEOUT 或减少 max_tokens")
            else:
                logger.error(f"❌ [VLLM Client] 生成失败: {e}")
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
        results = []
        start_time = time.time()
        
        # 串行处理（vLLM server 本身支持并发，但这里简化实现）
        for prompt in prompts:
            try:
                text, metrics = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
                results.append((text, metrics))
            except Exception as e:
                logger.error(f"批量生成中单个请求失败: {e}")
                results.append(("", {}))
        
        total_time = time.time() - start_time
        
        # 更新每个结果的 metrics（使用平均时间）
        for i, (text, metrics) in enumerate(results):
            if metrics:
                metrics["total_time"] = total_time / len(prompts)
                metrics["tokens_per_second"] = (
                    metrics.get("tokens_generated", 0) / metrics["total_time"]
                    if metrics["total_time"] > 0 else 0
                )
        
        return results
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        流式生成文本（生成器）
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            **kwargs: 其他采样参数
        
        Yields:
            生成的文本片段
        """
        # 构造消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 准备请求参数
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.7,
            "max_tokens": max_tokens if max_tokens is not None else 512,
            "stream": True,  # 启用流式输出
        }
        
        # 添加其他参数
        if "top_p" in kwargs:
            request_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            request_params["top_k"] = kwargs["top_k"]
        
        try:
            stream = self.client.chat.completions.create(**request_params)
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                logger.error("❌ [VLLM Client] 无法连接到 vLLM 服务")
                logger.error("   请确保已启动 vLLM 服务")
            else:
                logger.error(f"❌ [VLLM Client] 流式生成失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理引擎统计信息"""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "mode": "client-server",
        }


# 兼容性：保持 VLLM_AVAILABLE 变量（现在总是 True，因为只需要 openai 库）
VLLM_AVAILABLE = OPENAI_AVAILABLE
