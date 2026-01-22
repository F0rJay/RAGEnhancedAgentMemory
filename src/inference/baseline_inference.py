"""
基线推理模块

使用标准 transformers/HuggingFace 进行 LLM 推理，作为 vLLM 优化的基线对比。
"""

import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 在导入 transformers 之前设置镜像源（如果配置了）
# 这样可以确保 huggingface_hub 使用正确的 endpoint
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

# 注意：AutoAWQ 已被弃用，但 HuggingFace transformers 现在支持直接加载 AWQ 模型
# 我们不需要导入 AutoAWQForCausalLM，直接使用 AutoModelForCausalLM 即可
AWQ_AVAILABLE = True  # transformers 已支持 AWQ

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ..config import get_settings


class BaselineInference:
    """基线推理引擎（使用 transformers 或 OpenAI-compatible API）"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """
        初始化基线推理引擎
        
        支持两种模式：
        1. 本地模型模式：使用 transformers 加载本地模型或 HuggingFace 模型
        2. API 模式：通过 OpenAI-compatible API 调用（如 DeepSeek API）
        
        Args:
            model_path: 模型路径（本地路径、HuggingFace 模型 ID 或 API 模型名称）
            device: 运行设备 ("cuda" 或 "cpu")，仅用于本地模型模式
            base_url: API 服务地址（如果提供，则使用 API 模式）
            api_key: API 密钥（API 模式需要）
            timeout: 请求超时时间（秒，API 模式需要）
        """
        settings = get_settings()
        
        # 使用配置或参数
        self.model_path = model_path or settings.vllm_model or settings.vllm_model_path
        self.base_url = base_url or settings.vllm_base_url
        self.api_key = api_key if api_key is not None else settings.vllm_api_key
        self.timeout = timeout if timeout is not None else settings.vllm_timeout
        
        # 判断使用 API 模式还是本地模型模式
        # 如果提供了 base_url 或检测到 API 模型名称，使用 API 模式
        api_model_names = ["deepseek-chat", "deepseek-reasoner", "gpt-3.5-turbo", "gpt-4", "claude"]
        is_api_model = self.model_path and any(api_name in self.model_path.lower() for api_name in api_model_names)
        
        # 特殊处理：如果 base_url 是本地 vLLM 服务（http://localhost:8000/v1），
        # 且模型路径是本地路径（不是 API 模型名称），则强制使用本地 transformers 模式
        # 这是因为基线系统应该使用本地 transformers，而不是通过 vLLM API
        is_local_vllm_service = self.base_url and (
            "localhost" in self.base_url.lower() or 
            "127.0.0.1" in self.base_url.lower()
        )
        is_local_model_path = self.model_path and (
            self.model_path.startswith("models/") or
            "/" in self.model_path or
            Path(self.model_path).exists()
        )
        
        # 如果 base_url 是本地 vLLM 服务，且模型路径是本地路径，强制使用本地模式
        if is_local_vllm_service and is_local_model_path and not is_api_model:
            # 强制使用本地 transformers 模式（基线系统不应该通过 vLLM API）
            self.use_api = False
            self._init_local_mode(device, **kwargs)
        elif self.base_url or is_api_model:
            # API 模式：使用 OpenAI-compatible API
            self.use_api = True
            self._init_api_mode()
        else:
            # 本地模型模式：使用 transformers
            self.use_api = False
            self._init_local_mode(device, **kwargs)
    
    def _init_api_mode(self):
        """初始化 API 模式"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai 库未安装。API 模式需要 openai 库。请运行: pip install openai>=1.0.0"
            )
        
        if not self.base_url:
            raise ValueError(
                "API 模式需要 base_url。请在 .env 文件中设置 VLLM_BASE_URL 或传递 base_url 参数"
            )
        
        if not self.model_path:
            raise ValueError(
                "API 模式需要 model 名称。请在 .env 文件中设置 VLLM_MODEL 或传递 model_path 参数"
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        
        logger.info(f"初始化基线推理引擎（API 模式）...")
        logger.info(f"  服务地址: {self.base_url}")
        logger.info(f"  模型: {self.model_path}")
        logger.info("✓ 基线推理引擎初始化成功（API 模式）")
    
    def _init_local_mode(self, device: str = "cuda", **kwargs):
        """初始化本地模型模式"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers 未安装。本地模型模式需要 transformers 库。请运行: pip install transformers torch"
            )
        
        # 配置 HuggingFace 镜像源（如果设置了）
        settings = get_settings()
        if settings.hf_endpoint:
            logger.info(f"使用 HuggingFace 镜像源: {settings.hf_endpoint}")
        
        if not self.model_path:
            raise ValueError(
                "未指定模型路径。请在 .env 文件中设置 VLLM_MODEL 或传递 model_path 参数"
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
        
        self.device = device
        self.model = None
        self.tokenizer = None
        self.use_awq = False
        
        # 检测是否为 AWQ 模型
        is_awq_model = "awq" in self.model_path.lower()
        
        logger.info(f"初始化基线推理引擎...")
        logger.info(f"  模型路径: {self.model_path}")
        logger.info(f"  设备: {self.device}")
        if is_awq_model:
            logger.info(f"  检测到 AWQ 量化模型")
        
        try:
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            
            # 使用 transformers 加载模型（支持 AWQ 量化模型）
            # HuggingFace transformers 现在可以直接加载 AWQ 模型，无需 autoawq
            if is_awq_model:
                logger.info("  加载 AWQ 量化模型（使用 transformers）...")
            
            # 使用 AutoModelForCausalLM 加载（transformers 会自动识别 AWQ 格式）
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else "cpu",
                **kwargs
            )
            
            # 如果 device_map="auto" 没有自动设置，手动移动到设备
            if device == "cuda" and torch.cuda.is_available():
                if not hasattr(self.model, "device") or str(self.model.device) == "cpu":
                    self.model = self.model.cuda()
            
            # 设置为评估模式
            self.model.eval()
            
            if is_awq_model:
                logger.info("✓ 基线推理引擎初始化成功（AWQ 量化模型）")
            else:
                logger.info("✓ 基线推理引擎初始化成功")
        except Exception as e:
            logger.error(f"基线推理引擎初始化失败: {e}")
            if is_awq_model:
                logger.error("提示: HuggingFace transformers 应该支持直接加载 AWQ 模型")
                logger.error("如果失败，可能是 transformers 版本过旧，请升级: pip install --upgrade transformers")
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
            **kwargs: 其他生成参数
        
        Returns:
            (生成的文本, 性能指标)
        """
        if self.use_api:
            return self._generate_api(prompt, system_prompt, max_tokens, temperature, **kwargs)
        else:
            return self._generate_local(prompt, system_prompt, max_tokens, temperature, **kwargs)
    
    def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """API 模式生成（与 VLLMInference 类似）"""
        # 构造消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 估算输入 tokens 并动态调整 max_tokens（与 vllm_inference.py 保持一致）
        total_input_text = ""
        for msg in messages:
            total_input_text += msg.get("content", "")
        
        # 粗略估算输入 tokens（保守估计：2 tokens/字符）
        estimated_input_tokens = int(len(total_input_text) * 2)
        max_model_len = 1024  # 默认值，与 vLLM 服务保持一致
        available_tokens = max_model_len - estimated_input_tokens - 10  # 留 10 tokens 缓冲
        
        default_max_tokens = min(512, max(50, available_tokens))  # 最多 512（max_model_len=1024 时）
        
        if max_tokens is None:
            max_tokens = default_max_tokens
        else:
            max_tokens = min(max_tokens, min(512, available_tokens))  # 最多 512
        
        max_tokens = max(50, max_tokens)  # 至少 50
        
        # 准备请求参数
        request_params = {
            "model": self.model_path,
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
        
        try:
            logger.debug(f"开始生成（API 模式），提示长度: {len(prompt)} 字符")
            
            # 调用 OpenAI API
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
                    tokens_generated = int(len(generated_text) * 0.3)
                
                # 估算 TTFT（API 通常不提供，使用总时间的 10% 作为估算）
                ttft_estimate = generation_time * 0.1
                
                metrics = {
                    "ttft": ttft_estimate,
                    "total_time": generation_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0,
                }
                
                return generated_text, metrics
            else:
                raise ValueError("API 返回空响应")
                
        except Exception as e:
            logger.error(f"API 生成失败: {e}")
            raise
    
    def _generate_local(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """本地模型模式生成（原有逻辑）"""
        # 构建完整提示
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n用户: {prompt}\n助手: "
        else:
            full_prompt = prompt
        
        # 使用 tokenizer() 而不是 encode()，以获取 attention_mask
        # 这样可以避免 pad_token 和 eos_token 相同导致的警告
        tokenizer_output = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        inputs = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output.get("attention_mask", None)
        
        if self.device == "cuda":
            inputs = inputs.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
        
        # 动态调整 max_tokens（避免超过模型最大长度）
        # 估算输入 tokens
        input_length = inputs.shape[1]
        # 假设模型最大长度是 2048（Qwen2.5-7B 默认）
        max_model_len = 2048
        available_tokens = max_model_len - input_length - 10  # 留 10 tokens 缓冲
        
        # 设置默认 max_tokens
        default_max_tokens = min(512, max(50, available_tokens))
        
        if max_tokens is None:
            max_tokens = default_max_tokens
        else:
            max_tokens = min(max_tokens, available_tokens)
        
        max_tokens = max(50, max_tokens)  # 至少 50
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature or 0.7,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        }
        
        # 如果 attention_mask 存在，添加到生成参数中
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        
        # 添加其他参数
        gen_kwargs.update(kwargs)
        
        # 生成（测量延迟）
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(inputs, **gen_kwargs)
            
            # 计算首次 token 时间（简化：使用总时间的 10% 作为 TTFT 近似）
            first_token_time = time.time() - start_time
            ttft_estimate = first_token_time * 0.1  # 简化估算
            
            # 解码输出（只取新生成的 token）
            generated_ids = outputs[0][input_length:]  # 只取新生成的 token
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            total_time = time.time() - start_time
            
            metrics = {
                "ttft": ttft_estimate,  # 估算值
                "total_time": total_time,
                "tokens_generated": len(generated_ids),
                "tokens_per_second": len(generated_ids) / total_time if total_time > 0 else 0,
            }
            
            return generated_text, metrics
        
        except Exception as e:
            logger.error(f"生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理引擎统计信息"""
        stats = {
            "model_path": self.model_path,
            "use_api": self.use_api,
        }
        if self.use_api:
            stats.update({
                "base_url": self.base_url,
                "mode": "api",
            })
        else:
            stats.update({
                "device": self.device,
                "mode": "local",
            })
        return stats
