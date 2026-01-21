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
    """基线推理引擎（使用 transformers）"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ):
        """
        初始化基线推理引擎
        
        Args:
            model_path: 模型路径（本地路径或 HuggingFace 模型 ID）
            device: 运行设备 ("cuda" 或 "cpu")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers 未安装。请运行: pip install transformers torch"
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
        # 构建完整提示
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n用户: {prompt}\n助手: "
        else:
            full_prompt = prompt
        
        # 编码输入
        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.cuda()
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": max_tokens or 512,
            "temperature": temperature or 0.7,
            "do_sample": True,
            **kwargs
        }
        
        # 生成（测量延迟）
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(inputs, **gen_kwargs)
            
            # 计算首次 token 时间（简化：使用总时间的 10% 作为 TTFT 近似）
            first_token_time = time.time() - start_time
            ttft_estimate = first_token_time * 0.1  # 简化估算
            
            # 解码输出
            generated_ids = outputs[0][inputs.shape[1]:]  # 只取新生成的 token
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
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理引擎统计信息"""
        return {
            "model_path": self.model_path,
            "device": self.device,
        }
