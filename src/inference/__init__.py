"""
推理模块

提供 LLM 推理接口，支持：
- vLLM 高性能推理（PagedAttention + Prefix Caching）
- 基线推理（标准 transformers）
"""

# 延迟导入，避免模块级导入触发 torch
# 注意：不在模块级别定义这些变量，让 from ... import 触发 __getattr__
# 这样确保延迟导入能正常工作

def __getattr__(name: str):
    """
    延迟导入推理引擎，避免模块级导入触发 torch
    
    支持用法：
        from src.inference import VLLMInference, VLLM_AVAILABLE
        from src.inference import BaselineInference, TRANSFORMERS_AVAILABLE
    """
    if name == "VLLMInference":
        from .vllm_inference import VLLMInference
        return VLLMInference
    elif name == "VLLM_AVAILABLE":
        try:
            from .vllm_inference import VLLM_AVAILABLE
            return VLLM_AVAILABLE
        except ImportError:
            return False
    elif name == "BaselineInference":
        from .baseline_inference import BaselineInference
        return BaselineInference
    elif name == "TRANSFORMERS_AVAILABLE":
        try:
            from .baseline_inference import TRANSFORMERS_AVAILABLE
            return TRANSFORMERS_AVAILABLE
        except ImportError:
            return False
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    """返回模块的公共属性列表"""
    return [
        "VLLMInference",
        "BaselineInference",
        "VLLM_AVAILABLE",
        "TRANSFORMERS_AVAILABLE",
    ]

__all__ = [
    "VLLMInference",
    "BaselineInference",
    "VLLM_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]
