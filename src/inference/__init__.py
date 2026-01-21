"""
推理模块

提供 LLM 推理接口，支持：
- vLLM 高性能推理（PagedAttention + Prefix Caching）
- 基线推理（标准 transformers）
"""

try:
    from .vllm_inference import VLLMInference, VLLM_AVAILABLE
except ImportError:
    VLLMInference = None
    VLLM_AVAILABLE = False

try:
    from .baseline_inference import BaselineInference, TRANSFORMERS_AVAILABLE
except ImportError:
    BaselineInference = None
    TRANSFORMERS_AVAILABLE = False

__all__ = [
    "VLLMInference",
    "BaselineInference",
    "VLLM_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]
