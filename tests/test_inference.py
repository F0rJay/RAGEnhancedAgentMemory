"""
测试推理模块

包括 vLLM 推理引擎和基线推理引擎的测试。
由于推理模块需要加载实际模型，测试主要使用 Mock 来验证逻辑。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# 在导入之前patch相关模块
with patch('src.inference.vllm_inference.VLLM_AVAILABLE', True), \
     patch('src.inference.baseline_inference.TRANSFORMERS_AVAILABLE', True):
    from src.inference.vllm_inference import VLLMInference, VLLM_AVAILABLE
    from src.inference.baseline_inference import BaselineInference, TRANSFORMERS_AVAILABLE


# ==================== VLLMInference 测试 ====================

@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_initialization(mock_sampling_params, mock_llm_class):
    """测试 vLLM 推理引擎初始化"""
    mock_llm_instance = Mock()
    mock_llm_class.return_value = mock_llm_instance
    mock_sampling_params_instance = Mock()
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference(
            model_path="test-model",
            gpu_memory_utilization=0.7,
            max_model_len=2048,
            enable_prefix_caching=True,
        )

        assert inference.model_path == "test-model"
        assert inference.gpu_memory_utilization == 0.7
        assert inference.max_model_len == 2048
        assert inference.enable_prefix_caching is True
        assert inference.llm == mock_llm_instance
        mock_llm_class.assert_called_once()


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
def test_vllm_inference_initialization_no_vllm():
    """测试没有 vLLM 时的初始化失败"""
    with patch('src.inference.vllm_inference.VLLM_AVAILABLE', False):
        with pytest.raises(ImportError, match="vLLM 未安装"):
            VLLMInference(model_path="test-model")


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_initialization_default_values(mock_sampling_params, mock_llm_class):
    """测试使用默认值的初始化"""
    mock_llm_instance = Mock()
    mock_llm_class.return_value = mock_llm_instance
    mock_sampling_params_instance = Mock()
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "default-model"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.8
        mock_settings.return_value.vllm_max_model_len = 4096
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference()

        assert inference.model_path == "default-model"
        assert inference.gpu_memory_utilization == 0.8
        assert inference.max_model_len == 4096


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_quantization_detection(mock_sampling_params, mock_llm_class):
    """测试量化方法自动检测"""
    mock_llm_instance = Mock()
    mock_llm_class.return_value = mock_llm_instance
    mock_sampling_params_instance = Mock()
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model-awq"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference(model_path="test-model-awq")

        assert inference.quantization == "awq"
        # 检查是否传递了量化参数
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs.get("quantization") == "awq"


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_generate(mock_sampling_params, mock_llm_class):
    """测试 vLLM 生成方法"""
    mock_llm_instance = Mock()
    mock_output = Mock()
    mock_output.outputs = [Mock()]
    mock_output.outputs[0].text = "生成的文本"
    mock_output.outputs[0].token_ids = [1, 2, 3, 4, 5]
    mock_output.metrics = Mock()
    mock_output.metrics.time_to_first_token = 100.0  # 毫秒
    
    mock_llm_instance.generate.return_value = [mock_output]
    mock_llm_class.return_value = mock_llm_instance
    
    mock_sampling_params_instance = Mock()
    mock_sampling_params_instance.temperature = 0.7
    mock_sampling_params_instance.top_p = 0.9
    mock_sampling_params_instance.max_tokens = 512
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference(model_path="test-model")
        
        text, metrics = inference.generate("测试提示")

        assert text == "生成的文本"
        assert "ttft" in metrics
        assert "total_time" in metrics
        assert "tokens_generated" in metrics
        assert "tokens_per_second" in metrics
        assert metrics["tokens_generated"] == 5


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_generate_with_system_prompt(mock_sampling_params, mock_llm_class):
    """测试带系统提示的生成"""
    mock_llm_instance = Mock()
    mock_output = Mock()
    mock_output.outputs = [Mock()]
    mock_output.outputs[0].text = "生成的文本"
    mock_output.outputs[0].token_ids = [1, 2, 3]
    mock_output.metrics = Mock()
    # 设置time_to_first_token为数值，而不是Mock对象
    mock_output.metrics.time_to_first_token = 50.0  # 毫秒
    mock_output.metrics.ttft = None  # 确保使用time_to_first_token
    
    mock_llm_instance.generate.return_value = [mock_output]
    mock_llm_class.return_value = mock_llm_instance
    
    mock_sampling_params_instance = Mock()
    mock_sampling_params_instance.temperature = 0.7
    mock_sampling_params_instance.top_p = 0.9
    mock_sampling_params_instance.max_tokens = 512
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference(model_path="test-model")
        
        text, metrics = inference.generate(
            "测试提示",
            system_prompt="你是一个助手"
        )

        assert text == "生成的文本"
        # 检查是否包含了系统提示
        call_args = mock_llm_instance.generate.call_args[0][0]
        assert "你是一个助手" in call_args[0]


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_generate_custom_params(mock_sampling_params, mock_llm_class):
    """测试自定义参数的生成"""
    mock_llm_instance = Mock()
    mock_output = Mock()
    mock_output.outputs = [Mock()]
    mock_output.outputs[0].text = "生成的文本"
    mock_output.outputs[0].token_ids = [1, 2]
    mock_output.metrics = None  # 测试没有 metrics 的情况
    
    mock_llm_instance.generate.return_value = [mock_output]
    mock_llm_class.return_value = mock_llm_instance
    
    mock_sampling_params_instance = Mock()
    mock_sampling_params_instance.temperature = 0.7
    mock_sampling_params_instance.top_p = 0.9
    mock_sampling_params_instance.max_tokens = 512
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference(model_path="test-model")
        
        text, metrics = inference.generate(
            "测试提示",
            max_tokens=256,
            temperature=0.5
        )

        assert text == "生成的文本"
        # 检查是否使用了自定义参数
        # SamplingParams 应该被调用，传入新的参数
        assert mock_sampling_params.called


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_get_stats(mock_sampling_params, mock_llm_class):
    """测试获取统计信息"""
    mock_llm_instance = Mock()
    mock_llm_class.return_value = mock_llm_instance
    mock_sampling_params_instance = Mock()
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference(model_path="test-model")
        
        stats = inference.get_stats()

        assert "model_path" in stats
        assert stats["model_path"] == "test-model"
        assert "gpu_memory_utilization" in stats
        assert "max_model_len" in stats


# ==================== BaselineInference 测试 ====================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
@patch('src.inference.baseline_inference.AutoModelForCausalLM')
@patch('src.inference.baseline_inference.AutoTokenizer')
def test_baseline_inference_initialization(mock_tokenizer_class, mock_model_class):
    """测试基线推理引擎初始化"""
    mock_tokenizer = Mock()
    mock_model = Mock()
    mock_model_instance = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_model_instance.cuda = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)

    with patch('src.inference.baseline_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.hf_endpoint = None

        with patch('src.inference.baseline_inference.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float16 = "float16"

            inference = BaselineInference(
                model_path="test-model",
                device="cuda"
            )

            assert inference.model_path == "test-model"
            assert inference.device == "cuda"
            assert inference.model == mock_model_instance
            assert inference.tokenizer == mock_tokenizer


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
def test_baseline_inference_initialization_no_transformers():
    """测试没有 transformers 时的初始化失败"""
    with patch('src.inference.baseline_inference.TRANSFORMERS_AVAILABLE', False):
        with pytest.raises(ImportError, match="transformers 未安装"):
            BaselineInference(model_path="test-model")


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
@patch('src.inference.baseline_inference.AutoModelForCausalLM')
@patch('src.inference.baseline_inference.AutoTokenizer')
def test_baseline_inference_initialization_default_values(mock_tokenizer_class, mock_model_class):
    """测试使用默认值的初始化"""
    mock_tokenizer = Mock()
    mock_model_instance = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_model_instance.cuda = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)

    with patch('src.inference.baseline_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "default-model"
        mock_settings.return_value.hf_endpoint = None

        with patch('src.inference.baseline_inference.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float16 = "float16"

            inference = BaselineInference()

            assert inference.model_path == "default-model"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
@patch('src.inference.baseline_inference.AutoModelForCausalLM')
@patch('src.inference.baseline_inference.AutoTokenizer')
def test_baseline_inference_generate(mock_tokenizer_class, mock_model_class):
    """测试基线推理引擎生成方法"""
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = Mock()
    mock_tokenizer.decode.return_value = "生成的文本"
    
    mock_model_instance = Mock()
    mock_outputs = Mock()
    mock_outputs.shape = [1, 10]
    # Mock generate 返回包含输入和新生成的token
    mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_model_instance.cuda = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)

    with patch('src.inference.baseline_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.hf_endpoint = None

        with patch('src.inference.baseline_inference.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float16 = "float16"
            mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
            mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
            
            # Mock inputs 的形状
            mock_inputs = Mock()
            mock_inputs.shape = [1, 10]
            mock_inputs.cuda = Mock(return_value=mock_inputs)
            mock_tokenizer.encode.return_value = mock_inputs

            inference = BaselineInference(model_path="test-model", device="cuda")
            
            text, metrics = inference.generate("测试提示")

            assert text == "生成的文本"
            assert "ttft" in metrics
            assert "total_time" in metrics
            assert "tokens_generated" in metrics
            assert "tokens_per_second" in metrics


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
@patch('src.inference.baseline_inference.AutoModelForCausalLM')
@patch('src.inference.baseline_inference.AutoTokenizer')
def test_baseline_inference_generate_with_system_prompt(mock_tokenizer_class, mock_model_class):
    """测试带系统提示的生成"""
    mock_tokenizer = Mock()
    mock_inputs = Mock()
    mock_inputs.shape = [1, 5]
    mock_inputs.cuda = Mock(return_value=mock_inputs)
    mock_tokenizer.encode.return_value = mock_inputs
    mock_tokenizer.decode.return_value = "生成的文本"
    
    mock_model_instance = Mock()
    mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_model_instance.cuda = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)

    with patch('src.inference.baseline_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.hf_endpoint = None

        with patch('src.inference.baseline_inference.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float16 = "float16"
            mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
            mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)

            inference = BaselineInference(model_path="test-model", device="cuda")
            
            text, metrics = inference.generate(
                "测试提示",
                system_prompt="你是一个助手"
            )

            assert text == "生成的文本"
            # 检查是否编码了包含系统提示的完整提示
            assert mock_tokenizer.encode.called


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
@patch('src.inference.baseline_inference.AutoModelForCausalLM')
@patch('src.inference.baseline_inference.AutoTokenizer')
def test_baseline_inference_get_stats(mock_tokenizer_class, mock_model_class):
    """测试获取统计信息"""
    mock_tokenizer = Mock()
    mock_model_instance = Mock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_model_instance.cuda = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)

    with patch('src.inference.baseline_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.hf_endpoint = None

        with patch('src.inference.baseline_inference.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float16 = "float16"

            inference = BaselineInference(model_path="test-model", device="cuda")
            
            stats = inference.get_stats()

            assert "model_path" in stats
            assert stats["model_path"] == "test-model"
            assert "device" in stats
            assert stats["device"] == "cuda"


# ==================== 边界情况和错误处理测试 ====================

@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_no_model_path(mock_sampling_params, mock_llm_class):
    """测试未指定模型路径的情况"""
    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = None
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        with pytest.raises(ValueError, match="未指定模型路径"):
            VLLMInference()


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
def test_baseline_inference_no_model_path():
    """测试未指定模型路径的情况"""
    with patch('src.inference.baseline_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = None
        mock_settings.return_value.hf_endpoint = None

        with pytest.raises(ValueError, match="未指定模型路径"):
            BaselineInference()


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM 未安装")
@patch('src.inference.vllm_inference.LLM')
@patch('src.inference.vllm_inference.SamplingParams')
def test_vllm_inference_generate_empty_output(mock_sampling_params, mock_llm_class):
    """测试生成空输出的情况"""
    mock_llm_instance = Mock()
    mock_llm_instance.generate.return_value = []  # 空输出
    mock_llm_class.return_value = mock_llm_instance
    
    mock_sampling_params_instance = Mock()
    mock_sampling_params_instance.temperature = 0.7
    mock_sampling_params_instance.top_p = 0.9
    mock_sampling_params_instance.max_tokens = 512
    mock_sampling_params.return_value = mock_sampling_params_instance

    with patch('src.inference.vllm_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.vllm_gpu_memory_utilization = 0.7
        mock_settings.return_value.vllm_max_model_len = 2048
        mock_settings.return_value.hf_endpoint = None

        inference = VLLMInference(model_path="test-model")
        
        # 空输出应该抛出异常
        with pytest.raises(ValueError, match="生成结果为空"):
            inference.generate("测试提示")


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers 未安装")
@patch('src.inference.baseline_inference.AutoModelForCausalLM')
@patch('src.inference.baseline_inference.AutoTokenizer')
def test_baseline_inference_generate_exception(mock_tokenizer_class, mock_model_class):
    """测试生成过程中的异常处理"""
    mock_tokenizer = Mock()
    mock_inputs = Mock()
    mock_inputs.cuda = Mock(return_value=mock_inputs)
    mock_tokenizer.encode.return_value = mock_inputs
    
    mock_model_instance = Mock()
    mock_model_instance.generate.side_effect = RuntimeError("生成失败")
    
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_model_instance.cuda = Mock(return_value=mock_model_instance)
    mock_model_instance.eval = Mock(return_value=mock_model_instance)

    with patch('src.inference.baseline_inference.get_settings') as mock_settings:
        mock_settings.return_value.vllm_model_path = "test-model"
        mock_settings.return_value.hf_endpoint = None

        with patch('src.inference.baseline_inference.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float16 = "float16"
            mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
            mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)

            inference = BaselineInference(model_path="test-model", device="cuda")
            
            with pytest.raises(RuntimeError, match="生成失败"):
                inference.generate("测试提示")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
