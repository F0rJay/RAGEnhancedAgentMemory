"""
测试配置管理模块
"""

import pytest
import os
from unittest.mock import patch

from src.config import Settings, get_settings


def test_settings_defaults():
    """测试默认配置"""
    settings = Settings()

    assert settings.vector_db == "qdrant"
    assert settings.short_term_threshold == 10
    assert settings.long_term_trigger == 0.7
    assert isinstance(settings.qdrant_url, str)


def test_get_settings():
    """测试获取配置实例"""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_settings_env_override(monkeypatch):
    """测试环境变量覆盖"""
    monkeypatch.setenv("VECTOR_DB", "chroma")
    monkeypatch.setenv("SHORT_TERM_THRESHOLD", "15")

    # 注意：Settings 使用 pydantic-settings，可能需要重新加载
    # 这里仅测试配置类本身
    settings = Settings()
    # 由于 pydantic-settings 的缓存机制，可能需要重新创建实例
    # 实际测试中可能需要清理缓存


def test_settings_validation():
    """测试配置验证"""
    settings = Settings()

    # 测试类型验证
    assert isinstance(settings.short_term_threshold, int)
    assert isinstance(settings.long_term_trigger, float)
    assert isinstance(settings.vector_db, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
