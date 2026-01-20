"""
Pytest 配置文件

提供测试所需的 fixtures 和配置。
"""

import pytest
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def test_data_dir():
    """测试数据目录"""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_env_vars(monkeypatch):
    """模拟环境变量"""
    env_vars = {
        "VECTOR_DB": "qdrant",
        "QDRANT_URL": "http://localhost:6333",
        "EMBEDDING_MODEL": "BAAI/bge-large-en-v1.5",
        "RERANK_MODEL": "BAAI/bge-reranker-large",
        "SHORT_TERM_THRESHOLD": "10",
        "LONG_TERM_TRIGGER": "0.7",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
