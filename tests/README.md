# 测试说明

## 测试结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # Pytest 配置和 fixtures
├── utils.py                 # 测试工具函数
├── test_state.py            # 状态定义测试
├── test_config.py           # 配置管理测试
├── test_graph.py            # LangGraph 相关测试
├── test_short_term_memory.py # 短期记忆测试
├── test_long_term_memory.py  # 长期记忆测试
├── test_routing.py           # 路由逻辑测试
├── test_retrieval.py         # 检索模块测试
├── test_core.py              # 核心系统测试
├── test_evaluation.py        # 评估体系测试
└── test_integration.py       # 集成测试
```

## 运行测试

### 运行所有测试

```bash
pytest tests/
```

### 运行特定测试文件

```bash
pytest tests/test_state.py
```

### 运行特定测试函数

```bash
pytest tests/test_state.py::test_agent_state_creation
```

### 查看覆盖率

```bash
pytest --cov=src --cov-report=html tests/
```

## 测试注意事项

1. **依赖要求**：某些测试需要实际的数据库连接（如 Qdrant、Chroma），这些测试会被跳过
2. **Mock 对象**：使用 `tests/utils.py` 中的工具函数创建 mock 对象
3. **环境变量**：测试使用 `conftest.py` 中的 `mock_env_vars` fixture 设置环境变量
4. **集成测试**：集成测试可能需要外部服务，建议在 CI/CD 环境中运行

## 测试覆盖率目标

- 单元测试覆盖率：> 80%
- 集成测试覆盖率：> 60%
- 核心功能覆盖率：100%
