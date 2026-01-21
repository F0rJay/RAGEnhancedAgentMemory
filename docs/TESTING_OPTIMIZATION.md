# 测试性能优化指南

## 问题：为什么覆盖率测试很慢？

当使用 `--cov=src` 时，pytest-coverage 会：

1. **分析整个 src 目录**：即使只运行一个测试文件，也会尝试分析所有模块
2. **导入所有模块**：为了跟踪覆盖率，会导入 src 目录下的所有 Python 模块
3. **触发大型库初始化**：
   - `vllm` 库导入需要 **7+ 秒**
   - `transformers` 库导入需要 **2-3 秒**
   - `sentence-transformers` 库导入需要 **1-2 秒**
   - `torch` 等深度学习库初始化很慢

### 为什么会这样？

当 coverage 跟踪 `src/inference/__init__.py` 时，会执行：
```python
from .vllm_inference import VLLMInference  # 触发 vLLM 导入，耗时 7+ 秒
from .baseline_inference import BaselineInference  # 触发 transformers 导入
```

即使测试中根本没有使用这些模块！

## 解决方案

### 1. 使用精确的覆盖率范围（推荐）

只覆盖实际测试的模块，而不是整个 src 目录：

```bash
# 只测试 edge_cases，只覆盖相关模块
pytest tests/test_edge_cases.py \
  --cov=src/memory/short_term \
  --cov=src/memory/routing \
  --cov=src/retrieval/hybrid \
  --cov-report=term

# 测试评估模块，只覆盖评估模块
pytest tests/test_evaluation.py \
  --cov=src/evaluation \
  --cov-report=term

# 测试推理模块，只覆盖推理模块
pytest tests/test_inference.py \
  --cov=src/inference \
  --cov-report=term
```

### 2. 禁用分支覆盖率（更快）

分支覆盖率会让测试更慢：

```bash
pytest tests/test_edge_cases.py \
  --cov=src \
  --cov-branch=false \
  --cov-report=term
```

### 3. 使用并行测试（pytest-xdist）

安装 `pytest-xdist` 后使用：

```bash
pytest tests/ --cov=src -n auto --cov-report=term
```

### 4. 使用 HTML 报告（按需生成）

term 报告每次都会生成，HTML 报告可以按需查看：

```bash
# 快速运行，只生成 HTML（不显示 term）
pytest tests/ --cov=src --cov-report=html --cov-report=

# 查看 HTML 报告
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

### 5. 跳过大型库的导入（测试时）

在测试中使用 Mock 避免导入大型库：

```python
# 在测试文件顶部
from unittest.mock import patch

# Mock 大型库的导入
@patch('src.inference.vllm_inference.VLLM_AVAILABLE', False)
def test_something():
    # 测试不会导入 vLLM
    pass
```

## 性能对比

| 方法 | 时间 | 说明 |
|------|------|------|
| `--cov=src` (完整) | **30-60秒** | 导入所有模块，包括 vLLM |
| `--cov=src/memory` | **5-10秒** | 只覆盖记忆模块 |
| `--cov=src/evaluation` | **2-3秒** | 只覆盖评估模块 |
| 无覆盖率 | **1-2秒** | 最快，但无法查看覆盖率 |

## 推荐工作流程

### 日常开发（快速反馈）
```bash
# 只运行特定测试，不生成覆盖率
pytest tests/test_edge_cases.py -v
```

### 提交前（查看覆盖率）
```bash
# 运行所有测试，生成 HTML 报告
pytest tests/ --cov=src --cov-report=html --cov-report=

# 查看报告
xdg-open htmlcov/index.html
```

### CI/CD（完整报告）
```bash
# 生成所有报告
pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-report=xml
```

## 针对 test_edge_cases.py 的建议

由于 `test_edge_cases.py` 主要测试边界条件，不涉及推理模块，可以：

```bash
# 只覆盖实际使用的模块（推荐）
pytest tests/test_edge_cases.py \
  --cov=src/memory \
  --cov=src/retrieval/hybrid \
  --cov=src/config \
  --cov=src/graph/state \
  --cov-report=term

# 或者完全排除推理模块
pytest tests/test_edge_cases.py \
  --cov=src \
  --cov-config=.coveragerc \
  --cov-report=term
```

创建 `.coveragerc` 文件排除推理模块：

```ini
[run]
omit = 
    */inference/*
    */tests/*
    */venv/*
    */env/*
```
