# 存储效率优化功能实现说明

## ✅ 已实现功能

### 1. 语义相似度去重（阈值 0.95）

**实现位置：** `src/memory/long_term.py`

**功能：**
- 在归档时，检查新内容是否与已有内容语义相似（相似度 ≥ 0.95）
- 使用余弦相似度计算嵌入向量之间的相似度
- 如果相似度达到阈值，跳过重复内容的存储

**关键方法：**
- `_check_semantic_similarity()`: 检查语义相似度
- `_get_recent_contents()`: 获取最近的内容用于相似度检查
- `archive_from_short_term()`: 支持 `semantic_dedup` 参数

**配置参数：**
- `SEMANTIC_DEDUP_THRESHOLD=0.95`（在 `config.py` 中）

### 2. 低价值信息过滤

**实现位置：** `src/core.py`

**功能：**
- 在 `save_context()` 中过滤低价值信息
- 过滤规则：
  - 关键词匹配：过滤"你好"、"谢谢"、"好的"等寒暄
  - 长度过滤：过滤过短的消息（≤2字符）
  - 信息熵过滤：过滤字符种类少且长度短的消息

**关键方法：**
- `_is_low_value_message()`: 判断是否为低价值信息
- `save_context()`: 在保存前进行过滤检查

**配置参数：**
- `ENABLE_LOW_VALUE_FILTER=True`（在 `config.py` 中）

### 3. 时间加权衰减机制

**实现位置：** `src/memory/long_term.py::search()`

**功能：**
- 在检索时应用时间衰减公式：`S = S_semantic * e^(-λ * t)`
- 时间衰减系数 λ（默认 0.001/秒）
- 访问计数强化：`1 + α * log(1 + access_count)`
- 综合分数：`adjusted_score = original_score * time_decay * reinforcement`

**公式说明：**
- **时间衰减**：`e^(-λ * t)`，其中 t 是年龄（秒）
- **访问强化**：`1 + α * log(1 + access_count)`，频繁访问的记忆会被强化
- **综合分数**：`original_score * time_decay * reinforcement`

**配置参数：**
- `ENABLE_TIME_DECAY=True`
- `TIME_DECAY_LAMBDA=0.001`（单位：1/秒）
- `ACCESS_COUNT_ALPHA=0.1`（强化系数）

### 4. 访问计数跟踪

**实现位置：** `src/memory/long_term.py`

**功能：**
- 每次检索时自动增加 `access_count`
- 访问计数影响检索分数的调整（强化机制）
- 在数据库中持久化访问计数

**关键方法：**
- `_update_access_counts()`: 更新数据库中的访问计数
- `search()`: 在检索时更新访问计数

**元数据字段：**
- `access_count`: 访问次数（初始值为 0）

---

## 📊 配置说明

### 环境变量配置

在 `.env` 文件中添加以下配置：

```bash
# 存储优化配置
SEMANTIC_DEDUP_THRESHOLD=0.95        # 语义去重相似度阈值 (0-1)
ENABLE_LOW_VALUE_FILTER=True         # 是否启用低价值信息过滤
ENABLE_TIME_DECAY=True               # 是否启用时间加权衰减
TIME_DECAY_LAMBDA=0.001              # 时间衰减系数 (1/秒)
ACCESS_COUNT_ALPHA=0.1               # 访问计数强化系数
```

### 默认配置

如果未设置环境变量，使用以下默认值：
- `semantic_dedup_threshold`: 0.95
- `enable_low_value_filter`: True
- `enable_time_decay`: True
- `time_decay_lambda`: 0.001
- `access_count_alpha`: 0.1

---

## 🔧 使用方法

### 基本使用

```python
from src.core import RAGEnhancedAgentMemory

# 创建记忆系统（自动启用所有优化功能）
memory = RAGEnhancedAgentMemory(
    vector_db="qdrant",
    session_id="test_session",
)

# 保存对话（自动过滤低价值信息）
memory.save_context(
    inputs={"input": "你好"},
    outputs={"generation": "你好！"}
)  # 低价值信息，会被过滤

memory.save_context(
    inputs={"input": "我的名字是张三"},
    outputs={"generation": "已记录您的名字"}
)  # 有价值信息，会被保存

# 归档到长期记忆（自动进行语义去重）
archived_ids = memory.archive_short_term_to_long_term()

# 搜索记忆（自动应用时间衰减和访问计数强化）
results = memory.search("我的名字是什么？", top_k=5)
```

### 禁用某些功能

```python
# 通过环境变量禁用
import os
os.environ["ENABLE_LOW_VALUE_FILTER"] = "False"
os.environ["ENABLE_TIME_DECAY"] = "False"

# 或者在代码中修改配置
from src.config import get_settings
settings = get_settings()
settings.enable_low_value_filter = False
settings.enable_time_decay = False
```

---

## 📈 预期效果

### 存储降低率

**测试场景：**
- 100 条对话，包含 45 条重复/低价值信息

**预期结果：**
- 对照组（不过滤）：存储 100 条
- 实验组（过滤）：存储 55 条
- **存储降低率：45%** ✅

### 检索精度提升

**时间衰减效果：**
- 新记忆（1小时前）得分更高（时间衰减少）
- 老记忆（1个月前）得分较低（时间衰减多）
- 但频繁访问的记忆会被强化（访问计数机制）

**访问强化效果：**
- 访问 10 次的记忆：强化系数 ≈ 1.30
- 访问 1 次的记忆：强化系数 ≈ 1.00
- 差异：30%

---

## 🧪 测试验证

### 运行存储优化测试

```bash
# 运行完整测试
python scripts/benchmark/storage_optimization_test.py

# 运行单元测试
pytest tests/test_storage_optimization.py -v
```

### 验证指标

1. **存储降低率**
   - 运行对照组和实验组测试
   - 计算：`(对照组 - 实验组) / 对照组 * 100%`
   - 目标：≥ 45%

2. **检索精度**
   - 使用 Ragas 评估 Context Precision 和 Faithfulness
   - 对比过滤前后的评估结果
   - 目标：Context Precision 和 Faithfulness 提升

---

## 🔍 实现细节

### 语义相似度检查

```python
def _check_semantic_similarity(self, new_content, existing_contents, threshold=0.95):
    # 1. 计算新内容的嵌入向量
    # 2. 计算与已有内容的相似度（余弦相似度）
    # 3. 如果相似度 >= threshold，返回 True（表示相似）
    # 4. 否则返回 False
```

### 低价值信息过滤

```python
def _is_low_value_message(self, message):
    # 1. 检查是否完全是低价值关键词
    # 2. 检查消息长度（≤2字符）
    # 3. 检查信息熵（字符种类和长度）
    # 4. 返回 True/False
```

### 时间加权衰减

```python
def search(self, query, ...):
    # 1. 获取原始相似度分数
    # 2. 计算时间衰减：e^(-λ * age_seconds)
    # 3. 计算访问强化：1 + α * log(1 + access_count)
    # 4. 综合分数：original * time_decay * reinforcement
    # 5. 重新排序
    # 6. 更新访问计数
```

---

## ⚠️ 注意事项

### 1. NumPy 依赖

语义相似度检查需要 NumPy。如果未安装，系统会回退到哈希去重：
```bash
pip install numpy
```

### 2. 性能考虑

- 语义相似度检查需要计算嵌入向量，会有一定延迟
- 建议限制 `_get_recent_contents()` 的 `limit` 参数（默认 100）
- 大批量归档时，可以考虑分批处理

### 3. 配置调优

**时间衰减系数 λ：**
- 较小值（如 0.0001）：衰减慢，老记忆影响时间长
- 较大值（如 0.01）：衰减快，新记忆优先度高
- 默认 0.001：平衡新老记忆

**访问计数强化系数 α：**
- 较小值（如 0.05）：强化效果弱
- 较大值（如 0.2）：强化效果强
- 默认 0.1：适中强化

### 4. 数据库兼容性

- **Qdrant**: 完全支持访问计数更新（`set_payload`）
- **Chroma**: 需要更新 metadata，可能需要额外配置

---

## 🎯 后续优化方向

### 1. 更智能的低价值过滤
- 基于分类器（BERT-based）的智能过滤
- 信息熵的精确计算
- 上下文感知的过滤（某些寒暄在特定场景下是有价值的）

### 2. 自适应阈值调整
- 根据实际使用情况自动调整相似度阈值
- 学习用户行为模式

### 3. 批量优化
- 批量语义相似度检查
- 并行处理优化

### 4. Ragas 集成
- 集成 Ragas 评估框架
- 自动评估过滤效果

---

**最后更新：** 2026-01-22  
**维护者：** 项目团队
