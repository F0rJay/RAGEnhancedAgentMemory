# 📁 项目文件结构

```
RAGEnhancedAgentMemory/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── config.py                 # 配置管理（Pydantic）
│   ├── core.py                   # 核心系统集成
│   ├── core_example.py           # 使用示例
│   ├── memory/                   # 记忆管理模块
│   │   ├── __init__.py
│   │   ├── short_term.py         # 短期记忆
│   │   ├── long_term.py          # 长期记忆（向量数据库）
│   │   └── routing.py            # 自适应路由
│   ├── retrieval/                # 检索模块
│   │   ├── __init__.py
│   │   ├── hybrid.py             # 混合检索
│   │   └── reranker.py           # 重排序
│   ├── graph/                    # LangGraph 集成
│   │   ├── __init__.py
│   │   └── state.py              # Agent 状态定义
│   └── evaluation/               # 评估模块
│       ├── __init__.py
│       ├── ragas_eval.py         # Ragas 评估
│       └── needle_test.py        # Needle-in-a-Haystack 测试
│
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── conftest.py               # pytest 配置
│   ├── utils.py                  # 测试工具
│   ├── README.md                 # 测试说明
│   ├── test_core.py              # 核心模块测试
│   ├── test_config.py            # 配置测试
│   ├── test_evaluation.py        # 评估模块测试
│   ├── test_graph.py             # Graph 测试
│   ├── test_integration.py       # 集成测试
│   ├── test_long_term_memory.py  # 长期记忆测试
│   ├── test_long_term_memory_full.py  # 完整长期记忆测试
│   ├── test_retrieval.py         # 检索测试
│   ├── test_routing.py           # 路由测试
│   ├── test_short_term_memory.py # 短期记忆测试
│   ├── test_state.py             # 状态测试
│   └── test_storage_optimization.py  # 存储优化测试
│
├── scripts/                      # 脚本目录
│   ├── benchmark/                # 基准测试
│   │   ├── baseline_agents.py    # 基线系统实现
│   │   ├── baseline_test.py      # 基线测试
│   │   ├── conversation_dataset.py  # 对话数据集生成
│   │   ├── enhanced_test.py      # 增强系统测试
│   │   ├── long_conversation_test.py  # 长对话对比测试
│   │   └── storage_optimization_test.py  # 存储优化测试
│   ├── benchmark_results/        # 测试结果
│   │   ├── baseline_results.json
│   │   ├── enhanced_results.json
│   │   ├── comparison_report.json
│   │   └── storage_optimization/  # 存储优化测试结果
│   ├── verify_functionality.py   # 功能验证脚本
│   └── download_embedding_model.py  # 模型下载脚本
│
├── docs/                         # 文档目录
│   ├── BASELINE_ANALYSIS.md      # 基线系统分析
│   ├── PROJECT_STATUS.md         # 项目状态报告（最新）
│   ├── STORAGE_OPTIMIZATION_REPORT.md  # 存储优化技术报告
│   ├── STORAGE_OPTIMIZATION_IMPLEMENTATION.md  # 存储优化实现
│   ├── STORAGE_OPTIMIZATION_TESTING.md  # 存储优化测试
│   ├── ENV_CONFIG.md             # 环境变量配置说明
│   ├── DOCKER_SETUP.md           # Docker 配置指南
│   ├── POSTGRESQL_SETUP.md       # PostgreSQL 配置指南
│   └── USAGE.md                  # 使用文档
│
├── .gitignore                    # Git 忽略文件
├── LICENSE                       # MIT 许可证
├── README.md                     # 项目主文档
├── QUICKSTART.md                 # 快速开始指南
├── env.example                   # 环境变量示例
├── requirements.txt              # Python 依赖
├── pyproject.toml                # 项目配置
├── py.typed                      # 类型提示标记
├── Untitled.pdf                  # 项目愿景文档
│
└── (运行时生成的目录，已添加到 .gitignore)
    ├── __pycache__/              # Python 缓存
    ├── .pytest_cache/            # pytest 缓存
    ├── htmlcov/                  # 测试覆盖率报告
    ├── .coverage                 # coverage 数据
    ├── checkpoints/              # LangGraph 检查点
    ├── logs/                     # 日志文件
    ├── models/                   # 下载的模型文件
    └── qdrant_db/                # Qdrant 向量数据库
```

## 📝 目录说明

### 核心目录

- **`src/`**: 源代码目录，包含所有核心模块
- **`tests/`**: 测试代码，覆盖率达到 73%（核心模块 85%+）
- **`scripts/`**: 脚本和工具，包括基准测试和数据生成
- **`docs/`**: 项目文档，包括技术报告和分析

### 运行时目录（已忽略）

以下目录在 `.gitignore` 中，不会被提交到 Git：

- **`__pycache__/`**: Python 字节码缓存
- **`.pytest_cache/`**: pytest 测试缓存
- **`htmlcov/`**: 测试覆盖率 HTML 报告
- **`.coverage`**: coverage.py 数据文件
- **`checkpoints/`**: LangGraph 会话检查点
- **`logs/`**: 应用日志文件
- **`models/`**: 下载的模型文件（大小约 28GB）
- **`qdrant_db/`**: Qdrant 本地嵌入式数据库（约 7.2MB）


