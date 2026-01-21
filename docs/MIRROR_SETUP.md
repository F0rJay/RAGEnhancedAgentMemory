# 镜像源配置指南

本项目支持配置 pip 和 HuggingFace 镜像源，以加速依赖包安装和模型下载。

## 快速开始

运行镜像源配置脚本：

```bash
bash scripts/setup_mirrors.sh
```

按照提示选择镜像源即可完成配置。

## 配置选项

### 1. pip 镜像源

配置 pip 镜像源可以加速 Python 包的安装。

#### 可用的镜像源：

- **清华大学**（推荐）：`https://pypi.tuna.tsinghua.edu.cn/simple`
- **阿里云**：`https://mirrors.aliyun.com/pypi/simple/`
- **中科大**：`https://pypi.mirrors.ustc.edu.cn/simple/`
- **豆瓣**：`https://pypi.douban.com/simple/`
- **官方源**：`https://pypi.org/simple`

#### 手动配置

配置文件位置：`~/.pip/pip.conf`（Linux/Mac）或 `%APPDATA%\pip\pip.ini`（Windows）

**Linux/Mac：**
```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

**Windows：**
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

### 2. HuggingFace 镜像源

配置 HuggingFace 镜像源可以加速模型下载（包括嵌入模型、LLM 模型等）。

#### 可用的镜像源：

- **HF-Mirror**（推荐，国内）：`https://hf-mirror.com`
- **官方源**：`https://huggingface.co`

#### 配置方法

在项目的 `.env` 文件中添加：

```env
HF_ENDPOINT=https://hf-mirror.com
```

或者在 `env.example` 中配置，然后复制到 `.env`。

#### 环境变量方式

也可以直接设置环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或者在启动 Python 脚本前：

```bash
HF_ENDPOINT=https://hf-mirror.com python your_script.py
```

## 使用场景

### 场景 1：安装项目依赖

配置好 pip 镜像源后，安装依赖时会自动使用镜像：

```bash
pip install -r requirements.txt
```

### 场景 2：下载嵌入模型

配置好 HuggingFace 镜像源后，下载模型时会自动使用镜像：

```bash
python scripts/download_embedding_model.py
```

### 场景 3：使用 vLLM 加载模型

在代码中使用 vLLM 或基线推理时，如果配置了 `HF_ENDPOINT`，会自动使用镜像：

```python
from src.inference.vllm_inference import VLLMInference

# 会自动使用 .env 中配置的 HF_ENDPOINT
inference = VLLMInference()
```

## 验证配置

### 验证 pip 镜像源

```bash
pip config list
```

应该看到类似输出：
```
global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'
```

### 验证 HuggingFace 镜像源

检查 `.env` 文件：

```bash
grep HF_ENDPOINT .env
```

或在 Python 中：

```python
import os
print(os.environ.get("HF_ENDPOINT", "未设置（使用官方源）"))
```

## 常见问题

### Q1: 配置镜像源后仍然下载很慢？

- 检查网络连接是否正常
- 尝试切换到其他镜像源
- 某些镜像源可能在特定时间段较慢，可以尝试使用代理

### Q2: HuggingFace 镜像源不生效？

- 确保在 `.env` 文件中正确配置了 `HF_ENDPOINT`
- 确保代码中使用了 `python-dotenv` 加载 `.env` 文件
- 检查是否有其他环境变量覆盖了设置

### Q3: 如何临时使用官方源？

对于 pip，可以直接指定：

```bash
pip install -i https://pypi.org/simple package_name
```

对于 HuggingFace，可以临时取消设置：

```bash
unset HF_ENDPOINT
python your_script.py
```

### Q4: 镜像源配置会影响哪些操作？

- **pip 镜像源**：影响所有 `pip install` 操作
- **HuggingFace 镜像源**：影响所有通过 HuggingFace Hub 下载的模型，包括：
  - `sentence-transformers` 模型
  - `transformers` 模型
  - `huggingface_hub.snapshot_download()` 下载的模型
  - vLLM 加载的 HuggingFace 模型

## 技术说明

### pip 镜像源原理

pip 会读取配置文件 `~/.pip/pip.conf`（Linux/Mac）或 `%APPDATA%\pip\pip.ini`（Windows），使用配置的 `index-url` 作为包下载源。

### HuggingFace 镜像源原理

HuggingFace 相关的库（如 `huggingface_hub`、`transformers`、`sentence-transformers`）会检查 `HF_ENDPOINT` 环境变量。如果设置了该变量，所有对 `huggingface.co` 的请求会被重定向到镜像源。

## 参考链接

- [清华大学 pip 镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [HF-Mirror 镜像站](https://hf-mirror.com/)
- [HuggingFace 文档](https://huggingface.co/docs/hub/models-downloading)
