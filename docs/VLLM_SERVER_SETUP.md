# vLLM 服务启动指南

## 🚀 快速开始（使用本地模型）

如果您已经下载了本地模型（如 `models/Qwen--Qwen2.5-7B-Instruct`），可以直接启动：

```bash
# 方式 1：使用快速启动脚本（推荐）
./start_local_vllm.sh

# 方式 2：使用启动脚本
python scripts/launch_vllm_server.py --model models/Qwen--Qwen2.5-7B-Instruct --port 8000
```

服务启动后，插件会自动连接到 `http://localhost:8000/v1`。

### 📥 下载模型

如果还没有下载模型，可以使用下载脚本：

```bash
# 下载 Qwen-7B-Instruct 模型到 models/ 目录
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --output models/
```

下载完成后，模型会保存在 `models/Qwen--Qwen2.5-7B-Instruct/` 目录中（约 15GB）。

---

## 设计理念

**重要**：本插件是优化插件，应该适配用户已有的模型，而不是预设模型。

- ✅ **推荐**：用户指定自己的本地模型路径
- ✅ **支持**：使用 HuggingFace 缓存中的模型（自动检测）
- ✅ **支持**：使用云端 API（如果用户已有 API 服务）
- ❌ **不推荐**：自动从 HuggingFace 下载模型（需要网络，可能失败）

## 模型选择方式

### 方式 1：使用本地模型路径（最推荐）

```bash
# 直接指定本地模型路径
python scripts/launch_vllm_server.py --model /path/to/your/local/model --port 8000

# 或使用相对路径
python scripts/launch_vllm_server.py --model ./models/my-model --port 8000
```

**优点**：
- 不依赖网络
- 启动速度快
- 完全由用户控制

### 方式 2：使用 HuggingFace 缓存中的模型（自动检测）

```bash
# 启动脚本会自动检测 ~/.cache/huggingface/hub/ 中的模型
python scripts/launch_vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

**工作原理**：
1. 首先检查 HuggingFace 缓存目录（`~/.cache/huggingface/hub/`）
2. 如果找到本地缓存，直接使用
3. 如果找不到，才会尝试从 HuggingFace 下载（需要网络）

### 方式 3：使用云端 API（如果用户已有 API 服务）

```bash
# 不需要启动 vLLM，直接配置 VLLM_BASE_URL 指向用户的 API 服务
# 在 .env 中设置：
VLLM_BASE_URL=http://your-api-server:8000/v1
VLLM_MODEL=your-model-name  # 必须与 API 服务中的模型名称一致
```

**优点**：
- 不需要本地 GPU
- 可以使用云端推理服务
- 完全由用户控制

## 模型路径检测优先级

启动脚本会按以下顺序查找本地模型：

1. **指定的路径**（如果是绝对路径或相对路径）
2. **HuggingFace 缓存目录**（`~/.cache/huggingface/hub/`）
   - 格式：`models--org--model_name/snapshots/hash/`
3. **常见模型存储位置**：
   - `~/models/{model_name}`
   - `~/autodl-tmp/models/{model_name}`
   - `/root/models/{model_name}`
   - `/root/autodl-tmp/models/{model_name}`
   - `./models/{model_name}`
4. **如果都找不到**，使用 HuggingFace ID（可能需要下载，需要网络）

## 常见问题

### 问题 1：GPU 占用一直是 0GB，启动超时

**原因**：vLLM 无法连接到 HuggingFace 下载模型（网络不可达）

**解决方案**：

1. **使用本地模型**（最推荐）：
   ```bash
   python scripts/launch_vllm_server.py --model /path/to/your/local/model --port 8000
   ```

2. **配置 HuggingFace 镜像源**：
   ```bash
   # 在 .env 文件中设置
   HF_ENDPOINT=https://hf-mirror.com
   
   # 或临时设置
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **检查网络连接**：
   ```bash
   ping huggingface.co
   # 或
   curl https://huggingface.co
   ```

### 问题 2：duplicate template name 错误

如果启动 vLLM 服务时遇到 `AssertionError: duplicate template name` 错误，这是因为 vLLM 内部使用 `@torch.compile` 导致的已知问题。

**解决方案**：

启动脚本已经自动处理了这个问题，会：
- 设置 `TORCH_COMPILE_DISABLE=1`
- 设置 `TORCHDYNAMO_DISABLE=1`
- 使用 `--enforce-eager` 参数

如果手动启动，需要：

```bash
# 设置环境变量
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_vllm_$$

# 启动服务（使用 --enforce-eager）
vllm serve /path/to/model \
    --port 8000 \
    --enforce-eager \
    --gpu-memory-utilization 0.4 \
    --max-model-len 512
```

## 配置说明

### .env 文件配置

```bash
# vLLM 服务地址（如果使用本地 vLLM 服务，保持默认）
VLLM_BASE_URL=http://localhost:8000/v1

# 模型名称（必须与 vLLM server 启动时指定的模型一致）
# 如果使用本地模型，这里填写模型名称（用于 API 调用）
# 如果使用云端 API，这里填写 API 服务中的模型名称
VLLM_MODEL=

# HuggingFace 镜像源（如果需要从 HuggingFace 下载模型）
HF_ENDPOINT=https://hf-mirror.com
```

### 启动脚本参数

```bash
python scripts/launch_vllm_server.py \
    --model /path/to/model \          # 模型路径（本地路径或 HuggingFace ID）
    --port 8000 \                      # 服务端口
    --gpu-memory-utilization 0.4 \    # GPU 内存使用率
    --max-model-len 512 \              # 最大模型长度
    --enforce-eager                    # 禁用 CUDA graph（避免错误）
```

## 最佳实践

1. **优先使用本地模型**：
   - 下载模型到本地
   - 使用绝对路径启动 vLLM
   - 不依赖网络，启动更快

2. **配置 HuggingFace 镜像源**（如果需要下载）：
   - 在 `.env` 文件中设置 `HF_ENDPOINT`
   - 或使用环境变量 `export HF_ENDPOINT=https://hf-mirror.com`

3. **使用云端 API**（如果已有 API 服务）：
   - 直接配置 `VLLM_BASE_URL` 指向 API 服务
   - 不需要启动本地 vLLM

4. **让用户自由选择**：
   - 不要预设模型路径
   - 提供清晰的文档说明
   - 支持多种部署方式
