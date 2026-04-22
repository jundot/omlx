# oMLX - Mac 优化的 LLM 推理引擎

> 最后更新：2026-04-16
> 当前版本：v0.3.5
> 上游仓库：https://github.com/jundot/omlx
> **变更日志**：[CHANGELOG.md](CHANGELOG.md)

## 项目概述

oMLX 是专为 Apple Silicon 优化的 LLM 推理引擎，提供连续批处理（continuous batching）和分层 KV 缓存（tiered KV caching）功能。通过菜单栏应用或 CLI 管理，支持本地运行大语言模型。

### 核心特性

- **DFlash 推测解码**：块扩散推测解码引擎，实现 3-4 倍更快的贪婪解码
- **连续批处理**：基于 vLLM 风格的调度器，高效处理并发请求
- **分层 KV 缓存**：热缓存（RAM）+ 冷缓存（SSD）双层架构
- **多模型服务**：同时加载 LLM、VLM、嵌入模型和重排序模型
- **OpenAI 兼容 API**：无需修改即可集成现有工具
- **macOS 原生应用**：PyObjC 菜单栏应用（非 Electron）
- **Web 管理面板**：实时监控、模型管理、聊天界面
- **Claude Code 优化**：专为 Claude Code 等工具优化

### 技术栈

**核心框架：**
- Python 3.10+
- MLX >= 0.31.1（Apple 机器学习框架）
- FastAPI + Uvicorn（Web 服务器）
- mlx-lm（LLM 推理）
- mlx-vlm（视觉语言模型）
- mlx-embeddings（嵌入模型）

**关键依赖：**
```toml
mlx>=0.31.1
mlx-lm @ git+https://github.com/ml-explore/mlx-lm@dcbf6e3
mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@23e1dff
mlx-embeddings @ git+https://github.com/Blaizzy/mlx-embeddings@32981fa
dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx@814c4a1 (v0.1.3)
fastapi>=0.108.0
uvicorn>=0.23.0
transformers>=5.0.0,<5.4.0
```

## 架构设计

### 系统架构

```
FastAPI Server (OpenAI / Anthropic API)
    │
    ├── EnginePool (多模型管理、LRU 驱逐、TTL、手动加载/卸载)
    │   ├── DFlashEngine (推测解码，3-4x 加速)
    │   ├── BatchedEngine (LLM，连续批处理)
    │   ├── VLMEngine (视觉语言模型)
    │   ├── EmbeddingEngine (嵌入模型)
    │   └── RerankerEngine (重排序模型)
    │
    ├── ProcessMemoryEnforcer (总内存限制、TTL 检查)
    │
    ├── Scheduler (FCFS，可配置并发)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack (缓存栈)
        ├── PagedCacheManager (GPU，基于块，CoW，前缀共享)
        ├── Hot Cache (内存热层，写回)
        └── PagedSSDCacheManager (SSD 冷层，safetensors 格式)
```

### 核心组件

#### 1. 引擎池（EnginePool）

**文件：** `omlx/engine_pool.py`

管理多个模型引擎的生命周期：

- **LRU 驱逐**：内存不足时自动驱逐最近最少使用的模型
- **手动加载/卸载**：通过管理面板交互式控制
- **模型固定**：将常用模型固定在内存中
- **每模型 TTL**：设置空闲超时自动卸载
- **进程内存强制**：总内存限制（默认：系统 RAM - 8GB）

#### 2. 调度器（Scheduler）

**文件：** `omlx/scheduler.py`

基于 vLLM 的连续批处理调度器：

- **FCFS 调度**：先来先服务策略
- **可配置并发**：默认最大 8 个并发请求
- **mlx-lm BatchGenerator**：利用 MLX 的批处理能力

#### 3. 缓存系统

**热缓存（RAM）：**
- 频繁访问的块保留在内存中快速访问
- 写回策略

**冷缓存（SSD）：**
- 热缓存满时，块卸载到 SSD
- safetensors 格式存储
- 下次匹配前缀时从磁盘恢复而非重新计算
- 服务器重启后仍然可用

#### 4. API 适配器

**文件：** `omlx/api/`

- **OpenAI 适配器**：`omlx/api/openai_adapter.py`
- **Anthropic 适配器**：`omlx/api/anthropic_adapter.py`
- 支持流式传输、工具调用、视觉输入

#### 5. Web 管理面板

**目录：** `omlx/admin/`

- 实时监控仪表板
- 模型管理（加载/卸载/下载）
- 内置聊天界面
- 性能基准测试
- 每模型设置
- 支持英语、韩语、日语、中文

#### 6. DFlash 推测解码引擎

**文件：** `omlx/engine/dflash.py`

基于块扩散的推测解码引擎：

- **3-4 倍加速**：短/中等长度上下文使用推测解码
- **自动回退**：长上下文（>4096 tokens）自动切换到 BatchedEngine
- **草稿模型**：支持配置草稿模型和量化位数
- **温度采样**：支持温度采样
- **流式输出**：正确的 CJK/UTF-8 处理
- **指标日志**：token/s、接受率、周期数

## 目录结构

```
omlx/
├── __init__.py              # 公共 API 导出
├── _version.py              # 版本号
├── cli.py                   # CLI 入口点
├── config.py                # 配置管理
├── settings.py              # 持久化设置
├── server.py                # FastAPI 服务器主文件
├── scheduler.py             # 连续批处理调度器
├── engine_core.py           # 引擎核心
├── engine_pool.py           # 多模型引擎池
├── model_discovery.py       # 模型自动发现
├── model_settings.py        # 每模型配置
├── memory_monitor.py        # 内存监控
├── process_memory_enforcer.py # 进程内存强制
├── request.py               # 请求/响应数据结构
├── exceptions.py            # 自定义异常
├── logging_config.py        # 日志配置
├── oq.py                    # oQ 量化（TurboQuant KV）
├── turboquant_kv.py         # TurboQuant KV 优化
│
├── api/                     # API 适配器
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   ├── openai_models.py
│   └── ...
├── admin/                   # Web 管理面板
│   ├── templates/
│   ├── static/
│   └── i18n/
├── cache/                   # 缓存系统
│   ├── paged_cache.py
│   ├── paged_ssd_cache.py
│   ├── prefix_cache.py
│   └── stats.py
├── engine/                  # 引擎实现
│   ├── batched_engine.py
│   ├── dflash.py           # DFlash 推测解码引擎
│   ├── vlm_engine.py
│   ├── embedding_engine.py
│   └── reranker_engine.py
├── models/                  # 模型相关
├── adapter/                 # 适配器
├── mcp/                     # MCP (Model Context Protocol) 支持
├── patches/                 # 补丁
├── utils/                   # 工具函数
└── eval/                    # 评估脚本
```

## 开发指南

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/jundot/omlx.git
cd omlx

# 开发安装
pip install -e ".[dev]"

# 运行测试（排除慢速测试）
pytest -m "not slow"

# 包含 MCP 支持
pip install -e ".[mcp]"

# 包含语法约束解码
pip install -e ".[grammar]"

# 包含音频模型支持
pip install -e ".[audio]"
```

### 代码质量工具

项目使用以下工具维护代码质量：

- **Black**：代码格式化（line-length: 88）
- **Ruff**：Linter
- **MyPy**：类型检查
- **Pytest**：测试框架

```bash
# 格式化代码
black omlx tests

# 运行 linter
ruff check omlx tests

# 类型检查
mypy omlx
```

### 运行服务器

```bash
# 基本启动
omlx serve --model-dir ~/models

# 自定义配置
omlx serve --model-dir ~/models \
  --max-model-memory 32GB \
  --max-process-memory 80% \
  --paged-ssd-cache-dir ~/.omlx/cache \
  --hot-cache-max-size 20% \
  --max-concurrent-requests 16

# 使用 MCP 工具
omlx serve --model-dir ~/models --mcp-config mcp.json

# API 密钥认证
omlx serve --model-dir ~/models --api-key your-secret-key
```

### 测试

```bash
# 运行所有测试（排除慢速）
pytest -m "not slow"

# 运行特定测试
pytest tests/test_api_utils.py

# 运行集成测试
pytest tests/integration/

# 运行慢速测试
pytest -m "slow"
```

### macOS 应用构建

```bash
cd packaging

# 完整构建（venvstacks + app bundle + DMG）
python build.py

# 跳过 venvstacks（仅代码更改）
python build.py --skip-venv

# 仅 DMG
python build.py --dmg-only
```

## 工作流程

### 添加新功能

1. 确定功能位置（引擎、API、缓存等）
2. 创建功能分支
3. 实现功能并添加测试
4. 运行测试套件
5. 更新文档（如需要）

### 修复 Bug

1. 在 `tests/` 中创建重现测试
2. 修复问题
3. 验证修复
4. 提交 PR

### 性能优化

1. 使用 `omlx/eval/` 中的基准测试工具
2. 分析性能瓶颈
3. 实现优化
4. 验证性能提升

## 相关资源

- **上游仓库**：https://github.com/jundot/omlx
- **MLX 框架**：https://github.com/ml-explore/mlx
- **mlx-lm**：https://github.com/ml-explore/mlx-lm
- **mlx-vlm**：https://github.com/Blaizzy/mlx-vlm
- **dflash-mlx**：https://github.com/bstnxbt/dflash-mlx
- **vllm-mlx**（起源）：https://github.com/waybarrios/vllm-mlx

## 贡献指南

欢迎贡献！特别欢迎：

- Bug 修复和改进
- 性能优化
- 文档改进

请参阅 [Contributing Guide](docs/CONTRIBUTING.md) 了解详情。

## 许可证

Apache 2.0
