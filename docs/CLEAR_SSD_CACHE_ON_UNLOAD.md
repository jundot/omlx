# 自动清理 SSD Cache 功能

## 概述

当进行多模型基准测评时，每个模型测评结束后，SSD cache 默认不会被清理。这可能导致后续模型加载时出现 "SSD write queue full" 警告。

本功能允许在模型卸载时自动清理 SSD cache，适用于基准测评等不需要 cache 持久化的场景。

## 使用方法

### 1. CLI 参数

```bash
# 启用自动清理
omlx serve --model-dir ~/models --paged-ssd-cache-clear-on-unload

# 结合其他 cache 参数
omlx serve \
  --model-dir ~/models \
  --paged-ssd-cache-dir ~/.omlx/cache \
  --paged-ssd-cache-max-size 100GB \
  --paged-ssd-cache-clear-on-unload
```

### 2. 环境变量

```bash
export OMLX_PAGED_SSD_CACHE_CLEAR_ON_UNLOAD=true
omlx serve --model-dir ~/models
```

### 3. 配置文件

在 `~/.omlx/settings.json` 中添加：

```json
{
  "paged_ssd_cache": {
    "enabled": true,
    "cache_dir": "~/.omlx/cache",
    "max_size": "100GB",
    "clear_on_unload": true
  }
}
```

## 使用场景

### 推荐启用的场景

- **基准测评**：多个模型依次进行性能测试
- **模型比较**：需要在相同条件下测试不同模型
- **一次性任务**：不需要跨会话的 cache 持久化

### 推荐禁用的场景（默认）

- **生产环境**：需要 cache 持久化以提高响应速度
- **模型重用**：频繁加载/卸载相同模型
- **长上下文应用**：cache 命中率很重要

## 工作原理

当 `clear_ssd_cache_on_unload` 启用时：

1. 模型卸载前，系统会清理该模型的 SSD cache
2. 清理在引擎停止之前执行，避免 cache 干扰
3. 清理操作是异步的，不会阻塞模型卸载流程
4. 如果清理失败，会记录警告但不会影响模型卸载

## 性能影响

- **内存占用**：无额外内存占用
- **卸载时间**：可能增加 1-2 秒（取决于 cache 大小）
- **加载时间**：后续模型加载不会受到旧 cache 影响

## 日志输出

启用后会看到以下日志：

```
INFO - Unloading model: llama-3b (immediate abort)
INFO - Cleared SSD cache for llama-3b: 127 files
INFO - Unloaded model: llama-3b, freed=3.2GB (expected>=3.0GB), active_memory: 8.5GB (settled)
```

## 故障排查

### 警告：Failed to clear SSD cache

如果看到此警告，通常是因为：
- 模型没有使用 SSD cache
- SSD cache manager 未初始化
- 权限问题

这些警告不会影响模型卸载，可以安全忽略。

### 仍然出现 "SSD write queue full"

如果启用此功能后仍然看到该警告：
1. 确认功能已启用（检查日志中的 "Cleared SSD cache" 消息）
2. 检查是否有多个模型同时运行
3. 考虑增大 SSD cache 大小（`--paged-ssd-cache-max-size`）

## 相关配置

- `--paged-ssd-cache-dir`：SSD cache 存储目录
- `--paged-ssd-cache-max-size`：SSD cache 最大大小
- `--hot-cache-max-size`：内存热缓存大小
- `--no-cache`：完全禁用 SSD cache

## 示例：基准测评脚本

```python
import asyncio
import httpx

async def benchmark_with_auto_clear():
    """基准测评时自动清理 cache"""

    models = [
        "mlx-community/Llama-3.2-3B-Instruct",
        "mlx-community/Qwen2.5-7B-Instruct",
        "mlx-community/Mistral-7B-Instruct",
    ]

    async with httpx.AsyncClient() as client:
        for model in models:
            print(f"测试模型: {model}")

            # 加载模型（通过 API）
            await client.post("/api/load", json={"model": model})

            # 运行基准测试
            await run_benchmark(model)

            # 卸载模型
            await client.post("/api/unload", json={"model": model})

            # SSD cache 会自动清理（如果启用了 --paged-ssd-cache-clear-on-unload）
            print(f"完成测试: {model}")

if __name__ == "__main__":
    asyncio.run(benchmark_with_auto_clear())
```

## 技术细节

### 实现位置

- **配置**：`omlx/config.py`、`omlx/scheduler.py`
- **CLI**：`omlx/cli.py`
- **执行**：`omlx/engine_pool.py::_unload_engine()`

### 清理时机

```python
# 在模型卸载流程中的位置：
async def _unload_engine(self, model_id: str) -> None:
    # 1. 清理 SSD cache（如果启用）
    if self._scheduler_config.clear_ssd_cache_on_unload:
        ssd_manager.clear()

    # 2. 停止引擎
    await entry.engine.stop()

    # 3. 内存清理
    mx.clear_cache()
    # ...
```

### 错误处理

```python
try:
    # 安全地访问嵌套属性
    async_core = getattr(entry.engine, "_engine", None)
    if async_core is not None:
        core = getattr(async_core, "engine", None)
        # ...
        ssd_manager.clear()
except Exception as e:
    # 记录警告但不中断卸载流程
    logger.warning(f"Failed to clear SSD cache: {e}")
```

## 版本历史

- **v0.3.8**：首次引入此功能
