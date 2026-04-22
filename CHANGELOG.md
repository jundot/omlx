# oMLX 变更日志

> 最后更新：2026-04-16

本文档记录 omlx 项目的所有重要变更和提交历史。

## 最新提交（2026-04-16）

1. **3af2e1e** - formula: bump to v0.3.5
2. **ed7a46a** - fix: update dflash-mlx pin to valid commit (8e1df22 was force-pushed away)
3. **92aab5c** - fix: preserve tool_calls/tool_responses in VLM message formatting
4. **58b3ca5** - fix(dflash): lazy fallback to prevent double model loading
5. **3d1ac71** - fix(admin): hide reasoning parser dropdown when xgrammar is not installed (#774)
6. **dd1093d** - deps: bump dflash-mlx to v0.1.3 (814c4a1)
7. **a045b6a** - fix: deep-copy tokenizer to prevent "Already borrowed" under concurrent load
8. **b10eaad** - fix: scale settle_tolerance with model size for large models (#768)
9. **6731a2c** - fix: revert TurboQuant KV conversion in external prefill (#771)
10. **b125bfc** - formula: bump to v0.3.5-rc1
11. **62c8f5f** - fix: remove false-positive RotatingKVCache stale offset warning
12. **fd55ab3** - bump version to 0.3.5
13. **95ea04a** - fix: address PR review — safer restore, consistent model detection
14. **fec8dc0** - fix: rename colliding params for Gemma 4
15. **6fa0a77** - fix: enrich Gemma 4 tool parameter descriptions
16. **8baa0af** - docs: add DFlash-MLX integration guide
17. **6b1029c** - fix(benchmark): skip batch test for DFlashEngine (#759)
18. **aca2de1** - fix: remove VoiceDesign hasattr routing from TTS engine
19. **b2b03b3** - test: add unit tests for cache probe endpoint (#720)
20. **3b4feb9** - feat(admin): add cache probe endpoint for prompt prefix lookup (#720)
21. **ccbf09e** - feat: allow skip API key verification on any host
22. **a34615e** - fix: verify actual Metal memory release before updating pool tracking
23. **5373256** - fix: add parse_json_output to responses API and streaming endpoints
24. **f378c9b** - fix(packaging): skip pip-stripped interpreters in _find_target_python
25. **4928761** - fix: cache inspect.signature for embedding input remapping
26. **1b60a90** - fix: enable_thinking toggle precedence + add tests
27. **e6f2626** - feat: add dedicated Enable Thinking toggle with auto-detection (#748)
28. **a4d17be** - fix(app): add reopen and termination delegate methods (#725)
29. **ed289d7** - bump mlx-vlm to 3472132, remove dead gemma4 patch, fix 14 stale tests

## 重要变更

### v0.3.5 (2026-04-16)

- **版本发布**：正式发布 v0.3.5 版本
- **DFlash-MLX 依赖更新**：升级到 v0.1.3 (814c4a1)，修复之前 force-pushed 的提交引用
- **VLM 消息格式化**：修复 VLM 消息格式化中 tool_calls 和 tool_responses 的保留问题
- **DFlash 懒加载回退**：实现懒回退机制，防止双模型加载问题
- **管理面板改进**：当 xgrammar 未安装时隐藏推理解析器下拉框 (#774)
- **Tokenizer 并发安全**：深度复制 tokenizer 以防止并发加载时的 "Already borrowed" 错误
- **大型模型内存管理**：针对 40GB 以上模型，settle_tolerance 从固定 2GB 改为根据模型大小动态缩放（约 5%），解决虚假 settle barrier 超时和不必要的紧急回收问题 (#768)
- **TurboQuant KV 修复**：回滚外部 prefill 中的 TurboQuantKV 转换，因为 TurboQuantKVCache 缺少 merge()/maybe_trim_front() 方法，导致量化 KV 模型（Llama-4-Scout、Nemotron-3-Super-120B）出现垃圾输出和 SIGABRT (#771)
- **版本更新**：升级到 v0.3.5-rc1（候选发布版本）
- **RotatingKVCache 优化**：移除虚假的 stale offset 警告，改进 specprefill system_end 计算
- **Gemma 4 支持**：修复参数冲突问题，丰富工具参数描述
- **DFlash-MLX 集成指南**：新增完整的集成文档
- **缓存探测端点**：管理面板新增提示前缀查找功能 (#720)
- **思维模式切换**：新增专用 Enable Thinking 切换按钮，支持自动检测 (#748)
- **API 密钥验证**：允许在任何主机上跳过 API 密钥验证
- **Metal 内存管理**：验证实际的 Metal 内存释放后再更新池跟踪
- **JSON 输出解析**：为响应 API 和流式端点添加 parse_json_output
- **mlx-vlm 更新**：升级到 3472132 版本，修复 14 个过时测试
- **应用委托方法**：添加重新打开和终止委托方法 (#725)
- **TurboQuantKV 优化**：
  - 添加 TurboQuantKVCache.merge monkey-patch 支持
  - 改进 mRoPE 实现，使用 PromptProcessingBatch.prompt
  - 修复 burst-completion bug (#557)
  - 添加 _apply_turboquant_kv_empty 方法
