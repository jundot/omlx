# Chinese (zh-Hans) i18n localization for oMLX

## Summary

This PR adds complete Simplified Chinese localization to oMLX, covering both the **Web Admin Dashboard** and the **macOS native SwiftUI interface**.

## What's Changed

### 1. Web Admin Dashboard (`omlx/admin/i18n/`)

**File:** `omlx/admin/i18n/zh.json`
- Completed translation of **842+ i18n keys**
- All model settings descriptions now display in Chinese (TurboQuant, SpecPrefill, DFlash, Lightning MTP, VLM MTP)
- Previously hardcoded English in templates has been translated or extracted to i18n keys

**Template patches (hardcoded English → Chinese):**
- `dashboard/_modal_model_settings.html` — TurboQuant KV 缓存 / SpecPrefill（稀疏预填充）/ DFlash（扩散推测解码）
- `dashboard/_bench.html` — Community Benchmark Upload → 社区基准测试上传
- `dashboard/_models.html` — About oQ Quantization → 关于 oQ 量化
- `dashboard/_settings.html` — Network → 网络 / HTTP Proxy → HTTP 代理
- `dashboard/_status.html` — Runtime Cache Observability → 运行时缓存可观测性
- `chat.html` — Thinking → 思考 / MODEL → 模型 / PROFILE → 配置档

### 2. macOS Native SwiftUI Interface

**New file:** `apps/omlx-mac/Resources/zh-Hans.lproj/Localizable.strings`
- Added **926 translated keys** for full Simplified Chinese support
- Covers all screens: Appearance, Integrations, Security, Server, Models, Network, Performance, Logs, Quantization, Benchmarks
- Example translations:
  - `sidebar.appearance` → `外观`
  - `sidebar.integrations` → `集成`
  - `sidebar.security` → `安全`
  - `appearance.row.refresh_interval` → `刷新间隔`
  - `server.hero.restart` → `重启`
  - `server.hero.stop` → `停止`

**Swift source changes:**
- Modified Swift files to use `String(localized: "key", defaultValue: "...")` pattern for dynamic locale switching
- All previously hardcoded English strings now support localization

### 3. Translation Rules

| Type | Rule | Example |
|------|------|---------|
| Regular UI text | Translate to Chinese | `Save` → `保存` |
| Technical terms (abbreviations) | English + Chinese in parentheses | `MCP` → `MCP（模型上下文协议）` |
| Brand / Product names | Keep original English | `oMLX`, `OpenAI`, `Qwen` |
| Code / Parameters / Placeholders | Keep original English | `enable_thinking`, `hf_...` |
| Model names | Keep original English | `DeepSeek-V3`, `Qwen3`, `Llama-3` |

### 4. Test Coverage

- [x] Web dashboard: All pages display correct Chinese text
- [x] macOS sidebar: Appearance、Integrations、Security、About oMLX translated
- [x] Appearance screen: Refresh Interval、Show Dock Icon、Live Activity、Average Session Activity、All Time Activity
- [x] Server screen: Running、Restart、Stop、Start Server
- [x] Models screen: Load、Unload、Favorites、Library
- [x] All 842+ i18n keys have corresponding Chinese translations
- [x] All 926 macOS Locale keys have corresponding Chinese translations

## Files Changed (Summary)

### Web i18n
- `omlx/admin/i18n/zh.json` (+842 translation keys)

### Template patches
- `omlx/admin/templates/dashboard/_modal_model_settings.html`
- `omlx/admin/templates/dashboard/_bench.html`
- `omlx/admin/templates/dashboard/_models.html`
- `omlx/admin/templates/dashboard/_settings.html`
- `omlx/admin/templates/dashboard/_status.html`
- `omlx/admin/templates/chat.html`

### macOS Native
- `apps/omlx-mac/Resources/zh-Hans.lproj/Localizable.strings` (new, 926 keys)
- `apps/omlx-mac/Sources/AppView/Screens/AppearanceScreen.swift` (localized)
- `apps/omlx-mac/Sources/AppView/Screens/SecurityScreen.swift` (localized)
- `apps/omlx-mac/Sources/AppView/Screens/IntegrationsScreen.swift` (localized)
- Additional SwiftUI screen files with hardcoded strings now localized

## Note on Localizable.strings Format

The `Localizable.strings` file is provided in **XML plist format** for GitHub diff/reviewability. To use it in the app bundle, convert back to binary format using macOS `plutil`:

```bash
plutil -convert binary1 -o Localizable.strings Localizable.strings.plist
```

## Screenshots

*(Note: Please add screenshots showing Chinese UI in both Web Dashboard and macOS app)*

## Checklist

- [x] All regular UI text is translated to Chinese
- [x] Technical abbreviations follow "English（中文）" format
- [x] Brand names, code, and placeholders are kept in original English
- [x] Web admin dashboard fully localized
- [x] macOS native interface fully localized
- [x] No placeholder text or untranslated strings remain
- [ ] Add Traditional Chinese (zh-TW) support (future enhancement)
- [ ] Add CI test to ensure all locale keys are translated (future enhancement)

## Notes for Maintainers

1. **Web backend:** The `zh.json` file should be merged as-is. It contains complete translations for all existing i18n keys.

2. **macOS native:** The new `zh-Hans.lproj/Localizable.strings` file contains 926 keys. This is a complete translation set. If new keys are added in future releases, they will fall back to the English `defaultValue` in Swift code until translated.

3. **Future updates:** When new UI strings are added:
   - Web: Add new keys to `omlx/admin/i18n/en.json` first, then translate to `zh.json`
   - macOS: Use `String(localized: "new.key", defaultValue: "English text")` so Chinese users see translations once added to `Localizable.strings`

---

Closes #(issue_number if applicable)
