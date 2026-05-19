# oMLX GUI Bundle Hack — 让上游 `/Applications/oMLX.app` 启动我们 fork 代码

**适用范围**：m2max + m5max
**应用日期**：2026-05-13
**前提**：用户决定**永远不升级** `/Applications/oMLX.app`（升级会冲掉 patch）。我们也已转为长期维护自己的 oMLX fork（在 `panwudi/omlx feat/gemma4-dflash` 分支，将来可能 promote 成 `flyto-main`）。

## 为什么 hack 而不是另写 launchd

候选方案对比（讨论于 2026-05-13 session）：

| 方案 | 优点 | 缺点 |
|---|---|---|
| **A. 各自写 launchd plist 跑命令行 fork server** | 不动 app bundle，干净 | 失去 GUI menubar 状态显示。曾试过 SwiftBar 手写脚本（ds4-server 那套），用户嫌"很丑很恶心" |
| **B. 自己 fork 整个 .app 包** | 长期最干净，可发布 | 1-2 天 build 工作量；fork 公开化前不值得 |
| **C. Hack 现有 bundle，把 server subprocess 路由到我们 venv**（本方案） | 一晚上活；GUI menubar 保留；解决重启自启问题 | 动了 app bundle，需要 ad-hoc 重签；app 升级会冲掉 |

选 C：用户明确不升级 app，副作用消失；菜单栏视觉质量保留。

## 启动链路

```
机器开机
  ↓
launchctl gui/501/com.omlx.app  ←  ~/Library/LaunchAgents/com.omlx.app.plist
  ↓
open -a /Applications/oMLX.app
  ↓
oMLX 二进制（带 ad-hoc 签名）启动
  ↓
omlx_app/__main__.py → omlx_app/app.py 起 menubar
  ↓
ServerConfig.start_server_on_launch=True → server_manager.start_server()
  ↓
[FLYTO PATCH] get_bundled_python() → 返回 ~/Code/omlx/.venv/bin/python
[FLYTO PATCH] env.pop("PYTHONHOME") + env.pop("PYTHONPATH")
  ↓
subprocess.Popen([fork_venv_python, "-m", "omlx.cli", "serve", ...])
  ↓
omlx 0.3.9.dev2 (我们 fork HEAD 18b4df6) 跑在 port 8000
```

## 三处 patch

### 1. `Contents/Resources/omlx_app/server_manager.py`

**get_bundled_python()** —— 把 server subprocess 的 Python 切到 fork venv：

```python
def get_bundled_python() -> str:
    """Get the path to the bundled Python executable."""
    # FLYTO PATCH: route server subprocess through our fork venv
    flyto_venv = "/Users/yuanwei/Code/omlx/.venv/bin/python"
    if Path(flyto_venv).exists():
        return flyto_venv
    exe = Path(sys.executable)
    # ... 原 fallback 逻辑保留 ...
```

**spawn 前的 env scrub** —— bundle launcher 把 `PYTHONHOME` / `PYTHONPATH` 注入到 env，会把 fork venv 的 python 强制走 bundle cpython-3.11 site-packages，导致 `ModuleNotFoundError: No module named 'encodings'` 致命错误。Scrub 这两个变量：

```python
env = os.environ.copy()
if "Code/omlx/.venv" in python_exe:
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
homebrew_paths = [...]  # 原代码
```

### 2. `Contents/Resources/omlx_app/updater.py`

**禁用自动更新**（保险，虽然用户已说不升级）：

```python
def start(self):
    """Start the update process in a background thread."""
    # FLYTO PATCH: auto-update disabled (we maintain our own fork)
    return
```

### 3. `Contents/Resources/omlx_app/server_manager.py` 第二次 patch — Session 复用（2026-05-13 加，upstream PR #1211）

GUI menubar 每 ~5 秒 health check 一次，**原版每次新建 `requests.Session()`** → TCP 连接打开 → 关闭 → 进 `TIME_WAIT` ~2 分钟。24/7 跑下来 ephemeral port range 耗尽，整机所有 app 出 TCP 失败，只能重启 Mac。

修法：`__init__` 建一次 Session 反复用：

```python
# in __init__:
self._health_session = requests.Session()
self._health_session.trust_env = False

# in check_health:
response = self._health_session.get(self._get_health_url(), timeout=2)
```

源码 cherry-pick 在 `panwudi/main` commit `1cdfbac`（保留作者 arthware-dev 署名）。bundle 上同步 live-patch。

### 4. `Contents/Resources/omlx_app/app.py` — NSStatusItem occlusion 检测 Tahoe 兼容（2026-05-13 加，**我们原创**）

GUI 启动后 3 秒做一次 "menubar icon visibility" 自检（`_is_status_item_hidden`），如果判定 hidden 会弹 "menubar not showing — auto-fix?" 对话框。原版用位测试 `occlusion & NSWindowOcclusionStateVisible (0x2)`。

**macOS 26 Tahoe 把 visible 状态搬到新的 bit `0x2000`**，0x2 不再被 set。于是 Tahoe 上每次启动稳报 `status item hidden: occlusion=0x2000`，弹假警报。Sequoia 还是 0x2 工作所以无声。

修法：把 `occlusion & 0x2` 改成 `occlusion != 0`（任何非零都视为 WindowServer 在 track 这个 window，等价于可见）。同时兼容 Sequoia (0x2) 和 Tahoe (0x2000)，cleared-to-zero 仍是真 hidden。

```python
# in _is_status_item_hidden():
occlusion_visible = occlusion != 0   # was: bool(occlusion & 0x2)
```

源码在 `panwudi/main` commit `9ef09e7`（我们自己写的，作者 yuanwei）。这条上游没人发现，**适合将来 fork 公开化后发 PR 给 jundot/omlx 当真贡献**。

### 5. `~/Library/Application Support/oMLX/config.json`

确保 GUI 启动时 auto-spawn server 且监听 8000：

```json
{
  "base_path": "/Users/yuanwei/.omlx",
  "port": 8000,
  "model_dir": "",
  "launch_at_login": true,
  "start_server_on_launch": true
}
```

`launch_at_login=true` 让 GUI 出现在用户登录项；`start_server_on_launch=true` 让 GUI 启动后立即 spawn server。

### 6. `~/Library/LaunchAgents/com.omlx.app.plist`

通用 launchd plist，机器开机即拉起 GUI（GUI 再 spawn server）：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.omlx.app</string>
    <key>ProgramArguments</key>
    <array>
        <string>open</string>
        <string>-a</string>
        <string>/Applications/oMLX.app</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

### 7. Ad-hoc 重签

改 bundle 内容会破坏原 code signature，macOS launchd 拒绝起 unsigned app：

```bash
codesign --force --deep --sign - /Applications/oMLX.app
```

`-s -` 是 ad-hoc 签名，satisfy launchd 但不验证身份。`--deep` 递归签 framework bundle。

## 备份策略

每个被 patch 的文件就地存 `.flyto-bak`：

```
/Applications/oMLX.app/Contents/Resources/omlx_app/server_manager.py.flyto-bak
/Applications/oMLX.app/Contents/Resources/omlx_app/updater.py.flyto-bak
```

**不另存**：app 升级会同时冲掉 patch 和 backup，所以另存意义不大。用户决定不升级 app，备份就是回退手段。

## 回退步骤

如某天要恢复原 bundle 行为：

```bash
cd /Applications/oMLX.app/Contents/Resources/omlx_app/
mv server_manager.py.flyto-bak server_manager.py
mv updater.py.flyto-bak updater.py
rm -rf __pycache__
codesign --force --deep --sign - /Applications/oMLX.app
# 重启 GUI
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.omlx.app.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.omlx.app.plist
```

## 验证

```bash
# 1. server 进程确实指向 fork venv
ssh yuanwei@m2max 'ps aux | grep omlx.cli | grep -v grep'
# 期待: yuanwei .../Code/omlx/.venv/bin/python -m omlx.cli serve --base-path /Users/yuanwei/.omlx --port 8000

# 2. omlx 包路径 + 版本
ssh yuanwei@m2max '~/Code/omlx/.venv/bin/python -c "import omlx; print(omlx.__file__, omlx._version.__version__)"'
# 期待: /Users/yuanwei/Code/omlx/omlx/__init__.py 0.3.9.dev2

# 3. Gemma 4 真生成（验证 Path A 代码路径）
curl -s http://m5max:8000/v1/chat/completions -H "X-API-Key: $OMLX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e4b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

## 已知坑

1. **`encodings` ModuleNotFoundError**：忘记 scrub `PYTHONHOME`/`PYTHONPATH` 就撞这个，server.log 里很明显
2. **`ad-hoc` 签名后第一次开 macOS 可能弹"未知来源"对话框**：右键 → 打开，允许一次即可
3. **app 升级会无声覆盖 patch**：所以禁了 updater + 用户承诺不升。如果某天非升不可，按 README 重新跑这套 patch
4. **GUI menubar adopt 功能**：如果 port 已被外部占用（比如手动 nohup 起了 server），GUI 会显示 adopt 提示而不自己 spawn。我们已 kill 所有手动 server，避免 adopt 路径
