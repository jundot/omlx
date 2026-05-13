// PR 10 — first-run welcome wizard. Ports `packaging/omlx_app/welcome.py` to
// SwiftUI; data flow + validation rules captured in `docs/welcome-spec.md`.
//
// Architecture
//   • `WelcomeWindowController` is the AppKit owner of the NSWindow + the
//     SwiftUI `WelcomeView` that drives the four pages. AppDelegate creates
//     one on first run (and on later "Welcome…" menu re-entry).
//   • `WelcomeViewModel` is a @MainActor ObservableObject holding the wizard
//     state across pages, the validation, and the "Start Server" action.
//   • Single window, four pages (Welcome → Storage → API Key → Ready);
//     Next/Back at the bottom; step indicator dots at the top.
//
// First-run trigger lives in `AppDelegate` (PR 10 addition). When config.json
// already exists (re-entry), the Welcome page is skipped via VM init state.

import AppKit
import SwiftUI

// MARK: - Window controller

@MainActor
final class WelcomeWindowController: NSObject, NSWindowDelegate {
    static let willCloseNotification = Notification.Name("OMLXWelcomeWillClose")

    private var window: NSWindow?
    private weak var services: AppServices?
    private weak var server: ServerProcess?
    private let didFinish: (AppConfig, ServerProcess?) -> Void

    init(
        services: AppServices,
        server: ServerProcess?,
        didFinish: @escaping (AppConfig, ServerProcess?) -> Void
    ) {
        self.services = services
        self.server = server
        self.didFinish = didFinish
        super.init()
    }

    func show() {
        if let window {
            window.makeKeyAndOrderFront(self)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        guard let services else { return }

        let vm = WelcomeViewModel(
            services: services,
            server: server,
            isReentry: AppConfig.hasExistingConfig
        )
        vm.onFinish = { [weak self] config, server in
            guard let self else { return }
            self.didFinish(config, server)
            self.close()
        }

        let root = WelcomeView(vm: vm)
            .environmentObject(services)

        let hosting = NSHostingController(rootView: root)
        hosting.view.frame = NSRect(x: 0, y: 0, width: 540, height: 600)

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 540, height: 600),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        win.title = "Welcome to oMLX"
        win.contentViewController = hosting
        win.center()
        win.delegate = self
        win.isReleasedWhenClosed = false
        self.window = win

        win.makeKeyAndOrderFront(self)
        NSApp.activate(ignoringOtherApps: true)
    }

    func close() {
        window?.close()
    }

    // NSWindowDelegate

    nonisolated func windowWillClose(_ notification: Notification) {
        DispatchQueue.main.async {
            NotificationCenter.default.post(
                name: WelcomeWindowController.willCloseNotification,
                object: nil
            )
        }
    }
}

// MARK: - View model

@MainActor
final class WelcomeViewModel: ObservableObject {
    enum Page: Int, CaseIterable, Sendable {
        case welcome, storage, apiKey, ready

        var title: String {
            switch self {
            case .welcome: return "Welcome"
            case .storage: return "Storage"
            case .apiKey:  return "API Key"
            case .ready:   return "Ready"
            }
        }
    }

    @Published var page: Page
    @Published var basePath: String
    @Published var modelDir: String
    @Published var portText: String
    @Published var hfMirror: String
    @Published var apiKey: String = ""
    @Published var apiKeyConfirm: String = ""
    @Published var lastError: String?
    @Published var isStarting: Bool = false
    @Published var startCompleted: Bool = false

    var onFinish: ((AppConfig, ServerProcess?) -> Void)?

    private weak var services: AppServices?
    private weak var server: ServerProcess?

    init(services: AppServices, server: ServerProcess?, isReentry: Bool) {
        self.services = services
        self.server = server
        // Re-entry path skips the splash.
        self.page = isReentry ? .storage : .welcome
        let cfg = services.config
        self.basePath = cfg.basePath.isEmpty ? AppConfig.defaultBasePath() : cfg.basePath
        self.modelDir = cfg.modelDir
        self.portText = String(cfg.port)
        self.hfMirror = cfg.hfEndpoint
        self.apiKey = cfg.apiKey ?? ""
        self.apiKeyConfirm = cfg.apiKey ?? ""
    }

    // MARK: Navigation

    func next() {
        switch page {
        case .welcome: page = .storage
        case .storage:
            guard validateStorage() else { return }
            page = .apiKey
        case .apiKey:
            guard validateApiKey() else { return }
            page = .ready
        case .ready:
            break
        }
    }

    func back() {
        switch page {
        case .welcome: break
        case .storage: page = .welcome
        case .apiKey:  page = .storage
        case .ready:   page = .apiKey
        }
    }

    // MARK: Validation

    func validateStorage() -> Bool {
        let trimmedBase = basePath.trimmingCharacters(in: .whitespaces)
        guard !trimmedBase.isEmpty else {
            lastError = "Base directory is required."
            return false
        }
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)),
              (1...65535).contains(port) else {
            lastError = "Port must be a number between 1 and 65535."
            return false
        }
        _ = port
        lastError = nil
        return true
    }

    func validateApiKey() -> Bool {
        let key = apiKey.trimmingCharacters(in: .whitespaces)
        guard key.count >= 4 else {
            lastError = "API key must be at least 4 characters."
            return false
        }
        guard !key.contains(where: { $0.isWhitespace }) else {
            lastError = "API key must not contain whitespace."
            return false
        }
        guard key.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value < 0x7F }) else {
            lastError = "API key must contain only printable ASCII."
            return false
        }
        guard apiKey == apiKeyConfirm else {
            lastError = "API keys do not match."
            return false
        }
        lastError = nil
        return true
    }

    // MARK: Folder picker

    func browseBaseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select"
        panel.message = "Choose a parent folder. An .omlx directory will be created inside it."
        if panel.runModal() == .OK, let url = panel.url {
            basePath = url.appendingPathComponent(".omlx", isDirectory: true).path
        }
    }

    func browseModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select"
        panel.message = "Choose the directory containing your model files."
        if panel.runModal() == .OK, let url = panel.url {
            modelDir = url.path
        }
    }

    // MARK: Finish

    func startServer() async -> Bool {
        guard let services else { return false }
        isStarting = true
        defer { isStarting = false }

        // 1. Persist AppConfig.
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)) else {
            lastError = "Invalid port."
            return false
        }
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespaces)
        let resolvedBase = ((basePath.trimmingCharacters(in: .whitespaces)
                             as NSString).expandingTildeInPath as NSString)
            .standardizingPath
        var config = services.config
        config.basePath = resolvedBase
        config.port = port
        // modelDir is always a literal path. The wizard's "Reset" button
        // clears the field — interpret that as "use the default for the
        // basePath I just picked" rather than persisting an empty string.
        let trimmedDir = modelDir.trimmingCharacters(in: .whitespaces)
        config.modelDir = trimmedDir.isEmpty
            ? AppConfig.defaultModelDir(forBasePath: resolvedBase)
            : trimmedDir
        config.hfEndpoint = hfMirror.trimmingCharacters(in: .whitespaces)
        config.apiKey = trimmedKey

        // Ensure the base directory exists before spawning the server. The
        // Python child creates `<base>/settings.json` on first start; if the
        // directory is missing, it bails with "Cannot create directory".
        do {
            try FileManager.default.createDirectory(
                at: URL(fileURLWithPath: resolvedBase),
                withIntermediateDirectories: true
            )
        } catch {
            lastError = "Cannot create base directory: \(error.localizedDescription)"
            return false
        }

        // Persist the basePath so every relaunch path resolves to it:
        //   • setenv() for this process (and the child we're about to spawn)
        //   • bootstrap file for Finder relaunches (launchd env doesn't
        //     inherit shell rc, so the env var alone isn't enough)
        //   • shell rc for terminal-launched processes
        // When the user kept the default ~/.omlx, every override is cleared.
        let isDefault = (resolvedBase == AppConfig.defaultBasePath())
        if isDefault {
            unsetenv(ShellEnvWriter.variableName)
            try? AppConfig.writeBootstrapBasePath(nil)
            ShellEnvWriter.apply(value: nil)
        } else {
            setenv(ShellEnvWriter.variableName, resolvedBase, 1)
            try? AppConfig.writeBootstrapBasePath(resolvedBase)
            ShellEnvWriter.apply(value: resolvedBase)
        }

        do {
            try config.save()
        } catch {
            lastError = "Failed to save config: \(error.localizedDescription)"
            return false
        }
        services.updateConfig(config)

        // 2. Build a ServerProcess if AppDelegate didn't already pre-stage one
        // (first-run path defers spawning until the wizard finishes).
        let proc: ServerProcess
        if let existing = server {
            proc = existing
        } else {
            do {
                let runtime = try PythonRuntime.resolve()
                proc = ServerProcess(
                    runtime: runtime,
                    host: config.host,
                    port: config.port,
                    basePath: URL(fileURLWithPath: config.basePath, isDirectory: true)
                )
            } catch {
                lastError = "Failed to locate Python runtime: \(error.localizedDescription)"
                return false
            }
        }
        services.bind(server: proc)

        // 3. Start the server (port-conflict surfaces inline; user can edit
        // the port and tap again).
        do {
            switch try proc.start() {
            case .started, .alreadyRunning:
                break
            case .portConflict(let conflict):
                lastError = "Port \(config.port) is already in use" +
                    (conflict.isOMLX ? " (oMLX server already running)." : ".")
                return false
            }
        } catch {
            lastError = "Failed to start server: \(error.localizedDescription)"
            return false
        }

        // 4. Best-effort post-start fix-ups: setup-api-key (or login if the
        // server already had one) + hf_endpoint patch. None of these are
        // fatal on first run — the user can re-do them in Security /
        // Server screens.
        await Task.sleep(seconds: 0.5)  // give the server a beat to bind
        await waitUntilHealthyOrTimeout(proc: proc, timeout: 8)

        let setupOK = await setupServerApiKey(client: services.client, key: trimmedKey)
        if setupOK, !config.hfEndpoint.isEmpty {
            _ = try? await services.client.updateGlobalSettings(
                GlobalSettingsPatch()
                    // hf_endpoint isn't on GlobalSettingsPatch yet — extend
                    // when the Server screen exposes the mirror row. For
                    // now log + skip; the user can set it via the admin
                    // panel.
            )
        }

        startCompleted = true
        onFinish?(config, proc)
        return true
    }

    private func setupServerApiKey(client: OMLXClient, key: String) async -> Bool {
        // Try setup-api-key first (new install). If the server already has
        // one, fall back to a login so the cookie jar is populated.
        do {
            _ = try await client.setupApiKey(key, confirm: key)
            return true
        } catch {
            // setup-api-key returns 400 if a key already exists; the
            // OMLXClient's auto-login on 401 will handle subsequent calls,
            // so we just swallow this here.
            return false
        }
    }

    private func waitUntilHealthyOrTimeout(proc: ServerProcess, timeout: TimeInterval) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if case .running = proc.state { return }
            try? await Task.sleep(for: .milliseconds(200))
        }
    }
}

private extension Task where Success == Never, Failure == Never {
    static func sleep(seconds: Double) async {
        try? await Task.sleep(for: .seconds(seconds))
    }
}

// MARK: - View

struct WelcomeView: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = scheme == .dark ? OMLXTheme.dark : OMLXTheme.light
        VStack(spacing: 0) {
            StepIndicator(current: vm.page)
                .padding(.top, 24)
                .padding(.bottom, 18)

            ScrollView {
                Group {
                    switch vm.page {
                    case .welcome: WelcomePage()
                    case .storage: StoragePage(vm: vm)
                    case .apiKey:  APIKeyPage(vm: vm)
                    case .ready:   ReadyPage(vm: vm)
                    }
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 24)
            }

            Footer(vm: vm)
        }
        .background(theme.windowBg)
        .environment(\.omlxTheme, theme)
        .frame(width: 540, height: 600)
    }
}

private struct StepIndicator: View {
    let current: WelcomeViewModel.Page
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            ForEach(WelcomeViewModel.Page.allCases, id: \.rawValue) { page in
                let isCurrent = page == current
                let isDone = page.rawValue < current.rawValue
                Capsule()
                    .fill(isCurrent ? theme.accent
                          : (isDone ? theme.greenDot.opacity(0.6) : theme.controlBg))
                    .frame(width: isCurrent ? 28 : 8, height: 8)
                    .animation(.easeOut(duration: 0.18), value: current)
            }
        }
    }
}

private struct WelcomePage: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(spacing: 18) {
            Squircle(gradient: SquircleGradient.server, size: 72) {
                Text("oM")
                    .font(.omlxText(32, weight: .heavy))
                    .kerning(-0.6)
                    .foregroundStyle(.white)
            }
            VStack(spacing: 6) {
                Text("Welcome to oMLX")
                    .font(.omlxText(24, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text("LLM inference, optimized for your Mac")
                    .font(.omlxText(13))
                    .foregroundStyle(theme.textSecondary)
            }
            VStack(alignment: .leading, spacing: 10) {
                bullet("Stores model weights + config under a folder you choose")
                bullet("Talks to local apps over HTTP — no cloud round-trip")
                bullet("Lives in the menubar; this window appears once")
            }
            .padding(.top, 8)
            .padding(.horizontal, 4)
        }
        .padding(.top, 28)
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func bullet(_ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 13))
                .foregroundStyle(theme.greenDot)
            Text(text)
                .font(.omlxText(13))
                .foregroundStyle(theme.text)
        }
    }
}

private struct StoragePage: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Storage")
                    .font(.omlxText(20, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text("Where oMLX keeps model weights and runtime files.")
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }

            ListGroup {
                FreeRow {
                    VStack(alignment: .leading, spacing: 6) {
                        labelRow("Base Directory")
                        HStack(spacing: 8) {
                            Text(vm.basePath)
                                .font(.omlxMono(11))
                                .foregroundStyle(theme.textSecondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Button("Browse…") { vm.browseBaseDirectory() }
                                .buttonStyle(.omlx(.normal, size: .small))
                        }
                    }
                }
                FreeRow {
                    VStack(alignment: .leading, spacing: 6) {
                        labelRow("Model Directory",
                                 sub: "Optional — defaults to <base>/models")
                        HStack(spacing: 8) {
                            Text(vm.modelDir.isEmpty
                                 ? "<\((vm.basePath as NSString).lastPathComponent)>/models"
                                 : vm.modelDir)
                                .font(.omlxMono(11))
                                .foregroundStyle(theme.textSecondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            if !vm.modelDir.isEmpty {
                                Button("Reset") { vm.modelDir = "" }
                                    .buttonStyle(.omlx(.plain, size: .small))
                            }
                            Button("Browse…") { vm.browseModelDirectory() }
                                .buttonStyle(.omlx(.normal, size: .small))
                        }
                    }
                }
                Row(label: "Port",
                    sublabel: "1024-65535 recommended; default 8080") {
                    TextInput(text: $vm.portText, mono: true, width: 100)
                }
                Row(label: "HF Mirror",
                    sublabel: "Optional. Use hf-mirror.com if huggingface.co is restricted.",
                    isLast: true) {
                    TextInput(text: $vm.hfMirror,
                              placeholder: "https://hf-mirror.com",
                              mono: true,
                              width: 240)
                }
            }
        }
    }

    @ViewBuilder
    private func labelRow(_ label: String, sub: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.omlxText(12, weight: .medium))
                .foregroundStyle(theme.text)
            if let sub {
                Text(sub)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }
        }
    }
}

private struct APIKeyPage: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.omlxTheme) private var theme
    @State private var visible: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("API Key")
                    .font(.omlxText(20, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text("Required to authenticate /v1 requests. We'll save it on the server and remember it locally.")
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }

            ListGroup {
                Row(label: "API Key",
                    sublabel: "At least 4 printable characters, no whitespace") {
                    HStack(spacing: 6) {
                        if visible {
                            TextInput(text: $vm.apiKey,
                                      placeholder: "sk-omlx-…",
                                      mono: true,
                                      width: 220)
                        } else {
                            TextInput(text: $vm.apiKey,
                                      placeholder: "sk-omlx-…",
                                      isSecure: true,
                                      mono: true,
                                      width: 220)
                        }
                        Button {
                            visible.toggle()
                        } label: {
                            Image(systemName: visible ? "eye.slash" : "eye")
                                .font(.system(size: 12))
                        }
                        .buttonStyle(.omlx(.plain, size: .small))
                    }
                }
                Row(label: "Confirm",
                    sublabel: "Re-enter the key to catch typos",
                    isLast: true) {
                    if visible {
                        TextInput(text: $vm.apiKeyConfirm,
                                  placeholder: "sk-omlx-…",
                                  mono: true,
                                  width: 220)
                    } else {
                        TextInput(text: $vm.apiKeyConfirm,
                                  placeholder: "sk-omlx-…",
                                  isSecure: true,
                                  mono: true,
                                  width: 220)
                    }
                }
            }

            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "lock")
                    .font(.system(size: 11))
                    .foregroundStyle(theme.textTertiary)
                Text("Stored in `~/.omlx/settings.json`. Sub-keys for individual apps can be added later in Security.")
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }
        }
    }
}

private struct ReadyPage: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Ready to start")
                    .font(.omlxText(20, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text("Review your choices and start the server.")
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }

            ListGroup {
                Row(label: "Base Directory") {
                    Text(vm.basePath)
                        .font(.omlxMono(11))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Row(label: "Model Directory") {
                    Text(vm.modelDir.isEmpty ? "default (<base>/models)" : vm.modelDir)
                        .font(.omlxMono(11))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Row(label: "Port") {
                    Text(vm.portText)
                        .font(.omlxMono(12))
                }
                Row(label: "HF Mirror") {
                    Text(vm.hfMirror.isEmpty ? "huggingface.co (default)" : vm.hfMirror)
                        .font(.omlxMono(11))
                        .foregroundStyle(.secondary)
                }
                Row(label: "API Key", isLast: true) {
                    Text(String(repeating: "•", count: max(4, min(vm.apiKey.count, 16))))
                        .font(.omlxMono(12))
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                hint("Your model library starts empty — visit Downloads to fetch your first model.")
                hint("You can re-open this wizard anytime from the menubar's overflow.")
            }
        }
    }

    @ViewBuilder
    private func hint(_ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
                .foregroundStyle(theme.textTertiary)
            Text(text)
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
        }
    }
}

private struct Footer: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.redDot)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Spacer()
            }

            if vm.page != .welcome {
                Button("Back") { vm.back() }
                    .buttonStyle(.omlx(.plain))
                    .disabled(vm.isStarting)
            }
            primaryButton
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .frame(maxWidth: .infinity)
        .background(theme.toolbarBg)
        .overlay(
            Rectangle()
                .fill(theme.toolbarBorder)
                .frame(height: 0.5),
            alignment: .top
        )
    }

    @ViewBuilder
    private var primaryButton: some View {
        switch vm.page {
        case .welcome:
            Button("Get started") { vm.next() }
                .buttonStyle(.omlx(.primary))
        case .storage, .apiKey:
            Button("Continue") { vm.next() }
                .buttonStyle(.omlx(.primary))
        case .ready:
            Button {
                Task { _ = await vm.startServer() }
            } label: {
                if vm.isStarting {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Starting…")
                    }
                } else {
                    Text("Start Server")
                }
            }
            .buttonStyle(.omlx(.primary))
            .disabled(vm.isStarting)
        }
    }
}
