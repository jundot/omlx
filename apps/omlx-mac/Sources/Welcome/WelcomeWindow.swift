// First-run welcome wizard. Three-step flow: product intro, setup, and
// success. The setup step persists config and spawns the server.
//
// Architecture
//   • `WelcomeWindowController` is the AppKit owner of the NSWindow + the
//     SwiftUI `WelcomeView`. AppDelegate creates one on first run only —
//     returning users never see this window.
//   • `WelcomeViewModel` is a @MainActor ObservableObject holding the wizard
//     state across pages, the validation, and the "Start Server" action.
//
// First-run trigger lives in `AppDelegate` (PR 10 addition). When settings.json
// already exists (re-entry), the Welcome page is skipped via app boot flow.

import AppKit
import SwiftUI

// MARK: - Window controller

@MainActor
final class WelcomeWindowController: NSObject, NSWindowDelegate {
    static let willCloseNotification = Notification.Name("OMLXWelcomeWillClose")

    private var window: NSWindow?
    private var vm: WelcomeViewModel?
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

        let vm = WelcomeViewModel(services: services, server: server)
        vm.onFinish = { [weak self] config, server in
            guard let self else { return }
            self.didFinish(config, server)
        }
        vm.onOpenDashboard = { [weak self] in
            self?.close()
        }
        vm.onClose = { [weak self] in
            self?.close()
        }
        self.vm = vm

        let root = WelcomeView(vm: vm)
            .environment(services)

        let hosting = NSHostingController(rootView: root)
        hosting.view.frame = NSRect(x: 0, y: 0, width: 680, height: 620)

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 680, height: 620),
            styleMask: [.titled, .closable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        win.title = String(localized: "welcome.window.title",
                           defaultValue: "Welcome to oMLX",
                           comment: "Window title bar text for the Welcome wizard")
        win.titleVisibility = .hidden
        win.titlebarAppearsTransparent = true
        win.titlebarSeparatorStyle = .none
        win.backgroundColor = .windowBackgroundColor
        win.isMovableByWindowBackground = true
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
            MainActor.assumeIsolated {
                self.handleWillClose()
            }
        }
    }

    /// Closing before Start Server is cancellation, not partial setup. Do not
    /// write settings.json or create the base path; AppDelegate will terminate
    /// so the next launch starts from the Welcome intro again.
    private func handleWillClose() {
        NotificationCenter.default.post(
            name: WelcomeWindowController.willCloseNotification,
            object: nil
        )
    }
}

// MARK: - View model

enum WelcomeStep: Equatable, Sendable {
    case intro
    case setup
    case complete
}

@MainActor
final class WelcomeViewModel: ObservableObject {
    @Published var step: WelcomeStep = .intro
    @Published var basePath: String
    @Published var modelDir: String
    @Published var portText: String
    @Published var apiKey: String = ""
    @Published var lastError: String?
    @Published var isStarting: Bool = false
    @Published var startCompleted: Bool = false

    var onFinish: ((AppConfig, ServerProcess?) -> Void)?
    var onOpenDashboard: (() -> Void)?
    var onClose: (() -> Void)?

    private weak var services: AppServices?
    private weak var server: ServerProcess?

    init(services: AppServices, server: ServerProcess?) {
        self.services = services
        self.server = server
        let cfg = services.config
        self.basePath = cfg.basePath.isEmpty ? AppConfig.defaultBasePath() : cfg.basePath
        self.modelDir = cfg.modelDir
        self.portText = String(cfg.port)
        self.apiKey = cfg.apiKey ?? ""
    }

    /// Single-page validation gate — runs Storage + API-key checks in
    /// sequence and surfaces the first failure into `lastError`.
    func validateSetup() -> Bool {
        validateStorage() && validateApiKey()
    }

    func beginSetup() {
        step = .setup
        lastError = nil
    }

    func backToIntro() {
        guard !isStarting else { return }
        step = .intro
        lastError = nil
    }

    func requestClose() {
        onClose?()
    }

    // MARK: Validation

    func validateStorage() -> Bool {
        let trimmedBase = basePath.trimmingCharacters(in: .whitespaces)
        guard !trimmedBase.isEmpty else {
            lastError = String(localized: "welcome.error.base_dir_required",
                               defaultValue: "Base directory is required.",
                               comment: "Welcome wizard validation: empty base path")
            return false
        }
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)),
              (1...65535).contains(port) else {
            lastError = String(localized: "welcome.error.port_out_of_range",
                               defaultValue: "Port must be a number between 1 and 65535.",
                               comment: "Welcome wizard validation: port not in valid range")
            return false
        }
        _ = port
        lastError = nil
        return true
    }

    func validateApiKey() -> Bool {
        let key = apiKey.trimmingCharacters(in: .whitespaces)
        guard key.count >= 4 else {
            lastError = String(localized: "welcome.error.key_too_short",
                               defaultValue: "API key must be at least 4 characters.",
                               comment: "Welcome wizard validation: api key below min length")
            return false
        }
        guard !key.contains(where: { $0.isWhitespace }) else {
            lastError = String(localized: "welcome.error.key_whitespace",
                               defaultValue: "API key must not contain whitespace.",
                               comment: "Welcome wizard validation: api key contains spaces")
            return false
        }
        guard key.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value < 0x7F }) else {
            lastError = String(localized: "welcome.error.key_non_ascii",
                               defaultValue: "API key must contain only printable ASCII.",
                               comment: "Welcome wizard validation: api key has non-printable or non-ASCII chars")
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
        panel.prompt = String(localized: "welcome.browse.prompt",
                              defaultValue: "Select",
                              comment: "NSOpenPanel button label for the Welcome wizard's folder pickers")
        panel.message = String(localized: "welcome.browse.base_message",
                               defaultValue: "Choose a parent folder. An .omlx directory will be created inside it.",
                               comment: "NSOpenPanel message when picking the Base Directory in Welcome wizard")
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
        panel.prompt = String(localized: "welcome.browse.prompt",
                              defaultValue: "Select",
                              comment: "NSOpenPanel button label for the Welcome wizard's folder pickers")
        panel.message = String(localized: "welcome.browse.model_message",
                               defaultValue: "Choose the directory containing your model files.",
                               comment: "NSOpenPanel message when picking the Model Directory in Welcome wizard")
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
            lastError = String(localized: "welcome.error.invalid_port",
                               defaultValue: "Invalid port.",
                               comment: "Welcome wizard: port field couldn't be parsed as an integer")
            return false
        }
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespaces)
        let resolvedBase = ((basePath.trimmingCharacters(in: .whitespaces)
                             as NSString).expandingTildeInPath as NSString)
            .standardizingPath
        var config = services.config
        config.bindAddress = "127.0.0.1"
        config.basePath = resolvedBase
        config.port = port
        // modelDir is always a literal path. The wizard's "Reset" button
        // clears the field — interpret that as "use the default for the
        // basePath I just picked" rather than persisting an empty string.
        let trimmedDir = modelDir.trimmingCharacters(in: .whitespaces)
        let resolvedModelDir = trimmedDir.isEmpty
            ? AppConfig.defaultModelDir(forBasePath: resolvedBase)
            : trimmedDir
        config.setModelDirs([resolvedModelDir])
        // hf_endpoint is set later from Downloads → "HF Mirror" — we don't
        // touch the existing value here so a returning user's mirror choice
        // survives a re-entry into the wizard.
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
            lastError = String(localized: "welcome.error.mkdir_failed",
                               defaultValue: "Cannot create base directory: \(error.localizedDescription)",
                               comment: "Welcome wizard: mkdir on the base path failed; placeholder is the system error message")
            return false
        }

        // When the user kept the default ~/.omlx, clear every override.
        let isDefault = (resolvedBase == AppConfig.defaultBasePath())
        AppConfig.persistBasePath(isDefault ? nil : resolvedBase)

        do {
            try config.save()
        } catch {
            lastError = String(localized: "welcome.error.save_config_failed",
                               defaultValue: "Failed to save config: \(error.localizedDescription)",
                               comment: "Welcome wizard: writing settings.json failed; placeholder is the system error message")
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
                    bindAddress: config.bindAddress,
                    port: config.port,
                    basePath: URL(fileURLWithPath: config.basePath, isDirectory: true)
                )
            } catch {
                lastError = String(localized: "welcome.error.python_runtime_failed",
                                   defaultValue: "Failed to locate Python runtime: \(error.localizedDescription)",
                                   comment: "Welcome wizard: PythonRuntime.resolve() threw; placeholder is the system error message")
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
                lastError = conflict.isOMLX
                    ? String(localized: "welcome.error.port_in_use_omlx",
                             defaultValue: "Port \(String(config.port)) is already in use (oMLX server already running).",
                             comment: "Welcome wizard: bind() failed because another oMLX instance owns the port")
                    : String(localized: "welcome.error.port_in_use",
                             defaultValue: "Port \(String(config.port)) is already in use.",
                             comment: "Welcome wizard: bind() failed because some other process owns the port")
                return false
            }
        } catch {
            lastError = String(localized: "welcome.error.start_server_failed",
                               defaultValue: "Failed to start server: \(error.localizedDescription)",
                               comment: "Welcome wizard: ServerProcess.start() threw; placeholder is the system error message")
            return false
        }

        // 4. Best-effort post-start fix-ups: setup-api-key (or login if the
        // server already had one) + hf_endpoint patch. None of these are
        // fatal on first run — the user can re-do them in Security /
        // Server screens.
        // Give the server a beat to bind, then wait until the health-check
        // loop has confirmed /health 200 (cap 8s so a hung server doesn't
        // freeze the wizard).
        try? await Task.sleep(for: .milliseconds(500))
        await waitUntilHealthyOrTimeout(proc: proc, timeout: 8)

        _ = await setupServerApiKey(client: services.client, key: trimmedKey)

        startCompleted = true
        step = .complete
        onFinish?(config, proc)
        return true
    }

    /// Opens the admin dashboard after the setup step has started the server.
    @discardableResult
    func openWebDashboard() -> Bool {
        guard let services else { return false }
        guard let url = AppConfig.httpURL(
            host: services.config.host,
            port: services.config.port,
            path: "/admin/dashboard"
        ) else {
            return false
        }
        NSWorkspace.shared.open(url)
        onOpenDashboard?()
        return true
    }

    private func setupServerApiKey(client: OMLXClient, key: String) async -> Bool {
        // Try setup-api-key (fresh install). When the server already has a
        // key set, the endpoint returns 400 — we swallow that and let
        // `OMLXClient`'s 401 auto-login handle the next authenticated call.
        // The server is local-only on first run, so we don't need an
        // explicit login round-trip here.
        do {
            _ = try await client.setupApiKey(key, confirm: key)
            return true
        } catch {
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

// MARK: - View

struct WelcomeView: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = scheme == .dark ? OMLXTheme.dark : OMLXTheme.light
        ZStack {
            WelcomeBackdrop()
                .ignoresSafeArea()

            VStack(spacing: 0) {
                Group {
                    switch vm.step {
                    case .intro:
                        WelcomeIntroBody()
                    case .setup:
                        WelcomeSetupBody(vm: vm)
                    case .complete:
                        WelcomeCompleteBody(vm: vm)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                WelcomeFooter(vm: vm)
            }
        }
        .environment(\.omlxTheme, theme)
        .frame(width: 680, height: 620)
    }
}

// MARK: - Welcome redesign

private enum WelcomeStyle {
    static let bg = Color(nsColor: .windowBackgroundColor)
    static let panel = Color(nsColor: .controlBackgroundColor)
    static let panelBorder = Color(nsColor: .separatorColor)
    static let text = Color(nsColor: .labelColor)
    static let muted = Color(nsColor: .secondaryLabelColor)
    static let faint = Color(nsColor: .tertiaryLabelColor)
    static let fill = Color(nsColor: .quaternaryLabelColor).opacity(0.16)
    static let accent = Color.accentColor
}

private struct WelcomeBackdrop: View {
    var body: some View {
        WelcomeStyle.bg
    }
}

private struct WelcomeIntroBody: View {
    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 18)
            WelcomeIcon(size: 88)

            VStack(spacing: 14) {
                Text(String(localized: "welcome.header.title",
                            defaultValue: "oMLX",
                            comment: "Main heading shown on the Welcome wizard"))
                    .font(.omlxDisplay(48, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                Text(String(localized: "welcome.header.subtitle",
                            defaultValue: "Local AI, no more waiting.",
                            comment: "Short tagline under the Welcome wizard's main heading"))
                    .font(.omlxDisplay(25, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                    .multilineTextAlignment(.center)
                Text(String(localized: "welcome.header.tagline",
                            defaultValue: "macOS-native MLX server with smart caching.\nClaude Code, OpenClaw, and Cursor respond in 5 seconds, not 90.",
                            comment: "Sub-tagline under the Welcome wizard's main heading"))
                    .font(.omlxText(15))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(5)
                    .frame(maxWidth: 540)
                    .padding(.top, 6)
            }

            HStack(spacing: 10) {
                FeaturePill(icon: "lock.fill", title: "Localhost")
                FeaturePill(icon: "memorychip", title: "MLX")
                FeaturePill(icon: "bolt.fill", title: "Smart caching")
            }
            .padding(.top, 8)

            Text(String(localized: "welcome.header.meta",
                        defaultValue: "Apple Silicon · macOS-native · Apache 2.0",
                        comment: "Short metadata line on the Welcome intro page"))
                .font(.omlxText(12))
                .foregroundStyle(WelcomeStyle.faint)

            Spacer(minLength: 28)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 54)
    }
}

private struct WelcomeSetupBody: View {
    @ObservedObject var vm: WelcomeViewModel
    @State private var keyVisible: Bool = false

    var body: some View {
        VStack(spacing: 22) {
            WelcomeIcon(size: 70)

            VStack(spacing: 6) {
                Text(String(localized: "welcome.setup.title",
                            defaultValue: "Set up your local server",
                            comment: "Heading on the Welcome setup page"))
                    .font(.omlxDisplay(24, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text(String(localized: "welcome.intro",
                            defaultValue: "Choose where models live, pick a port, and create the API key you'll use from apps and the web dashboard.",
                            comment: "Intro paragraph at the top of the Welcome wizard's setup body"))
                    .font(.omlxText(13))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
                    .frame(maxWidth: 480)
            }

            VStack(alignment: .leading, spacing: 18) {
                WelcomeNotice(
                    icon: "memorychip.fill",
                    title: String(localized: "welcome.setup.local.title",
                                  defaultValue: "Local server",
                                  comment: "Title for the local server setup card"),
                    text: String(localized: "welcome.setup.local.body",
                                 defaultValue: "oMLX binds to 127.0.0.1 on first run, so clients on this Mac can use it without exposing the server to your network.",
                                 comment: "Body for the local server setup card")
                )

                VStack(spacing: 0) {
                    SettingRow(
                        icon: "number",
                        title: String(localized: "welcome.storage.port.label",
                                      defaultValue: "Port",
                                      comment: "Row label for the server port field in Welcome wizard"),
                        subtitle: String(localized: "welcome.storage.port.sub",
                                         defaultValue: "Default 8000. Change this only if the port is already in use.",
                                         comment: "Sublabel for the port field with the recommended default")
                    ) {
                        TextInput(text: $vm.portText, mono: true, width: 96)
                    }

                    WelcomeDivider()

                    SettingRow(
                        icon: "folder",
                        title: String(localized: "welcome.storage.model_dir.label",
                                      defaultValue: "Model Directory",
                                      comment: "Row label for the Model Directory picker in Welcome wizard"),
                        subtitle: String(localized: "welcome.storage.model_dir.sub",
                                         defaultValue: "Where downloaded models are stored.",
                                         comment: "Sublabel explaining the model directory")
                    ) {
                        HStack(spacing: 8) {
                            Text(vm.modelDir.isEmpty
                                 ? AppConfig.defaultModelDir(forBasePath: vm.basePath)
                                 : vm.modelDir)
                                .font(.omlxMono(11))
                                .foregroundStyle(WelcomeStyle.muted)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(width: 232, alignment: .trailing)
                            Button {
                                vm.browseModelDirectory()
                            } label: {
                                Image(systemName: "folder")
                                    .font(.system(size: 13, weight: .semibold))
                            }
                            .buttonStyle(.omlx(.normal, size: .small))
                            .disabled(vm.isStarting)
                            .help(String(localized: "welcome.button.browse",
                                         defaultValue: "Browse...",
                                         comment: "Folder picker trigger button in Welcome wizard"))
                        }
                    }

                    WelcomeDivider()

                    SettingRow(
                        icon: "key",
                        title: String(localized: "welcome.api_key.label",
                                  defaultValue: "API Key",
                                  comment: "Row label for the primary API key field in Welcome wizard"),
                        subtitle: String(localized: "welcome.api_key.sub",
                                         defaultValue: "This key is also your Web Dashboard login password.",
                                         comment: "Sublabel explaining API key usage")
                    ) {
                        HStack(spacing: 0) {
                            TextInput("welcome.api_key.placeholder", text: $vm.apiKey, placeholder: "sk-omlx-…", isSecure: !keyVisible, mono: true, width: 210)
                            Button {
                                keyVisible.toggle()
                            } label: {
                                Image(systemName: keyVisible ? "eye.slash" : "eye")
                                    .font(.system(size: 13, weight: .semibold))
                            }
                            .disabled(vm.isStarting)
                            .help(keyVisible
                                  ? String(localized: "welcome.api_key.hide",
                                           defaultValue: "Hide key",
                                           comment: "Tooltip on the eye-slash button that masks the API key field")
                                  : String(localized: "welcome.api_key.show",
                                           defaultValue: "Show key",
                                           comment: "Tooltip on the eye button that unmasks the API key field"))

                            Button {
                                vm.apiKey = APIKeyGenerator.random()
                                keyVisible = true
                            } label: {
                                Image(systemName: "arrow.triangle.2.circlepath")
                            }
                            .help(String(localized: "security.api_key.generate",
                                         defaultValue: "Generate a random key",
                                         comment: "Tooltip on the API key regenerate button"))
                        }
                        .buttonStyle(.omlx(.plain, size: .small))
                    }
                }
                .background(WelcomeStyle.panel)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(WelcomeStyle.panelBorder.opacity(0.45), lineWidth: 0.5)
                )

                Text(String(localized: "welcome.hint.settings_path",
                            defaultValue: "Settings are stored in ~/.omlx/settings.json.",
                            comment: "Hint line under the API key section pointing to settings.json"))
                    .font(.omlxText(11))
                    .foregroundStyle(WelcomeStyle.faint)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .frame(width: 560)
        }
        .padding(.horizontal, 48)
        .padding(.top, 18)
        .padding(.bottom, 6)
        .disabled(vm.isStarting)
    }
}

private struct WelcomeCompleteBody: View {
    @ObservedObject var vm: WelcomeViewModel

    var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 34)
            WelcomeIcon(size: 84)

            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 42, weight: .semibold))
                .foregroundStyle(WelcomeStyle.accent)
                .accessibilityHidden(true)

            VStack(spacing: 10) {
                Text(String(localized: "welcome.complete.title",
                            defaultValue: "All set!",
                            comment: "Heading on the Welcome completion page"))
                    .font(.omlxDisplay(30, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text(String(localized: "welcome.complete.description",
                            defaultValue: "oMLX is running locally at http://127.0.0.1:\(vm.portText). Open the web dashboard to download your first model, manage settings, and connect your coding tools.",
                            comment: "Description on the Welcome completion page"))
                    .font(.omlxText(14))
                    .foregroundStyle(WelcomeStyle.muted)
                    .multilineTextAlignment(.center)
                    .lineSpacing(3)
                    .frame(maxWidth: 500)
                Text(String(localized: "welcome.complete.status",
                            defaultValue: "Server started. Dashboard is ready.",
                            comment: "Short success status on the Welcome completion page"))
                    .font(.omlxText(12, weight: .medium))
                    .foregroundStyle(WelcomeStyle.faint)
                    .padding(.top, 4)
            }
            Spacer(minLength: 40)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 54)
    }
}

private struct WelcomeFooter: View {
    @ObservedObject var vm: WelcomeViewModel

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            if vm.step == .setup {
                Button {
                    vm.backToIntro()
                } label: {
                    Label(String(localized: "common.back",
                                 defaultValue: "Back",
                                 comment: "Back button label"),
                          systemImage: "chevron.left")
                }
                .buttonStyle(.omlx(.plain))
                .disabled(vm.isStarting)
            } else {
                Spacer()
                    .frame(width: 90)
            }

            Spacer()

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11.5))
                    .foregroundStyle(Color(nsColor: .systemRed))
                    .lineLimit(2)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 320, alignment: .trailing)
            }

            primaryButton
        }
        .padding(.horizontal, 28)
        .frame(height: 72)
        .background(WelcomeStyle.panel.opacity(0.68))
        .overlay(alignment: .top) {
            Rectangle()
                .fill(WelcomeStyle.panelBorder.opacity(0.5))
                .frame(height: 0.5)
        }
    }

    @ViewBuilder
    private var primaryButton: some View {
        switch vm.step {
        case .intro:
            WelcomeCTA(
                title: String(localized: "welcome.button.get_started",
                              defaultValue: "Get Started",
                              comment: "Primary footer button that advances from the intro to setup"),
                systemImage: "arrow.right",
                width: 142
            ) {
                vm.beginSetup()
            }

        case .setup:
            WelcomeCTA(
                title: vm.isStarting
                    ? String(localized: "welcome.button.starting",
                             defaultValue: "Starting Server...",
                             comment: "Footer button label shown while the server is being spawned")
                    : String(localized: "welcome.button.start_server",
                             defaultValue: "Start Server",
                             comment: "Primary footer button that spawns the server"),
                systemImage: vm.isStarting ? nil : "arrow.right",
                isBusy: vm.isStarting,
                width: 160
            ) {
                Task {
                    guard vm.validateSetup() else { return }
                    _ = await vm.startServer()
                }
            }
            .disabled(vm.isStarting)

        case .complete:
            WelcomeCTA(
                title: String(localized: "welcome.button.open_dashboard",
                              defaultValue: "Open Web Dashboard",
                              comment: "Footer button that opens the local web dashboard"),
                systemImage: "arrow.up.right",
                width: 210
            ) {
                _ = vm.openWebDashboard()
            }
        }
    }
}

private struct WelcomeCTA: View {
    let title: String
    var systemImage: String?
    var isBusy: Bool = false
    var width: CGFloat
    let action: () -> Void

    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isBusy {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(title)
                    .font(.omlxText(13, weight: .medium))
                    .lineLimit(1)
                    .minimumScaleFactor(0.86)
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: 12, weight: .semibold))
                }
            }
            .foregroundStyle(Color.white)
            .frame(width: width, height: 32)
            .background(WelcomeStyle.accent)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .opacity(isEnabled ? 1.0 : 0.55)
        }
        .buttonStyle(.plain)
    }
}

private struct WelcomeIcon: View {
    let size: CGFloat

    var body: some View {
        Image("AppLogo")
            .resizable()
            .interpolation(.high)
            .frame(width: size, height: size)
            .clipShape(RoundedRectangle(cornerRadius: size * 0.22, style: .continuous))
            .shadow(color: Color.black.opacity(0.10), radius: 12, y: 6)
            .accessibilityLabel("oMLX")
    }
}

private struct FeaturePill: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
            Text(title)
                .font(.omlxText(12, weight: .medium))
        }
        .foregroundStyle(WelcomeStyle.muted)
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .background(WelcomeStyle.fill)
        .clipShape(Capsule())
    }
}

private struct WelcomeNotice: View {
    let icon: String
    let title: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(WelcomeStyle.accent)
                .frame(width: 24, height: 24)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.omlxText(13, weight: .semibold))
                    .foregroundStyle(WelcomeStyle.text)
                Text(text)
                    .font(.omlxText(12))
                    .foregroundStyle(WelcomeStyle.muted)
                    .lineSpacing(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .background(WelcomeStyle.fill)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct SettingRow<Content: View>: View {
    let icon: String
    let title: String
    let subtitle: String
    let content: () -> Content

    init(
        icon: String,
        title: String,
        subtitle: String,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.icon = icon
        self.title = title
        self.subtitle = subtitle
        self.content = content
    }

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(WelcomeStyle.muted)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.omlxText(13, weight: .medium))
                    .foregroundStyle(WelcomeStyle.text)
                Text(subtitle)
                    .font(.omlxText(11.5))
                    .foregroundStyle(WelcomeStyle.faint)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 16)
            content()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
    }
}

private struct WelcomeDivider: View {
    var body: some View {
        Rectangle()
            .fill(WelcomeStyle.panelBorder.opacity(0.42))
            .frame(height: 0.5)
            .padding(.leading, 52)
    }
}
