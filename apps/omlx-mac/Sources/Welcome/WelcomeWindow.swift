// First-run welcome wizard. Single-page Storage + API-key setup; spawns
// the server on confirm.
//
// Architecture
//   • `WelcomeWindowController` is the AppKit owner of the NSWindow + the
//     SwiftUI `WelcomeView` that drives the four pages. AppDelegate creates
//     one on first run only — returning users never see this window.
//   • `WelcomeViewModel` is a @MainActor ObservableObject holding the wizard
//     state across pages, the validation, and the "Start Server" action.
//   • Single window, four pages (Welcome → Storage → API Key → Ready);
//     Next/Back at the bottom; step indicator dots at the top.
//
// First-run trigger lives in `AppDelegate` (PR 10 addition). When config.json
// already exists (re-entry), the Welcome page is skipped via VM init state.

import AppKit
import Security
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
    private let didSkip: ((AppConfig) -> Void)?

    init(
        services: AppServices,
        server: ServerProcess?,
        didFinish: @escaping (AppConfig, ServerProcess?) -> Void,
        didSkip: ((AppConfig) -> Void)? = nil
    ) {
        self.services = services
        self.server = server
        self.didFinish = didFinish
        self.didSkip = didSkip
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
            self.close()
        }
        self.vm = vm

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
            MainActor.assumeIsolated {
                self.handleWillClose()
            }
        }
    }

    /// Skip path: the user dismissed the wizard without running Start Server.
    /// Spec §State machine says the current Storage values should be written
    /// (with an empty API key when not validated) so the next launch lands
    /// on AppView's API-key-not-configured banner instead of re-firing the
    /// wizard. Triggered only when `vm.startCompleted` is false — otherwise
    /// `onFinish` already wrote a complete config.
    private func handleWillClose() {
        if let vm, !vm.startCompleted, let didSkip {
            let snapshot = vm.skipSnapshot()
            didSkip(snapshot)
        }
        NotificationCenter.default.post(
            name: WelcomeWindowController.willCloseNotification,
            object: nil
        )
    }
}

// MARK: - View model

@MainActor
final class WelcomeViewModel: ObservableObject {
    @Published var basePath: String
    @Published var modelDir: String
    @Published var portText: String
    @Published var apiKey: String = ""
    @Published var apiKeyConfirm: String = ""
    @Published var lastError: String?
    @Published var isStarting: Bool = false
    @Published var startCompleted: Bool = false

    var onFinish: ((AppConfig, ServerProcess?) -> Void)?

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
        self.apiKeyConfirm = cfg.apiKey ?? ""
    }

    /// Single-page validation gate — runs Storage + API-key checks in
    /// sequence and surfaces the first failure into `lastError`.
    func validateSetup() -> Bool {
        validateStorage() && validateApiKey()
    }

    // MARK: API key generation

    /// Build a fresh API key of the form `sk-omlx-<32 hex chars>`. Uses
    /// `SecRandomCopyBytes` for cryptographic randomness (16 bytes → 32 hex
    /// chars + 8-char prefix = 40 chars total, comfortably above the
    /// server-side ≥4 minimum). Writes the new value into both `apiKey` and
    /// `apiKeyConfirm` so the confirm field stays in sync without forcing
    /// the user to retype.
    func generateApiKey() {
        var bytes = [UInt8](repeating: 0, count: 16)
        let result = bytes.withUnsafeMutableBytes { buf -> Int32 in
            guard let base = buf.baseAddress else { return errSecAllocate }
            return SecRandomCopyBytes(kSecRandomDefault, buf.count, base)
        }
        let hex: String
        if result == errSecSuccess {
            hex = bytes.map { String(format: "%02x", $0) }.joined()
        } else {
            // Fallback (very unlikely): use Swift's RNG. Not crypto-grade
            // but the wizard's threat model is "user types the same key on
            // both fields", not "attacker predicts the key".
            hex = (0..<16).map { _ in String(format: "%02x", UInt8.random(in: 0...255)) }.joined()
        }
        let key = "sk-omlx-\(hex)"
        apiKey = key
        apiKeyConfirm = key
        lastError = nil
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
            lastError = "Cannot create base directory: \(error.localizedDescription)"
            return false
        }

        // When the user kept the default ~/.omlx, clear every override.
        let isDefault = (resolvedBase == AppConfig.defaultBasePath())
        AppConfig.persistBasePath(isDefault ? nil : resolvedBase)

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

        _ = await setupServerApiKey(client: services.client, key: trimmedKey)

        startCompleted = true
        onFinish?(config, proc)
        return true
    }

    /// Drives Start Server **and** opens the admin dashboard in the user's
    /// default browser. Spec §Flow page 4 splits the Ready action into two:
    /// "Start Server" (Welcome closes; AppView opens) and "Open Admin Panel
    /// & Close" (browser opens to the local dashboard). Implementation just
    /// runs `startServer()` then hands the URL to NSWorkspace.
    @discardableResult
    func startServerAndOpenAdmin() async -> Bool {
        let ok = await startServer()
        guard ok, let services else { return ok }
        let port = services.config.port
        let host = services.config.host
        guard let url = URL(string: "http://\(host):\(port)/admin/dashboard") else {
            return ok
        }
        NSWorkspace.shared.open(url)
        return ok
    }

    /// Spec §State machine — early close: write the current Storage values
    /// (keep `apiKey` blank if not validated) so the user lands on AppView
    /// with the API-key-not-configured banner instead of looping back into
    /// the wizard. Called by `WelcomeWindowController` on `windowWillClose`
    /// when `startCompleted` is false.
    func skipSnapshot() -> AppConfig {
        guard let services else { return AppConfig.default }
        var cfg = services.config
        let trimmedBase = ((basePath.trimmingCharacters(in: .whitespaces)
                            as NSString).expandingTildeInPath as NSString)
            .standardizingPath
        if !trimmedBase.isEmpty { cfg.basePath = trimmedBase }
        if let port = Int(portText.trimmingCharacters(in: .whitespaces)),
           (1...65535).contains(port) {
            cfg.port = port
        }
        let trimmedDir = modelDir.trimmingCharacters(in: .whitespaces)
        cfg.modelDir = trimmedDir.isEmpty
            ? AppConfig.defaultModelDir(forBasePath: cfg.basePath)
            : trimmedDir
        // Per spec: an unvalidated API key is dropped, so the user lands on
        // the API-key-not-configured banner rather than persisting garbage.
        if validateApiKey() {
            cfg.apiKey = apiKey.trimmingCharacters(in: .whitespaces)
        } else {
            cfg.apiKey = ""
        }
        return cfg
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
            ScrollView {
                VStack(spacing: 24) {
                    WelcomeHeader()
                    SetupBody(vm: vm)
                }
                .padding(.horizontal, 32)
                .padding(.top, 28)
                .padding(.bottom, 24)
            }

            Footer(vm: vm)
        }
        .background(theme.windowBg)
        .environment(\.omlxTheme, theme)
        .frame(width: 540, height: 640)
    }
}

/// Top splash band — logo squircle, headline, tagline, and the three
/// "what this app does" bullets. Static; appears on every wizard open
/// (first-run and re-entry alike) since there's now only one page.
private struct WelcomeHeader: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(spacing: 12) {
            // AppLogo's SVG has a 10pt margin inside a 160pt viewBox; the
            // 73×73 frame (≈64 × 160/140) reads at the same visible ~64pt
            // size the previous Squircle did. Matches AboutScreen/ServerScreen.
            Image("AppLogo")
                .resizable()
                .interpolation(.high)
                .frame(width: 73, height: 73)
            VStack(spacing: 4) {
                Text("Welcome to oMLX")
                    .font(.omlxText(22, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text("LLM inference, optimized for your Mac")
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }
        }
        .frame(maxWidth: .infinity)
    }
}

/// Single-page setup: Storage rows, API Key rows, hints. The footer's
/// "Start Server" / "Open Admin Panel & Close" actions live on the
/// outer `Footer` so this view is purely the editable body.
private struct SetupBody: View {
    @ObservedObject var vm: WelcomeViewModel
    @Environment(\.omlxTheme) private var theme
    @State private var keyVisible: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Confirm where weights live and pick an API key. You can change either later in Settings.")
                .font(.omlxText(12))
                .foregroundStyle(theme.textSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Storage
            VStack(alignment: .leading, spacing: 6) {
                sectionLabel("Storage")
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
                        sublabel: "1024-65535 recommended; default 8080",
                        isLast: true) {
                        TextInput(text: $vm.portText, mono: true, width: 100)
                    }
                }
            }

            // API Key
            VStack(alignment: .leading, spacing: 6) {
                sectionLabel("API Key")
                ListGroup {
                    FreeRow {
                        VStack(alignment: .leading, spacing: 6) {
                            labelRow("API Key",
                                     sub: "At least 4 printable characters, no whitespace")
                            HStack(spacing: 6) {
                                keyField($vm.apiKey)
                                Button {
                                    keyVisible.toggle()
                                } label: {
                                    Image(systemName: keyVisible ? "eye.slash" : "eye")
                                        .font(.system(size: 12))
                                }
                                .buttonStyle(.omlx(.plain, size: .small))
                                .help(keyVisible ? "Hide key" : "Show key")
                                Button {
                                    vm.generateApiKey()
                                    keyVisible = true
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "sparkles")
                                            .font(.system(size: 11))
                                        Text("Generate")
                                    }
                                }
                                .buttonStyle(.omlx(.normal, size: .small))
                                .help("Generate a random 40-char API key")
                            }
                        }
                    }
                    FreeRow(isLast: true) {
                        VStack(alignment: .leading, spacing: 6) {
                            labelRow("Confirm", sub: "Re-enter the key to catch typos")
                            keyField($vm.apiKeyConfirm)
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                hint("Stored in `~/.omlx/settings.json`. Sub-keys for individual apps can be added later in Security.")
                hint("Your model library starts empty — visit Downloads to fetch your first model.")
                hint("You can re-open this wizard anytime from the menubar.")
            }
        }
    }

    @ViewBuilder
    private func keyField(_ binding: Binding<String>) -> some View {
        if keyVisible {
            TextInput(text: binding, placeholder: "sk-omlx-…", mono: true, width: 260)
        } else {
            TextInput(text: binding, placeholder: "sk-omlx-…",
                      isSecure: true, mono: true, width: 260)
        }
    }

    @ViewBuilder
    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.omlxText(10, weight: .semibold))
            .foregroundStyle(theme.textTertiary)
            .textCase(.uppercase)
            .kerning(0.6)
            .padding(.horizontal, 14)
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

            // Two actions side-by-side. "Open Admin Panel & Close" performs
            // the same start, then redirects the user to the local
            // /admin/dashboard. Sits to the left of the primary Start Server
            // button (macOS HIG: alternative on the left of primary).
            Button("Open Admin Panel & Close") {
                Task {
                    guard vm.validateSetup() else { return }
                    _ = await vm.startServerAndOpenAdmin()
                }
            }
            .buttonStyle(.omlx(.normal))
            .disabled(vm.isStarting)

            Button {
                Task {
                    guard vm.validateSetup() else { return }
                    _ = await vm.startServer()
                }
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
}
