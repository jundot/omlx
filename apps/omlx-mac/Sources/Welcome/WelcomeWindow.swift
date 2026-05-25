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
        win.title = String(localized: "welcome.window.title",
                           defaultValue: "Welcome to oMLX",
                           comment: "Window title bar text for the Welcome wizard")
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

    /// Mint a fresh API key via the shared `APIKeyGenerator` and mirror it
    /// into both fields so the Confirm row stays in sync. Shared with the
    /// Security screen so both surfaces produce the same `sk-omlx-<...>`
    /// shape.
    func generateApiKey() {
        let key = APIKeyGenerator.random()
        apiKey = key
        apiKeyConfirm = key
        lastError = nil
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
        guard apiKey == apiKeyConfirm else {
            lastError = String(localized: "welcome.error.key_mismatch",
                               defaultValue: "API keys do not match.",
                               comment: "Welcome wizard validation: confirm field differs")
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
                    host: config.host,
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
                Text(String(localized: "welcome.header.title",
                            defaultValue: "Welcome to oMLX",
                            comment: "Main heading shown on the Welcome wizard"))
                    .font(.omlxText(22, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(String(localized: "welcome.header.tagline",
                            defaultValue: "LLM inference, optimized for your Mac",
                            comment: "Sub-tagline under the Welcome wizard's main heading"))
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
            Text(String(localized: "welcome.intro",
                        defaultValue: "Confirm where weights live and pick an API key. You can change either later in Settings.",
                        comment: "Intro paragraph at the top of the Welcome wizard's setup body"))
                .font(.omlxText(12))
                .foregroundStyle(theme.textSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Storage
            VStack(alignment: .leading, spacing: 6) {
                sectionLabel(String(localized: "welcome.storage.section",
                                    defaultValue: "Storage",
                                    comment: "Section heading above the Storage rows in Welcome wizard"))
                ListGroup {
                    FreeRow {
                        VStack(alignment: .leading, spacing: 6) {
                            labelRow(String(localized: "welcome.storage.base_dir.label",
                                            defaultValue: "Base Directory",
                                            comment: "Row label for the Base Directory picker in Welcome wizard"))
                            HStack(spacing: 8) {
                                Text(vm.basePath)
                                    .font(.omlxMono(11))
                                    .foregroundStyle(theme.textSecondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Button(String(localized: "welcome.button.browse",
                                              defaultValue: "Browse…",
                                              comment: "Folder picker trigger button in Welcome wizard")) {
                                    vm.browseBaseDirectory()
                                }
                                    .buttonStyle(.omlx(.normal, size: .small))
                            }
                        }
                    }
                    FreeRow {
                        VStack(alignment: .leading, spacing: 6) {
                            labelRow(String(localized: "welcome.storage.model_dir.label",
                                            defaultValue: "Model Directory",
                                            comment: "Row label for the Model Directory picker in Welcome wizard"),
                                     sub: String(localized: "welcome.storage.model_dir.sub",
                                                 defaultValue: "Optional — defaults to <base>/models",
                                                 comment: "Sublabel hinting that Model Directory is optional"))
                            HStack(spacing: 8) {
                                Text(vm.modelDir.isEmpty
                                     ? String(localized: "welcome.storage.model_dir.placeholder",
                                              defaultValue: "<\((vm.basePath as NSString).lastPathComponent)>/models",
                                              comment: "Placeholder string shown when Model Directory is unset; placeholder is the base path's leaf name")
                                     : vm.modelDir)
                                    .font(.omlxMono(11))
                                    .foregroundStyle(theme.textSecondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                if !vm.modelDir.isEmpty {
                                    Button(String(localized: "welcome.button.reset",
                                                  defaultValue: "Reset",
                                                  comment: "Clear the Model Directory override")) {
                                        vm.modelDir = ""
                                    }
                                        .buttonStyle(.omlx(.plain, size: .small))
                                }
                                Button(String(localized: "welcome.button.browse",
                                              defaultValue: "Browse…",
                                              comment: "Folder picker trigger button in Welcome wizard")) {
                                    vm.browseModelDirectory()
                                }
                                    .buttonStyle(.omlx(.normal, size: .small))
                            }
                        }
                    }
                    Row(label: String(localized: "welcome.storage.port.label",
                                      defaultValue: "Port",
                                      comment: "Row label for the server port field in Welcome wizard"),
                        sublabel: String(localized: "welcome.storage.port.sub",
                                         defaultValue: "1024-65535 recommended; default 8080",
                                         comment: "Sublabel for the port field with the recommended range"),
                        isLast: true) {
                        TextInput(text: $vm.portText, mono: true, width: 100)
                    }
                }
            }

            // API Key
            VStack(alignment: .leading, spacing: 6) {
                sectionLabel(String(localized: "welcome.api_key.section",
                                    defaultValue: "API Key",
                                    comment: "Section heading above the API key rows in Welcome wizard"))
                ListGroup {
                    FreeRow {
                        VStack(alignment: .leading, spacing: 6) {
                            labelRow(String(localized: "welcome.api_key.label",
                                            defaultValue: "API Key",
                                            comment: "Row label for the primary API key field in Welcome wizard"),
                                     sub: String(localized: "welcome.api_key.sub",
                                                 defaultValue: "At least 4 printable characters, no whitespace",
                                                 comment: "Sublabel describing API key format requirements"))
                            HStack(spacing: 6) {
                                keyField($vm.apiKey)
                                Button {
                                    keyVisible.toggle()
                                } label: {
                                    Image(systemName: keyVisible ? "eye.slash" : "eye")
                                        .font(.system(size: 12))
                                }
                                .buttonStyle(.omlx(.plain, size: .small))
                                .help(keyVisible
                                      ? String(localized: "welcome.api_key.hide",
                                               defaultValue: "Hide key",
                                               comment: "Tooltip on the eye-slash button that masks the API key field")
                                      : String(localized: "welcome.api_key.show",
                                               defaultValue: "Show key",
                                               comment: "Tooltip on the eye button that unmasks the API key field"))
                                Button {
                                    vm.generateApiKey()
                                    keyVisible = true
                                } label: {
                                    HStack(spacing: 4) {
                                        Image(systemName: "sparkles")
                                            .font(.system(size: 11))
                                        Text(String(localized: "welcome.api_key.generate",
                                                    defaultValue: "Generate",
                                                    comment: "Button label that mints a random API key"))
                                    }
                                }
                                .buttonStyle(.omlx(.normal, size: .small))
                                .help(String(localized: "welcome.api_key.generate.help",
                                             defaultValue: "Generate a random 40-char API key",
                                             comment: "Tooltip on the Generate API key button"))
                            }
                        }
                    }
                    FreeRow(isLast: true) {
                        VStack(alignment: .leading, spacing: 6) {
                            labelRow(String(localized: "welcome.api_key.confirm.label",
                                            defaultValue: "Confirm",
                                            comment: "Row label for the API key confirmation field"),
                                     sub: String(localized: "welcome.api_key.confirm.sub",
                                                 defaultValue: "Re-enter the key to catch typos",
                                                 comment: "Sublabel for the Confirm API key field"))
                            keyField($vm.apiKeyConfirm)
                        }
                    }
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                HintLine(text: String(localized: "welcome.hint.settings_path",
                                      defaultValue: "Stored in `~/.omlx/settings.json`. Sub-keys for individual apps can be added later in Security.",
                                      comment: "Hint line under the API key section pointing to settings.json"))
                HintLine(text: String(localized: "welcome.hint.models_library",
                                      defaultValue: "Your model library starts empty — visit Downloads to fetch your first model.",
                                      comment: "Hint line pointing the user to Downloads after first-run"))
                HintLine(text: String(localized: "welcome.hint.reopen",
                                      defaultValue: "You can re-open this wizard anytime from the menubar.",
                                      comment: "Hint line telling the user how to get back to the Welcome wizard"))
            }
        }
    }

    @ViewBuilder
    private func keyField(_ binding: Binding<String>) -> some View {
        let placeholder = String(localized: "welcome.api_key.placeholder",
                                 defaultValue: "sk-omlx-…",
                                 comment: "Placeholder text inside the API key text fields")
        if keyVisible {
            TextInput(text: binding, placeholder: placeholder, mono: true, width: 260)
        } else {
            TextInput(text: binding, placeholder: placeholder,
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
            Button(String(localized: "welcome.button.open_admin",
                          defaultValue: "Open Admin Panel & Close",
                          comment: "Secondary footer button: start server and open the browser dashboard")) {
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
                        Text(String(localized: "welcome.button.starting",
                                    defaultValue: "Starting…",
                                    comment: "Footer button label shown while the server is being spawned"))
                    }
                } else {
                    Text(String(localized: "welcome.button.start_server",
                                defaultValue: "Start Server",
                                comment: "Primary footer button that spawns the server and closes the wizard"))
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
