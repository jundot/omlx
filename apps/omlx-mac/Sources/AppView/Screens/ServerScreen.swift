// PR 7 — real Server screen. ServerHero shows live ServerProcess.State and
// drives Start / Stop / Restart through AppServices. Network + Logging rows
// read/write `/admin/api/global-settings`.
//
// Scope vs design: rows whose backing field doesn't exist server-side yet
// (CORS, HTTPS, Request Timeout, Telemetry, GPU memory, KV-cache quant) are
// deferred until those settings are added to GlobalSettingsRequest. We keep
// the shipped surface honest: every row in this screen is fully wired.

import SwiftUI
import AppKit

struct ServerScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = ServerScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ServerHeroCard(vm: vm)

            BackendSection()

            SectionHeader(String(localized: "server.section.network",
                                  defaultValue: "Network",
                                  comment: "Section heading for the Network rows in Server screen"))
            ListGroup {
                Row(label: String(localized: "server.row.listen_address",
                                  defaultValue: "Listen Address",
                                  comment: "Row label for the listen-address picker in Server screen")) {
                    Popup(
                        selection: vm.bind($vm.host, save: { vm.saveHost(services: services) }),
                        width: .controlMedium,
                        options: [
                            ("127.0.0.1", String(localized: "server.host.local_only",
                                                  defaultValue: "127.0.0.1 (Local only)",
                                                  comment: "Listen-address popup option for loopback only")),
                            ("0.0.0.0", String(localized: "server.host.ipv4_only",
                                                defaultValue: "0.0.0.0 (IPv4 only)",
                                                comment: "Listen-address popup option for binding to IPv4 interfaces only")),
                            ("::", String(localized: "server.host.all_networks",
                                          defaultValue: "0.0.0.0 & :: (All Networks)",
                                          comment: "Listen-address popup option for binding to all interfaces; the IPv6 wildcard also accepts IPv4 (dual-stack)")),
                            ("localhost", String(localized: "server.host.localhost",
                                                  defaultValue: "localhost",
                                                  comment: "Listen-address popup option for localhost")),
                        ]
                    )
                }
                Row(
                    label: String(localized: "server.row.port",
                                  defaultValue: "Port",
                                  comment: "Row label for the server port field"),
                    sublabel: String(localized: "server.row.port.sub",
                                     defaultValue: "Default 8000. Server restarts on save.",
                                     comment: "Sublabel under the Port field"),
                    isLast: true
                ) {
                    TextInput(text: $vm.portText, mono: true, width: .controlNarrow)
                }
            }

            SectionHeader(String(localized: "server.section.endpoints",
                                  defaultValue: "API Endpoints",
                                  comment: "Section heading for the API endpoints list"))
            APIEndpointsList(host: vm.effectiveHost, port: vm.effectivePort,
                             remote: services.config.endpoint.isRemote ? services.config.endpoint : nil)

            SectionHeader(
                String(localized: "server.section.default_profile",
                       defaultValue: "Default Profile",
                       comment: "Section heading for the default sampling profile editor"),
                subtitle: String(localized: "server.section.default_profile.sub",
                                 defaultValue: "Fallback values used when a model has no profile, or when a profile leaves a field empty",
                                 comment: "Subtitle for the Default Profile section")
            )
            // Deep-link target for the per-model Profiles tab's "Edit on
            // Server →" link (see AppServices.ServerAnchor.defaultProfile).
            .id(ServerAnchor.defaultProfile.rawValue)
            ServerDefaultProfileEditor(vm: vm)

            SectionHeader(String(localized: "server.section.startup",
                                  defaultValue: "Server Startup",
                                  comment: "Section heading for server startup behavior settings"))
            ListGroup {
                Row(
                    label: String(localized: "server.row.auto_start_on_launch",
                                  defaultValue: "Automatically start server on launch",
                                  comment: "Row label for automatically starting the managed server when the macOS app launches"),
                    sublabel: String(localized: "server.row.auto_start_on_launch.sub",
                                     defaultValue: "When disabled, the menu bar app opens without starting the server.",
                                     comment: "Sublabel explaining the auto-start on launch setting"),
                    isLast: true
                ) {
                    RowSwitch(isOn: vm.bind($vm.autoStartOnLaunch, save: {
                        vm.saveAutoStartOnLaunch(services: services)
                    }))
                }
            }

            SectionHeader(String(localized: "server.section.logging",
                                  defaultValue: "Logging",
                                  comment: "Section heading for the Logging rows"))
            ListGroup {
                Row(label: String(localized: "server.row.log_level",
                                  defaultValue: "Log Level",
                                  comment: "Row label for the log level picker"),
                    isLast: true) {
                    Popup(
                        selection: vm.bind($vm.logLevel, save: vm.saveLogLevel),
                        width: .controlCompact,
                        options: [
                            ("error",   String(localized: "server.log_level.error",
                                                defaultValue: "Error",
                                                comment: "Log level popup option")),
                            ("warning", String(localized: "server.log_level.warning",
                                                defaultValue: "Warning",
                                                comment: "Log level popup option")),
                            ("info",    String(localized: "server.log_level.info",
                                                defaultValue: "Info",
                                                comment: "Log level popup option")),
                            ("debug",   String(localized: "server.log_level.debug",
                                                defaultValue: "Debug",
                                                comment: "Log level popup option")),
                            ("trace",   String(localized: "server.log_level.trace",
                                                defaultValue: "Trace",
                                                comment: "Log level popup option")),
                        ]
                    )
                }
            }

            SectionHeader(String(localized: "server.section.usage",
                                  defaultValue: "Usage History",
                                  comment: "Section heading for the local usage history switch in Server screen"))
            // Deep-link target for the Status screen's "Usage history is off"
            // notice (see AppServices.ServerAnchor.usageHistory).
            .id(ServerAnchor.usageHistory.rawValue)
            ListGroup {
                Row(
                    label: String(localized: "server.row.usage_history",
                                  defaultValue: "Record usage history",
                                  comment: "Row label for the switch that records local hourly usage history"),
                    sublabel: String(localized: "server.row.usage_history.sub",
                                     defaultValue: "Stores hourly per-model token totals in usage.sqlite3. Turning this off keeps existing history.",
                                     comment: "Sublabel explaining the usage history switch"),
                    isLast: true
                ) {
                    RowSwitch(isOn: vm.bind($vm.usageHistoryEnabled, save: vm.saveUsageHistory))
                }
            }

            SectionHeader(
                String(localized: "server.section.storage",
                       defaultValue: "Storage",
                       comment: "Section heading for storage rows in Server screen"),
                subtitle: String(localized: "server.section.storage.sub",
                                 defaultValue: "Where models, settings, logs, and the SSD cache live.",
                                 comment: "Subtitle for the Storage section in Server screen")
            )
            ListGroup {
                Row(
                    label: String(localized: "server.row.base_path",
                                  defaultValue: "Base Path",
                                  comment: "Row label for the Base Path text input"),
                    sublabel: String(localized: "server.row.base_path.sub",
                                     defaultValue: "OMLX_BASE_PATH. Files move and the server restarts when this changes.",
                                     comment: "Sublabel under the Base Path field")
                ) {
                    TextInput(text: $vm.basePathText, mono: true, width: .controlWide)
                }
                FreeRow {
                    ModelDirectoriesEditor(vm: vm)
                }
                Row(
                    label: String(localized: "server.row.hf_cache",
                                  defaultValue: "Use Hugging Face local cache",
                                  comment: "Row label for enabling standard Hugging Face Hub cache model discovery"),
                    sublabel: String(localized: "server.row.hf_cache.sub",
                                     defaultValue: "Discover MLX-compatible models from the standard Hugging Face Hub cache.",
                                     comment: "Sublabel for enabling Hugging Face local cache discovery"),
                    isLast: true
                ) {
                    RowSwitch(isOn: $vm.hfCacheEnabled)
                }
            }
            ServerAdvancedSection(vm: vm)

            FooterBar(
                hint: String(localized: "server.footer.hint",
                             defaultValue: "Listen Address, Log Level, and SSE Keep-Alive Mode apply the moment you change them. The rest (port, default profile, storage, aliases) commits when you click Apply. Port and storage changes take effect after a server restart.",
                             comment: "Hint footer text under the Server screen explaining which controls apply immediately vs. via the Apply button"),
                error: vm.lastError
            ) {
                Button(String(localized: "settings.button.reset_defaults",
                              defaultValue: "Reset Defaults",
                              comment: "Fill this screen with defaults before applying")) {
                    Task { await vm.resetDefaults(client: services.client) }
                }
                .buttonStyle(.omlx(.normal))
                .disabled(vm.isMovingBasePath || vm.isLoading || vm.isResetting || services.canSaveSettingsOffline)
                .help(String(localized: "settings.reset_defaults.help",
                             defaultValue: "Restore default values. Paths and API keys are kept. Click Apply to save."))
                Button(String(localized: "server.button.apply",
                              defaultValue: "Apply",
                              comment: "Button to apply pending server settings: port, default profile, storage, and aliases")) {
                    vm.applyServerSettings(services: services)
                }
                    .buttonStyle(.omlx(.primary))
                    .disabled(!vm.hasPendingServerChanges(services: services)
                              || vm.isMovingBasePath || vm.isResetting)
            }
        }
        .alert(String(localized: "settings.reset_defaults.title",
                      defaultValue: "Settings Reset"), isPresented: $vm.showResetNotice) {
            Button(String(localized: "common.cancel", defaultValue: "Cancel"), role: .cancel) {
                vm.cancelReset()
            }
            Button(String(localized: "common.ok", defaultValue: "OK")) {
                vm.confirmReset()
            }
        } message: {
            Text(String(localized: "settings.reset_defaults.message",
                        defaultValue: "Settings have been reset to defaults. Click Apply to save the changes."))
        }
        .task {
            // services.config is already populated by AppDelegate before this
            // view is mounted, so .onChange never fires for the initial value —
            // mirror it explicitly on first appearance.
            vm.applyConfig(services.config)
            await vm.load(client: services.client)
        }
        .onChange(of: services.config) { _, _ in
            vm.applyConfig(services.config)
        }
        .onChange(of: services.serverState) { _, _ in
            // After a restart triggered by saving host/port, reload to pick
            // up the new effective values.
            Task { await vm.load(client: services.client) }
        }
    }
}

private struct ModelDirectoriesEditor: View {
    var vm: ServerScreenVM
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(localized: "server.row.models_directories",
                                defaultValue: "Model Directories",
                                comment: "Row label for the model directories editor"))
                        .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                    Text(String(localized: "server.row.models_directories.sub",
                                defaultValue: "The first path is the download target. All paths are scanned for local models.",
                                comment: "Sublabel under the model directories editor"))
                        .font(.omlxText(11.5))
                        .foregroundStyle(theme.textSecondary)
                }
                Spacer(minLength: 12)
                Button {
                    vm.addModelDirectory()
                } label: {
                    Image(systemName: "plus")
                }
                .buttonStyle(.omlx(.normal, size: .small))
                .help(String(localized: "server.model_dirs.add.help",
                             defaultValue: "Add model directory",
                             comment: "Tooltip for the add model directory button"))
            }

            VStack(spacing: 7) {
                ForEach(Array(vm.modelDirTexts.indices), id: \.self) { index in
                    modelDirRow(index: index)
                }
            }
        }
    }

    private func modelDirRow(index: Int) -> some View {
        HStack(spacing: 7) {
            Text(index == 0
                 ? String(localized: "server.model_dirs.primary",
                          defaultValue: "Primary",
                          comment: "Badge for the first model directory")
                 : String(format: "#%d", index + 1))
                .font(.omlxMono(DesignTokens.FontSize.caption, weight: .semibold))
                .foregroundStyle(index == 0 ? theme.accent : theme.textSecondary)
                .frame(width: 52, alignment: .leading)

            TextInput(text: Binding(
                get: { vm.modelDirText(at: index) },
                set: { vm.setModelDirText($0, at: index) }
            ), mono: true, width: .controlWide)

            Button {
                vm.browseModelDirectory(at: index)
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .help(String(localized: "server.model_dirs.browse.help",
                         defaultValue: "Choose folder",
                         comment: "Tooltip for choosing a model directory"))

            Button {
                vm.removeModelDirectory(at: index)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .help(String(localized: "server.model_dirs.remove.help",
                         defaultValue: "Remove model directory",
                         comment: "Tooltip for removing a model directory"))
        }
    }
}

// MARK: - Server hero

/// Hero card shared between the Server and Status screens (omlx-screens.jsx
/// uses the same component on both, lines 75 and 833). When a `vm` is wired
/// in (Server screen), the Restart button folds any pending port/host edits
/// into the restart. On the Status screen there is no such VM, so it just
/// asks AppServices to bounce the cached endpoint.
struct ServerHeroCard: View {
    var vm: ServerScreenVM? = nil

    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 16) {
            // App logo — `AppLogo` asset ships the rounded omlx mark (same
            // artwork as the README's hero icon, light/dark variants).
            // Replaces the previous gradient squircle + "oM" placeholder
            // so the hero card on Server / Status reads as oMLX rather
            // than a generic Apple-style server icon.
            Image("AppLogo")
                .resizable()
                .interpolation(.high)
                .frame(width: 52, height: 52)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 10) {
                    Text(title)
                        .font(.omlxText(18, weight: .semibold))
                        .foregroundStyle(theme.text)
                    StatusPill(status: pillStatus)
                }
                Text(subtitle)
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            }
            Spacer(minLength: 12)
            buttons
        }
        .padding(18)
        // Hero card surface follows the grouped Settings style. Status is
        // carried by the pill/actions, not by a colored background wash.
        .background(heroBackground)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .padding(.horizontal, 14)
        .padding(.bottom, 14)
    }

    @ViewBuilder
    private var buttons: some View {
        if services.isRemoteBackend {
            // The remote server is managed on its own Mac: nothing to
            // start, stop or restart from here.
            Button {
                Task { await services.refreshRemoteBackend() }
            } label: {
                Label(String(localized: "server.hero.check_now",
                             defaultValue: "Check now",
                             comment: "Hero card button that re-checks whether the other oMLX server answers"),
                      systemImage: "arrow.clockwise")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.omlx(.normal))
        } else {
            localButtons
        }
    }

    @ViewBuilder
    private var localButtons: some View {
        switch services.serverState {
        case .running, .unresponsive:
            HStack(spacing: 6) {
                Button {
                    // Pick up any pending edits in Listen Address / Port that
                    // weren't committed via Enter/blur, so this button is a
                    // single "apply + restart" affordance rather than only
                    // restarting on the cached endpoint. When the hero is
                    // mounted without a VM (Status screen) we simply bounce.
                    if let vm {
                        vm.restart(services: services)
                    } else {
                        Task { try? await services.restartServer() }
                    }
                } label: {
                    Label(String(localized: "server.hero.restart",
                                 defaultValue: "Restart",
                                 comment: "Hero card button to restart the server"),
                          systemImage: "arrow.clockwise")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.normal))

                Button {
                    Task { await services.stopServer() }
                } label: {
                    Label(String(localized: "server.hero.stop",
                                 defaultValue: "Stop",
                                 comment: "Hero card button to stop the server"),
                          systemImage: "stop.fill")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.destructive))
            }

        case .starting, .stopping:
            Button(String(localized: "server.hero.working",
                          defaultValue: "Working…",
                          comment: "Hero card button label shown while the server is transitioning between states")) { }
                .buttonStyle(.omlx(.normal))
                .disabled(true)

        case .stopped, .failed:
            Button {
                _ = try? services.startServer()
            } label: {
                Label(String(localized: "server.hero.start",
                             defaultValue: "Start Server",
                             comment: "Hero card button to start the server"),
                      systemImage: "play.fill")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.omlx(.primary))
            .disabled(!services.hasServer)
        }
    }

    private var pillStatus: StatusPill.Status {
        if services.isRemoteBackend {
            switch services.remoteState {
            case .reachable:      return .running
            case .checking:       return .starting
            case .unauthorized:   return .custom(color: theme.amberDot,
                                                  label: String(localized: "server.hero.pill.key_rejected",
                                                                defaultValue: "Key rejected",
                                                                comment: "Status pill label when the other oMLX server answers but does not accept the API key"),
                                                  fillBg: true)
            case .unreachable, .invalidAddress: return .error
            case .idle:           return .stopped
            }
        }
        switch services.serverState {
        case .running:      return .running
        case .starting:     return .starting
        case .stopping:     return .stopping
        case .stopped:      return .stopped
        case .unresponsive: return .custom(color: theme.amberDot,
                                            label: String(localized: "server.hero.pill.unresponsive",
                                                          defaultValue: "Unresponsive",
                                                          comment: "Status pill label when the server process is alive but not answering health checks"),
                                            fillBg: true)
        case .failed:       return .error
        }
    }

    private var remoteSubtitle: String {
        let address = services.config.endpoint.displayAddress
        switch services.remoteState {
        case .reachable:
            return String(localized: "server.hero.subtitle.remote_using",
                          defaultValue: "Using the server at \(address)",
                          comment: "Hero subtitle when the app uses another oMLX server and it answers; the placeholder is its address")
        case .checking:
            return String(localized: "server.hero.subtitle.remote_checking",
                          defaultValue: "Checking \(address)…",
                          comment: "Hero subtitle while the app checks whether the other oMLX server answers; the placeholder is its address")
        case .unauthorized:
            return String(localized: "server.hero.subtitle.remote_unauthorized",
                          defaultValue: "The server at \(address) did not accept the API key",
                          comment: "Hero subtitle when the other oMLX server answers but rejects the API key; the placeholder is its address")
        case .invalidAddress:
            return String(localized: "server.hero.subtitle.remote_invalid",
                          defaultValue: "Remote server address is not valid",
                          comment: "Hero subtitle when the other oMLX server's address cannot be parsed")
        case .unreachable, .idle:
            return String(localized: "server.hero.subtitle.remote_unreachable",
                          defaultValue: "Can't reach \(address)",
                          comment: "Hero subtitle when the other oMLX server does not answer; the placeholder is its address")
        }
    }

    private var subtitle: String {
        if services.isRemoteBackend { return remoteSubtitle }
        let host = services.config.host
        let port = services.config.port
        switch services.serverState {
        case .running, .unresponsive:
            return String(localized: "server.hero.subtitle.listening",
                          defaultValue: "Listening on \(host):\(String(port))",
                          comment: "Hero subtitle while server is running; placeholders are host and port (port is plain integer, no grouping)")
        case .starting:
            return String(localized: "server.hero.subtitle.starting",
                          defaultValue: "Starting on \(host):\(String(port))…",
                          comment: "Hero subtitle while server is starting up; placeholders are host and port (port is plain integer, no grouping)")
        case .stopping:
            return String(localized: "server.hero.subtitle.stopping",
                          defaultValue: "Stopping…",
                          comment: "Hero subtitle while server is shutting down")
        case .stopped:
            return String(localized: "server.hero.subtitle.not_running",
                          defaultValue: "Not running",
                          comment: "Hero subtitle when server is stopped")
        case .failed(let m):
            return m
        }
    }

    @ViewBuilder
    private var heroBackground: some View {
        theme.groupBg
    }

    private var title: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
        let trimmed = version?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? "oMLX" : "oMLX \(trimmed)"
    }
}

// MARK: - Backend

/// Which oMLX server this app uses: the one it runs on this Mac, or one on
/// another Mac. The choice is read at launch, so Apply writes it and
/// relaunches the app (`AppServices.applyRemoteBackend`).
private struct BackendSection: View {
    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme

    @State private var useRemote = false
    @State private var address = ""
    @State private var apiKey = ""
    @State private var showKey = false
    @State private var check = ConnectionCheck.idle
    @State private var confirming = false
    @State private var isApplying = false
    @State private var applyError: String?

    private enum ConnectionCheck: Equatable {
        case idle
        case checking
        case connected(models: Int)
        /// Reachable and the key lists models, but the admin API refuses it.
        case adminDenied
        case failed(String)
    }

    private var pendingEndpoint: OMLXEndpoint? {
        OMLXEndpoint(remoteAddress: address, apiKey: apiKey)
    }

    private var hasPendingChange: Bool {
        let config = services.config
        if useRemote != config.remoteEnabled { return true }
        guard useRemote else { return false }
        return address.trimmingCharacters(in: .whitespacesAndNewlines)
            != config.remoteAddress.trimmingCharacters(in: .whitespacesAndNewlines)
            || AppConfig.cleanedKey(apiKey) != config.remoteAPIKey
    }

    private var canApply: Bool {
        hasPendingChange && !isApplying && (!useRemote || pendingEndpoint != nil)
    }

    var body: some View {
        SectionHeader(
            String(localized: "server.section.backend",
                   defaultValue: "Backend",
                   comment: "Section heading for choosing which oMLX server this app uses"),
            subtitle: String(localized: "server.section.backend.sub",
                             defaultValue: "The oMLX server this app uses. With a remote server, nothing runs on this Mac: models, settings and logs are that server's.",
                             comment: "Subtitle for the Backend section in Server screen")
        )
        ListGroup {
            Row(
                label: String(localized: "server.backend.row.server",
                              defaultValue: "Server",
                              comment: "Row label for choosing the oMLX server this app uses"),
                isLast: !useRemote
            ) {
                Segmented(
                    selection: $useRemote,
                    titleKey: "server.backend.row.server",
                    options: [
                        (false, String(localized: "server.backend.this_mac",
                                       defaultValue: "This Mac",
                                       comment: "Segment: the app uses the oMLX server it runs on this Mac")),
                        (true, String(localized: "server.backend.remote",
                                      defaultValue: "Remote server",
                                      comment: "Segment: the app uses an oMLX server on another Mac")),
                    ]
                )
                .frame(width: .controlMedium)
            }
            if useRemote {
                Row(
                    label: String(localized: "server.backend.row.address",
                                  defaultValue: "Address",
                                  comment: "Row label for the other oMLX server's address"),
                    sublabel: String(localized: "server.backend.row.address.sub",
                                     defaultValue: "A host, host:port, or https:// address. The default port is 8000.",
                                     comment: "Sublabel for the other oMLX server's address")
                ) {
                    TextInput(text: $address, placeholder: "studio.local:8000",
                              mono: true, width: .controlWide)
                }
                Row(
                    label: String(localized: "server.backend.row.key",
                                  defaultValue: "API key",
                                  comment: "Row label for the other oMLX server's API key"),
                    sublabel: String(localized: "server.backend.row.key.sub",
                                     defaultValue: "The main API key of that server. Stored in settings.json with the rest of this install's settings, like the local key.",
                                     comment: "Sublabel for the other oMLX server's API key")
                ) {
                    HStack(spacing: 4) {
                        TextInput(text: $apiKey, placeholder: "sk-omlx-…", isSecure: !showKey,
                                  mono: true, width: .controlMedium)
                        Button {
                            showKey.toggle()
                        } label: {
                            Image(systemName: showKey ? "eye.slash" : "eye")
                                .frame(width: 26, height: 26)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help(showKey
                            ? String(localized: "server.backend.key.hide",
                                     defaultValue: "Hide the API key",
                                     comment: "Tooltip on the eye button when the other server's API key is visible")
                            : String(localized: "server.backend.key.show",
                                     defaultValue: "Show the API key",
                                     comment: "Tooltip on the eye button when the other server's API key is masked"))
                    }
                }
                Row(
                    label: String(localized: "server.backend.row.connection",
                                  defaultValue: "Connection",
                                  comment: "Row label for the button that checks the other oMLX server"),
                    sublabel: connectionSublabel,
                    warning: connectionWarning,
                    isLast: true
                ) {
                    Button(String(localized: "server.backend.test",
                                  defaultValue: "Test connection",
                                  comment: "Button that checks the other oMLX server's address and API key")) {
                        Task { await testConnection() }
                    }
                    .buttonStyle(.omlx(.normal, size: .small))
                    .disabled(check == .checking)
                }
            }
        }
        .onChange(of: useRemote) { check = .idle }
        .onChange(of: address) { check = .idle }
        .onChange(of: apiKey) { check = .idle }

        if useRemote, pendingEndpoint?.sendsKeyInTheClear == true {
            HintLine(
                text: String(localized: "server.backend.insecure",
                             defaultValue: "This address uses http, so the API key and everything this app sends to that server are not encrypted on the network. Use a network you trust, or an https address.",
                             comment: "Warning when the app is set to use another oMLX server over plain http"),
                icon: "exclamationmark.triangle"
            )
            .padding(.horizontal, 28)
            .padding(.top, 6)
        }

        FooterBar(error: applyError) {
            Button(String(localized: "server.backend.apply",
                          defaultValue: "Apply and Relaunch",
                          comment: "Button that saves the backend choice (this Mac or another server) and relaunches the app")) {
                confirming = true
            }
            .buttonStyle(.omlx(.primary))
            .disabled(!canApply)
        }
        .onAppear(perform: seed)
        .onChange(of: services.config) { seed() }
        .alert(
            String(localized: "server.backend.confirm.title",
                   defaultValue: "Switch server?",
                   comment: "Title of the confirmation shown before switching the oMLX server this app uses"),
            isPresented: $confirming
        ) {
            Button(String(localized: "common.cancel", defaultValue: "Cancel"), role: .cancel) {}
            Button(String(localized: "common.ok", defaultValue: "OK")) { apply() }
        } message: {
            Text(confirmMessage)
        }
    }

    private var confirmMessage: String {
        if useRemote, let endpoint = pendingEndpoint {
            let shown = endpoint.displayAddress
            return String(localized: "server.backend.confirm.message.remote",
                          defaultValue: "oMLX will stop the server on this Mac and reopen using \(shown).",
                          comment: "Confirmation message before switching to another oMLX server; the placeholder is its address")
        }
        return String(localized: "server.backend.confirm.message.local",
                      defaultValue: "oMLX will reopen and start the server on this Mac.",
                      comment: "Confirmation message before switching back to the oMLX server on this Mac")
    }

    private func seed() {
        useRemote = services.config.remoteEnabled
        address = services.config.remoteAddress
        apiKey = services.config.remoteAPIKey ?? ""
    }

    private func apply() {
        let enabled = useRemote
        let address = address.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = AppConfig.cleanedKey(apiKey)
        applyError = nil
        isApplying = true
        Task {
            do {
                try await services.applyRemoteBackend(enabled: enabled, address: address, apiKey: key)
            } catch {
                applyError = error.omlxDescription
                isApplying = false
            }
        }
    }

    private var connectionSublabel: String? {
        switch check {
        case .idle, .failed:
            nil
        case .checking:
            String(localized: "server.backend.test.checking",
                   defaultValue: "Checking…",
                   comment: "Shown while the app checks the other oMLX server")
        case .connected(let models):
            String(localized: "server.backend.test.connected",
                   defaultValue: "Connected. Chat models: \(models)",
                   comment: "Shown when the other oMLX server answered; the argument is the number of chat models")
        case .adminDenied:
            String(localized: "server.backend.test.admin_denied",
                   defaultValue: "Connected, but this key cannot open the admin API. Use that server's main API key, not a sub key.",
                   comment: "Shown when the other oMLX server accepts the key for models but not for its admin API")
        }
    }

    private var connectionWarning: String? {
        if case .failed(let message) = check { message } else { nil }
    }

    /// Parse, then `/health`, the model list, and the admin login: the
    /// screens need the main key, which sub keys do not satisfy.
    private func testConnection() async {
        guard let endpoint = pendingEndpoint else {
            check = .failed(String(localized: "server.backend.test.invalid_address",
                                   defaultValue: "Enter a valid address, such as studio.local:8000.",
                                   comment: "Shown when the other oMLX server's address is empty or not valid"))
            return
        }
        check = .checking
        do {
            try await endpoint.checkHealth()
            let all = try await endpoint.models()
            do {
                try await endpoint.checkAdminLogin()
            } catch OMLXClientError.unauthenticated {
                check = .adminDenied
                return
            }
            check = .connected(models: all.filter {
                AskAIScreen.isChatModel(type: $0.modelType, virtual: $0.virtual)
            }.count)
        } catch {
            check = .failed(Self.message(for: error))
        }
    }

    private static func message(for error: Error) -> String {
        if let error = error as? OMLXClientError {
            switch error {
            case .unauthenticated:
                return String(localized: "server.backend.test.needs_key",
                              defaultValue: "That server asks for an API key.",
                              comment: "Shown when the other oMLX server asks for an API key and none is set")
            case .http(let status, _) where status == 401 || status == 403:
                return String(localized: "server.backend.test.key_rejected",
                              defaultValue: "That server did not accept this API key.",
                              comment: "Shown when the other oMLX server rejects the API key")
            default:
                break
            }
        }
        let detail = error.omlxDescription
        return String(localized: "server.backend.test.unreachable",
                      defaultValue: "Can't reach that server: \(detail)",
                      comment: "Shown when the other oMLX server does not answer; the argument is the error")
    }
}

// MARK: - Default profile editor

/// Editor for `GlobalSettings.sampling` (server-wide defaults).
///
/// The HTML design surfaces 15 fields here. The server's `SamplingSettings`
/// dataclass currently backs 6 (context, max-tokens, temperature, top_p,
/// top_k, repetition_penalty). The other 9 (min_p, presence_penalty, TTL,
/// thinking, force sampling, pin, etc) are per-model only — we render
/// them disabled with a "Per-model only" tag so the user knows where to
/// look. Expander mirrors the design's "Show all fields…" affordance.
private struct ServerDefaultProfileEditor: View {
    @Bindable var vm: ServerScreenVM

    @State private var expanded: Bool = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ListGroup {
                Row(label: String(localized: "server.profile.context_window",
                                  defaultValue: "Context Window",
                                  comment: "Row label for the context window field in Default Profile"),
                    sublabel: String(localized: "server.profile.context_window.sub",
                                     defaultValue: "Maximum prompt + completion tokens.",
                                     comment: "Sublabel for the context window field")) {
                    TextInput(text: $vm.samplingContextText, mono: true, suffix: "tk", width: .controlCompact)
                }
                Row(label: String(localized: "server.profile.max_tokens",
                                  defaultValue: "Max Tokens",
                                  comment: "Row label for the max tokens field"),
                    sublabel: String(localized: "server.profile.max_tokens.sub",
                                     defaultValue: "Server-wide cap on generated tokens.",
                                     comment: "Sublabel for the max tokens field")) {
                    TextInput(text: $vm.samplingMaxTokensText, mono: true, suffix: "tk", width: .controlCompact)
                }
                Row(label: String(localized: "server.profile.temperature",
                                  defaultValue: "Temperature",
                                  comment: "Row label for the temperature field"),
                    sublabel: String(localized: "server.profile.temperature.sub",
                                     defaultValue: "Sampling randomness (0–2).",
                                     comment: "Sublabel for the temperature field")) {
                    TextInput(text: $vm.samplingTemperatureText, placeholder: "0.7", mono: true, width: .controlNarrow)
                }
                Row(label: String(localized: "server.profile.top_p",
                                  defaultValue: "Top P",
                                  comment: "Row label for the Top P field"),
                    sublabel: String(localized: "server.profile.top_p.sub",
                                     defaultValue: "Nucleus sampling cutoff (0–1).",
                                     comment: "Sublabel for the Top P field")) {
                    TextInput(text: $vm.samplingTopPText, mono: true, width: .controlNarrow)
                }
                Row(label: String(localized: "server.profile.top_k",
                                  defaultValue: "Top K",
                                  comment: "Row label for the Top K field"),
                    sublabel: String(localized: "server.profile.top_k.sub",
                                     defaultValue: "Limit candidates to top K. 0 = disabled.",
                                     comment: "Sublabel for the Top K field")) {
                    TextInput(text: $vm.samplingTopKText, mono: true, width: .controlNarrow)
                }
                Row(label: String(localized: "server.profile.repetition_penalty",
                                  defaultValue: "Repetition Penalty",
                                  comment: "Row label for the repetition penalty field"),
                    sublabel: String(localized: "server.profile.repetition_penalty.sub",
                                     defaultValue: "Penalize repeated tokens.",
                                     comment: "Sublabel for the repetition penalty field"),
                    isLast: !expanded
                ) {
                    TextInput(text: $vm.samplingRepetitionPenaltyText, mono: true, width: .controlNarrow)
                }
                if expanded {
                    // The remaining design rows aren't server-backed yet —
                    // surfaced disabled with a "Per-model only" pill so the
                    // user knows to set them in the per-model Advanced tab.
                    perModelOnlyRow(label: String(localized: "server.profile.min_p",
                                                  defaultValue: "Min P",
                                                  comment: "Disabled row label for Min P"),
                                    note: String(localized: "server.profile.min_p.note",
                                                 defaultValue: "Server defaults don't include min_p; set on a model profile.",
                                                 comment: "Note explaining Min P is per-model only"))
                    perModelOnlyRow(label: String(localized: "server.profile.presence_penalty",
                                                  defaultValue: "Presence Penalty",
                                                  comment: "Disabled row label for Presence Penalty"),
                                    note: String(localized: "server.profile.presence_penalty.note",
                                                 defaultValue: "Per-model only.",
                                                 comment: "Note marking Presence Penalty as per-model only"))
                    perModelOnlyRow(label: String(localized: "server.profile.ttl",
                                                  defaultValue: "TTL",
                                                  comment: "Disabled row label for TTL"),
                                    note: String(localized: "server.profile.ttl.note",
                                                 defaultValue: "Per-model only — see Models → [model] → Basic.",
                                                 comment: "Note explaining TTL is per-model only and where to find it"))
                    perModelOnlyRow(label: String(localized: "server.profile.enable_thinking",
                                                  defaultValue: "Enable Thinking",
                                                  comment: "Disabled row label for Enable Thinking"),
                                    note: String(localized: "server.profile.enable_thinking.note",
                                                 defaultValue: "Per-model only — set on a profile.",
                                                 comment: "Note marking Enable Thinking as per-model only"))
                    perModelOnlyRow(label: String(localized: "server.profile.limit_tool_output",
                                                  defaultValue: "Limit Tool Output",
                                                  comment: "Disabled row label for Limit Tool Output"),
                                    note: String(localized: "server.profile.limit_tool_output.note",
                                                 defaultValue: "Per-model only.",
                                                 comment: "Note marking Limit Tool Output as per-model only"))
                    perModelOnlyRow(label: String(localized: "server.profile.force_sampling",
                                                  defaultValue: "Force Sampling",
                                                  comment: "Disabled row label for Force Sampling"),
                                    note: String(localized: "server.profile.force_sampling.note",
                                                 defaultValue: "Per-model only.",
                                                 comment: "Note marking Force Sampling as per-model only"))
                    perModelOnlyRow(label: String(localized: "server.profile.pin_in_memory",
                                                  defaultValue: "Pin in memory",
                                                  comment: "Disabled row label for Pin in memory"),
                                    note: String(localized: "server.profile.pin_in_memory.note",
                                                 defaultValue: "Per-model only - see Models > [model] > Advanced.",
                                                 comment: "Note explaining where to configure Pin in memory"))
                    perModelOnlyRow(label: String(localized: "server.profile.speculative_decoding",
                                                  defaultValue: "Speculative decoding",
                                                  comment: "Disabled row label for Speculative decoding"),
                                    note: String(localized: "server.profile.speculative_decoding.note",
                                                 defaultValue: "Per-model only — see Models → [model] → Advanced.",
                                                 comment: "Note explaining Speculative decoding is per-model only and where to find it"),
                                    isLast: true)
                }
            }
            HStack {
                Spacer()
                Button {
                    expanded.toggle()
                } label: {
                    Text(expanded
                         ? String(localized: "server.profile.show_fewer",
                                  defaultValue: "Show fewer",
                                  comment: "Toggle label to collapse the advanced sampling fields list")
                         : String(localized: "server.profile.show_all",
                                  defaultValue: "Show all fields…",
                                  comment: "Toggle label to expand the advanced sampling fields list"))
                        .font(.omlxText(11.5, weight: .medium))
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .padding(.horizontal, 14)
                .padding(.top, 4)
                .padding(.bottom, 10)
            }
        }
    }

    @ViewBuilder
    private func perModelOnlyRow(label: String, note: String, isLast: Bool = false) -> some View {
        Row(label: label, sublabel: note, isLast: isLast) {
            Text(String(localized: "server.profile.per_model_only",
                        defaultValue: "Per-model only",
                        comment: "Pill text marking a sampling field as configurable only on individual model profiles"))
                .font(.omlxText(10.5, weight: .heavy))
                .kerning(0.6)
                .textCase(.uppercase)
                .foregroundStyle(theme.textTertiary)
                .padding(.horizontal, 8)
                .frame(height: 22)
                .background(
                    Capsule().fill(theme.codeBg)
                )
                .overlay(
                    Capsule().strokeBorder(theme.inputBorder, lineWidth: 0.5)
                )
        }
    }
}

// MARK: - Endpoints

private struct APIEndpointsList: View {
    let host: String
    let port: Int
    /// The other server this app uses, when there is one: its URLs replace
    /// the local host and port (and keep its scheme).
    var remote: OMLXEndpoint? = nil

    var body: some View {
        ListGroup {
            Row(label: String(localized: "server.endpoint.openai",
                              defaultValue: "OpenAI-compatible",
                              comment: "API endpoint row label for the OpenAI-compatible base URL")) {
                CodeChip(value: endpoint(path: "/v1"))
            }
            Row(label: String(localized: "server.endpoint.anthropic",
                              defaultValue: "Anthropic / Claude Code",
                              comment: "API endpoint row label for the Anthropic/Claude Code base URL")) {
                CodeChip(value: endpoint(path: ""))
            }
            Row(label: String(localized: "server.endpoint.health",
                              defaultValue: "Health probe",
                              comment: "API endpoint row label for the health probe URL")) {
                CodeChip(value: endpoint(path: "/health"))
            }
            Row(label: String(localized: "server.endpoint.metrics",
                              defaultValue: "Metrics (Prometheus)",
                              comment: "API endpoint row label for the Prometheus metrics URL"),
                isLast: true) {
                CodeChip(value: endpoint(path: "/metrics"))
            }
        }
    }

    /// Build via `AppConfig.httpURL` so an IPv6 literal host is bracketed
    /// (`http://[::1]:8000/v1`) instead of the malformed `http://::1:8000/v1`
    /// raw interpolation would produce. Fall back to interpolation only if the
    /// URL is somehow unconstructable.
    private func endpoint(path: String) -> String {
        if let remote, let url = remote.httpURL(path: path) { return url.absoluteString }
        return AppConfig.httpURL(host: host, port: port, path: path)?.absoluteString
            ?? "http://\(host):\(port)\(path)"
    }
}

// MARK: - Advanced disclosure

/// Phase 4 — Server identity / protocol knobs that the average user never
/// touches: request limits, `server_aliases` (extra host names the server
/// identifies as for cookie + host-header purposes), and SSE keep-alive.
/// Hidden behind a chevron so they don't crowd the main ServerScreen surface,
/// but rendered inline (not in a popover) so power users can scroll-find them.
private struct ServerAdvancedSection: View {
    @Bindable var vm: ServerScreenVM

    @State private var expanded: Bool = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Custom disclosure header — DisclosureGroup's default styling
            // doesn't match the rest of the screen (uses SF Pro vs omlxText,
            // adds its own padding). Roll our own to stay consistent with
            // SectionHeader.
            Button {
                expanded.toggle()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(theme.textSecondary)
                        .rotationEffect(.degrees(expanded ? 90 : 0))
                        .animation(.easeOut(duration: 0.12), value: expanded)
                    Text(String(localized: "server.section.advanced",
                                defaultValue: "Advanced",
                                comment: "Disclosure header for the advanced server settings"))
                        .font(.omlxText(DesignTokens.FontSize.subheadline, weight: .semibold))
                        .foregroundStyle(theme.textSecondary)
                        .textCase(.uppercase)
                        .kerning(0.6)
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            // Matches SectionHeader's indent (card gutter + row inset).
            .padding(.horizontal, 28)
            .padding(.top, 22)
            .padding(.bottom, 8)

            if expanded {
                ListGroup {
                    Row(
                        label: String(localized: "server.advanced.max_audio_upload_size",
                                      defaultValue: "Maximum Audio Upload Size",
                                      comment: "Advanced row label for the audio upload size limit"),
                        sublabel: String(localized: "server.advanced.max_audio_upload_size.sub",
                                         defaultValue: "Maximum file size accepted by audio transcription and processing endpoints. Higher values can increase memory use for concurrent requests.",
                                         comment: "Sublabel describing the audio upload size limit")
                    ) {
                        TextInput(
                            text: $vm.maxAudioUploadSizeText,
                            placeholder: "100MB",
                            mono: true,
                            width: .controlCompact
                        )
                    }
                    Row(
                        label: String(localized: "server.advanced.sse_keepalive",
                                      defaultValue: "SSE Keep-Alive Mode",
                                      comment: "Advanced row label for the SSE keep-alive mode picker"),
                        sublabel: String(localized: "server.advanced.sse_keepalive.sub",
                                         defaultValue: "How the server keeps long-lived SSE streams open. \"chunk\" emits an empty data line, \"comment\" emits `: ping`, \"off\" disables.",
                                         comment: "Sublabel describing the SSE keep-alive mode options")
                    ) {
                        Popup(
                            selection: vm.bind($vm.sseKeepaliveMode, save: vm.saveSseKeepaliveMode),
                            width: .controlCompact,
                            options: [
                                ("chunk",   String(localized: "server.advanced.sse_keepalive.chunk",
                                                    defaultValue: "Chunk",
                                                    comment: "SSE keep-alive mode option: empty data chunk")),
                                ("comment", String(localized: "server.advanced.sse_keepalive.comment",
                                                    defaultValue: "Comment",
                                                    comment: "SSE keep-alive mode option: comment ping")),
                                ("off",     String(localized: "server.advanced.sse_keepalive.off",
                                                    defaultValue: "Off",
                                                    comment: "SSE keep-alive mode option: disabled")),
                            ]
                        )
                    }
                    Row(
                        label: String(localized: "server.advanced.aliases",
                                      defaultValue: "Server Aliases",
                                      comment: "Advanced row label for the server aliases input"),
                        sublabel: String(localized: "server.advanced.aliases.sub",
                                         defaultValue: "Extra host names the server identifies as. Comma-separated. Used for cookie / Host header matching.",
                                         comment: "Sublabel describing the server aliases input format"),
                        isLast: true
                    ) {
                        TextInput(
                            text: $vm.serverAliasesText,
                            placeholder: "omlx.local, oMLX.lan",
                            mono: true,
                            width: .controlWide
                        )
                    }
                }
            }
        }
    }
}


