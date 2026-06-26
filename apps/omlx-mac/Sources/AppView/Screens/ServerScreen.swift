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

            SectionHeader(String(localized: "server.section.network",
                                  defaultValue: "Network",
                                  comment: "Section heading for the Network rows in Server screen"))
            ListGroup {
                Row(label: String(localized: "server.row.listen_address",
                                  defaultValue: "Listen Address",
                                  comment: "Row label for the listen-address picker in Server screen")) {
                    Popup(
                        selection: vm.bind($vm.host, save: { vm.saveHost(services: services) }),
                        width: 220,
                        options: [
                            ("127.0.0.1", String(localized: "server.host.local_only",
                                                  defaultValue: "127.0.0.1 (Local only)",
                                                  comment: "Listen-address popup option for loopback only")),
                            ("0.0.0.0", String(localized: "server.host.all_networks",
                                                defaultValue: "0.0.0.0 (All networks)",
                                                comment: "Listen-address popup option for binding to all interfaces")),
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
                    TextInput(text: $vm.portText, mono: true, width: 90)
                }
            }

            SectionHeader(String(localized: "server.section.endpoints",
                                  defaultValue: "API Endpoints",
                                  comment: "Section heading for the API endpoints list"))
            APIEndpointsList(host: vm.effectiveHost, port: vm.effectivePort)

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
                    Toggle("", isOn: vm.bind($vm.autoStartOnLaunch, save: {
                        vm.saveAutoStartOnLaunch(services: services)
                    }))
                    .labelsHidden()
                    .toggleStyle(.switch)
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
                        width: 130,
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
                    TextInput(text: $vm.basePathText, mono: true, width: 280)
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
                    Toggle("", isOn: $vm.hfCacheEnabled)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
            }
            ServerAdvancedSection(vm: vm)

            HStack {
                Spacer()
                Button(String(localized: "server.button.apply",
                              defaultValue: "Apply",
                              comment: "Button to apply pending server settings: port, default profile, storage, and aliases")) {
                    vm.applyServerSettings(services: services)
                }
                    .buttonStyle(.omlx(.primary))
                    .disabled(!vm.hasPendingServerChanges(services: services)
                              || vm.isMovingBasePath)
            }
            .padding(.horizontal, 18)
            .padding(.top, 6)

            HintFooter(error: vm.lastError)
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
                        .font(.omlxText(13, weight: .medium))
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
                .font(.omlxMono(10, weight: .semibold))
                .foregroundStyle(index == 0 ? theme.accent : theme.textSecondary)
                .frame(width: 52, alignment: .leading)

            TextInput(text: Binding(
                get: { vm.modelDirText(at: index) },
                set: { vm.setModelDirText($0, at: index) }
            ), mono: true, width: 340)

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
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .padding(.horizontal, 14)
        .padding(.bottom, 14)
    }

    @ViewBuilder
    private var buttons: some View {
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

    private var subtitle: String {
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
                    TextInput(text: $vm.samplingContextText, mono: true, suffix: "tk", width: 110)
                }
                Row(label: String(localized: "server.profile.max_tokens",
                                  defaultValue: "Max Tokens",
                                  comment: "Row label for the max tokens field"),
                    sublabel: String(localized: "server.profile.max_tokens.sub",
                                     defaultValue: "Server-wide cap on generated tokens.",
                                     comment: "Sublabel for the max tokens field")) {
                    TextInput(text: $vm.samplingMaxTokensText, mono: true, suffix: "tk", width: 110)
                }
                Row(label: String(localized: "server.profile.temperature",
                                  defaultValue: "Temperature",
                                  comment: "Row label for the temperature field"),
                    sublabel: String(localized: "server.profile.temperature.sub",
                                     defaultValue: "Sampling randomness (0–2).",
                                     comment: "Sublabel for the temperature field")) {
                    TextInput(text: $vm.samplingTemperatureText, placeholder: "0.7", mono: true, width: 90)
                }
                Row(label: String(localized: "server.profile.top_p",
                                  defaultValue: "Top P",
                                  comment: "Row label for the Top P field"),
                    sublabel: String(localized: "server.profile.top_p.sub",
                                     defaultValue: "Nucleus sampling cutoff (0–1).",
                                     comment: "Sublabel for the Top P field")) {
                    TextInput(text: $vm.samplingTopPText, mono: true, width: 90)
                }
                Row(label: String(localized: "server.profile.top_k",
                                  defaultValue: "Top K",
                                  comment: "Row label for the Top K field"),
                    sublabel: String(localized: "server.profile.top_k.sub",
                                     defaultValue: "Limit candidates to top K. 0 = disabled.",
                                     comment: "Sublabel for the Top K field")) {
                    TextInput(text: $vm.samplingTopKText, mono: true, width: 90)
                }
                Row(label: String(localized: "server.profile.repetition_penalty",
                                  defaultValue: "Repetition Penalty",
                                  comment: "Row label for the repetition penalty field"),
                    sublabel: String(localized: "server.profile.repetition_penalty.sub",
                                     defaultValue: "Penalize repeated tokens.",
                                     comment: "Sublabel for the repetition penalty field"),
                    isLast: !expanded
                ) {
                    TextInput(text: $vm.samplingRepetitionPenaltyText, mono: true, width: 90)
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
                                                 defaultValue: "Per-model only.",
                                                 comment: "Note marking Pin in memory as per-model only"))
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

    var body: some View {
        ListGroup {
            Row(label: String(localized: "server.endpoint.openai",
                              defaultValue: "OpenAI-compatible",
                              comment: "API endpoint row label for the OpenAI-compatible base URL")) {
                CodeChip(value: "http://\(host):\(port)/v1")
            }
            Row(label: String(localized: "server.endpoint.anthropic",
                              defaultValue: "Anthropic / Claude Code",
                              comment: "API endpoint row label for the Anthropic/Claude Code base URL")) {
                CodeChip(value: "http://\(host):\(port)")
            }
            Row(label: String(localized: "server.endpoint.health",
                              defaultValue: "Health probe",
                              comment: "API endpoint row label for the health probe URL")) {
                CodeChip(value: "http://\(host):\(port)/health")
            }
            Row(label: String(localized: "server.endpoint.metrics",
                              defaultValue: "Metrics (Prometheus)",
                              comment: "API endpoint row label for the Prometheus metrics URL"),
                isLast: true) {
                CodeChip(value: "http://\(host):\(port)/metrics")
            }
        }
    }
}

// MARK: - Advanced disclosure

/// Phase 4 — Server identity / protocol knobs that the average user never
/// touches: `server_aliases` (extra host names the server identifies as for
/// cookie + host-header purposes) and `sse_keepalive_mode`. Hidden behind a
/// chevron so they don't crowd the main ServerScreen surface, but rendered
/// inline (not in a popover) so power users can scroll-find them.
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
                        .font(.omlxText(11, weight: .semibold))
                        .foregroundStyle(theme.textSecondary)
                        .textCase(.uppercase)
                        .kerning(0.6)
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 18)
            .padding(.top, 22)
            .padding(.bottom, 8)

            if expanded {
                ListGroup {
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
                            width: 130,
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
                            width: 320
                        )
                    }
                }
            }
        }
    }
}

// MARK: - Footer hint

private struct HintFooter: View {
    let error: String?
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HintLine(text: String(localized: "server.footer.hint",
                                  defaultValue: "Listen Address, Log Level, and SSE Keep-Alive Mode apply the moment you change them. The rest (port, default profile, storage, aliases) commits when you click Apply. Port and storage changes take effect after a server restart.",
                                  comment: "Hint footer text under the Server screen explaining which controls apply immediately vs. via the Apply button"))
            if let error {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.redDot)
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
    }
}
