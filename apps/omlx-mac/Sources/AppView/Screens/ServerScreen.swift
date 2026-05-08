// PR 7 — real Server screen. ServerHero shows live ServerProcess.State and
// drives Start / Stop / Restart through AppServices. Network + Logging rows
// read/write `/admin/api/global-settings`.
//
// Scope vs design: rows whose backing field doesn't exist server-side yet
// (CORS, HTTPS, Request Timeout, Telemetry, GPU memory, KV-cache quant) are
// deferred until those settings are added to GlobalSettingsRequest. We keep
// the shipped surface honest: every row in this screen is fully wired.

import SwiftUI

struct ServerScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = ServerScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ServerHero()

            SectionHeader("Network")
            ListGroup {
                Row(label: "Listen Address") {
                    Popup(
                        selection: vm.bind($vm.host, save: vm.saveHost),
                        width: 220,
                        options: [
                            ("127.0.0.1", "127.0.0.1 (Local only)"),
                            ("0.0.0.0", "0.0.0.0 (All networks)"),
                            ("localhost", "localhost"),
                        ]
                    )
                }
                Row(
                    label: "Port",
                    sublabel: "Default 8080. Restart server to apply."
                ) {
                    TextInput(text: $vm.portText, mono: true, width: 90)
                        .onSubmit(vm.savePort)
                }
                Row(
                    label: "Max Concurrent Requests",
                    sublabel: "Cap on simultaneous /v1 requests.",
                    isLast: true
                ) {
                    TextInput(text: $vm.maxConcurrentText, mono: true, width: 90)
                        .onSubmit(vm.saveMaxConcurrent)
                }
            }

            SectionHeader("API Endpoints")
            APIEndpointsList(host: vm.effectiveHost, port: vm.effectivePort)

            SectionHeader("Logging")
            ListGroup {
                Row(label: "Log Level", isLast: true) {
                    Popup(
                        selection: vm.bind($vm.logLevel, save: vm.saveLogLevel),
                        width: 130,
                        options: [
                            ("error",   "Error"),
                            ("warning", "Warning"),
                            ("info",    "Info"),
                            ("debug",   "Debug"),
                            ("trace",   "Trace"),
                        ]
                    )
                }
            }

            HintFooter(error: vm.lastError)
        }
        .task { await vm.load(client: services.client) }
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

// MARK: - Server hero

private struct ServerHero: View {
    @EnvironmentObject private var services: AppServices
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 16) {
            Squircle(gradient: SquircleGradient.server, size: 52) {
                Text("oM")
                    .font(.omlxText(22, weight: .heavy))
                    .kerning(-0.5)
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 10) {
                    Text("oMLX Server")
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
        .background(heroBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .padding(.horizontal, 14)
        .padding(.bottom, 14)
    }

    @ViewBuilder
    private var buttons: some View {
        switch services.serverState {
        case .running, .unresponsive:
            HStack(spacing: 6) {
                Button {
                    Task { try? await services.restartServer() }
                } label: {
                    Label("Restart", systemImage: "arrow.clockwise")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.normal))

                Button {
                    Task { await services.stopServer() }
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.destructive))
            }

        case .starting, .stopping:
            Button("Working…") { }
                .buttonStyle(.omlx(.normal))
                .disabled(true)

        case .stopped, .failed:
            Button {
                _ = try? services.startServer()
            } label: {
                Label("Start Server", systemImage: "play.fill")
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
        case .unresponsive: return .custom(color: theme.amberDot, label: "Unresponsive", fillBg: true)
        case .failed:       return .error
        }
    }

    private var subtitle: String {
        let host = services.config.host
        let port = services.config.port
        switch services.serverState {
        case .running, .unresponsive:
            return "Listening on \(host):\(port)"
        case .starting:
            return "Starting on \(host):\(port)…"
        case .stopping:
            return "Stopping…"
        case .stopped:
            return "Not running"
        case .failed(let m):
            return m
        }
    }

    @ViewBuilder
    private var heroBackground: some View {
        if theme.isDark {
            LinearGradient(
                colors: [
                    Color(rgb24: 0x30D158, opacity: 0.08),
                    Color(rgb24: 0x0A84FF, opacity: 0.06),
                ],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
        } else {
            LinearGradient(
                colors: [
                    Color(rgb24: 0xF4FAF6),
                    Color(rgb24: 0xF4F7FC),
                ],
                startPoint: .topLeading, endPoint: .bottomTrailing
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
            Row(label: "OpenAI-compatible") {
                CodeChip(value: "http://\(host):\(port)/v1")
            }
            Row(label: "Anthropic / Claude Code") {
                CodeChip(value: "http://\(host):\(port)")
            }
            Row(label: "Health probe") {
                CodeChip(value: "http://\(host):\(port)/health")
            }
            Row(label: "Metrics (Prometheus)", isLast: true) {
                CodeChip(value: "http://\(host):\(port)/metrics")
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
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "info.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(theme.textTertiary)
                Text("Endpoints update live as you change Listen Address and Port. Some changes (host, port) take effect after a server restart.")
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }
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

// MARK: - View model

@MainActor
final class ServerScreenVM: ObservableObject {
    @Published var host: String = "127.0.0.1"
    @Published var portText: String = "8080"
    @Published var maxConcurrentText: String = "8"
    @Published var logLevel: String = "info"
    @Published var lastError: String?

    /// Last applied (effective) values used to build endpoint URLs. Distinct
    /// from `host`/`portText` so the URLs don't flicker mid-edit.
    @Published var effectiveHost: String = "127.0.0.1"
    @Published var effectivePort: Int = 8080

    private weak var client: OMLXClient?
    private var hasLoaded = false

    func load(client: OMLXClient) async {
        self.client = client
        do {
            let dto = try await client.getGlobalSettings()
            self.host = dto.server.host
            self.portText = String(dto.server.port)
            self.logLevel = canonicalize(level: dto.server.logLevel)
            if let mc = dto.scheduler?.maxConcurrentRequests {
                self.maxConcurrentText = String(mc)
            }
            self.effectiveHost = dto.server.host
            self.effectivePort = dto.server.port
            self.lastError = nil
            self.hasLoaded = true
        } catch {
            self.lastError = describe(error)
        }
    }

    func applyConfig(_ config: AppConfig) {
        if !hasLoaded {
            self.host = config.host
            self.portText = String(config.port)
            self.effectiveHost = config.host
            self.effectivePort = config.port
        }
    }

    func saveHost() {
        Task {
            await commit(GlobalSettingsPatch(host: host))
            self.effectiveHost = host
        }
    }

    func savePort() {
        guard let port = Int(portText.trimmingCharacters(in: .whitespaces)),
              (1...65535).contains(port) else {
            self.lastError = "Port must be a number between 1 and 65535."
            return
        }
        Task {
            await commit(GlobalSettingsPatch(port: port))
            self.effectivePort = port
        }
    }

    func saveLogLevel() {
        Task { await commit(GlobalSettingsPatch(logLevel: logLevel)) }
    }

    func saveMaxConcurrent() {
        guard let value = Int(maxConcurrentText.trimmingCharacters(in: .whitespaces)),
              value > 0 else {
            self.lastError = "Max concurrent must be a positive integer."
            return
        }
        Task { await commit(GlobalSettingsPatch(maxConcurrentRequests: value)) }
    }

    /// Build a `Binding` that calls `save` after the value changes. Used for
    /// Popups that have no `onSubmit` hook.
    func bind<T: Equatable>(
        _ binding: Binding<T>,
        save: @escaping () -> Void
    ) -> Binding<T> {
        Binding(
            get: { binding.wrappedValue },
            set: { newValue in
                let changed = binding.wrappedValue != newValue
                binding.wrappedValue = newValue
                if changed { save() }
            }
        )
    }

    private func commit(_ patch: GlobalSettingsPatch) async {
        guard let client else { return }
        do {
            _ = try await client.updateGlobalSettings(patch)
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
    }

    private func canonicalize(level raw: String) -> String {
        switch raw.lowercased() {
        case "warn":   return "warning"
        default:       return raw.lowercased()
        }
    }
}
