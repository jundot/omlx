// PR 9 — Integrations.
//
// Routes Claude Code requests to local models or to the cloud, exposes the
// other named integrations (Codex / OpenCode / OpenClaw / Pi) as model
// popups, and renders a copyable `omlx launch claude` command derived from
// the current selection. The model popups read their options from
// /admin/api/models so the user can only pick something the server
// actually has on disk.
//
// The OpenAI Compatibility section + Connected Apps from the design canvas
// are skipped: there are no matching server fields. We keep every shipped
// row honestly wired.

import SwiftUI

struct IntegrationsScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = IntegrationsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ClaudeCodeSection(vm: vm, client: services.client)
            SetupCommandSection(command: vm.claudeLaunchCommand)
            OtherIntegrationsSection(vm: vm, client: services.client)

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 8)
            }
        }
        .task { await vm.load(client: services.client) }
    }
}

// MARK: - Claude Code

private struct ClaudeCodeSection: View {
    @ObservedObject var vm: IntegrationsScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader(
            "Claude Code",
            subtitle: "Route Claude Code requests to local models or the cloud"
        )

        ListGroup {
            Row(label: "Mode") {
                Segmented(
                    selection: vm.bind($vm.claudeMode, save: {
                        Task { await vm.save(.claudeMode, client: client) }
                    }),
                    options: [("cloud", "Cloud"), ("local", "Local")]
                )
                .frame(width: 160)
            }
            if vm.claudeMode == "local" {
                Row(label: "Opus tier") {
                    Popup(
                        selection: vm.bind($vm.opusModel, save: {
                            Task { await vm.save(.opusModel, client: client) }
                        }),
                        width: 220,
                        options: vm.modelOptions
                    )
                }
                Row(label: "Sonnet tier") {
                    Popup(
                        selection: vm.bind($vm.sonnetModel, save: {
                            Task { await vm.save(.sonnetModel, client: client) }
                        }),
                        width: 220,
                        options: vm.modelOptions
                    )
                }
                Row(
                    label: "Haiku tier",
                    sublabel: "Used for background tasks and tool calls"
                ) {
                    Popup(
                        selection: vm.bind($vm.haikuModel, save: {
                            Task { await vm.save(.haikuModel, client: client) }
                        }),
                        width: 220,
                        options: vm.modelOptions
                    )
                }
            }
            Row(
                label: "Context scaling",
                sublabel: "Stretch context windows for long agentic sessions",
                isLast: true
            ) {
                Toggle("", isOn: vm.bind($vm.contextScaling, save: {
                    Task { await vm.save(.contextScaling, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
        }
    }
}

// MARK: - Setup command

private struct SetupCommandSection: View {
    let command: String
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader("Setup Command")

        ZStack(alignment: .topTrailing) {
            VStack(alignment: .leading, spacing: 4) {
                Text("$ Terminal")
                    .font(.omlxText(10, weight: .semibold))
                    .foregroundStyle(theme.textTertiary)
                    .textCase(.uppercase)
                    .kerning(0.6)
                Text(command)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(12)
            .background(theme.codeBg)
            .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                    .strokeBorder(theme.groupBorder, lineWidth: 0.5)
            )
            .padding(.horizontal, 14)

            CopyButton(value: command)
                .padding(.top, 18)
                .padding(.trailing, 22)
        }
    }
}

private struct CopyButton: View {
    let value: String
    @State private var copied = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Button {
            let pb = NSPasteboard.general
            pb.clearContents()
            pb.setString(value, forType: .string)
            copied = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) {
                copied = false
            }
        } label: {
            Image(systemName: copied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(copied ? theme.successText : theme.textSecondary)
                .padding(5)
                .background(theme.controlBg)
                .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Other integrations

private struct OtherIntegrationsSection: View {
    @ObservedObject var vm: IntegrationsScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader(
            "Other Integrations",
            subtitle: "Default model used by each named integration's launcher"
        )

        ListGroup {
            Row(label: "Codex") {
                Popup(
                    selection: vm.bind($vm.codexModel, save: {
                        Task { await vm.save(.codexModel, client: client) }
                    }),
                    width: 220,
                    options: vm.modelOptions
                )
            }
            Row(label: "OpenCode") {
                Popup(
                    selection: vm.bind($vm.opencodeModel, save: {
                        Task { await vm.save(.opencodeModel, client: client) }
                    }),
                    width: 220,
                    options: vm.modelOptions
                )
            }
            Row(label: "OpenClaw") {
                Popup(
                    selection: vm.bind($vm.openclawModel, save: {
                        Task { await vm.save(.openclawModel, client: client) }
                    }),
                    width: 220,
                    options: vm.modelOptions
                )
            }
            Row(label: "OpenClaw tools profile",
                sublabel: "Which built-in MCP tools the OpenClaw launcher exposes") {
                Popup(
                    selection: vm.bind($vm.openclawToolsProfile, save: {
                        Task { await vm.save(.openclawToolsProfile, client: client) }
                    }),
                    width: 160,
                    options: [
                        ("minimal",    "Minimal"),
                        ("coding",     "Coding"),
                        ("messaging",  "Messaging"),
                        ("full",       "Full"),
                    ]
                )
            }
            Row(label: "Pi", isLast: true) {
                Popup(
                    selection: vm.bind($vm.piModel, save: {
                        Task { await vm.save(.piModel, client: client) }
                    }),
                    width: 220,
                    options: vm.modelOptions
                )
            }
        }
    }
}

// MARK: - View model

@MainActor
final class IntegrationsScreenVM: ObservableObject {
    enum Field: Sendable {
        case claudeMode, opusModel, sonnetModel, haikuModel, contextScaling
        case codexModel, opencodeModel, openclawModel, piModel, openclawToolsProfile
    }

    // Claude Code
    @Published var claudeMode: String = "cloud"
    @Published var opusModel: String = ""
    @Published var sonnetModel: String = ""
    @Published var haikuModel: String = ""
    @Published var contextScaling: Bool = false

    // Other integrations
    @Published var codexModel: String = ""
    @Published var opencodeModel: String = ""
    @Published var openclawModel: String = ""
    @Published var piModel: String = ""
    @Published var openclawToolsProfile: String = "coding"

    @Published private(set) var availableModels: [String] = []
    @Published var lastError: String?

    /// Popup options: a leading "Select model…" placeholder + every model id.
    var modelOptions: [(String, String)] {
        var out: [(String, String)] = [("", "Select model…")]
        for id in availableModels {
            out.append((id, id))
        }
        return out
    }

    /// Composed `omlx launch claude` command — kept reactive on field
    /// changes so the user sees their selection immediately.
    var claudeLaunchCommand: String {
        if claudeMode == "cloud" {
            return "omlx launch claude"
        }
        let opus   = opusModel.isEmpty   ? "<opus-model>"   : opusModel
        let sonnet = sonnetModel.isEmpty ? "<sonnet-model>" : sonnetModel
        let haiku  = haikuModel.isEmpty  ? "<haiku-model>"  : haikuModel
        return "omlx launch claude --opus \(opus) --sonnet \(sonnet) --haiku \(haiku)"
    }

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

    func load(client: OMLXClient) async {
        do {
            // Settings
            let settings = try await client.getGlobalSettings()
            if let cc = settings.claudeCode {
                self.claudeMode      = cc.mode ?? "cloud"
                self.opusModel       = cc.opusModel ?? ""
                self.sonnetModel     = cc.sonnetModel ?? ""
                self.haikuModel      = cc.haikuModel ?? ""
                self.contextScaling  = cc.contextScalingEnabled ?? false
            }
            if let it = settings.integrations {
                self.codexModel           = it.codexModel ?? ""
                self.opencodeModel        = it.opencodeModel ?? ""
                self.openclawModel        = it.openclawModel ?? ""
                self.piModel              = it.piModel ?? ""
                self.openclawToolsProfile = it.openclawToolsProfile ?? "coding"
            }

            // Available models
            let models = try await client.listModels().models
            self.availableModels = models.map { $0.id }
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    func save(_ field: Field, client: OMLXClient) async {
        var patch = GlobalSettingsPatch()
        switch field {
        case .claudeMode:           patch.claudeCodeMode = claudeMode
        case .opusModel:            patch.claudeCodeOpusModel = opusModel
        case .sonnetModel:          patch.claudeCodeSonnetModel = sonnetModel
        case .haikuModel:           patch.claudeCodeHaikuModel = haikuModel
        case .contextScaling:       patch.claudeCodeContextScalingEnabled = contextScaling
        case .codexModel:           patch.integrationsCodexModel = codexModel
        case .opencodeModel:        patch.integrationsOpencodeModel = opencodeModel
        case .openclawModel:        patch.integrationsOpenclawModel = openclawModel
        case .piModel:              patch.integrationsPiModel = piModel
        case .openclawToolsProfile: patch.integrationsOpenclawToolsProfile = openclawToolsProfile
        }
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
}
