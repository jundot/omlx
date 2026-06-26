import SwiftUI

@MainActor
@Observable
final class IntegrationsScreenVM {
    enum Field: Sendable {
        case claudeMode, opusModel, sonnetModel, haikuModel, contextScaling, targetContextSize
        case codexModel, opencodeModel, openclawModel, piModel, openclawToolsProfile
        case hermesModel, copilotModel
        case mcpConfig
    }

    // Claude Code
    var claudeMode: String = "cloud"
    var opusModel: String = ""
    var sonnetModel: String = ""
    var haikuModel: String = ""
    var contextScaling: Bool = false
    /// Free-text editor backing for `claude_code.target_context_size`. The
    /// server stores an `int`; we keep the screen field as a string so the
    /// user can type/clear without intermediate parse errors and we validate
    /// on save.
    var targetContextSizeText: String = "200000"
    /// Last value persisted to the server. Drives the per-section Apply
    /// button under Target Context Size — diverges from
    /// `targetContextSizeText` whenever the user has unsaved edits,
    /// converges on a successful save. Mirrors `mcpConfigLoaded` below.
    private(set) var targetContextSizeLoaded: String = "200000"

    // Other integrations
    var codexModel: String = ""
    var opencodeModel: String = ""
    var openclawModel: String = ""
    var piModel: String = ""
    var openclawToolsProfile: String = "coding"
    var hermesModel: String = ""
    var copilotModel: String = ""

    // MCP
    var mcpConfigPath: String = ""
    /// Last value persisted to the server. Drives the Apply button's
    /// enabled state — diverges from `mcpConfigPath` whenever the user
    /// has unsaved edits, converges on a successful save.
    private(set) var mcpConfigLoaded: String = ""

    private(set) var availableModels: [String] = []
    var lastError: String?

    // Server-resolved fields used by the command builders. Populated from
    // /admin/api/stats so the shell strings reflect whatever host/port/key
    // the running server actually advertises (instead of the local config,
    // which can drift after a hot-reload).
    private(set) var serverHost: String = "127.0.0.1"
    private(set) var serverPort: Int = 8000
    private(set) var serverApiKey: String = ""
    private(set) var cliPrefix: String = "omlx"

    /// Popup options: a leading "Select model…" placeholder + every model id.
    var modelOptions: [(String, String)] {
        var out: [(String, String)] = [
            ("", String(localized: "integrations.model.select_placeholder",
                        defaultValue: "Select model…",
                        comment: "Placeholder option shown at the top of every per-integration model picker"))
        ]
        for id in availableModels {
            out.append((id, id))
        }
        return out
    }

    /// Composed `omlx launch claude` command. Claude tier selections are
    /// persisted in settings, so the launcher reads them without extra flags.
    var claudeLaunchCommand: String {
        "\(cliCommandPrefix) launch claude"
    }

    /// Env-var recipe that runs the real `claude` binary directly. Mirrors
    /// `claudeCodeCommand` in dashboard.js — cloud form unsets the Anthropic
    /// vars, local form sets them to point at the oMLX server.
    var claudeEnvRecipe: String {
        if claudeMode == "cloud" {
            return "env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN "
                 + "-u ANTHROPIC_DEFAULT_OPUS_MODEL -u ANTHROPIC_DEFAULT_SONNET_MODEL "
                 + "-u ANTHROPIC_DEFAULT_HAIKU_MODEL -u API_TIMEOUT_MS "
                 + "-u CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC claude"
        }
        let opus   = opusModel.isEmpty   ? "select-a-model" : opusModel
        let sonnet = sonnetModel.isEmpty ? "select-a-model" : sonnetModel
        let haiku  = haikuModel.isEmpty  ? "select-a-model" : haikuModel
        var parts: [String] = []
        parts.append(Self.shellEnvAssign("ANTHROPIC_BASE_URL",
                                         "http://\(formatDisplayHost(serverHost)):\(serverPort)"))
        if !serverApiKey.isEmpty {
            parts.append(Self.shellEnvAssign("ANTHROPIC_AUTH_TOKEN", serverApiKey))
        }
        parts.append(Self.shellEnvAssign("ANTHROPIC_DEFAULT_OPUS_MODEL",   opus))
        parts.append(Self.shellEnvAssign("ANTHROPIC_DEFAULT_SONNET_MODEL", sonnet))
        parts.append(Self.shellEnvAssign("ANTHROPIC_DEFAULT_HAIKU_MODEL",  haiku))
        parts.append("API_TIMEOUT_MS=3000000")
        parts.append("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1")
        parts.append("claude")
        return parts.joined(separator: " ")
    }

    var codexCommand: String    { "\(cliCommandPrefix) launch codex" }
    var codexAppCommand: String { "\(cliCommandPrefix) launch codex_app" }
    var opencodeCommand: String { "\(cliCommandPrefix) launch opencode" }
    var openclawCommand: String {
        let profile = openclawToolsProfile.isEmpty ? "coding" : openclawToolsProfile
        return "\(cliCommandPrefix) launch openclaw --tools-profile \(profile)"
    }
    var hermesCommand: String   { "\(cliCommandPrefix) launch hermes" }
    var piCommand: String       { "\(cliCommandPrefix) launch pi" }
    var copilotCommand: String  { "\(cliCommandPrefix) launch copilot" }

    var hasPendingMCPChanges: Bool {
        mcpConfigPath.trimmingCharacters(in: .whitespaces) != mcpConfigLoaded
    }

    /// True when the Target Context Size draft diverges from the saved
    /// baseline. The per-section Apply button under that field uses this
    /// to gate its `disabled` state.
    var hasPendingContextSizeChange: Bool {
        targetContextSizeText.trimmingCharacters(in: .whitespaces) != targetContextSizeLoaded
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
                if let target = cc.targetContextSize {
                    let s = String(target)
                    self.targetContextSizeText = s
                    self.targetContextSizeLoaded = s
                }
            }
            if let it = settings.integrations {
                self.codexModel           = it.codexModel ?? ""
                self.opencodeModel        = it.opencodeModel ?? ""
                self.openclawModel        = it.openclawModel ?? ""
                self.piModel              = it.piModel ?? ""
                self.openclawToolsProfile = it.openclawToolsProfile ?? "coding"
                self.hermesModel          = it.hermesModel ?? ""
                self.copilotModel         = it.copilotModel ?? ""
            }
            if let mcp = settings.mcp {
                let path = mcp.configPath ?? ""
                self.mcpConfigPath = path
                self.mcpConfigLoaded = path
            }

            // Available models
            let models = try await client.listModels().models
            self.availableModels = models.map { $0.id }

            // Stats — host/port/api_key/cli_prefix for the command builders.
            // Failure here is non-fatal: the screen still works against the
            // default `omlx` prefix and 127.0.0.1:8000.
            if let stats = try? await client.getStats() {
                if let host = stats.host, !host.isEmpty { self.serverHost = host }
                if let port = stats.port               { self.serverPort = port }
                self.serverApiKey = stats.apiKey ?? ""
                if let prefix = stats.cliPrefix, !prefix.isEmpty {
                    self.cliPrefix = prefix
                }
            }
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
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
        case .targetContextSize:
            let trimmed = targetContextSizeText.trimmingCharacters(in: .whitespaces)
            guard let n = Int(trimmed), n > 0 else {
                self.lastError = String(localized: "integrations.error.target_context_invalid",
                                        defaultValue: "Target context size must be a positive integer.",
                                        comment: "Integrations screen error when the target context size input is invalid")
                return
            }
            patch.claudeCodeTargetContextSize = n
        case .codexModel:           patch.integrationsCodexModel = codexModel
        case .opencodeModel:        patch.integrationsOpencodeModel = opencodeModel
        case .openclawModel:        patch.integrationsOpenclawModel = openclawModel
        case .piModel:              patch.integrationsPiModel = piModel
        case .openclawToolsProfile: patch.integrationsOpenclawToolsProfile = openclawToolsProfile
        case .hermesModel:          patch.integrationsHermesModel = hermesModel
        case .copilotModel:         patch.integrationsCopilotModel = copilotModel
        case .mcpConfig:
            patch.mcpConfig = mcpConfigPath.trimmingCharacters(in: .whitespaces)
        }
        do {
            _ = try await client.updateGlobalSettings(patch)
            self.lastError = nil
            switch field {
            case .mcpConfig:
                self.mcpConfigLoaded = mcpConfigPath.trimmingCharacters(in: .whitespaces)
            case .targetContextSize:
                self.targetContextSizeLoaded = targetContextSizeText.trimmingCharacters(in: .whitespaces)
            default:
                break
            }
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    // MARK: - Shell helpers

    private var cliCommandPrefix: String {
        cliPrefix == "omlx" ? cliPrefix : Self.shellQuote(cliPrefix)
    }

    /// POSIX single-quote escape — mirrors `shellQuote` in dashboard.js so
    /// the rendered command can be copy-pasted into bash/zsh as-is.
    private static func shellQuote(_ value: String) -> String {
        if value.isEmpty { return "''" }
        return "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    private static func shellEnvAssign(_ name: String, _ value: String) -> String {
        "\(name)=\(shellQuote(value))"
    }

    /// Wrap an IPv6 host in brackets so the URL parses. IPv4 / hostnames
    /// pass through unchanged.
    private func formatDisplayHost(_ host: String) -> String {
        let unwrapped = host.hasPrefix("[") && host.hasSuffix("]")
            ? String(host.dropFirst().dropLast())
            : host
        return unwrapped.contains(":") ? "[\(unwrapped)]" : unwrapped
    }
}
