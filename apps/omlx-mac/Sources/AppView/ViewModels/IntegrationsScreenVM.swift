import AppKit
import SwiftUI

@MainActor
@Observable
final class IntegrationsScreenVM {
    enum Field: Hashable, Sendable {
        case claudeMode, opusModel, sonnetModel, haikuModel, desktopEnabled
        case codexModel, opencodeModel, openclawModel, piModel, openclawToolsProfile
        case hermesModel, copilotModel, dshModel, vibeModel
        case mcpConfig
        case splashPath, splashLimits, splashKvFormat, splashAnePrefill
        case markitdownEnabled, markitdownExposeModel, markitdownMaxFileSize
        case markitdownMaxFiles, markitdownPdfProcessingEngine
    }

    // Claude Code
    var claudeMode: String = "cloud"
    var opusModel: String = ""
    var sonnetModel: String = ""
    var haikuModel: String = ""
    var desktopEnabled: Bool = false

    // Other integrations
    var codexModel: String = ""
    var opencodeModel: String = ""
    var openclawModel: String = ""
    var piModel: String = ""
    var vibeModel: String = ""
    var openclawToolsProfile: String = "coding"
    var hermesModel: String = ""
    var copilotModel: String = ""
    var dshModel: String = ""

    // External engines: Splash
    private(set) var splash: GlobalSettingsDTO.SplashStatus?
    /// Path saved in settings; empty when the server detects the launcher.
    private(set) var splashPathSetting: String = ""
    var splashPathDraft: String = ""
    var editingSplashPath = false
    var splashKvFormat: String = "int8"
    var splashAnePrefill = true
    /// Whole GB as text; empty means automatic.
    var splashMaxMemoryGb: String = ""
    var splashSsdCacheGb: String = ""
    private var splashLimitsLoaded = ("", "")

    // MarkItDown
    var markitdownEnabled = true
    var markitdownExposeModel = false
    var markitdownMaxFileSizeText = "25"
    var markitdownMaxFilesText = "5"
    var markitdownPdfProcessingEngine = "markitdown"
    private var markitdownLoaded = (
        enabled: true, exposeModel: false, maxFileSizeMb: 25,
        maxFilesPerRequest: 5, pdfProcessingEngine: "markitdown"
    )
    private(set) var savingMarkitdownFields: Set<Field> = []
    private(set) var markitdownOcrModels: [String] = []

    // Web search values stay in the form even when their provider hides them.
    enum WebSearchField: CaseIterable {
        case provider, braveApiKey, searxngUrl, ddgsBackends, maxResults
        case contentMode, contentTruncate, contentMaxChars
    }

    struct WebSearchForm: Equatable {
        var provider = "ddgs"
        var braveApiKey = ""
        var searxngUrl = ""
        var ddgsBackends = ""
        var maxResults = "3"
        var contentMode = "snippet"
        var contentTruncate = true
        var contentMaxChars = "20000"
    }

    static let ddgsBackends = ["brave", "duckduckgo", "grokipedia", "mojeek", "wikipedia", "yahoo", "yandex"]
    var webSearch = WebSearchForm() {
        didSet {
            if webSearch != oldValue {
                webSearchRevision += 1
                webSearchTestMessage = nil
                webSearchTestOK = nil
            }
        }
    }
    private var webSearchLoaded = WebSearchForm()
    private var webSearchRevision = 0
    private(set) var savingWebSearchFields: Set<WebSearchField> = []
    private(set) var webSearchTesting = false
    private(set) var webSearchTestMessage: String?
    private(set) var webSearchTestOK: Bool?

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

    var markitdownModelVisible: Bool {
        get { markitdownEnabled && markitdownExposeModel }
        set {
            if markitdownEnabled { markitdownExposeModel = newValue }
        }
    }

    var isSavingMarkitdownVisibility: Bool {
        savingMarkitdownFields.contains(.markitdownEnabled)
            || savingMarkitdownFields.contains(.markitdownExposeModel)
    }

    var markitdownOcrModelMissing: Bool {
        markitdownPdfProcessingEngine != "markitdown"
            && !availableModels.contains(markitdownPdfProcessingEngine)
    }

    var markitdownPdfOptions: [(String, String)] {
        var options = [("markitdown", "MarkItDown")]
        if markitdownOcrModelMissing {
            options.append((markitdownPdfProcessingEngine, markitdownPdfProcessingEngine))
        }
        options += markitdownOcrModels.map { ($0, $0) }
        return options
    }

    var markitdownPdfWarning: String? {
        guard markitdownPdfProcessingEngine == "markitdown" || markitdownOcrModelMissing else { return nil }
        let warning = String(localized: "integrations.markitdown.pdf_processing_markitdown_warning",
                             defaultValue: "Scanned or image-only PDFs will fail when MarkItDown is selected, and PDFs with tables may not be processed correctly.",
                             comment: "Warning when PDFs are processed by MarkItDown without OCR")
        guard markitdownOcrModelMissing else { return warning }
        let missing = String(localized: "integrations.markitdown.pdf_processing_missing_model",
                             defaultValue: "The selected OCR model is unavailable. PDFs use MarkItDown until the model is available again or you select another engine.",
                             comment: "Warning when the configured PDF OCR model is unavailable")
        return "\(missing)\n\(warning)"
    }

    /// Composed `omlx launch claude` command. Claude tier selections are
    /// persisted in settings, so the launcher reads them without extra flags.
    var claudeLaunchCommand: String {
        "\(cliCommandPrefix) launch claude"
    }

    /// Composed `omlx launch claude_desktop` command. Launches Claude Desktop
    /// pointed at the oMLX gateway. Mirrors `claudeDesktopCommand` in
    /// `omlx/admin/static/js/dashboard.js`.
    var claudeDesktopLaunchCommand: String {
        "\(cliCommandPrefix) launch claude_desktop"
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
        // Deny LSP: its schema joins the tools array mid-session and
        // re-prefills the whole conversation on a caching server (#2349).
        parts.append("claude --disallowedTools LSP")
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
    var vibeCommand: String {
        let command = "\(cliCommandPrefix) launch vibe"
        return vibeModel.isEmpty ? command : "\(command) --model \(Self.shellQuote(vibeModel))"
    }
    var copilotCommand: String  { "\(cliCommandPrefix) launch copilot" }
    var dshCommand: String      { "\(cliCommandPrefix) launch dsh" }

    var hasPendingMCPChanges: Bool {
        mcpConfigPath.trimmingCharacters(in: .whitespaces) != mcpConfigLoaded
    }

    /// Show the path field when no launcher was found or the user edits it.
    var showsSplashPathField: Bool {
        editingSplashPath || splash?.available != true
    }

    var hasPendingSplashLimits: Bool {
        (Self.trimmed(splashMaxMemoryGb), Self.trimmed(splashSsdCacheGb)) != splashLimitsLoaded
    }

    func beginEditingSplashPath() {
        splashPathDraft = splashPathSetting.isEmpty ? (splash?.path ?? "") : splashPathSetting
        editingSplashPath = true
    }

    func chooseSplashLauncher() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.treatsFilePackagesAsDirectories = true
        panel.prompt = String(localized: "integrations.splash.browse.prompt",
                              defaultValue: "Select",
                              comment: "NSOpenPanel button label for choosing the Splash launcher")
        panel.message = String(localized: "integrations.splash.browse.message",
                               defaultValue: "Choose the splash or splash-m1 launcher.",
                               comment: "NSOpenPanel message for choosing the Splash launcher")
        if panel.runModal() == .OK, let url = panel.url {
            splashPathDraft = url.path
        }
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
                self.desktopEnabled  = cc.desktopEnabled ?? false
            }
            if let it = settings.integrations {
                self.codexModel           = it.codexModel ?? ""
                self.opencodeModel        = it.opencodeModel ?? ""
                self.openclawModel        = it.openclawModel ?? ""
                self.piModel              = it.piModel ?? ""
                self.vibeModel            = it.vibeModel ?? ""
                self.openclawToolsProfile = it.openclawToolsProfile ?? "coding"
                self.hermesModel          = it.hermesModel ?? ""
                self.copilotModel         = it.copilotModel ?? ""
                self.dshModel             = it.dshModel ?? ""
                self.splashPathSetting    = it.splashPath ?? ""
                self.splashKvFormat       = it.splashKvFormat ?? "int8"
                self.splashAnePrefill     = it.splashAnePrefill ?? true
                self.splashMaxMemoryGb    = Self.gigabytesText(it.splashMaxMemoryGb)
                self.splashSsdCacheGb     = Self.gigabytesText(it.splashSsdCacheGb)
                self.splashLimitsLoaded   = (self.splashMaxMemoryGb, self.splashSsdCacheGb)
            }
            self.markitdownEnabled = settings.integrations?.markitdownEnabled ?? true
            self.markitdownExposeModel = settings.integrations?.markitdownExposeModel ?? false
            let maxFileSize = settings.integrations?.markitdownMaxFileSizeMb ?? 25
            let maxFiles = settings.integrations?.markitdownMaxFilesPerRequest ?? 5
            self.markitdownMaxFileSizeText = String(maxFileSize)
            self.markitdownMaxFilesText = String(maxFiles)
            self.markitdownPdfProcessingEngine = settings.integrations?.markitdownPdfProcessingEngine ?? "markitdown"
            self.markitdownLoaded = (markitdownEnabled, markitdownExposeModel, maxFileSize,
                                    maxFiles, markitdownPdfProcessingEngine)
            self.webSearch = WebSearchForm(
                provider: settings.integrations?.webSearchProvider ?? "ddgs",
                braveApiKey: settings.integrations?.webSearchBraveApiKey ?? "",
                searxngUrl: settings.integrations?.webSearchSearxngUrl ?? "",
                ddgsBackends: settings.integrations?.webSearchDdgsBackends ?? "",
                maxResults: String(settings.integrations?.webSearchMaxResults ?? 3),
                contentMode: settings.integrations?.webSearchContentMode ?? "snippet",
                contentTruncate: settings.integrations?.webSearchContentTruncate ?? true,
                contentMaxChars: String(settings.integrations?.webSearchContentMaxChars ?? 20000)
            )
            self.webSearchLoaded = webSearch
            self.splash = settings.externalEngines?.splash
            if let mcp = settings.mcp {
                let path = mcp.configPath ?? ""
                self.mcpConfigPath = path
                self.mcpConfigLoaded = path
            }

            // Available models
            try await refreshModels(client: client)

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
        guard !savingMarkitdownFields.contains(field) else { return }
        var patch = GlobalSettingsPatch()
        switch field {
        case .claudeMode:           patch.claudeCodeMode = claudeMode
        case .opusModel:            patch.claudeCodeOpusModel = opusModel
        case .sonnetModel:          patch.claudeCodeSonnetModel = sonnetModel
        case .haikuModel:           patch.claudeCodeHaikuModel = haikuModel
        case .desktopEnabled:       patch.claudeCodeDesktopEnabled = desktopEnabled
        case .codexModel:           patch.integrationsCodexModel = codexModel
        case .opencodeModel:        patch.integrationsOpencodeModel = opencodeModel
        case .openclawModel:        patch.integrationsOpenclawModel = openclawModel
        case .piModel:              patch.integrationsPiModel = piModel
        case .vibeModel:            patch.integrationsVibeModel = vibeModel
        case .openclawToolsProfile: patch.integrationsOpenclawToolsProfile = openclawToolsProfile
        case .hermesModel:          patch.integrationsHermesModel = hermesModel
        case .copilotModel:         patch.integrationsCopilotModel = copilotModel
        case .dshModel:             patch.integrationsDshModel = dshModel
        case .mcpConfig:
            patch.mcpConfig = mcpConfigPath.trimmingCharacters(in: .whitespaces)
        case .splashPath:
            patch.splashPath = Self.trimmed(splashPathDraft)
        case .splashKvFormat:       patch.splashKvFormat = splashKvFormat
        case .splashAnePrefill:     patch.splashAnePrefill = splashAnePrefill
        case .splashLimits:
            guard let memory = Self.gigabytesPatch(splashMaxMemoryGb, minimum: 1),
                  let ssd = Self.gigabytesPatch(splashSsdCacheGb, minimum: 0) else {
                self.lastError = String(localized: "integrations.splash.limits.invalid",
                                        defaultValue: "Enter whole gigabytes, or leave the field empty for automatic.",
                                        comment: "Error when a Splash memory or SSD cache size is not a whole number")
                return
            }
            patch.splashMaxMemoryGb = memory
            patch.splashSsdCacheGb = ssd
        case .markitdownEnabled:
            guard !isSavingMarkitdownVisibility, markitdownEnabled != markitdownLoaded.enabled else { return }
            patch.markitdownEnabled = markitdownEnabled
        case .markitdownExposeModel:
            guard markitdownEnabled, !isSavingMarkitdownVisibility,
                  markitdownExposeModel != markitdownLoaded.exposeModel else { return }
            patch.markitdownExposeModel = markitdownExposeModel
        case .markitdownMaxFileSize, .markitdownMaxFiles:
            let text = field == .markitdownMaxFileSize ? markitdownMaxFileSizeText : markitdownMaxFilesText
            guard let value = Int(Self.trimmed(text)), value > 0 else {
                self.lastError = String(localized: "integrations.markitdown.limits.invalid",
                                        defaultValue: "Enter a whole number greater than zero.",
                                        comment: "Error when a MarkItDown file limit is not a positive integer")
                return
            }
            if field == .markitdownMaxFileSize {
                if value == markitdownLoaded.maxFileSizeMb {
                    lastError = nil
                    return
                }
                patch.markitdownMaxFileSizeMb = value
            } else {
                if value == markitdownLoaded.maxFilesPerRequest {
                    lastError = nil
                    return
                }
                patch.markitdownMaxFilesPerRequest = value
            }
        case .markitdownPdfProcessingEngine:
            guard markitdownPdfProcessingEngine != markitdownLoaded.pdfProcessingEngine else { return }
            patch.markitdownPdfProcessingEngine = markitdownPdfProcessingEngine
        }
        switch field {
        case .markitdownEnabled, .markitdownExposeModel, .markitdownMaxFileSize,
             .markitdownMaxFiles, .markitdownPdfProcessingEngine:
            savingMarkitdownFields.insert(field)
        default:
            break
        }
        defer { savingMarkitdownFields.remove(field) }
        do {
            _ = try await client.updateGlobalSettings(patch)
            self.lastError = nil
            switch field {
            case .mcpConfig:
                self.mcpConfigLoaded = mcpConfigPath.trimmingCharacters(in: .whitespaces)
            case .splashPath:
                self.editingSplashPath = false
                // Reload for the launcher status the server resolved.
                await load(client: client)
            case .splashLimits:
                self.splashLimitsLoaded = (Self.trimmed(splashMaxMemoryGb), Self.trimmed(splashSsdCacheGb))
            case .markitdownEnabled:
                if let value = patch.markitdownEnabled { markitdownLoaded.enabled = value }
                try await refreshModels(client: client)
            case .markitdownExposeModel:
                if let value = patch.markitdownExposeModel { markitdownLoaded.exposeModel = value }
                try await refreshModels(client: client)
            case .markitdownMaxFileSize:
                if let value = patch.markitdownMaxFileSizeMb {
                    markitdownLoaded.maxFileSizeMb = value
                    markitdownMaxFileSizeText = String(value)
                }
            case .markitdownMaxFiles:
                if let value = patch.markitdownMaxFilesPerRequest {
                    markitdownLoaded.maxFilesPerRequest = value
                    markitdownMaxFilesText = String(value)
                }
            case .markitdownPdfProcessingEngine:
                if let value = patch.markitdownPdfProcessingEngine { markitdownLoaded.pdfProcessingEngine = value }
            default:
                break
            }
        } catch {
            switch field {
            case .markitdownEnabled: markitdownEnabled = markitdownLoaded.enabled
            case .markitdownExposeModel: markitdownExposeModel = markitdownLoaded.exposeModel
            case .markitdownPdfProcessingEngine: markitdownPdfProcessingEngine = markitdownLoaded.pdfProcessingEngine
            default: break
            }
            self.lastError = error.omlxDescription
        }
    }

    func ddgsBackendChecked(_ name: String) -> Bool {
        webSearch.ddgsBackends.split(separator: ",").contains { Self.trimmed(String($0)) == name }
    }

    func setDdgsBackend(_ name: String, selected: Bool) {
        var backends = webSearch.ddgsBackends.split(separator: ",").map { Self.trimmed(String($0)) }
        backends.removeAll { $0 == name || $0.isEmpty }
        if selected { backends.append(name) }
        webSearch.ddgsBackends = backends.joined(separator: ",")
    }

    func saveWebSearch(_ field: WebSearchField, client: OMLXClient) async {
        guard !savingWebSearchFields.contains(field) else { return }
        let draft = webSearch
        var submitted = draft
        var patch = GlobalSettingsPatch()
        switch field {
        case .provider:
            guard submitted.provider != webSearchLoaded.provider else { lastError = nil; return }
            patch.webSearchProvider = submitted.provider
        case .braveApiKey:
            submitted.braveApiKey = Self.trimmed(draft.braveApiKey)
            guard submitted.braveApiKey != webSearchLoaded.braveApiKey else { lastError = nil; return }
            patch.webSearchBraveApiKey = submitted.braveApiKey
        case .searxngUrl:
            submitted.searxngUrl = Self.trimmed(draft.searxngUrl)
            guard Self.validSearxngURL(submitted.searxngUrl) else {
                lastError = String(localized: "integrations.websearch.url_invalid",
                                   defaultValue: "Enter an HTTP or HTTPS URL with a host, or leave it empty.",
                                   comment: "Invalid SearXNG instance URL")
                return
            }
            while submitted.searxngUrl.hasSuffix("/") { submitted.searxngUrl.removeLast() }
            guard submitted.searxngUrl != webSearchLoaded.searxngUrl else { lastError = nil; return }
            patch.webSearchSearxngUrl = submitted.searxngUrl
        case .ddgsBackends:
            guard submitted.ddgsBackends != webSearchLoaded.ddgsBackends else { lastError = nil; return }
            patch.webSearchDdgsBackends = submitted.ddgsBackends
        case .maxResults:
            guard let value = Int(Self.trimmed(draft.maxResults)), (1...10).contains(value) else {
                lastError = String(localized: "integrations.websearch.max_results_invalid",
                                   defaultValue: "Enter a whole number between 1 and 10.",
                                   comment: "Invalid web search numeric setting")
                return
            }
            submitted.maxResults = String(value)
            guard submitted.maxResults != webSearchLoaded.maxResults else { lastError = nil; return }
            patch.webSearchMaxResults = value
        case .contentMode:
            guard submitted.contentMode != webSearchLoaded.contentMode else { lastError = nil; return }
            patch.webSearchContentMode = submitted.contentMode
        case .contentTruncate:
            guard submitted.contentTruncate != webSearchLoaded.contentTruncate else { lastError = nil; return }
            patch.webSearchContentTruncate = submitted.contentTruncate
        case .contentMaxChars:
            guard let value = Int(Self.trimmed(draft.contentMaxChars)), value > 0 else {
                lastError = String(localized: "integrations.websearch.content_limit_invalid",
                                   defaultValue: "Enter a whole number greater than zero.",
                                   comment: "Invalid web search numeric setting")
                return
            }
            submitted.contentMaxChars = String(value)
            guard submitted.contentMaxChars != webSearchLoaded.contentMaxChars else { lastError = nil; return }
            patch.webSearchContentMaxChars = value
        }
        savingWebSearchFields.insert(field)
        defer { savingWebSearchFields.remove(field) }
        do {
            _ = try await client.updateGlobalSettings(patch)
            switch field {
            case .provider:
                webSearchLoaded.provider = submitted.provider
                if webSearch.provider == draft.provider { webSearch.provider = submitted.provider }
            case .braveApiKey:
                webSearchLoaded.braveApiKey = submitted.braveApiKey
                if webSearch.braveApiKey == draft.braveApiKey { webSearch.braveApiKey = submitted.braveApiKey }
            case .searxngUrl:
                webSearchLoaded.searxngUrl = submitted.searxngUrl
                if webSearch.searxngUrl == draft.searxngUrl { webSearch.searxngUrl = submitted.searxngUrl }
            case .ddgsBackends:
                webSearchLoaded.ddgsBackends = submitted.ddgsBackends
                if webSearch.ddgsBackends == draft.ddgsBackends { webSearch.ddgsBackends = submitted.ddgsBackends }
            case .maxResults:
                webSearchLoaded.maxResults = submitted.maxResults
                if webSearch.maxResults == draft.maxResults { webSearch.maxResults = submitted.maxResults }
            case .contentMode:
                webSearchLoaded.contentMode = submitted.contentMode
                if webSearch.contentMode == draft.contentMode { webSearch.contentMode = submitted.contentMode }
            case .contentTruncate:
                webSearchLoaded.contentTruncate = submitted.contentTruncate
                if webSearch.contentTruncate == draft.contentTruncate { webSearch.contentTruncate = submitted.contentTruncate }
            case .contentMaxChars:
                webSearchLoaded.contentMaxChars = submitted.contentMaxChars
                if webSearch.contentMaxChars == draft.contentMaxChars { webSearch.contentMaxChars = submitted.contentMaxChars }
            }
            lastError = nil
        } catch {
            switch field {
            case .provider:
                if webSearch.provider == draft.provider { webSearch.provider = webSearchLoaded.provider }
            case .ddgsBackends:
                if webSearch.ddgsBackends == draft.ddgsBackends { webSearch.ddgsBackends = webSearchLoaded.ddgsBackends }
            case .contentMode:
                if webSearch.contentMode == draft.contentMode { webSearch.contentMode = webSearchLoaded.contentMode }
            case .contentTruncate:
                if webSearch.contentTruncate == draft.contentTruncate { webSearch.contentTruncate = webSearchLoaded.contentTruncate }
            case .braveApiKey, .searxngUrl, .maxResults, .contentMaxChars:
                // Keep typed drafts available for correction or retry.
                break
            }
            lastError = error.omlxDescription
        }
    }

    func testWebSearch(client: OMLXClient) async {
        guard !webSearchTesting else { return }
        webSearchTestMessage = nil
        webSearchTestOK = nil
        guard let count = Int(Self.trimmed(webSearch.maxResults)), (1...10).contains(count) else {
            webSearchTestOK = false
            webSearchTestMessage = String(localized: "integrations.websearch.max_results_invalid",
                                         defaultValue: "Enter a whole number between 1 and 10.",
                                         comment: "Invalid web search numeric setting")
            return
        }
        let request = WebSearchTestRequest(
            provider: webSearch.provider, braveApiKey: Self.trimmed(webSearch.braveApiKey),
            searxngUrl: Self.trimmed(webSearch.searxngUrl), ddgsBackends: webSearch.ddgsBackends,
            maxResults: count
        )
        let revision = webSearchRevision
        webSearchTesting = true
        defer { webSearchTesting = false }
        do {
            let response = try await client.testWebSearch(request)
            guard revision == webSearchRevision else { return }
            webSearchTestOK = response.ok
            if response.ok {
                webSearchTestMessage = String(localized: "integrations.websearch.test_success",
                                             defaultValue: "Search OK: \(response.results?.count ?? 0) results",
                                             comment: "Successful web search connectivity test; argument is result count")
            } else {
                webSearchTestMessage = response.error?.message
                    ?? String(localized: "integrations.websearch.test_failed",
                              defaultValue: "Search test failed",
                              comment: "Fallback error for a web search connectivity test")
            }
        } catch {
            guard revision == webSearchRevision else { return }
            webSearchTestOK = false
            webSearchTestMessage = error.omlxDescription
        }
    }

    private static func validSearxngURL(_ text: String) -> Bool {
        if text.isEmpty { return true }
        guard let url = URLComponents(string: text),
              let scheme = url.scheme, ["http", "https"].contains(scheme),
              let host = url.host, !host.isEmpty else { return false }
        return !text.contains(where: { $0.isWhitespace })
    }

    private func refreshModels(client: OMLXClient) async throws {
        let models = try await client.listModels().models
        availableModels = models.map { $0.id }
        markitdownOcrModels = models.filter { ($0.configModelType ?? "").lowercased().contains("ocr") }.map { $0.id }
    }

    private static func trimmed(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespaces)
    }

    private static func gigabytesText(_ value: Double?) -> String {
        guard let value else { return "" }
        return String(Int(value))
    }

    /// `.null` for an empty field, the value for whole GB, nil when invalid.
    private static func gigabytesPatch(_ text: String, minimum: Int) -> PatchOptionalInt? {
        let value = trimmed(text)
        if value.isEmpty { return .null }
        guard let gigabytes = Int(value), gigabytes >= minimum else { return nil }
        return .value(gigabytes)
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
