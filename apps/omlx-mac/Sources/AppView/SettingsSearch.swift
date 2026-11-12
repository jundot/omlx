import Foundation

struct SettingsSearchEntry: Identifiable, Sendable {
    let labelKey: String
    let section: AppSection
    let askAIPage: AskAIPage?
    let pathKeys: [String]
    let aliases: String

    var id: String { (pathKeys + [labelKey]).joined(separator: ":") }

    func title(bundle: Bundle = .main) -> String {
        SettingsSearch.localized(labelKey, bundle: bundle)
    }

    func breadcrumb(bundle: Bundle = .main) -> String {
        pathKeys.map { SettingsSearch.localized($0, bundle: bundle) }.joined(separator: " › ")
    }
}

enum SettingsSearch {
    static let englishBundle = Bundle.main.url(forResource: "en", withExtension: "lproj")
        .flatMap { Bundle(url: $0) } ?? .main

    static func localized(_ key: String, bundle: Bundle = .main) -> String {
        let english = englishBundle.localizedString(forKey: key, value: nil, table: nil)
        return bundle.localizedString(forKey: key, value: english, table: nil)
    }

    static func results(for query: String, bundle: Bundle = .main) -> [SettingsSearchEntry] {
        let terms = query.split(whereSeparator: \.isWhitespace).map(String.init)
        guard !terms.isEmpty else { return [] }
        return entries.filter { entry in
            let keys = entry.pathKeys + [entry.labelKey]
            let text = (keys.map { localized($0, bundle: bundle) }
                        + keys.map { localized($0, bundle: englishBundle) }
                        + [entry.aliases]).joined(separator: " ")
            return terms.allSatisfy {
                text.range(of: $0, options: [.caseInsensitive, .diacriticInsensitive],
                           locale: .current) != nil
            }
        }
    }

    // Add new global settings here alongside their UI. Only labels and aliases belong in search.
    static let entries: [SettingsSearchEntry] = {
        var entries: [SettingsSearchEntry] = []
        func add(_ section: AppSection, _ path: [String], page: AskAIPage? = nil,
                 _ fields: [(String, String)]) {
            entries += fields.map { key, aliases in
                SettingsSearchEntry(labelKey: key, section: section, askAIPage: page,
                                    pathKeys: ["sidebar.\(section.rawValue)"] + path, aliases: aliases)
            }
        }

        add(.status, ["status.updates.title"], [
            ("status.updates.channel", "stable beta release candidate dev"),
            ("status.updates.auto_check", "automatic daily check"),
            ("status.updates.background_download", "background automatic download"),
            ("status.updates.auto_download", "notification release notes"),
        ])
        add(.server, ["server.section.network"], [
            ("server.row.listen_address", "host bind localhost IPv4 IPv6 LAN"),
            ("server.row.port", "TCP HTTP listen"),
        ])
        add(.server, ["server.section.default_profile"], [
            ("server.profile.context_window", "context length tokens sampling"),
            ("server.profile.max_tokens", "output length response sampling"),
            ("server.profile.temperature", "sampling randomness"),
            ("server.profile.top_p", "sampling nucleus"),
            ("server.profile.top_k", "sampling"),
            ("server.profile.repetition_penalty", "sampling repeat"),
        ])
        add(.server, ["server.section.startup"], [
            ("server.row.auto_start_on_launch", "startup automatic launch"),
        ])
        add(.server, ["server.section.logging"], [("server.row.log_level", "debug trace verbosity")])
        add(.server, ["server.section.usage"], [("server.row.usage_history", "statistics metrics requests")])
        add(.server, ["server.section.storage"], [
            ("server.row.base_path", "directory folder data settings logs"),
            ("server.row.models_directories", "folder storage model path"),
            ("server.row.hf_cache", "HF hub discovery"),
        ])
        add(.server, ["server.section.advanced"], [
            ("server.advanced.max_audio_upload_size", "file limit MB speech"),
            ("server.advanced.sse_keepalive", "streaming heartbeat timeout"),
            ("server.advanced.aliases", "hostname names"),
        ])
        add(.network, ["network.section.proxies.title"], [
            ("network.row.http_proxy.label", "connection endpoint"),
            ("network.row.https_proxy.label", "connection endpoint"),
            ("network.row.no_proxy.label", "bypass exclusions hosts"),
        ])
        add(.network, ["network.section.tls.title"], [("network.row.ca_bundle.label", "SSL certificate trust")])
        add(.performance, ["performance.section.scheduler"], [
            ("performance.scheduler.max_concurrent", "parallel requests batch concurrency"),
            ("performance.scheduler.embedding_batch_size", "embeddings batch"),
            ("performance.scheduler.chunked_prefill", "prompt processing chunk size"),
            ("performance.scheduler.prefill_priority", "prompt processing speed max context"),
        ])
        add(.performance, ["performance.section.memory"], [
            ("performance.memory.prefill_guard", "RAM OOM admission limit"),
            ("performance.memory.guard_tier", "RAM safe balanced aggressive custom"),
            ("performance.memory.custom_ceiling", "RAM limit GB"),
            ("performance.memory.idle_timeout", "unload TTL eviction seconds"),
            ("performance.memory.model_fallback", "default model missing"),
        ])
        add(.performance, ["performance.section.cache"], [
            ("performance.cache.enabled", "KV prefix caching"),
            ("performance.cache.hot_only", "RAM memory SSD spillover"),
            ("performance.cache.hot_size", "RAM memory limit"),
            ("performance.cache.ssd_dir", "disk directory folder path"),
            ("performance.cache.ssd_size", "disk storage limit"),
            ("performance.cache.initial_blocks", "preallocation startup"),
        ])
        add(.appearance, ["appearance.section.general"], [
            ("appearance.row.refresh_interval", "menu bar update frequency polling"),
            ("appearance.row.show_dock_icon", "application icon"),
            ("appearance.row.number_scale", "exact digits compact units count"),
        ])
        add(.appearance, ["appearance.section.model_library"], [("appearance.row.visible_models", "favorites menu bar")])
        add(.appearance, ["appearance.section.menubar_items"], [
            ("appearance.row.live_activity", "LIV tokens speed"),
            ("appearance.row.average_activity", "AVG tokens speed"),
            ("appearance.row.alltime_activity", "ALL tokens speed"),
            ("appearance.row.cpu_item", "processor usage"),
            ("appearance.row.gpu_item", "graphics usage"),
            ("appearance.row.memory_item", "RAM MEM usage"),
        ])
        add(.appearance, [], [("appearance.row.enhanced_readability", "contrast accessibility text")])
        add(.askAI, [], [
            ("askai.row.enabled", "ask via text enable"),
            ("talkai.row.enabled", "ask via voice enable speech microphone"),
        ])
        add(.askAI, ["askai.section.server"], [
            ("askai.row.server", "remote local this Mac another server"),
            ("askai.row.server_address", "remote host port URL endpoint"),
            ("askai.row.server_key", "remote authentication credential"),
        ])
        add(.askAI, ["askai.section.text"], [
            ("askai.row.hotkey", "keyboard shortcut"),
            ("askai.row.model", "default LLM"),
            ("askai.row.gpu_power_limit", "watts W battery energy cap"),
        ])
        add(.askAI, ["askai.section.text"], page: .presets, [("askai.section.presets", "slash commands manage")])
        add(.askAI, ["askai.section.text", "askai.section.presets"], page: .presets, [
            ("askai.preset.command", "name slash"),
            ("askai.row.model", "LLM"),
            ("askai.row.thinking", "reasoning"),
            ("askai.row.auto_copy", "clipboard"),
            ("askai.tools.row", "functions"),
            ("askai.tools.web_search", "internet"),
            ("askai.tools.read_links", "URL web pages"),
            ("askai.tools.screen", "screenshot vision VLM"),
            ("askai.row.system_prompt", "instructions"),
        ])
        add(.askAI, ["askai.section.text", "askai.section.advanced"], page: .textAdvanced, [
            ("askai.row.send_on_enter", "return keyboard"),
            ("askai.row.new_conversation", "idle reset timeout history"),
            ("askai.row.thinking", "reasoning"),
            ("askai.row.auto_copy", "clipboard"),
            ("askai.row.system_prompt", "instructions"),
            ("askai.tools.row", "functions"),
            ("askai.tools.web_search", "internet"),
            ("askai.tools.read_links", "URL web pages"),
            ("askai.tools.screen", "screenshot vision VLM"),
        ])
        add(.askAI, ["talkai.section"], [
            ("talkai.row.hotkey", "keyboard shortcut"),
            ("askai.row.model", "speech audio"),
            ("talkai.row.speed", "playback audio rate"),
            ("askai.row.gpu_power_limit", "watts W battery energy cap"),
        ])
        add(.askAI, ["talkai.section", "askai.section.advanced"], page: .voiceAdvanced, [
            ("talkai.row.voice", "speaker speech audio"),
            ("talkai.prompt.model", "system instructions"),
            ("talkai.prompt.general", "system instructions"),
            ("askai.tools.row", "functions"),
            ("askai.tools.web_search", "internet"),
            ("askai.tools.read_links", "URL web pages"),
            ("askai.tools.screen", "screenshot vision VLM"),
        ])
        add(.downloads, ["downloads.source.hf"], [("downloads.mirror.label", "mirror endpoint URL HF hub")])
        add(.downloads, ["downloads.source.ms"], [("downloads.mirror.label", "mirror endpoint URL")])
        add(.downloads, ["downloads.source.hf"], [("downloads.format.label", "MLX GGUF filter suggested models")])
        add(.integrations, ["integrations.section.external_engines"], [("integrations.splash.binary", "launcher executable path")])
        add(.integrations, ["integrations.section.external_engines", "integrations.splash.binary"], [
            ("integrations.splash.kv_format", "cache quantization precision"),
            ("integrations.splash.max_memory", "RAM GB cap"),
            ("integrations.splash.ssd_cache", "disk storage size"),
            ("integrations.splash.ane_prefill", "ANE prompt processing"),
        ])
        add(.integrations, ["integrations.websearch.section_label"], [
            ("integrations.websearch.provider", "internet DDGS DuckDuckGo Brave SearXNG"),
            ("integrations.websearch.ddgs_backends", "DDGS custom engines grokipedia mojeek wikipedia yahoo yandex"),
            ("integrations.websearch.brave_api_key", "Brave credential token"),
            ("integrations.websearch.searxng_url", "SearXNG instance endpoint"),
            ("integrations.websearch.max_results", "sources count"),
            ("integrations.websearch.content_mode", "snippets full page"),
            ("integrations.websearch.content_truncate", "truncate page context"),
            ("integrations.websearch.content_max_chars", "characters limit"),
        ])
        add(.integrations, ["integrations.section.claude_code"], [
            ("integrations.claude.mode", "local cloud Anthropic"),
            ("integrations.claude.opus", "model tier"),
            ("integrations.claude.sonnet", "model tier"),
            ("integrations.claude.haiku", "model tier"),
            ("integrations.claude.desktop", "aliases"),
        ])
        add(.integrations, ["integrations.section.other"], [
            ("integrations.tool.codex", "default model CLI desktop"),
            ("integrations.tool.opencode", "default model"),
            ("integrations.tool.openclaw", "default model"),
            ("integrations.openclaw.profile_label", "OpenClaw MCP tools profile"),
            ("integrations.tool.hermes", "default model"),
            ("integrations.tool.pi", "default model"),
            ("integrations.tool.vibe", "default model"),
            ("integrations.tool.copilot", "default model"),
            ("integrations.tool.dsh", "default model"),
        ])
        add(.integrations, ["integrations.section.mcp"], [
            ("integrations.mcp.config_path", "configuration file JSON tools"),
            ("integrations.mcp.expose_tools", "backend tools clients chat completions"),
        ])
        add(.security, [], [
            ("security.api_key.row_label", "authentication credential token"),
            ("security.auth.disable_verify", "disable authentication verification"),
            ("security.auth.keyless_inference", "public unauthenticated API"),
            ("security.section.sub_keys", "scoped API keys manage revoke"),
        ])
        add(.security, ["security.section.sub_keys"], [
            ("security.sub_keys.name_label", "API key label"),
            ("security.sub_keys.key_label", "API token credential"),
        ])
        return entries
    }()
}
