// PR 9 — Integrations.
//
// Routes Claude Code requests to local models or to the cloud, exposes the
// other named integrations (Codex / OpenCode / OpenClaw / Hermes / Pi /
// Copilot) as model popups with per-tool launch commands, and renders the
// Claude Code setup command — both the simple `omlx launch claude` form and
// an "Advanced" env-var recipe that targets the real `claude` binary
// directly. The model popups read their options from /admin/api/models so
// the user can only pick something the server actually has on disk.
//
// The OpenAI Compatibility section + Connected Apps from the design canvas
// are skipped: there are no matching server fields. We keep every shipped
// row honestly wired.

import SwiftUI

struct IntegrationsScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = IntegrationsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ExternalEnginesSection(vm: vm, client: services.client)
            ClaudeCodeSection(vm: vm, client: services.client)
            ClaudeSetupCommandSection(vm: vm)
            OtherIntegrationsSection(vm: vm, client: services.client)
            MCPSection(vm: vm, client: services.client)

            FooterBar(error: vm.lastError)
        }
        .task { await vm.load(client: services.client) }
    }
}

// MARK: - External engines

/// Engines that serve a model from their own process. Each model picks its
/// engine in Model Settings > Advanced; this section holds what they share.
private struct ExternalEnginesSection: View {
    @Bindable var vm: IntegrationsScreenVM
    let client: OMLXClient

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "integrations.section.external_engines",
                   defaultValue: "External Engines",
                   comment: "Section header for engines that serve models from their own process"),
            subtitle: String(localized: "integrations.section.external_engines.sub",
                             defaultValue: "Serve a model with another inference engine in its own process. Choose the engine per model in Model Settings → Advanced.",
                             comment: "Subtitle for the External Engines section")
        )

        ListGroup {
            if vm.showsSplashPathField {
                Row(
                    label: String(localized: "integrations.splash.binary",
                                  defaultValue: "Splash",
                                  comment: "Row label for the Splash launcher"),
                    sublabel: vm.splash?.available == true
                        ? String(localized: "integrations.splash.binary.sub",
                                 defaultValue: "Path to the splash or splash-m1 launcher.",
                                 comment: "Sublabel for the Splash launcher path field")
                        : (vm.splash?.reason ?? "")
                ) {
                    HStack(spacing: 7) {
                        TextInput(text: $vm.splashPathDraft,
                                  placeholder: "/opt/homebrew/bin/splash",
                                  mono: true,
                                  width: .controlMedium)
                        Button {
                            vm.chooseSplashLauncher()
                        } label: {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.omlx(.plain, size: .small))
                        .help(String(localized: "integrations.splash.browse.help",
                                     defaultValue: "Choose launcher",
                                     comment: "Tooltip for choosing the Splash launcher"))
                        Button(String(localized: "integrations.splash.apply",
                                      defaultValue: "Apply",
                                      comment: "Apply button for the Splash launcher path")) {
                            Task { await vm.save(.splashPath, client: client) }
                        }
                        .buttonStyle(.omlx(.primary, size: .small))
                        .disabled(vm.splashPathDraft.trimmingCharacters(in: .whitespaces).isEmpty)
                        if !vm.splashPathSetting.isEmpty {
                            Button(String(localized: "integrations.splash.use_auto",
                                          defaultValue: "Detect Automatically",
                                          comment: "Button that clears the Splash path so the server detects the launcher")) {
                                vm.splashPathDraft = ""
                                Task { await vm.save(.splashPath, client: client) }
                            }
                            .buttonStyle(.omlx(.normal, size: .small))
                        }
                        if vm.editingSplashPath {
                            Button(String(localized: "integrations.splash.cancel",
                                          defaultValue: "Cancel",
                                          comment: "Cancel editing the Splash launcher path")) {
                                vm.editingSplashPath = false
                            }
                            .buttonStyle(.omlx(.plain, size: .small))
                        }
                    }
                }
            } else if let splash = vm.splash {
                Row(
                    label: String(localized: "integrations.splash.binary",
                                  defaultValue: "Splash",
                                  comment: "Row label for the Splash launcher"),
                    sublabel: splash.path ?? ""
                ) {
                    HStack(spacing: 7) {
                        StatusPill(status: .custom(
                            color: theme.greenDot,
                            label: "\(splash.label ?? "Splash") \(splash.version ?? "")",
                            fillBg: true
                        ))
                        .help(splash.autoDetected == true
                              ? String(localized: "integrations.splash.detected",
                                       defaultValue: "Found automatically",
                                       comment: "Tooltip when the Splash launcher was detected")
                              : String(localized: "integrations.splash.configured",
                                       defaultValue: "Set by path",
                                       comment: "Tooltip when the Splash launcher path was set by the user"))
                        Button(String(localized: "integrations.splash.change",
                                      defaultValue: "Change",
                                      comment: "Button to edit the detected Splash launcher path")) {
                            vm.beginEditingSplashPath()
                        }
                        .buttonStyle(.omlx(.normal, size: .small))
                    }
                }
            }

            Row(label: String(localized: "integrations.splash.kv_format",
                              defaultValue: "KV Cache Precision",
                              comment: "Row label for the Splash KV cache format"),
                sublabel: String(localized: "integrations.splash.kv_format.sub",
                                 defaultValue: "int8 holds about twice the context of bf16 in the same memory.",
                                 comment: "Sublabel for the Splash KV cache format")) {
                Popup(selection: vm.bind($vm.splashKvFormat, save: {
                    Task { await vm.save(.splashKvFormat, client: client) }
                }), width: .controlCompact, options: [("int8", "int8"), ("bf16", "bf16")])
            }

            Row(label: String(localized: "integrations.splash.max_memory",
                              defaultValue: "Memory Limit",
                              comment: "Row label for the Splash memory limit"),
                sublabel: String(localized: "integrations.splash.max_memory.sub",
                                 defaultValue: "Empty gives Splash enough for the full context window; a larger limit gives its prefix cache more room.",
                                 comment: "Sublabel for the Splash memory limit")) {
                TextInput(text: $vm.splashMaxMemoryGb,
                          placeholder: String(localized: "integrations.splash.auto",
                                              defaultValue: "Auto",
                                              comment: "Placeholder for an automatic Splash size"),
                          suffix: "GB",
                          width: .controlNarrow)
            }

            Row(label: String(localized: "integrations.splash.ssd_cache",
                              defaultValue: "SSD Cache",
                              comment: "Row label for the Splash SSD cache size"),
                sublabel: String(localized: "integrations.splash.ssd_cache.sub",
                                 defaultValue: "Splash's prefix cache in the oMLX SSD cache folder. Empty uses the oMLX SSD cache size; 0 turns it off.",
                                 comment: "Sublabel for the Splash SSD cache size")) {
                TextInput(text: $vm.splashSsdCacheGb,
                          placeholder: String(localized: "integrations.splash.auto",
                                              defaultValue: "Auto",
                                              comment: "Placeholder for an automatic Splash size"),
                          suffix: "GB",
                          width: .controlNarrow)
            }

            let running = vm.splash?.models ?? []
            Row(label: String(localized: "integrations.splash.ane_prefill",
                              defaultValue: "Neural Engine Prefill",
                              comment: "Row label for Splash Neural Engine prefill"),
                sublabel: String(localized: "integrations.splash.ane_prefill.sub",
                                 defaultValue: "Let Splash run part of long prompts on the Neural Engine when that is faster.",
                                 comment: "Sublabel for Splash Neural Engine prefill"),
                isLast: running.isEmpty) {
                RowSwitch(isOn: vm.bind($vm.splashAnePrefill, save: {
                    Task { await vm.save(.splashAnePrefill, client: client) }
                }))
            }

            ForEach(Array(running.enumerated()), id: \.element.id) { index, model in
                Row(label: model.id,
                    sublabel: model.logPath,
                    isLast: index == running.count - 1) {
                    let used = UInt64(max(0, model.memoryBytes ?? 0))
                    let budget = UInt64(max(0, model.memoryBudgetBytes ?? 0))
                    Text("\(SystemMetricsPoller.formatBytesAsGiB(used)) / \(SystemMetricsPoller.formatBytesAsGiB(budget)) GB")
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textSecondary)
                }
            }
        }
        FooterBar {
            Button(String(localized: "integrations.splash.limits.apply",
                          defaultValue: "Apply",
                          comment: "Apply button for the Splash memory and SSD cache sizes")) {
                Task { await vm.save(.splashLimits, client: client) }
            }
            .buttonStyle(.omlx(.primary))
            .disabled(!vm.hasPendingSplashLimits)
        }
    }
}

// MARK: - Claude Code

private struct ClaudeCodeSection: View {
    @Bindable var vm: IntegrationsScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader(
            String(localized: "integrations.section.claude_code",
                   defaultValue: "Claude Code",
                   comment: "Section header for the Claude Code integration"),
            subtitle: String(localized: "integrations.section.claude_code.sub",
                             defaultValue: "Route Claude Code requests to local models or the cloud",
                             comment: "Subtitle for the Claude Code section")
        )

        ListGroup {
            Row(label: String(localized: "integrations.claude.mode",
                              defaultValue: "Mode",
                              comment: "Row label for the Claude Code mode segmented control")) {
                Segmented(
                    selection: vm.bind($vm.claudeMode, save: {
                        Task { await vm.save(.claudeMode, client: client) }
                    }),
                    options: [
                        ("cloud", String(localized: "integrations.claude.mode.cloud",
                                          defaultValue: "Cloud",
                                          comment: "Claude Code mode option: route to cloud")),
                        ("local", String(localized: "integrations.claude.mode.local",
                                          defaultValue: "Local",
                                          comment: "Claude Code mode option: route to local server")),
                    ]
                )
            }
            if vm.claudeMode == "local" {
                Row(label: String(localized: "integrations.claude.opus",
                                  defaultValue: "Opus tier",
                                  comment: "Row label for the Opus model picker")) {
                    Popup(
                        selection: vm.bind($vm.opusModel, save: {
                            Task { await vm.save(.opusModel, client: client) }
                        }),
                        width: .controlMedium,
                        options: vm.modelOptions
                    )
                }
                Row(label: String(localized: "integrations.claude.sonnet",
                                  defaultValue: "Sonnet tier",
                                  comment: "Row label for the Sonnet model picker")) {
                    Popup(
                        selection: vm.bind($vm.sonnetModel, save: {
                            Task { await vm.save(.sonnetModel, client: client) }
                        }),
                        width: .controlMedium,
                        options: vm.modelOptions
                    )
                }
                Row(
                    label: String(localized: "integrations.claude.haiku",
                                  defaultValue: "Haiku tier",
                                  comment: "Row label for the Haiku model picker"),
                    sublabel: String(localized: "integrations.claude.haiku.sub",
                                     defaultValue: "Used for background tasks and tool calls",
                                     comment: "Sublabel for the Haiku tier picker"),
                    isLast: true
                ) {
                    Popup(
                        selection: vm.bind($vm.haikuModel, save: {
                            Task { await vm.save(.haikuModel, client: client) }
                        }),
                        width: .controlMedium,
                        options: vm.modelOptions
                    )
                }
                Row(
                    label: String(localized: "integrations.claude.desktop",
                                  defaultValue: "Claude Desktop",
                                  comment: "Row label for the Claude Desktop tier-alias toggle"),
                    sublabel: String(localized: "integrations.claude.desktop.sub",
                                     defaultValue: "Expose claude-opus / claude-sonnet / claude-haiku as aliases of the tier models",
                                     comment: "Sublabel explaining the Claude Desktop tier-alias toggle")
                ) {
                    Toggle("", isOn: vm.bind($vm.desktopEnabled, save: {
                        Task { await vm.save(.desktopEnabled, client: client) }
                    }))
                    .labelsHidden().toggleStyle(.switch)
                }
                if vm.desktopEnabled {
                    FreeRow {
                        CommandBlock(
                            command: vm.claudeDesktopLaunchCommand,
                            caption: String(localized: "integrations.claude.desktop.command",
                                            defaultValue: "Claude Desktop",
                                            comment: "Caption above the Claude Desktop launcher command block")
                        )
                    }
                }
            }
        }
    }
}

// MARK: - Setup command (Claude Code)

/// Houses both the primary `omlx launch claude` block and the "Advanced"
/// env-var recipe that points the real `claude` binary at the local server.
/// Mirrors `claudeCodeCommand` in `apps/omlx-web/omlx_web/static/js/dashboard.js`.
private struct ClaudeSetupCommandSection: View {
    var vm: IntegrationsScreenVM
    @State private var showAdvanced = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(String(localized: "integrations.section.setup_command",
                              defaultValue: "Setup Command",
                              comment: "Section header for the Claude Code setup command block"))

        VStack(alignment: .leading, spacing: 10) {
            CommandBlock(command: vm.claudeLaunchCommand)

            DisclosureGroup(isExpanded: $showAdvanced) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(vm.claudeMode == "cloud"
                         ? String(localized: "integrations.setup.advanced.cloud",
                                  defaultValue: "Resets Anthropic env vars so the real `claude` binary talks to the cloud.",
                                  comment: "Explanation of the advanced env recipe in cloud mode")
                         : String(localized: "integrations.setup.advanced.local",
                                  defaultValue: "Points the real `claude` binary at your local oMLX server.",
                                  comment: "Explanation of the advanced env recipe in local mode"))
                        .font(.omlxText(11.5))
                        .foregroundStyle(theme.textSecondary)
                    CommandBlock(command: vm.claudeEnvRecipe)
                }
                .padding(.top, 6)
            } label: {
                Text(String(localized: "integrations.setup.advanced.label",
                            defaultValue: "Advanced — run `claude` directly",
                            comment: "Disclosure label revealing the advanced env-var recipe for running claude directly"))
                    .font(.omlxText(12, weight: .medium))
                    .foregroundStyle(theme.textSecondary)
            }
            .padding(.horizontal, 4)
        }
        .padding(.horizontal, 14)
    }
}

/// Shared monospaced command block with a copy button. Used by both the
/// Claude Code section and each per-tool row in OtherIntegrationsSection.
private struct CommandBlock: View {
    let command: String
    var caption: String? = nil
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ZStack(alignment: .topTrailing) {
            VStack(alignment: .leading, spacing: 4) {
                Text(caption ?? String(localized: "integrations.command.terminal_caption",
                                       defaultValue: "$ Terminal",
                                       comment: "Caption above each shell command block"))
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

            // Same quiet copy affordance as `CodeChip`/`CopyIconButton`
            // everywhere else — the previous locally-drawn boxed button was a
            // second copy-button style.
            CopyIconButton(
                value: command,
                helpText: String(localized: "integrations.command.copy",
                                 defaultValue: "Copy command",
                                 comment: "Tooltip for the copy button on a shell command block")
            )
            .padding(.top, 4)
            .padding(.trailing, 4)
        }
    }
}

// MARK: - Other integrations

private struct OtherIntegrationsSection: View {
    @Bindable var vm: IntegrationsScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader(
            String(localized: "integrations.section.other",
                   defaultValue: "Other Integrations",
                   comment: "Section header for the additional integrations list"),
            subtitle: String(localized: "integrations.section.other.sub",
                             defaultValue: "Default model + launcher command for each named integration",
                             comment: "Subtitle for the Other Integrations section")
        )

        ListGroup {
            IntegrationRow(
                name: String(localized: "integrations.tool.codex",
                             defaultValue: "Codex",
                             comment: "Display name for the Codex integration"),
                modelBinding: vm.bind($vm.codexModel, save: {
                    Task { await vm.save(.codexModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.codexCommand,
                commandLabel: String(localized: "integrations.codex.cli",
                                     defaultValue: "CLI",
                                     comment: "Label for the Codex CLI launcher command"),
                secondaryCommand: vm.codexAppCommand,
                secondaryCommandLabel: String(localized: "integrations.codex.app",
                                             defaultValue: "Desktop App",
                                             comment: "Label for the Codex desktop app launcher command")
            )
            IntegrationRow(
                name: String(localized: "integrations.tool.opencode",
                             defaultValue: "OpenCode",
                             comment: "Display name for the OpenCode integration"),
                modelBinding: vm.bind($vm.opencodeModel, save: {
                    Task { await vm.save(.opencodeModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.opencodeCommand
            )
            IntegrationRow(
                name: String(localized: "integrations.tool.openclaw",
                             defaultValue: "OpenClaw",
                             comment: "Display name for the OpenClaw integration"),
                modelBinding: vm.bind($vm.openclawModel, save: {
                    Task { await vm.save(.openclawModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.openclawCommand,
                profileBinding: vm.bind($vm.openclawToolsProfile, save: {
                    Task { await vm.save(.openclawToolsProfile, client: client) }
                }),
                profileSublabel: String(localized: "integrations.openclaw.profile_sub",
                                         defaultValue: "Built-in MCP tools the OpenClaw launcher exposes",
                                         comment: "Sublabel for the OpenClaw tools-profile picker")
            )
            IntegrationRow(
                name: String(localized: "integrations.tool.hermes",
                             defaultValue: "Hermes Agent",
                             comment: "Display name for the Hermes Agent integration"),
                modelBinding: vm.bind($vm.hermesModel, save: {
                    Task { await vm.save(.hermesModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.hermesCommand
            )
            IntegrationRow(
                name: String(localized: "integrations.tool.pi",
                             defaultValue: "Pi",
                             comment: "Display name for the Pi integration"),
                modelBinding: vm.bind($vm.piModel, save: {
                    Task { await vm.save(.piModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.piCommand
            )
            IntegrationRow(
                name: String(localized: "integrations.tool.vibe",
                             defaultValue: "Mistral Vibe",
                             comment: "Display name for the Mistral Vibe integration"),
                modelBinding: vm.bind($vm.vibeModel, save: {
                    Task { await vm.save(.vibeModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.vibeCommand
            )
            IntegrationRow(
                name: String(localized: "integrations.tool.copilot",
                             defaultValue: "Copilot CLI",
                             comment: "Display name for the Copilot CLI integration"),
                modelBinding: vm.bind($vm.copilotModel, save: {
                    Task { await vm.save(.copilotModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.copilotCommand
            )
            IntegrationRow(
                name: String(localized: "integrations.tool.dsh",
                             defaultValue: "DeepSeek Harness",
                             comment: "Display name for the DeepSeek Harness (dsh) integration"),
                modelBinding: vm.bind($vm.dshModel, save: {
                    Task { await vm.save(.dshModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.dshCommand,
                isLast: true
            )
        }
    }
}

/// Per-integration FreeRow: name + model picker on the top half, monospaced
/// launcher command + copy button below. Optional OpenClaw tools-profile
/// popup folds in under the model picker.
private struct IntegrationRow: View {
    let name: String
    let modelBinding: Binding<String>
    let modelOptions: [(String, String)]
    let command: String
    var commandLabel: String? = nil
    var secondaryCommand: String? = nil
    var secondaryCommandLabel: String? = nil
    var profileBinding: Binding<String>? = nil
    var profileSublabel: String? = nil
    var isLast: Bool = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        FreeRow(isLast: isLast) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 12) {
                    Text(name)
                        .font(.omlxText(13, weight: .medium))
                        .foregroundStyle(theme.text)
                    Spacer(minLength: 12)
                    Popup(
                        selection: modelBinding,
                        width: .controlMedium,
                        options: modelOptions
                    )
                }
                if let profileBinding {
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(String(localized: "integrations.openclaw.profile_label",
                                        defaultValue: "Tools profile",
                                        comment: "Row label for the OpenClaw tools-profile picker"))
                                .font(.omlxText(12))
                                .foregroundStyle(theme.textSecondary)
                            if let profileSublabel {
                                Text(profileSublabel)
                                    .font(.omlxText(11))
                                    .foregroundStyle(theme.textTertiary)
                            }
                        }
                        Spacer(minLength: 12)
                        Popup(
                            selection: profileBinding,
                            width: .controlMedium,
                            options: [
                                ("minimal",   String(localized: "integrations.openclaw.profile.minimal",
                                                     defaultValue: "Minimal",
                                                     comment: "OpenClaw tools profile option: minimal")),
                                ("coding",    String(localized: "integrations.openclaw.profile.coding",
                                                     defaultValue: "Coding",
                                                     comment: "OpenClaw tools profile option: coding")),
                                ("messaging", String(localized: "integrations.openclaw.profile.messaging",
                                                     defaultValue: "Messaging",
                                                     comment: "OpenClaw tools profile option: messaging")),
                                ("full",      String(localized: "integrations.openclaw.profile.full",
                                                     defaultValue: "Full",
                                                     comment: "OpenClaw tools profile option: full")),
                            ]
                        )
                    }
                }
                if let secondaryCommand {
                    VStack(alignment: .leading, spacing: 6) {
                        CommandBlock(command: command, caption: commandLabel)
                        CommandBlock(command: secondaryCommand, caption: secondaryCommandLabel)
                    }
                } else {
                    CommandBlock(command: command, caption: commandLabel)
                }
            }
        }
    }
}

// MARK: - MCP

/// Path to an MCP server config file consumed by every integration launcher
/// (Claude Code, OpenClaw, Hermes, …). Lives at the bottom of Integrations
/// because it's a shared resource — putting it under any one integration
/// would mislead.
private struct MCPSection: View {
    @Bindable var vm: IntegrationsScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader(
            String(localized: "integrations.section.mcp",
                   defaultValue: "MCP",
                   comment: "Section header for the MCP config path row"),
            subtitle: String(localized: "integrations.section.mcp.sub",
                             defaultValue: "Path to an MCP server config file. Shared across all integration launchers.",
                             comment: "Subtitle for the MCP section")
        )

        ListGroup {
            Row(
                label: String(localized: "integrations.mcp.config_path",
                              defaultValue: "Config Path",
                              comment: "Row label for the MCP config path input"),
                sublabel: String(localized: "integrations.mcp.config_path.sub",
                                 defaultValue: "Absolute path to an MCP config JSON. Leave blank to disable.",
                                 comment: "Sublabel describing the MCP config path field"),
                isLast: true
            ) {
                TextInput(
                    text: $vm.mcpConfigPath,
                    placeholder: "/path/to/mcp.json",
                    mono: true,
                    width: .controlWide
                )
            }
        }
        FooterBar {
            Button(String(localized: "integrations.mcp.apply",
                          defaultValue: "Apply",
                          comment: "Apply button for the MCP config path")) {
                Task { await vm.save(.mcpConfig, client: client) }
            }
            .buttonStyle(.omlx(.primary))
            .disabled(!vm.hasPendingMCPChanges)
        }
    }
}
