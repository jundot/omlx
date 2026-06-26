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
            ClaudeCodeSection(vm: vm, client: services.client)
            ClaudeSetupCommandSection(vm: vm)
            OtherIntegrationsSection(vm: vm, client: services.client)
            MCPSection(vm: vm, client: services.client)

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
                        width: 220,
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
                        width: 220,
                        options: vm.modelOptions
                    )
                }
                Row(
                    label: String(localized: "integrations.claude.haiku",
                                  defaultValue: "Haiku tier",
                                  comment: "Row label for the Haiku model picker"),
                    sublabel: String(localized: "integrations.claude.haiku.sub",
                                     defaultValue: "Used for background tasks and tool calls",
                                     comment: "Sublabel for the Haiku tier picker")
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
                label: String(localized: "integrations.claude.context_scaling",
                              defaultValue: "Context scaling",
                              comment: "Row label for the Claude Code context scaling toggle"),
                sublabel: String(localized: "integrations.claude.context_scaling.sub",
                                 defaultValue: "Stretch context windows for long agentic sessions",
                                 comment: "Sublabel for the context scaling toggle"),
                isLast: !vm.contextScaling
            ) {
                Toggle("", isOn: vm.bind($vm.contextScaling, save: {
                    Task { await vm.save(.contextScaling, client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
            if vm.contextScaling {
                Row(
                    label: String(localized: "integrations.claude.target_context",
                                  defaultValue: "Target context size",
                                  comment: "Row label for the Claude Code target context size field"),
                    sublabel: String(localized: "integrations.claude.target_context.sub",
                                     defaultValue: "Per-request context window Claude Code will scale toward",
                                     comment: "Sublabel for the target context size field"),
                    isLast: true
                ) {
                    TextInput(
                        text: $vm.targetContextSizeText,
                        mono: true,
                        suffix: "tk",
                        width: 130
                    )
                }
            }
        }
        if vm.contextScaling {
            HStack {
                Spacer()
                Button(String(localized: "integrations.target_context.apply",
                              defaultValue: "Apply",
                              comment: "Apply button for the Claude Code target context size field")) {
                    Task { await vm.save(.targetContextSize, client: client) }
                }
                .buttonStyle(.omlx(.primary))
                .disabled(!vm.hasPendingContextSizeChange)
            }
            .padding(.horizontal, 18)
            .padding(.top, 6)
        }
    }
}

// MARK: - Setup command (Claude Code)

/// Houses both the primary `omlx launch claude` block and the "Advanced"
/// env-var recipe that points the real `claude` binary at the local server.
/// Mirrors `claudeCodeCommand` in `omlx/admin/static/js/dashboard.js`.
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

            CopyButton(value: command)
                .padding(.top, 6)
                .padding(.trailing, 8)
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
            Image(systemName: copied ? "checkmark" : "document.on.document")
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
                name: String(localized: "integrations.tool.copilot",
                             defaultValue: "Copilot CLI",
                             comment: "Display name for the Copilot CLI integration"),
                modelBinding: vm.bind($vm.copilotModel, save: {
                    Task { await vm.save(.copilotModel, client: client) }
                }),
                modelOptions: vm.modelOptions,
                command: vm.copilotCommand,
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
                        width: 220,
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
                            width: 160,
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
                    width: 320
                )
            }
        }
        HStack {
            Spacer()
            Button(String(localized: "integrations.mcp.apply",
                          defaultValue: "Apply",
                          comment: "Apply button for the MCP config path")) {
                Task { await vm.save(.mcpConfig, client: client) }
            }
            .buttonStyle(.omlx(.primary))
            .disabled(!vm.hasPendingMCPChanges)
        }
        .padding(.horizontal, 18)
        .padding(.top, 6)
    }
}
