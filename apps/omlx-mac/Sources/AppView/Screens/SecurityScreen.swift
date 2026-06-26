// PR 9 — Security.
//
// Three sections:
//   • API Key — editable masked field with show/regenerate/copy/save.
//                Routes through /admin/api/setup-api-key for first-time
//                setup, and /admin/api/global-settings (`api_key` field)
//                for updates to an already-configured key.
//   • Authentication — `skip_api_key_verification` toggle (no-auth mode)
//   • Sub Keys — list + create + delete via /admin/api/sub-keys (POST/DELETE)
//
// The main key is rendered as a SecureField with an explicit show toggle
// rather than the always-visible CodeChip — administrators routinely have
// this screen open over shoulder-surfable displays, and the copy button
// covers the "stash it into a client config" case without exposing the
// plaintext.

import SwiftUI
import AppKit

struct SecurityScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = SecurityScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            APIKeySection(vm: vm, client: services.client)
            AuthenticationSection(vm: vm, client: services.client)
            SubKeysSection(vm: vm, client: services.client)

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

// MARK: - API key section

private struct APIKeySection: View {
    var vm: SecurityScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader(
            String(localized: "security.section.api_key",
                   defaultValue: "API Key",
                   comment: "Section header for the API key editor"),
            subtitle: String(localized: "security.section.api_key.sub",
                             defaultValue: "Required to authenticate /v1 requests and admin sessions",
                             comment: "Subtitle for the API key section header")
        )

        ListGroup {
            APIKeyEditorRow(vm: vm, client: client)
        }
    }
}

private struct APIKeyEditorRow: View {
    var vm: SecurityScreenVM
    let client: OMLXClient

    @State private var draft: String = ""
    @State private var showKey: Bool = false
    @State private var copied: Bool = false
    @State private var saving: Bool = false

    @FocusState private var focused: Bool

    @Environment(\.omlxTheme) private var theme

    private var loaded: String { vm.apiKey ?? "" }
    private var isDirty: Bool { draft != loaded }
    private var trimmed: String { draft.trimmingCharacters(in: .whitespaces) }
    /// Server-side rule: ≥ 4 printable chars, no whitespace. We mirror the
    /// admin route's `validate_api_key` so the Save button can't enable for
    /// a key the server will reject.
    private var canSave: Bool {
        isDirty
            && trimmed.count >= 4
            && !draft.contains(where: { $0.isWhitespace })
            && !saving
    }

    private var sublabel: String {
        vm.apiKeySet
            ? String(localized: "security.api_key.sublabel.set",
                     defaultValue: "Used to authenticate /v1 and admin requests. ≥ 4 printable chars, no whitespace.",
                     comment: "Sublabel below the API key field when a key is already configured")
            : String(localized: "security.api_key.sublabel.unset",
                     defaultValue: "Set one before exposing the server. ≥ 4 printable chars, no whitespace.",
                     comment: "Sublabel below the API key field when no key is configured")
    }

    var body: some View {
        Row(label: String(localized: "security.api_key.row_label",
                          defaultValue: "API Key",
                          comment: "Row label for the API key editor"),
            sublabel: sublabel, isLast: true) {
            HStack(spacing: 6) {
                TextInput("security.api_key.row_label", text: $draft, placeholder: "sk-omlx-…", isSecure: !showKey, mono: true, width: 260)
                    .onSubmit { Task { await save() } }
                iconButton(systemName: showKey ? "eye.slash" : "eye",
                           help: showKey
                                ? String(localized: "security.api_key.hide",
                                          defaultValue: "Hide key",
                                          comment: "Tooltip on the eye button when the API key is visible")
                                : String(localized: "security.api_key.show",
                                          defaultValue: "Show key",
                                          comment: "Tooltip on the eye button when the API key is masked")) {
                    showKey.toggle()
                }
                iconButton(systemName: "arrow.triangle.2.circlepath",
                           help: String(localized: "security.api_key.generate",
                                        defaultValue: "Generate a random key",
                                        comment: "Tooltip on the API key regenerate button")) {
                    draft = APIKeyGenerator.random()
                    showKey = true
                }
                iconButton(systemName: copied ? "checkmark" : "document.on.document",
                           help: String(localized: "security.api_key.copy",
                                        defaultValue: "Copy to clipboard",
                                        comment: "Tooltip on the API key copy button"),
                           tint: copied ? theme.successText : theme.textSecondary,
                           disabled: draft.isEmpty) {
                    copyToClipboard()
                }
                if isDirty {
                    Button(saving
                           ? String(localized: "security.api_key.saving",
                                    defaultValue: "Saving…",
                                    comment: "API key save button label while a save is in progress")
                           : String(localized: "security.api_key.save",
                                    defaultValue: "Save",
                                    comment: "API key save button label when changes are pending")) {
                        Task { await save() }
                    }
                    .buttonStyle(.omlx(.primary, size: .small))
                    .disabled(!canSave)
                }
            }
        }
        .task(id: loaded) {
            if !focused { draft = loaded }
        }
    }

    @ViewBuilder
    private func iconButton(
        systemName: String,
        help: String,
        tint: Color? = nil,
        disabled: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(tint ?? theme.textSecondary)
                .frame(width: 26, height: 26)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .help(help)
    }

    private func copyToClipboard() {
        guard !draft.isEmpty else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(draft, forType: .string)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) { copied = false }
    }

    private func save() async {
        guard canSave else { return }
        saving = true
        defer { saving = false }
        let next = trimmed
        let ok = await vm.applyApiKey(next, client: client)
        if ok {
            // Drop focus so the next `.task(id:)` mirror picks up the fresh
            // loaded value without fighting an in-progress edit.
            focused = false
            draft = next
        }
    }

}

// MARK: - Authentication

private struct AuthenticationSection: View {
    @Bindable var vm: SecurityScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader(String(localized: "security.section.authentication",
                              defaultValue: "Authentication",
                              comment: "Section header for the authentication settings"))

        ListGroup {
            Row(
                label: String(localized: "security.auth.disable_verify",
                              defaultValue: "Disable API Key Verification",
                              comment: "Row label for the toggle that disables API key verification"),
                sublabel: String(localized: "security.auth.disable_verify.sub",
                                 defaultValue: "Allow unauthenticated /v1 and admin requests. Use for development only.",
                                 comment: "Sublabel for the disable API key verification toggle"),
                isLast: true
            ) {
                Toggle("", isOn: vm.bind($vm.skipApiKeyVerification, save: {
                    Task { await vm.saveSkipApiKeyVerification(client: client) }
                }))
                .labelsHidden().toggleStyle(.switch)
            }
        }
    }
}

// MARK: - Sub keys

private struct SubKeysSection: View {
    var vm: SecurityScreenVM
    let client: OMLXClient

    @State private var newName: String = ""
    @State private var newKey: String = ""
    @State private var showCreate: Bool = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "security.section.sub_keys",
                   defaultValue: "Sub Keys",
                   comment: "Section header for the sub-keys list"),
            subtitle: String(localized: "security.section.sub_keys.sub",
                             defaultValue: "Issue scoped keys for individual apps or users. Sub keys cannot grant admin access.",
                             comment: "Subtitle for the Sub Keys section")
        ) {
            Button {
                showCreate = true
            } label: {
                Label(String(localized: "security.sub_keys.new",
                             defaultValue: "New",
                             comment: "Button to open the sub-key create form"),
                      systemImage: "plus")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.omlx(.normal, size: .small))
        }

        ListGroup {
            if showCreate {
                Row(label: String(localized: "security.sub_keys.name_label",
                                  defaultValue: "Name",
                                  comment: "Row label for sub-key name input"),
                    sublabel: String(localized: "security.sub_keys.name_sub",
                                     defaultValue: "Optional human-readable label",
                                     comment: "Sublabel for sub-key name input")) {
                    TextInput(text: $newName,
                              placeholder: String(localized: "security.sub_keys.name_placeholder",
                                                  defaultValue: "Claude Code on laptop",
                                                  comment: "Placeholder text for sub-key name input"),
                              width: 220)
                }
                Row(label: String(localized: "security.sub_keys.key_label",
                                  defaultValue: "Key",
                                  comment: "Row label for sub-key value input")) {
                    TextInput(text: $newKey,
                              placeholder: String(localized: "security.sub_keys.key_placeholder",
                                                  defaultValue: "sk-omlx-sub-…",
                                                  comment: "Placeholder text inside the sub-key value input"),
                              mono: true, width: 220)
                }
                FreeRow {
                    HStack(spacing: 6) {
                        Spacer()
                        Button(String(localized: "common.cancel",
                                      defaultValue: "Cancel",
                                      comment: "Common cancel button label")) {
                            showCreate = false
                            newName = ""
                            newKey = ""
                        }
                        .buttonStyle(.omlx(.plain, size: .small))
                        Button(String(localized: "common.create",
                                      defaultValue: "Create",
                                      comment: "Common create button label")) {
                            Task {
                                let ok = await vm.createSubKey(
                                    key: newKey, name: newName, client: client
                                )
                                if ok {
                                    showCreate = false
                                    newName = ""
                                    newKey = ""
                                }
                            }
                        }
                        .buttonStyle(.omlx(.primary, size: .small))
                        .disabled(newKey.count < 4)
                    }
                }
            }

            if vm.subKeys.isEmpty && !showCreate {
                FreeRow(isLast: true) {
                    Text(String(localized: "security.sub_keys.empty",
                                defaultValue: "No sub keys yet.",
                                comment: "Empty-state text shown when there are no sub-keys"))
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 14)
                }
            } else {
                ForEach(Array(vm.subKeys.enumerated()), id: \.element.id) { idx, sub in
                    let isLast = idx == vm.subKeys.count - 1 && !showCreate
                    FreeRow(isLast: isLast) {
                        HStack(spacing: 10) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(sub.name.isEmpty
                                     ? String(localized: "security.sub_keys.unnamed",
                                              defaultValue: "(unnamed)",
                                              comment: "Fallback display name for a sub-key with no name set")
                                     : sub.name)
                                    .font(.omlxText(13, weight: .medium))
                                    .foregroundStyle(theme.text)
                                Text(formatCreatedAt(sub.createdAt))
                                    .font(.omlxText(11))
                                    .foregroundStyle(theme.textTertiary)
                            }
                            Spacer(minLength: 8)
                            CodeChip(value: sub.key, maxWidth: 220)
                            Button {
                                Task {
                                    await vm.deleteSubKey(key: sub.key, client: client)
                                }
                            } label: {
                                Image(systemName: "trash")
                                    .font(.system(size: 12))
                                    .foregroundStyle(theme.redDot)
                            }
                            .buttonStyle(.omlx(.plain, size: .small))
                            .help(String(localized: "security.sub_keys.revoke",
                                         defaultValue: "Revoke",
                                         comment: "Tooltip on the sub-key delete button"))
                        }
                    }
                }
            }
        }
    }

    private func formatCreatedAt(_ iso: String) -> String {
        guard !iso.isEmpty else {
            return String(localized: "security.sub_keys.created_unknown",
                          defaultValue: "Created · unknown",
                          comment: "Sub-key created-at text when the timestamp is unavailable")
        }
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let date = isoFormatter.date(from: iso) ?? ISO8601DateFormatter().date(from: iso)
        guard let date else {
            return String(localized: "security.sub_keys.created_raw",
                          defaultValue: "Created · \(iso)",
                          comment: "Sub-key created-at text when the timestamp can't be parsed; placeholder is the raw ISO string")
        }
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return String(localized: "security.sub_keys.created_at",
                      defaultValue: "Created · \(f.string(from: date))",
                      comment: "Sub-key created-at text; placeholder is the formatted date and time")
    }
}
