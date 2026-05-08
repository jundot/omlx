// PR 9 — Security.
//
// Three sections:
//   • API Key — copyable current key (when configured) + a setup form when
//                not yet configured (POST /admin/api/setup-api-key)
//   • Authentication — `skip_api_key_verification` toggle (no-auth mode)
//   • Sub Keys — list + create + delete via /admin/api/sub-keys (POST/DELETE)
//
// The main API key is shown as a CodeChip; the design hides it behind
// "show password" but admins copy it into client configs all the time, so
// click-to-copy is the right ergonomics. Sub keys are read+show-once (the
// key is only revealed in the row that lists them — same as the GET response).

import SwiftUI
import AppKit

struct SecurityScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = SecurityScreenVM()

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
    @ObservedObject var vm: SecurityScreenVM
    let client: OMLXClient

    @State private var setup1: String = ""
    @State private var setup2: String = ""

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            "API Key",
            subtitle: "Required to authenticate /v1 requests and admin sessions"
        )

        ListGroup {
            if vm.apiKeySet {
                Row(label: "Status", isLast: true) {
                    HStack(spacing: 8) {
                        StatusPill(status: .running)
                        if let key = vm.apiKey, !key.isEmpty {
                            CodeChip(value: key)
                        }
                    }
                }
            } else {
                FreeRow {
                    HStack(spacing: 8) {
                        StatusPill(status: .custom(
                            color: theme.amberDot, label: "Not configured", fillBg: true
                        ))
                        Text("Set an API key before exposing the server.")
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textSecondary)
                    }
                }
                Row(label: "New API Key",
                    sublabel: "At least 4 printable characters, no whitespace") {
                    TextInput(text: $setup1, placeholder: "sk-omlx-…", mono: true, width: 220)
                }
                Row(label: "Confirm") {
                    TextInput(text: $setup2, placeholder: "Re-enter", mono: true, width: 220)
                }
                FreeRow(isLast: true) {
                    HStack {
                        Spacer()
                        Button("Save API Key") {
                            Task {
                                let ok = await vm.setupApiKey(
                                    key: setup1, confirm: setup2, client: client
                                )
                                if ok {
                                    setup1 = ""
                                    setup2 = ""
                                }
                            }
                        }
                        .buttonStyle(.omlx(.primary))
                        .disabled(setup1.isEmpty || setup2.isEmpty || setup1 != setup2)
                    }
                }
            }
        }
    }
}

// MARK: - Authentication

private struct AuthenticationSection: View {
    @ObservedObject var vm: SecurityScreenVM
    let client: OMLXClient

    var body: some View {
        SectionHeader("Authentication")

        ListGroup {
            Row(
                label: "Disable API Key Verification",
                sublabel: "Allow unauthenticated /v1 and admin requests. Use for development only.",
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
    @ObservedObject var vm: SecurityScreenVM
    let client: OMLXClient

    @State private var newName: String = ""
    @State private var newKey: String = ""
    @State private var showCreate: Bool = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            "Sub Keys",
            subtitle: "Issue scoped keys for individual apps or users. Sub keys cannot grant admin access."
        ) {
            Button {
                showCreate = true
            } label: {
                Label("New", systemImage: "plus")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.omlx(.normal, size: .small))
        }

        ListGroup {
            if showCreate {
                Row(label: "Name", sublabel: "Optional human-readable label") {
                    TextInput(text: $newName, placeholder: "Claude Code on laptop", width: 220)
                }
                Row(label: "Key") {
                    TextInput(text: $newKey, placeholder: "sk-omlx-sub-…", mono: true, width: 220)
                }
                FreeRow {
                    HStack(spacing: 6) {
                        Spacer()
                        Button("Cancel") {
                            showCreate = false
                            newName = ""
                            newKey = ""
                        }
                        .buttonStyle(.omlx(.plain, size: .small))
                        Button("Create") {
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
                    Text("No sub keys yet.")
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
                                Text(sub.name.isEmpty ? "(unnamed)" : sub.name)
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
                            .help("Revoke")
                        }
                    }
                }
            }
        }
    }

    private func formatCreatedAt(_ iso: String) -> String {
        guard !iso.isEmpty else { return "Created · unknown" }
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let date = isoFormatter.date(from: iso) ?? ISO8601DateFormatter().date(from: iso)
        guard let date else { return "Created · \(iso)" }
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return "Created · \(f.string(from: date))"
    }
}

// MARK: - View model

@MainActor
final class SecurityScreenVM: ObservableObject {
    @Published var apiKeySet: Bool = false
    @Published var apiKey: String?
    @Published var skipApiKeyVerification: Bool = false
    @Published var subKeys: [SubKeyDTO] = []
    @Published var lastError: String?

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
            let settings = try await client.getGlobalSettings()
            self.apiKeySet = settings.auth?.apiKeySet ?? false
            self.apiKey = settings.auth?.apiKey
            self.skipApiKeyVerification = settings.auth?.skipApiKeyVerification ?? false
            self.subKeys = settings.auth?.subKeys ?? []
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    func setupApiKey(key: String, confirm: String, client: OMLXClient) async -> Bool {
        do {
            _ = try await client.setupApiKey(key, confirm: confirm)
            // Re-bootstrap the client so subsequent /admin/api/* calls auth
            // with the new key.
            client.configure(host: client.host, port: client.port, apiKey: key)
            await load(client: client)
            return true
        } catch {
            self.lastError = describe(error)
            return false
        }
    }

    func saveSkipApiKeyVerification(client: OMLXClient) async {
        do {
            _ = try await client.updateGlobalSettings(
                GlobalSettingsPatch(skipApiKeyVerification: skipApiKeyVerification)
            )
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    func createSubKey(key: String, name: String, client: OMLXClient) async -> Bool {
        do {
            _ = try await client.createSubKey(key: key, name: name)
            await load(client: client)
            return true
        } catch {
            self.lastError = describe(error)
            return false
        }
    }

    func deleteSubKey(key: String, client: OMLXClient) async {
        do {
            _ = try await client.deleteSubKey(key: key)
            await load(client: client)
        } catch {
            self.lastError = describe(error)
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
    }
}
