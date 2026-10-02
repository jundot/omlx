import SwiftUI

@MainActor
@Observable
final class SecurityScreenVM {
    var apiKeySet: Bool = false
    var apiKey: String?
    var skipApiKeyVerification: Bool = false
    var subKeys: [SubKeyDTO] = []
    var lastError: String?

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
            self.lastError = error.omlxDescription
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
            self.lastError = error.omlxDescription
            return false
        }
    }

    /// Unified write path for the editor row. Routes through /setup-api-key
    /// for first-time setup (server rejects the PATCH path when no key is
    /// configured) and through PATCH /global-settings for updates.
    func applyApiKey(_ key: String, client: OMLXClient) async -> Bool {
        if apiKeySet {
            do {
                _ = try await client.updateGlobalSettings(
                    GlobalSettingsPatch(apiKey: key)
                )
                client.configure(host: client.host, port: client.port, apiKey: key)
                await load(client: client)
                return true
            } catch {
                self.lastError = error.omlxDescription
                return false
            }
        } else {
            // First-time setup: the dedicated endpoint requires a confirm
            // value, which the editor row collapses into a single field. We
            // mirror the draft as the confirm so the server-side equality
            // check passes — typo protection lives in the field's own
            // show/copy affordances now, not in a duplicate input.
            return await setupApiKey(key: key, confirm: key, client: client)
        }
    }

    func saveSkipApiKeyVerification(client: OMLXClient) async {
        do {
            _ = try await client.updateGlobalSettings(
                GlobalSettingsPatch(skipApiKeyVerification: skipApiKeyVerification)
            )
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func createSubKey(key: String, name: String, client: OMLXClient) async -> Bool {
        do {
            _ = try await client.createSubKey(key: key, name: name)
            await load(client: client)
            return true
        } catch {
            self.lastError = error.omlxDescription
            return false
        }
    }

    func deleteSubKey(key: String, client: OMLXClient) async {
        do {
            _ = try await client.deleteSubKey(key: key)
            await load(client: client)
        } catch {
            self.lastError = error.omlxDescription
        }
    }

}
