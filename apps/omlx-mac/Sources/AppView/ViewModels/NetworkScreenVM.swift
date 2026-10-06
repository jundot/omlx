import SwiftUI

@MainActor
@Observable
final class NetworkScreenVM {
    // Editable drafts
    var httpProxy: String = ""
    var httpsProxy: String = ""
    var noProxy: String = ""
    var caBundle: String = ""

    // Last-loaded values. Drives the Apply button's enabled state.
    private(set) var loadedHttpProxy: String = ""
    private(set) var loadedHttpsProxy: String = ""
    private(set) var loadedNoProxy: String = ""
    private(set) var loadedCaBundle: String = ""

    private(set) var isSaving: Bool = false
    @ObservationIgnored
    private var restoreBeforeReset: (() -> Void)?
    var showResetNotice = false
    private(set) var isLoading = false
    private(set) var isResetting = false
    var lastError: String?

    /// Trimmed draft != loaded for at least one field. Whitespace-only edits
    /// don't count as changes.
    var hasPendingChanges: Bool {
        trim(httpProxy)  != loadedHttpProxy
        || trim(httpsProxy) != loadedHttpsProxy
        || trim(noProxy)    != loadedNoProxy
        || trim(caBundle)   != loadedCaBundle
    }

    func resetDefaults(client: OMLXClient) async {
        guard !isLoading, !isResetting, !showResetNotice else { return }
        isResetting = true
        defer { isResetting = false }
        let previous = (
            httpProxy: httpProxy,
            httpsProxy: httpsProxy,
            noProxy: noProxy,
            lastError: lastError
        )
        do {
            let settings = try await client.getGlobalSettingsDefaults()
            restoreBeforeReset = { [weak self] in
                guard let self else { return }
                self.httpProxy = previous.httpProxy
                self.httpsProxy = previous.httpsProxy
                self.noProxy = previous.noProxy
                self.lastError = previous.lastError
            }
            if let net = settings.network {
                self.httpProxy = net.httpProxy
                self.httpsProxy = net.httpsProxy
                self.noProxy = net.noProxy
            }

            self.lastError = nil
            self.showResetNotice = true
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func cancelReset() {
        restoreBeforeReset?()
        confirmReset()
    }

    func confirmReset() {
        restoreBeforeReset = nil
        showResetNotice = false
    }

    func load(client: OMLXClient) async {
        isLoading = true
        defer { isLoading = false }
        do {
            let settings = try await client.getGlobalSettings()
            if let net = settings.network {
                self.httpProxy = net.httpProxy
                self.httpsProxy = net.httpsProxy
                self.noProxy = net.noProxy
                self.caBundle = net.caBundle
                self.loadedHttpProxy  = net.httpProxy
                self.loadedHttpsProxy = net.httpsProxy
                self.loadedNoProxy    = net.noProxy
                self.loadedCaBundle   = net.caBundle
            }
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    func save(client: OMLXClient) async {
        // Only send fields the user actually changed so we don't clobber
        // values set out-of-band (e.g. CLI or another admin client).
        var patch = GlobalSettingsPatch()
        var touched: [String] = []
        if trim(httpProxy) != loadedHttpProxy {
            patch.networkHttpProxy = trim(httpProxy)
            touched.append("http_proxy")
        }
        if trim(httpsProxy) != loadedHttpsProxy {
            patch.networkHttpsProxy = trim(httpsProxy)
            touched.append("https_proxy")
        }
        if trim(noProxy) != loadedNoProxy {
            patch.networkNoProxy = trim(noProxy)
            touched.append("no_proxy")
        }
        if trim(caBundle) != loadedCaBundle {
            patch.networkCaBundle = trim(caBundle)
            touched.append("ca_bundle")
        }
        guard !touched.isEmpty else { return }

        isSaving = true
        defer { isSaving = false }
        do {
            _ = try await client.updateGlobalSettings(patch)
            // Converge loaded baselines on success.
            self.loadedHttpProxy  = trim(httpProxy)
            self.loadedHttpsProxy = trim(httpsProxy)
            self.loadedNoProxy    = trim(noProxy)
            self.loadedCaBundle   = trim(caBundle)
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

    private func trim(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespaces)
    }

}
