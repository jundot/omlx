import SwiftUI

@MainActor
@Observable
final class ModelsScreenVM {
    private(set) var allModels: [ModelDTO] = []
    var lastError: String?
    /// Library row the user just clicked "trash" on; non-nil drives the
    /// confirmation dialog. Cleared on cancel or after delete completes.
    var pendingRemoveID: String?
    /// While a delete is in flight, the row shows a spinner instead of the
    /// trash glyph and the whole row's button-stack is disabled to prevent
    /// double-tap deletes against a model the server is still unloading.
    private(set) var deletingID: String?

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    var activeModels: [ModelDTO] {
        allModels.filter { $0.loaded || $0.isLoading }
    }
    var libraryModels: [ModelDTO] { allModels }

    func start(client: OMLXClient) async {
        self.client = client
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.refresh(clearsError: false)
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func load(id: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.loadModel(id: id)
                await self?.refresh()
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func unload(id: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.unloadModel(id: id)
                await self?.refresh()
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func setFavorite(id: String, favorite: Bool, client: OMLXClient) {
        Task { [weak self] in
            do {
                var patch = ModelSettingsPatch()
                patch.isFavorite = favorite
                _ = try await client.updateModelSettings(id: id, patch: patch)
                await self?.refresh()
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func remove(id: String, client: OMLXClient) {
        pendingRemoveID = nil
        deletingID = id
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.deletingID = nil } }
            do {
                _ = try await client.deleteHFModel(modelName: id)
                await self?.refresh()
                self?.lastError = nil
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    /// `clearsError`: the 2s poll loop (below) must not touch `lastError` —
    /// it was wiping a load/unload/favorite/remove failure within that
    /// window (§G2). The action methods call this right after their own
    /// successful request, where touching lastError is legitimate
    /// action-triggered feedback, not a poll clobbering it.
    private func refresh(clearsError: Bool = true) async {
        guard let client else { return }
        do {
            self.allModels = sortModelsByName(try await client.listModels().models)
            if clearsError { self.lastError = nil }
        } catch {
            if clearsError { self.lastError = error.omlxDescription }
        }
    }

}
