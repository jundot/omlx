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
                await self.refresh()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func load(id: String, client: OMLXClient) {
        if let model = allModels.first(where: { $0.id == id }), model.isAttachableDrafter {
            attachDrafter(model, client: client)
            return
        }
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

    /// Attach an MTP/Assistant drafter to its chat model instead of
    /// loading it standalone: enable VLM MTP on the target with this
    /// drafter, then load the target. When the target is already loaded
    /// the settings change makes the server rebuild its engine with the
    /// drafter attached, so no explicit load call is needed.
    private func attachDrafter(_ drafter: ModelDTO, client: OMLXClient) {
        Task { [weak self] in
            do {
                guard let self else { return }
                guard let target = MtpDrafterPairing.resolveTarget(
                    drafter: drafter, in: self.allModels
                ) else {
                    self.lastError = String(
                        localized: "models.library.attach.no_target",
                        defaultValue: "Can't find a chat model to attach drafter \(drafter.id) to. Attach it manually via the chat model's settings instead.",
                        comment: "Error shown when clicking Load on a speculative-decoding drafter whose companion chat model can't be found; placeholder is the drafter model id")
                    return
                }
                var patch = ModelSettingsPatch()
                patch.vlmMtpEnabled = true
                patch.vlmMtpDraftModel = drafter.id
                _ = try await client.updateModelSettings(id: target.id, patch: patch)
                if !target.loaded && !target.isLoading {
                    _ = try await client.loadModel(id: target.id)
                }
                await self.refresh()
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

    private func refresh() async {
        guard let client else { return }
        do {
            self.allModels = sortModelsByName(try await client.listModels().models)
            self.lastError = nil
        } catch {
            self.lastError = error.omlxDescription
        }
    }

}
