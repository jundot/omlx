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
        allModels.filter {
            $0.virtual != true
                && $0.removalKind != "profile"
                && ($0.loaded || $0.isLoading)
        }
    }
    var libraryModels: [ModelDTO] { modelLibraryRows(allModels) }
    var pendingRemovalModel: ModelDTO? {
        guard let pendingRemoveID else { return nil }
        return libraryModels.first { $0.id == pendingRemoveID }
    }

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

    func remove(model: ModelDTO, client: OMLXClient) {
        pendingRemoveID = nil
        deletingID = model.id
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.deletingID = nil } }
            do {
                switch model.removalKind {
                case "profile":
                    guard let sourceModelID = model.sourceModelId,
                          let profileName = model.profileName else { return }
                    _ = try await client.deleteModelProfile(
                        id: sourceModelID,
                        name: profileName
                    )
                case "local_cache", "local_model":
                    _ = try await client.deleteHFModel(modelName: model.id)
                default:
                    return
                }
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

func modelLibraryRows(_ models: [ModelDTO]) -> [ModelDTO] {
    let rows = models.flatMap { model -> [ModelDTO] in
        let profiles = (model.exposedProfiles ?? []).map { profile in
            ModelDTO(
                id: profile.modelId ?? "\(model.id):\(profile.name)",
                displayName: profile.displayName,
                modelPath: nil,
                loaded: false,
                isLoading: false,
                estimatedSize: 0,
                estimatedSizeFormatted: "—",
                actualSize: 0,
                actualSizeFormatted: nil,
                pinned: false,
                isDefault: false,
                isFavorite: false,
                engineType: model.engineType,
                modelType: model.modelType,
                configModelType: model.configModelType,
                modelContextLength: model.modelContextLength,
                thinkingDefault: model.thinkingDefault,
                dflashCompatible: model.dflashCompatible,
                dflashCompatibilityReason: model.dflashCompatibilityReason,
                dflashSsdCacheAvailable: model.dflashSsdCacheAvailable,
                mtpCompatible: model.mtpCompatible,
                mtpCompatibilityReason: model.mtpCompatibilityReason,
                virtual: false,
                sourceType: "profile",
                sourceRepoId: nil,
                deletable: true,
                removalKind: "profile",
                sourceModelId: model.id,
                profileName: profile.name,
                exposedProfiles: nil,
                settings: nil
            )
        }
        return [model] + profiles
    }
    return sortModelsByName(rows)
}
