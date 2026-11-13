import SwiftUI

@MainActor
@Observable
final class ModelsScreenVM {
    private(set) var allModels: [ModelDTO] = []
    var librarySearch = ""
    var lastError: String?
    /// Library row the user just clicked "trash" on; non-nil drives the
    /// confirmation dialog. Cleared on cancel or after delete completes.
    var pendingRemoveID: String?
    /// While a delete is in flight, the row shows a spinner instead of the
    /// trash glyph and the whole row's button-stack is disabled to prevent
    /// double-tap deletes against a model the server is still unloading.
    private(set) var deletingID: String?

    /// Files tab of the model card opened from an update badge.
    var updateTarget: ModelCardTarget?
    private(set) var isCheckingUpdates = false
    private(set) var updateStatus: String?
    /// Results of the last update check, keyed by lowercased repo id.
    private var updates: [String: HFModelUpdateDTO] = [:]
    private var repoByModel: [String: String] = [:]

    @ObservationIgnored
    private weak var client: OMLXClient?
    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    var activeModels: [ModelDTO] {
        allModels.filter { $0.loaded || $0.isLoading }
    }
    var libraryModels: [ModelDTO] { allModels }

    func update(for id: String) -> HFModelUpdateDTO? {
        repoByModel[id].flatMap { updates[$0.lowercased()] }
    }

    func showUpdate(for id: String) {
        guard let repo = repoByModel[id] else { return }
        updateTarget = ModelCardTarget(repoId: repo, source: .huggingFace, startTab: .files)
    }

    func checkForUpdates(client: OMLXClient) async {
        isCheckingUpdates = true
        defer { isCheckingUpdates = false }
        do {
            // The server skips repos it did not download or still downloads.
            var repos: [String: String] = [:]
            for m in try await client.listHFLocalModels().models where m.sourceType == "local" {
                if let repo = m.repoId { repos[m.name] = repo }
            }
            repoByModel = repos
            let result = try await client.checkHFModelUpdates(repoIds: Set(repos.values).sorted())
            if result.checked == 0 {
                updates = [:]
                updateStatus = String(localized: "models.updates.nothing",
                                      defaultValue: "No eligible models to check.")
            } else if result.failed.count == result.checked {
                updateStatus = String(localized: "models.updates.check_failed",
                                      defaultValue: "Could not check for model updates. Previous results are last known results.")
            } else {
                updates = Dictionary(result.updates.map { ($0.key.lowercased(), $0.value) },
                                     uniquingKeysWith: { first, _ in first })
                updateStatus = String(localized: "models.updates.result",
                                      defaultValue: "Updates available: \(updates.count) · Checked: \(result.checked) · Failed: \(result.failed.count)")
            }
        } catch {
            updateStatus = String(localized: "models.updates.check_error",
                                  defaultValue: "Could not check for model updates: \(error.omlxDescription)")
        }
    }

    func startUpdate(repo: String, files: [String]?, client: OMLXClient) async -> Bool {
        do {
            _ = try await client.startHFDownload(repoId: repo, files: files)
            updates.removeValue(forKey: repo.lowercased())
            return true
        } catch {
            updateStatus = String(localized: "models.updates.start_error",
                                  defaultValue: "Could not start the model update: \(error.omlxDescription)")
            return false
        }
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
