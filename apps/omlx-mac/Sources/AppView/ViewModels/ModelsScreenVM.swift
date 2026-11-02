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

    var updatePreview: HFLocalModelDTO?
    private(set) var isCheckingUpdates = false
    private(set) var startingUpdateID: String?
    private(set) var updateError: String?
    private(set) var checkResult: HFUpdateCheckResponse?
    private(set) var updatesByRepo: [String: HFModelUpdateDTO] = [:]
    private(set) var staleUpdateRepos: Set<String> = []
    private var updateCandidates: [String: HFLocalModelDTO] = [:]
    private var tasksByRepo: [String: HFTaskDTO] = [:]
    private var updateContext: UpdateContext?

    private struct UpdateContext: Equatable {
        let host: String
        let port: Int
        let directory: String
    }

    @ObservationIgnored private var updateGeneration = 0

    @ObservationIgnored
    private var pollTask: Task<Void, Never>?

    var activeModels: [ModelDTO] {
        allModels.filter { $0.loaded || $0.isLoading }
    }
    var libraryModels: [ModelDTO] { allModels }

    var availableUpdateCount: Int {
        updatesByRepo.keys.filter { tasksByRepo[$0].map { $0.statusEnum == .completed } ?? true }.count
    }

    var updateStatus: String? {
        guard let checkResult else { return nil }
        if checkResult.checked == 0 {
            return String(localized: "models.updates.nothing", defaultValue: "No eligible models to check.")
        }
        return String(localized: "models.updates.result",
                      defaultValue: "Updates available: \(availableUpdateCount) · Checked: \(checkResult.checked) · Failed: \(checkResult.failed.count)")
    }

    func update(for id: String) -> HFModelUpdateDTO? {
        guard let repo = updateCandidates[id]?.repoId?.lowercased() else { return nil }
        return updatesByRepo[repo]
    }

    func updateIsStale(for id: String) -> Bool {
        guard let repo = updateCandidates[id]?.repoId?.lowercased() else { return false }
        return staleUpdateRepos.contains(repo)
    }

    func updateTask(for id: String) -> HFTaskDTO? {
        guard let repo = updateCandidates[id]?.repoId?.lowercased(),
              let task = tasksByRepo[repo], task.statusEnum != .completed else { return nil }
        return task
    }

    func isUpdating(_ id: String) -> Bool {
        startingUpdateID == id || updateTask(for: id)?.isActive == true
    }

    func isLoadBlocked(_ id: String) -> Bool {
        // Interrupted updates can leave files from different model revisions.
        startingUpdateID == id || updateTask(for: id) != nil
    }

    func showUpdateOverview(id: String) {
        guard update(for: id) != nil else { return }
        updatePreview = updateCandidates[id]
    }

    func downloadDisabledReason(for id: String) -> String? {
        if let reason = updateTargetError(for: id) { return reason }
        if isCheckingUpdates || startingUpdateID != nil {
            return String(localized: "models.updates.busy", defaultValue: "Wait for the current update operation to finish.")
        }
        return nil
    }

    private func updateTargetError(for id: String) -> String? {
        guard let model = allModels.first(where: { $0.id == id }), updateCandidates[id] != nil,
              deletingID != id else {
            return String(localized: "models.updates.unavailable",
                          defaultValue: "This model changed or is no longer available. Check for updates again.")
        }
        if model.loaded || model.isLoading {
            return String(localized: "models.updates.unload", defaultValue: "Unload this model before updating.")
        }
        if updateTask(for: id) != nil {
            return String(localized: "models.updates.existing_task",
                          defaultValue: "Manage this model's unfinished download in Downloads.")
        }
        return nil
    }

    func resetModelUpdates() {
        updateGeneration += 1
        clearUpdateResults()
        updateContext = nil
        isCheckingUpdates = false
        startingUpdateID = nil
    }

    private func clearUpdateResults() {
        updatePreview = nil
        updateError = nil
        checkResult = nil
        updatesByRepo = [:]
        staleUpdateRepos = []
        updateCandidates = [:]
        tasksByRepo = [:]
    }

    func checkForUpdates(client: OMLXClient) async {
        guard !isCheckingUpdates, startingUpdateID == nil else { return }
        isCheckingUpdates = true
        updateError = nil
        let generation = updateGeneration
        defer { if generation == updateGeneration { isCheckingUpdates = false } }
        do {
            try await refreshUpdateInventory(client: client)
            let context = updateContext
            let repos = Array(Set(updateCandidates.values.compactMap(\.repoId))).sorted()
            let result = repos.isEmpty
                ? HFUpdateCheckResponse(updates: [:], checked: 0, failed: [])
                : try await client.checkHFModelUpdates(repoIds: repos)
            guard generation == updateGeneration, context == updateContext,
                  context?.host == client.host, context?.port == client.port else { return }
            checkResult = result
            let failed = Set(result.failed.map { $0.lowercased() })
            if result.checked == 0 || failed.count == result.checked {
                staleUpdateRepos.formUnion(updatesByRepo.keys)
                if !failed.isEmpty {
                    updateError = String(localized: "models.updates.check_failed",
                                         defaultValue: "Could not check for model updates. Previous results are last known results.")
                }
                return
            }
            var updates = updatesByRepo.filter { failed.contains($0.key) }
            for (repo, update) in result.updates { updates[repo.lowercased()] = update }
            updatesByRepo = updates
            staleUpdateRepos = failed.intersection(updates.keys)
        } catch {
            guard generation == updateGeneration, !Task.isCancelled else { return }
            staleUpdateRepos.formUnion(updatesByRepo.keys)
            updateError = String(localized: "models.updates.check_error",
                                 defaultValue: "Could not check for model updates: \(error.omlxDescription)")
        }
    }

    // shortcut: guards this native flow; concurrent clients need server-side locking.
    func startUpdate(model: HFLocalModelDTO, files: [String]?, client: OMLXClient) async -> Bool {
        guard startingUpdateID == nil, !isCheckingUpdates else { return false }
        // A poll started before queueing must not erase the newly accepted task.
        updateGeneration += 1
        startingUpdateID = model.name
        updateError = nil
        let generation = updateGeneration
        defer { if generation == updateGeneration { startingUpdateID = nil } }
        do {
            try await refreshUpdateInventory(client: client)
            guard updateCandidates[model.name] == model, let repo = model.repoId else {
                updateError = String(localized: "models.updates.unavailable",
                                    defaultValue: "This model changed or is no longer available. Check for updates again.")
                return false
            }
            if let reason = updateTargetError(for: model.name) { updateError = reason; return false }
            let context = updateContext
            let response = try await client.startHFDownload(repoId: repo, files: files)
            guard generation == updateGeneration, context == updateContext,
                  context?.host == client.host, context?.port == client.port else { return false }
            guard response.success, let task = response.task,
                  task.repoId.caseInsensitiveCompare(repo) == .orderedSame else {
                throw OMLXClientError.invalidResponse
            }
            tasksByRepo[repo.lowercased()] = task
            if task.statusEnum == .completed { updatesByRepo.removeValue(forKey: repo.lowercased()) }
            return true
        } catch {
            guard generation == updateGeneration, !Task.isCancelled else { return false }
            updateError = String(localized: "models.updates.start_error",
                                 defaultValue: "Could not start the model update: \(error.omlxDescription)")
            return false
        }
    }

    private static func normalizedPath(_ path: String) -> String {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL.path
    }

    private func context(client: OMLXClient, settings: GlobalSettingsDTO) throws -> UpdateContext {
        guard let directory = settings.model?.modelDirs?.first ?? settings.model?.modelDir,
              !directory.isEmpty else { throw OMLXClientError.invalidResponse }
        return UpdateContext(host: client.host, port: client.port, directory: Self.normalizedPath(directory))
    }

    private func refreshUpdateInventory(client: OMLXClient) async throws {
        let generation = updateGeneration
        let host = client.host, port = client.port
        async let settings = client.getGlobalSettings()
        async let inventory = client.listHFLocalModels()
        async let models = client.listModels()
        async let tasks = client.listHFTasks()
        let (settingsValue, inventoryValue, modelsValue, tasksValue) = try await (settings, inventory, models, tasks)
        guard generation == updateGeneration, host == client.host, port == client.port, !Task.isCancelled else {
            throw CancellationError()
        }
        let nextContext = try context(client: client, settings: settingsValue)
        if updateContext != nextContext {
            clearUpdateResults()
            updateContext = nextContext
        }
        allModels = sortModelsByName(modelsValue.models)
        var candidates: [String: HFLocalModelDTO] = [:]
        for item in inventoryValue.models {
            guard item.sourceType == "local", item.modelFormat != "gguf",
                  let repo = item.repoId, repo.split(separator: "/").count == 2,
                  let path = item.path,
                  let model = allModels.first(where: { $0.id == item.name }), !model.isGGUF,
                  let modelPath = model.modelPath,
                  Self.normalizedPath(path) == Self.normalizedPath(modelPath),
                  Self.normalizedPath(path) == Self.normalizedPath(nextContext.directory + "/" + repo) else { continue }
            candidates[item.name] = item
        }
        updateCandidates = candidates
        let repos = Set(candidates.values.compactMap { $0.repoId?.lowercased() })
        updatesByRepo = updatesByRepo.filter { repos.contains($0.key) }
        staleUpdateRepos.formIntersection(repos)
        reconcileTasks(tasksValue.tasks)
    }

    private func reconcileTasks(_ tasks: [HFTaskDTO]) {
        let repos = Set(updateCandidates.values.compactMap { $0.repoId?.lowercased() })
        var latest: [String: HFTaskDTO] = [:]
        for task in tasks where repos.contains(task.repoId.lowercased()) {
            let repo = task.repoId.lowercased()
            if latest[repo].map({ $0.createdAt > task.createdAt }) != true { latest[repo] = task }
        }
        for (repo, previous) in tasksByRepo {
            if latest[repo] == nil || (latest[repo]?.statusEnum == .completed
                && (previous.statusEnum != .completed || previous.taskId != latest[repo]?.taskId)) {
                updatesByRepo.removeValue(forKey: repo)
                staleUpdateRepos.remove(repo)
            }
        }
        tasksByRepo = latest
    }

    func start(client: OMLXClient) async {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.refresh(client: client)
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
            guard self?.isLoadBlocked(id) == false else { return }
            do {
                _ = try await client.loadModel(id: id)
                await self?.refresh(client: client)
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
                await self?.refresh(client: client)
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
                await self?.refresh(client: client)
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func remove(id: String, client: OMLXClient) {
        pendingRemoveID = nil
        guard !isUpdating(id) else { return }
        deletingID = id
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.deletingID = nil } }
            do {
                _ = try await client.deleteHFModel(modelName: id)
                await self?.refresh(client: client)
                self?.lastError = nil
            } catch {
                guard let self else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    func refresh(client: OMLXClient) async {
        guard startingUpdateID == nil else { return }
        let generation = updateGeneration
        let previousContext = updateContext
        do {
            let models = try await client.listModels().models
            if let previousContext {
                async let settings = client.getGlobalSettings()
                async let tasks = client.listHFTasks()
                let (settingsValue, tasksValue) = try await (settings, tasks)
                guard generation == updateGeneration, previousContext == updateContext, !Task.isCancelled else { return }
                if try context(client: client, settings: settingsValue) != previousContext {
                    resetModelUpdates()
                } else {
                    updateCandidates = updateCandidates.filter { id, item in
                        models.contains { $0.id == id && $0.modelPath.map(Self.normalizedPath) == item.path.map(Self.normalizedPath) }
                    }
                    let repos = Set(updateCandidates.values.compactMap { $0.repoId?.lowercased() })
                    updatesByRepo = updatesByRepo.filter { repos.contains($0.key) }
                    reconcileTasks(tasksValue.tasks)
                }
            } else if generation != updateGeneration || Task.isCancelled { return }
            self.allModels = sortModelsByName(models)
            self.lastError = nil
        } catch {
            guard generation == updateGeneration, !Task.isCancelled else { return }
            self.lastError = error.omlxDescription
        }
    }

}
