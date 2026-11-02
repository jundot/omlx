import AppKit
import XCTest
@testable import oMLX

@MainActor
final class ModelsScreenUpdatesTests: XCTestCase {
    private let root = "/models"
    private let repo = "Owner/Model_With_Underscores"
    private let id = "Model_With_Underscores"

    func testCheckUsesInventoryIdentityAndPrimaryDirectory() async throws {
        let (client, server) = makeClient()
        let extra = [
            item("cache", repo: "Owner/cache", source: "hf_cache"),
            item("gguf", repo: "Owner/gguf", format: "gguf"),
            item("ms", repo: "Owner/ms", source: "modelscope"),
            item("external", repo: "Owner/external", path: "/external/Owner/external"),
            item("wrong", repo: "Owner/wrong"),
            item("unmatched", repo: "Owner/unmatched"),
        ]
        server.set("/hf/models", ["models": [item(id, repo: repo, path: "/models/./Owner/Model_With_Underscores")] + extra])
        server.set("/models", ["models": [model()] + extra.dropLast().map {
            model(id: $0["name"] as! String, path: $0["name"] as? String == "wrong" ? "/other/wrong" : $0["path"] as? String)
        }])
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        let request = try XCTUnwrap(server.requests(for: "/hf/updates").first)
        let body = try requestBody(request)
        XCTAssertEqual(body["repo_ids"] as? [String], [repo])
        XCTAssertEqual(body["hf_token"] as? String, "")
        XCTAssertEqual(vm.availableUpdateCount, 1)
        XCTAssertEqual(vm.update(for: id)?.size, 77_000_000_000)
        vm.showUpdateOverview(id: id)
        XCTAssertEqual(vm.updatePreview?.repoId, repo)
        XCTAssertTrue(server.requests(for: "/hf/download").isEmpty, "Opening the overview must not download")
    }

    func testPartialAndTotalFailureKeepLastKnownResultsAndPollingKeepsError() async {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        server.set("/hf/updates", ["updates": [:], "checked": 2, "failed": [repo.uppercased()]])
        await vm.checkForUpdates(client: client)
        XCTAssertNotNil(vm.update(for: id))
        XCTAssertTrue(vm.updateIsStale(for: id))
        server.set("/hf/updates", ["updates": [:], "checked": 1, "failed": [repo]])
        await vm.checkForUpdates(client: client)
        XCTAssertNotNil(vm.updateError)
        let error = vm.updateError
        await vm.refresh(client: client)
        XCTAssertEqual(vm.updateError, error)
        XCTAssertNotNil(vm.update(for: id))
        server.set("/hf/updates", [:], status: 502)
        await vm.checkForUpdates(client: client)
        XCTAssertNotNil(vm.update(for: id))
        XCTAssertTrue(vm.updateIsStale(for: id))
        server.set("/hf/updates", ["updates": [:], "checked": 1, "failed": []])
        await vm.checkForUpdates(client: client)
        XCTAssertNil(vm.update(for: id))
        XCTAssertNil(vm.updateError)
    }

    func testZeroEligibleChecksAndNoCandidatesAreDistinctFromCurrentModels() async {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        server.set("/hf/updates", ["updates": [:], "checked": 0, "failed": []])
        await vm.checkForUpdates(client: client)
        XCTAssertEqual(vm.checkResult?.checked, 0)
        XCTAssertEqual(vm.updateStatus, "No eligible models to check.")
        server.set("/hf/models", ["models": []])
        await vm.checkForUpdates(client: client)
        XCTAssertEqual(server.requests(for: "/hf/updates").count, 1)
        XCTAssertEqual(vm.checkResult?.checked, 0)
    }

    func testCheckTimeoutSurvivesAuthenticationRetry() async throws {
        let (client, server) = makeClient()
        server.rejectNextCheck()
        _ = try await client.checkHFModelUpdates(repoIds: [repo])
        let requests = server.requests(for: "/hf/updates")
        XCTAssertEqual(requests.count, 2)
        XCTAssertTrue(requests.allSatisfy { $0.timeoutInterval == 120 })
        XCTAssertEqual(server.requests(for: "/login").count, 1)
        _ = try await client.listModels()
        XCTAssertNotEqual(server.requests(for: "/models").last?.timeoutInterval, 120)
    }

    func testDuplicateChecksAreIgnoredAndPollingNeverChecksHub() async {
        let (client, server) = makeClient()
        server.setDelay(0.1, for: "/hf/updates")
        let vm = ModelsScreenVM()
        async let first: Void = vm.checkForUpdates(client: client)
        while !vm.isCheckingUpdates { await Task.yield() }
        await vm.checkForUpdates(client: client)
        await first
        await vm.refresh(client: client)
        XCTAssertEqual(server.requests(for: "/hf/updates").count, 1)
        XCTAssertFalse(vm.isCheckingUpdates)
    }

    func testLoadedLoadingAndChangedPathsRejectSubmissionAfterPreview() async throws {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.showUpdateOverview(id: id)
        let preview = try XCTUnwrap(vm.updatePreview)
        for state in ["loaded", "is_loading"] {
            var changed = model()
            changed[state] = true
            server.set("/models", ["models": [changed]])
            let started = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
            XCTAssertFalse(started)
            XCTAssertEqual(vm.updateError, "Unload this model before updating.")
            XCTAssertNotNil(vm.downloadDisabledReason(for: id))
        }
        server.set("/models", ["models": [model(path: "/other/" + repo)]])
        let moved = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
        XCTAssertFalse(moved)
        XCTAssertTrue(server.requests(for: "/hf/download").isEmpty)
    }

    func testQueuePassesOnlyUpdateFilesAndReconcilesCompletion() async throws {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.showUpdateOverview(id: id)
        let preview = try XCTUnwrap(vm.updatePreview)
        let files = ["config.json", "model-00001.safetensors"]
        let started = await vm.startUpdate(model: preview, files: files, client: client)
        XCTAssertTrue(started)
        let body = try requestBody(XCTUnwrap(server.requests(for: "/hf/download").first))
        XCTAssertEqual(body["files"] as? [String], files)
        XCTAssertEqual(body["repo_id"] as? String, repo)
        XCTAssertTrue(vm.isUpdating(id))
        XCTAssertEqual(vm.availableUpdateCount, 0)
        server.set("/hf/tasks", ["tasks": [task(status: "downloading")]])
        await vm.refresh(client: client)
        XCTAssertTrue(vm.isUpdating(id))
        vm.load(id: id, client: client)
        vm.remove(id: id, client: client)
        await Task.yield()
        XCTAssertTrue(server.requests(for: "/models/" + id + "/load").isEmpty)
        XCTAssertTrue(server.requests(for: "/hf/models/" + id).isEmpty)
        server.set("/hf/tasks", ["tasks": [task(status: "completed")]])
        await vm.refresh(client: client)
        XCTAssertFalse(vm.isUpdating(id))
        XCTAssertNil(vm.update(for: id))
        XCTAssertNil(vm.updateTask(for: id))
    }

    func testInterruptedUpdatesBlockLoadingButAllowDeletion() async throws {
        for status in ["failed", "cancelled", "paused"] {
            let (client, server) = makeClient()
            server.set("/hf/tasks", ["tasks": [task(status: status)]])
            server.set("/hf/models/" + id, ["success": true])
            let vm = ModelsScreenVM()
            await vm.checkForUpdates(client: client)
            XCTAssertFalse(vm.isUpdating(id))

            let load = expectation(description: "No load for \(status) update")
            load.isInverted = true
            server.expect(load, for: "/models/" + id + "/load")
            vm.load(id: id, client: client)
            await fulfillment(of: [load], timeout: 0.1)

            let deletion = expectation(description: "Delete \(status) update")
            server.expect(deletion, for: "/hf/models/" + id)
            vm.remove(id: id, client: client)
            await fulfillment(of: [deletion], timeout: 1)
        }
    }

    func testMenuBlocksUpdateLoadsAndAllowsLoadingAfterCompletion() async throws {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.showUpdateOverview(id: id)
        let preview = try XCTUnwrap(vm.updatePreview)
        let accepted = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
        XCTAssertTrue(accepted)
        let controller = MenubarController(server: nil, config: .default, modelUpdates: vm, client: client)
        let model = try XCTUnwrap(vm.allModels.first)
        let loadItem = try XCTUnwrap(controller.modelItem(for: model).submenu?.items.first)
        let action = try XCTUnwrap(loadItem.action)
        XCTAssertFalse(loadItem.isEnabled)

        let blockedLoad = expectation(description: "Menu cannot load an updating model")
        blockedLoad.isInverted = true
        server.expect(blockedLoad, for: "/models/" + id + "/load")
        _ = controller.perform(action, with: loadItem)
        await fulfillment(of: [blockedLoad], timeout: 0.1)

        server.set("/hf/tasks", ["tasks": [task(status: "completed")]])
        await vm.refresh(client: client)
        let completedItem = try XCTUnwrap(controller.modelItem(for: model).submenu?.items.first)
        XCTAssertTrue(completedItem.isEnabled)
        let completedLoad = expectation(description: "Menu loads the completed update")
        server.expect(completedLoad, for: "/models/" + id + "/load")
        server.set("/models/" + id + "/load", ["status": "ok"])
        _ = controller.perform(action, with: completedItem)
        await fulfillment(of: [completedLoad], timeout: 1)
    }

    func testQueuedLoadCannotBypassAnUpdateThatStartsBeforeItsTaskRuns() async throws {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.showUpdateOverview(id: id)
        let preview = try XCTUnwrap(vm.updatePreview)
        let load = expectation(description: "Queued load must recheck the update state")
        load.isInverted = true
        server.expect(load, for: "/models/" + id + "/load")

        vm.load(id: id, client: client)
        let accepted = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
        XCTAssertTrue(accepted)
        await fulfillment(of: [load], timeout: 0.1)
    }

    func testQueueFailurePreservesBadgeAndDuplicateSubmissionDoesNotQueueTwice() async throws {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.showUpdateOverview(id: id)
        let preview = try XCTUnwrap(vm.updatePreview)
        server.set("/hf/download", ["detail": "Download failed"], status: 500)
        let failed = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
        XCTAssertFalse(failed)
        XCTAssertNotNil(vm.update(for: id))
        let error = vm.updateError
        await vm.refresh(client: client)
        XCTAssertEqual(vm.updateError, error)
        server.set("/hf/download", ["success": true, "task": task()])
        server.setDelay(0.1, for: "/hf/download")
        async let first = vm.startUpdate(model: preview, files: ["config.json"], client: client)
        while vm.startingUpdateID == nil { await Task.yield() }
        let duplicate = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
        XCTAssertFalse(duplicate)
        let accepted = await first
        XCTAssertTrue(accepted)
        XCTAssertEqual(server.requests(for: "/hf/download").count, 2, "One rejected request and one accepted request")
    }

    func testPollStartedBeforeSubmissionCannotEraseAcceptedTask() async throws {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.showUpdateOverview(id: id)
        let preview = try XCTUnwrap(vm.updatePreview)
        let requestCount = server.requests(for: "/hf/tasks").count
        server.setDelay(0.15, for: "/hf/tasks")
        async let poll: Void = vm.refresh(client: client)
        while server.requests(for: "/hf/tasks").count == requestCount { await Task.yield() }
        server.setDelay(0, for: "/hf/tasks")
        let accepted = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
        XCTAssertTrue(accepted)
        await poll
        XCTAssertTrue(vm.isUpdating(id))
        XCTAssertEqual(vm.updateTask(for: id)?.taskId, "update-1")
    }

    func testUnfinishedDownloadsRouteToExistingTasksAndMissingTaskBecomesUnchecked() async throws {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.showUpdateOverview(id: id)
        let preview = try XCTUnwrap(vm.updatePreview)
        for status in ["pending", "downloading", "failed", "paused", "cancelled"] {
            server.set("/hf/tasks", ["tasks": [task(status: status)]])
            let accepted = await vm.startUpdate(model: preview, files: ["config.json"], client: client)
            XCTAssertFalse(accepted)
            XCTAssertEqual(vm.updateTask(for: id)?.status, status)
        }
        XCTAssertTrue(server.requests(for: "/hf/download").isEmpty)
        server.set("/hf/tasks", ["tasks": []])
        await vm.refresh(client: client)
        XCTAssertNil(vm.updateTask(for: id))
        XCTAssertNil(vm.update(for: id))
    }

    func testResultsSurviveNavigationButDirectoryAndServerChangesResetThem() async {
        let (client, server) = makeClient()
        let vm = ModelsScreenVM()
        await vm.checkForUpdates(client: client)
        vm.stop()
        await vm.start(client: client)
        vm.stop()
        XCTAssertNotNil(vm.update(for: id))
        server.set("/global-settings", settings(directory: "/new-root"))
        await vm.refresh(client: client)
        XCTAssertNil(vm.checkResult)
        XCTAssertNil(vm.update(for: id))
        server.set("/global-settings", settings())
        await vm.checkForUpdates(client: client)
        client.configure(host: client.host, port: 9001, apiKey: nil)
        await vm.refresh(client: client)
        XCTAssertNil(vm.checkResult)
        XCTAssertTrue(vm.updatesByRepo.isEmpty)
    }

    func testResetDiscardsAnInFlightCheck() async {
        let (client, server) = makeClient()
        server.setDelay(0.1, for: "/hf/updates")
        let vm = ModelsScreenVM()
        async let check: Void = vm.checkForUpdates(client: client)
        while server.requests(for: "/hf/updates").isEmpty { await Task.yield() }
        vm.resetModelUpdates()
        await check
        XCTAssertNil(vm.checkResult)
        XCTAssertFalse(vm.isCheckingUpdates)
        XCTAssertTrue(vm.updatesByRepo.isEmpty)
    }

    private func makeClient() -> (OMLXClient, ModelUpdateTestServer) {
        let server = ModelUpdateTestServer()
        server.set("/models", ["models": [model()]])
        server.set("/global-settings", settings())
        server.set("/hf/models", ["models": [item(id, repo: repo)]])
        server.set("/hf/updates", ["updates": [repo.lowercased(): ["files": 2, "size": 77_000_000_000]], "checked": 1, "failed": []])
        server.set("/hf/tasks", ["tasks": []])
        server.set("/hf/download", ["success": true, "task": task()])
        server.set("/login", ["success": true])
        let host = UUID().uuidString.lowercased() + ".test"
        ModelUpdateURLProtocol.register(server, host: host)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ModelUpdateURLProtocol.self]
        let client = OMLXClient(host: host, apiKey: "test-key", session: URLSession(configuration: configuration))
        return (client, server)
    }

    private func settings(directory: String = "/models") -> [String: Any] {
        ["server": ["host": "localhost", "port": 8000, "log_level": "info", "server_aliases": []],
         "model": ["model_dirs": [directory], "model_dir": directory]]
    }

    private func item(_ name: String, repo: String, source: String = "local", format: String = "mlx", path: String? = nil) -> [String: Any] {
        ["name": name, "path": path ?? root + "/" + repo, "repo_id": repo, "source_type": source, "model_format": format]
    }

    private func model(id: String = "Model_With_Underscores", path: String? = nil) -> [String: Any] {
        ["id": id, "display_name": "An alias, not a repo", "model_path": path ?? root + "/" + repo,
         "loaded": false, "is_loading": false, "estimated_size": 100]
    }

    private func task(status: String = "pending") -> [String: Any] {
        ["task_id": "update-1", "repo_id": repo.uppercased(), "status": status, "progress": 0,
         "total_size": 100, "downloaded_size": 0, "error": "", "created_at": 1,
         "started_at": 1, "completed_at": 0, "retry_count": 0]
    }

    private func requestBody(_ request: URLRequest) throws -> [String: Any] {
        let data: Data
        if let body = request.httpBody {
            data = body
        } else if let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var bytes = Data()
            var buffer = [UInt8](repeating: 0, count: 1024)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                bytes.append(buffer, count: count)
            }
            data = bytes
        } else {
            throw OMLXClientError.invalidResponse
        }
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

private final class ModelUpdateTestServer: @unchecked Sendable {
    struct Response {
        let status: Int
        let body: Data
        let delay: TimeInterval
    }
    private let lock = NSLock()
    private var responses: [String: (Int, Data)] = [:]
    private var recorded: [URLRequest] = []
    private var delays: [String: TimeInterval] = [:]
    private var expectations: [String: XCTestExpectation] = [:]
    private var rejectCheck = false

    func set(_ path: String, _ body: [String: Any], status: Int = 200) {
        let data = try! JSONSerialization.data(withJSONObject: body)
        lock.withLock { responses[path] = (status, data) }
    }
    func setDelay(_ delay: TimeInterval, for path: String) { lock.withLock { delays[path] = delay } }
    func rejectNextCheck() { lock.withLock { rejectCheck = true } }
    func expect(_ expectation: XCTestExpectation, for path: String) {
        lock.withLock { expectations[path] = expectation }
    }
    func requests(for path: String) -> [URLRequest] {
        lock.withLock { recorded.filter { $0.url?.path == "/admin/api" + path } }
    }
    func response(to request: URLRequest) -> Response {
        lock.withLock {
            recorded.append(request)
            let path = String((request.url?.path ?? "").dropFirst("/admin/api".count))
            expectations[path]?.fulfill()
            if path == "/hf/updates", rejectCheck {
                rejectCheck = false
                return Response(status: 401, body: Data(), delay: 0)
            }
            let (status, body) = responses[path] ?? (404, Data())
            return Response(status: status, body: body, delay: delays[path] ?? 0)
        }
    }
}

private final class ModelUpdateURLProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var servers: [String: ModelUpdateTestServer] = [:]

    static func register(_ server: ModelUpdateTestServer, host: String) {
        lock.withLock { servers[host] = server }
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let server = Self.lock.withLock({ Self.servers[request.url!.host!] }) else { return }
        let result = server.response(to: request)
        DispatchQueue.global().asyncAfter(deadline: .now() + result.delay) { [self] in
            let response = HTTPURLResponse(url: request.url!, statusCode: result.status, httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: result.body)
            client?.urlProtocolDidFinishLoading(self)
        }
    }
    override func stopLoading() {}
}
