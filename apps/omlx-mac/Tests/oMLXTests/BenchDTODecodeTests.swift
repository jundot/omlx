// Covers the bench results envelope's decode contract, in both directions of
// the app/server version skew.
//
// The forward case matters because `upload_state` gained `feature_flags` and
// each result gained `system_metrics` when accelerated runs started uploading
// instead of being withheld.
//
// The backward case matters more: `BenchUploadStateDTO.skippedFeatures` is
// non-optional, so if the server ever stopped sending that key, decoding the
// whole `BenchResultsResponse` would throw. `ThroughputBenchScreenVM` polls
// once per second for up to two minutes, so a decode failure there is not one
// error — it is 120 of them.

import XCTest
@testable import oMLX

final class BenchDTODecodeTests: XCTestCase {

    /// Mirrors the `OMLXClient` decoder configuration.
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    /// Same source-relative resolution DTOFixtureTests uses — the fixtures are
    /// git artifacts, not bundle resources.
    private func loadFixture(_ name: String) throws -> Data {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("\(name).json")
        return try Data(contentsOf: url)
    }

    // MARK: Current server

    func testEncodeBenchmarkContextProfile() throws {
        let request = BenchStartRequest(
            modelId: "model",
            contextProfile: .novelKorean,
            warmupMode: .ane2048,
            alignPromptToAne: true,
            promptLengths: [1024],
            generationLength: 128,
            batchSizes: [2],
            forceLmEngine: true,
            external: nil
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoder.encode(request))
                as? [String: Any]
        )
        XCTAssertEqual(object["context_profile"] as? String, "novel_ko")
        XCTAssertEqual(object["warmup_mode"] as? String, "ane_2048")
        XCTAssertEqual(object["align_prompt_to_ane"] as? Bool, true)
        XCTAssertEqual(object["force_lm_engine"] as? Bool, true)
        XCTAssertNil(object["external"])
    }

    @MainActor
    func testThroughputDefaultsMatchWeb() {
        let vm = ThroughputBenchScreenVM()
        XCTAssertEqual(vm.contextProfile, .codePython)
        XCTAssertEqual(vm.warmupMode, .quick)
        XCTAssertFalse(vm.alignPromptToAne)
        XCTAssertEqual(vm.promptLengths, [1024, 4096])
        XCTAssertEqual(vm.batchSizes, [2, 4])
        XCTAssertEqual(vm.genLength, "128")
        XCTAssertFalse(vm.forceLmEngine)
        XCTAssertFalse(vm.externalEnabled)
        XCTAssertTrue(vm.selectedModelId.isEmpty)
        XCTAssertTrue(vm.externalBaseUrl.isEmpty)
        XCTAssertTrue(vm.externalApiKey.isEmpty)
        XCTAssertTrue(vm.externalModel.isEmpty)
        XCTAssertFalse(vm.canRun)
        XCTAssertTrue(vm.exportText.hasPrefix("# Context: Code (Python)"))
        vm.selectedModelId = "local-model"
        vm.forceLmEngine = true
        XCTAssertTrue(vm.canRun)
        XCTAssertEqual(vm.startRequest?.forceLmEngine, true)
        XCTAssertNil(vm.startRequest?.external)
    }

    @MainActor
    func testExternalBenchmarkRequest() throws {
        let vm = ThroughputBenchScreenVM()
        vm.forceLmEngine = true
        vm.externalEnabled = true
        vm.externalBaseUrl = " https://example.test/v1\n"
        vm.externalModel = " remote-model\n"
        vm.externalApiKey = "  test-secret  "
        XCTAssertTrue(vm.canRun, "External runs do not require a local model")

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let request = try XCTUnwrap(vm.startRequest)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any]
        )
        XCTAssertEqual(object["model_id"] as? String, "remote-model")
        XCTAssertEqual(object["force_lm_engine"] as? Bool, false)
        let external = try XCTUnwrap(object["external"] as? [String: String])
        XCTAssertEqual(external, ["base_url": "https://example.test/v1",
                                  "api_key": "  test-secret  ", "model": "remote-model"])
        vm.externalApiKey = ""
        XCTAssertEqual(vm.startRequest?.external?.apiKey, "")
        for url in ["", "https://", "file:///tmp/model", "/v1"] {
            vm.externalBaseUrl = url
            XCTAssertNil(vm.startRequest, url)
        }
        vm.externalBaseUrl = "http://localhost:8080/v1"
        vm.externalModel = " \n"
        XCTAssertNil(vm.startRequest)
    }

    @MainActor
    func testBenchmarkErrorStopsPollingAndPreservesRunIdentity() async throws {
        BenchmarkURLProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BenchmarkURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let client = OMLXClient(host: "benchmark.test", session: session)
        let vm = ThroughputBenchScreenVM()
        defer { vm.stop() }
        vm.externalEnabled = true
        vm.externalBaseUrl = "https://user:password@example.test/v1?token=url-secret#fragment-secret"
        vm.externalModel = "remote-model"
        vm.externalApiKey = "test-secret"
        vm.runBenchmark(client: client)
        XCTAssertTrue(vm.running)
        XCTAssertFalse(vm.canRun)
        for _ in 0..<200 {
            if !vm.running { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(vm.running)
        XCTAssertEqual(vm.lastError, "Endpoint rejected the API key")

        let description = vm.runDescription
        vm.externalModel = "next-model"
        vm.externalBaseUrl = "https://next.test/v1"
        vm.contextProfile = .codeMixed
        XCTAssertEqual(vm.runDescription, description)
        XCTAssertTrue(vm.exportText.contains("remote-model"))
        XCTAssertTrue(vm.exportText.contains("https://example.test/v1"))
        XCTAssertTrue(vm.exportText.hasPrefix("# Context: Code (Python)"))
        for secret in ["test-secret", "user:password", "url-secret", "fragment-secret"] {
            XCTAssertFalse(vm.exportText.contains(secret))
        }
        try await Task.sleep(for: .milliseconds(1100))
        XCTAssertEqual(BenchmarkURLProtocol.resultsRequestCount, 1)
    }

    @MainActor
    func testSwitchingTargetsDuringUploadAndCancellingPendingStart() async throws {
        BenchmarkURLProtocol.reset(completesFirstRun: true)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BenchmarkURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let client = OMLXClient(host: "benchmark.test", session: session)
        let vm = ThroughputBenchScreenVM()
        defer { vm.stop() }
        vm.selectedModelId = "local-model"
        vm.forceLmEngine = true
        vm.runBenchmark(client: client)
        for _ in 0..<200 {
            if !vm.running { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(vm.running)
        XCTAssertEqual(vm.singleResults.count, 1)
        vm.contextProfile = .codeMixed
        try await Task.sleep(for: .milliseconds(1100))
        XCTAssertEqual(vm.contextProfile, .codeMixed, "Upload polling must not reset the next run's form")
        let previousPollCount = BenchmarkURLProtocol.resultsRequestCount
        XCTAssertGreaterThan(previousPollCount, 1, "Successful runs still poll for upload completion")

        vm.externalEnabled = true
        vm.externalBaseUrl = "https://example.test/v1"
        vm.externalModel = "remote-model"
        vm.runBenchmark(client: client)
        try await Task.sleep(for: .milliseconds(1100))
        XCTAssertTrue(vm.running)
        XCTAssertNil(vm.currentBenchId)
        XCTAssertTrue(vm.singleResults.isEmpty)
        XCTAssertEqual(BenchmarkURLProtocol.resultsRequestCount, previousPollCount)
        vm.cancelBenchmark(client: client)
        XCTAssertTrue(vm.running, "Wait for the new bench ID before cancelling on the server")
        BenchmarkURLProtocol.releaseStart()
        for _ in 0..<200 {
            if vm.lastError != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertNotNil(vm.lastError)
        XCTAssertTrue(vm.running, "A rejected cancellation must leave the run tracked and retryable")
        vm.cancelBenchmark(client: client)
        for _ in 0..<200 {
            if !vm.running { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertFalse(vm.running)
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(BenchmarkURLProtocol.cancelledBenchIds, ["b-2", "b-2"])
        vm.externalEnabled = false
        XCTAssertEqual(vm.startRequest?.modelId, "local-model")
        XCTAssertEqual(vm.startRequest?.forceLmEngine, true)
        XCTAssertNil(vm.startRequest?.external)
    }

    func testDecodeResultsContextProfile() throws {
        let json = """
        {
            "bench_id": "b-1",
            "status": "completed",
            "context_profile": "code_mixed",
            "results": []
        }
        """.data(using: .utf8)!
        let response = try decoder.decode(BenchResultsResponse.self, from: json)
        XCTAssertEqual(response.contextProfile, .codeMixed)
    }

    func testDecodeAcceleratedRun() throws {
        let response = try decoder.decode(
            BenchResultsResponse.self, from: try loadFixture("bench-results")
        )

        let upload = try XCTUnwrap(response.uploadState)
        XCTAssertEqual(upload.phase, "done")
        XCTAssertEqual(upload.successCount, 1)

        let flags = try XCTUnwrap(upload.featureFlags)
        XCTAssertEqual(flags.map(\.key), ["lightning_mtp", "turboquant_kv_4bit"])
        XCTAssertEqual(flags.first?.label, "Lightning MTP")
        XCTAssertEqual(flags.first?.detail, "3 draft tokens")
        // Identifiable uses the key, which the server guarantees unique.
        XCTAssertEqual(flags.first?.id, "lightning_mtp")
        XCTAssertNil(flags.last?.detail)
    }

    func testDecodeSystemMetrics() throws {
        let response = try decoder.decode(
            BenchResultsResponse.self, from: try loadFixture("bench-results")
        )
        let metrics = try XCTUnwrap(response.results.first?.systemMetrics)

        XCTAssertEqual(metrics.sampleCount, 42)
        XCTAssertEqual(metrics.intervalS, 1.0)
        XCTAssertEqual(metrics.cpu?.totalAvg, 38.2)
        XCTAssertEqual(metrics.cpu?.pAvg, 55.1)
        XCTAssertEqual(metrics.gpu?.utilMax, 99.0)
        // Raw OSThermalPressureLevel, not the four-valued Foundation enum.
        XCTAssertEqual(metrics.thermal?.start, 0)
        XCTAssertEqual(metrics.thermal?.max, 1)
        XCTAssertEqual(metrics.memory?.physFootprintPeak, 44.87)
        XCTAssertEqual(metrics.memory?.totalRam, 128)
    }

    func testBatchResultCarriesNoSystemMetricsWhenAbsent() throws {
        let response = try decoder.decode(
            BenchResultsResponse.self, from: try loadFixture("bench-results")
        )
        let batch = try XCTUnwrap(response.results.first { $0.testType == "batch" })
        XCTAssertNil(batch.systemMetrics)
        XCTAssertEqual(batch.batchSize, 4)
    }

    // MARK: Older server

    func testDecodeLegacyResponseWithoutNewKeys() throws {
        let response = try decoder.decode(
            BenchResultsResponse.self, from: try loadFixture("bench-results-legacy")
        )

        let upload = try XCTUnwrap(response.uploadState)
        XCTAssertEqual(upload.phase, "done")
        // Absent on an older server — must decode as nil, not throw.
        XCTAssertNil(upload.featureFlags)
        XCTAssertEqual(upload.skippedFeatures, [])
        XCTAssertNil(response.results.first?.systemMetrics)
    }

    func testDecodeSkippedExternalEndpoint() throws {
        // The only skip reason that survives now that accelerated runs upload.
        let json = """
        {
            "bench_id": "b-1",
            "status": "completed",
            "results": [],
            "upload_state": {
                "phase": "skipped",
                "results": [],
                "total": 0,
                "success_count": 0,
                "failed_count": 0,
                "owner_hash": null,
                "skipped_reason": "external_endpoint",
                "skipped_features": [],
                "feature_flags": []
            }
        }
        """.data(using: .utf8)!

        let response = try decoder.decode(BenchResultsResponse.self, from: json)
        let upload = try XCTUnwrap(response.uploadState)
        XCTAssertEqual(upload.skippedReason, "external_endpoint")
        XCTAssertEqual(upload.featureFlags, [])
    }

    func testMissingSkippedFeaturesKeyFailsLoudly() {
        // Documents why the server keeps sending an always-empty
        // `skipped_features`: without it the whole envelope fails to decode.
        let json = """
        {
            "bench_id": "b-1",
            "status": "completed",
            "results": [],
            "upload_state": {
                "phase": "done",
                "results": [],
                "total": 0,
                "success_count": 0,
                "failed_count": 0,
                "owner_hash": null,
                "skipped_reason": null
            }
        }
        """.data(using: .utf8)!

        XCTAssertThrowsError(try decoder.decode(BenchResultsResponse.self, from: json))
    }

    // MARK: Accuracy queue

    func testDecodeIdleAccuracyQueueStatus() throws {
        // What `queue/status` returns when idle and right after a cancel.
        // A decode failure here left the Running row stuck after Cancel (#4250).
        let json = """
        {
            "running": false,
            "current_model": null,
            "current_bench_id": null,
            "last_progress": null,
            "phase": null,
            "queue": []
        }
        """.data(using: .utf8)!

        let status = try decoder.decode(AccuracyQueueStatus.self, from: json)
        XCTAssertFalse(status.running)
        XCTAssertNil(status.currentModel)
        XCTAssertNil(status.currentBenchId)
        XCTAssertFalse(status.isActivelyEvaluating)
    }

    func testDecodeIdleAccuracyResults() throws {
        let json = """
        {
            "results": [],
            "running": false,
            "current_model": null,
            "current_bench_id": null
        }
        """.data(using: .utf8)!

        let response = try decoder.decode(AccuracyResultsResponse.self, from: json)
        XCTAssertFalse(response.running)
        XCTAssertNil(response.currentModel)
        XCTAssertNil(response.currentBenchId)
    }
}

private final class BenchmarkURLProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var count = 0
    nonisolated(unsafe) private static var startCount = 0
    nonisolated(unsafe) private static var completesFirstRun = false
    nonisolated(unsafe) private static var heldStart: BenchmarkURLProtocol?
    nonisolated(unsafe) private static var cancelledIds: [String] = []
    static var resultsRequestCount: Int { lock.withLock { count } }
    static var cancelledBenchIds: [String] { lock.withLock { cancelledIds } }
    static func reset(completesFirstRun: Bool = false) {
        lock.withLock {
            count = 0
            startCount = 0
            self.completesFirstRun = completesFirstRun
            heldStart = nil
            cancelledIds = []
        }
    }

    static func releaseStart() {
        let pending = lock.withLock {
            defer { heldStart = nil }
            return heldStart
        }
        pending?.respond(#"{"bench_id":"b-2","status":"started","total_tests":4}"#)
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "benchmark.test"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let body: String
        switch request.url?.path {
        case "/admin/api/bench/start":
            let hold = Self.lock.withLock {
                Self.startCount += 1
                if Self.completesFirstRun && Self.startCount == 2 {
                    Self.heldStart = self
                    return true
                }
                return false
            }
            if hold { return }
            body = #"{"bench_id":"b-1","status":"started","total_tests":4}"#
        case "/admin/api/bench/b-1/results":
            let completed = Self.lock.withLock {
                Self.count += 1
                return Self.completesFirstRun
            }
            body = completed
                ? #"{"bench_id":"b-1","status":"completed","context_profile":"code_python","results":[{"test_type":"single","pp":1024,"tg":128}]}"#
                : #"{"bench_id":"b-1","status":"error","results":[],"error":"Endpoint rejected the API key"}"#
        case "/admin/api/bench/b-2/results":
            body = #"{"bench_id":"b-2","status":"running","results":[]}"#
        case "/admin/api/bench/b-1/cancel", "/admin/api/bench/b-2/cancel":
            let benchId = request.url!.deletingLastPathComponent().lastPathComponent
            let rejectFirst = Self.lock.withLock {
                Self.cancelledIds.append(benchId)
                return Self.cancelledIds.count == 1
            }
            if rejectFirst {
                respond(#"{"detail":"Cancellation temporarily unavailable"}"#, status: 503)
                return
            }
            body = "{\"bench_id\":\"\(benchId)\",\"status\":\"cancelled\"}"
        default:
            XCTFail("Unexpected request: \(request.url?.path ?? "")")
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        respond(body)
    }

    private func respond(_ body: String, status: Int = 200) {
        let response = HTTPURLResponse(url: request.url!, statusCode: status,
                                       httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {
        Self.lock.withLock {
            if Self.heldStart === self { Self.heldStart = nil }
        }
    }
}
