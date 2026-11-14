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
            external: BenchExternalEndpoint(baseUrl: "https://example.test/v1", apiKey: "key", model: "remote")
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
        XCTAssertEqual(object["external"] as? [String: String],
                       ["base_url": "https://example.test/v1", "api_key": "key", "model": "remote"])
    }

    @MainActor
    func testThroughputContextDefaultsToPythonCode() {
        let vm = ThroughputBenchScreenVM()
        XCTAssertEqual(vm.contextProfile, .codePython)
        XCTAssertEqual(vm.warmupMode, .quick)
        XCTAssertFalse(vm.alignPromptToAne)
    }

    @MainActor
    func testPollEndsRunOnErrorStatusOrLostRun() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BenchStubProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let client = OMLXClient(host: "bench.test", session: session)
        // "failed" reports status=error. "lost" is unknown to the server, like after a restart.
        for benchId in ["failed", "lost"] {
            BenchStubProtocol.benchId = benchId
            let vm = ThroughputBenchScreenVM()
            vm.selectedModelId = "model"
            vm.runBenchmark(client: client)
            for _ in 0..<200 where vm.running {
                try await Task.sleep(for: .milliseconds(10))
            }
            XCTAssertFalse(vm.running, benchId)
            XCTAssertNotNil(vm.lastError, benchId)
            vm.stop()
        }
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

    // MARK: Throughput queue

    /// A queued start has no bench id yet: `benchId`/`totalTests` are absent
    /// and the app follows `queueId` until the server reaches it.
    func testDecodeQueuedStartResponse() throws {
        let json = """
        {
            "status": "queued",
            "queue_id": "q-1234abcd",
            "queue": [
                {"queue_id": "q-1234abcd", "model_id": "model-x", "external": false}
            ]
        }
        """.data(using: .utf8)!

        let response = try decoder.decode(BenchStartResponse.self, from: json)
        XCTAssertEqual(response.status, "queued")
        XCTAssertEqual(response.queueId, "q-1234abcd")
        XCTAssertNil(response.benchId)
        XCTAssertNil(response.totalTests)
        XCTAssertEqual(response.queue?.map(\.modelId), ["model-x"])
    }

    /// `/bench/active` carries the running run plus the waiting queue; the
    /// app compares `queueId` against its own to know when to attach.
    func testDecodeActiveThroughputBenchWithQueue() throws {
        let json = """
        {
            "running": true,
            "bench_id": "bench-abc",
            "model_id": "model-x",
            "context_profile": "code_python",
            "force_lm_engine": false,
            "external": false,
            "queue_id": "q-1234abcd",
            "queue": [
                {"queue_id": "q-5678efgh", "model_id": "model-y", "external": true}
            ]
        }
        """.data(using: .utf8)!

        let response = try decoder.decode(BenchActiveResponse.self, from: json)
        XCTAssertTrue(response.running)
        XCTAssertEqual(response.benchId, "bench-abc")
        XCTAssertEqual(response.queueId, "q-1234abcd")
        XCTAssertEqual(response.queue.count, 1)
        XCTAssertEqual(response.queue.first?.external, true)
    }
}

private final class BenchStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var benchId = ""

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "bench.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let (status, body): (Int, String) = switch request.url?.path {
        case "/admin/api/bench/start":
            (200, #"{"bench_id":"\#(Self.benchId)","status":"started","total_tests":1}"#)
        case "/admin/api/bench/failed/results":
            (200, #"{"bench_id":"failed","status":"error","results":[],"error":"Endpoint rejected the API key"}"#)
        default:
            (404, #"{"detail":"Benchmark not found"}"#)
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}
