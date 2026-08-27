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
            batchSizes: [2]
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
    }

    @MainActor
    func testThroughputContextDefaultsToPythonCode() {
        let vm = ThroughputBenchScreenVM()
        XCTAssertEqual(vm.contextProfile, .codePython)
        XCTAssertEqual(vm.warmupMode, .quick)
        XCTAssertFalse(vm.alignPromptToAne)
        XCTAssertTrue(vm.exportText.hasPrefix("# Context: Code (Python)"))
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
}
