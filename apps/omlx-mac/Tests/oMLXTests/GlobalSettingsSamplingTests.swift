// The server-side default profile lives in `GlobalSettings.sampling`.
// These tests pin the decode shape of the nested `sampling` object on the
// read side and the flat `sampling_*` keys on the patch side, so a future
// rename on either edge breaks the build instead of silently dropping the
// server defaults the Profiles tab and Server screen depend on.

import XCTest
import SwiftUI
@testable import oMLX

final class GlobalSettingsSamplingTests: XCTestCase {

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    @MainActor
    func testResetStagesEachScreenWithoutSavingOrReplacingPaths() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SettingsDefaultsURLProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let client = OMLXClient(host: "127.0.0.1", port: 18159, session: session)
        let server = ServerScreenVM()
        let performance = PerformanceScreenVM()
        let network = NetworkScreenVM()
        await server.load(client: client)
        await performance.load(client: client)
        await network.load(client: client)
        XCTAssertNil(server.lastError)
        XCTAssertNil(performance.lastError)
        XCTAssertNil(network.lastError)
        server.basePathText = "/custom/base"
        server.samplingTemperatureText = "0.23"
        performance.maxConcurrentText = "42"
        performance.idleTimeoutText = "900"
        network.httpProxy = "http://unsaved-proxy:8080"
        let modelDirs = server.modelDirTexts
        let cacheDir = performance.ssdCacheDir
        let caBundle = network.caBundle
        let loadedConcurrent = performance.loadedMaxConcurrent
        let loadedProxy = network.loadedHttpProxy

        await server.resetDefaults(client: client)
        await performance.resetDefaults(client: client)
        await network.resetDefaults(client: client)

        XCTAssertNil(server.lastError)
        XCTAssertNil(performance.lastError)
        XCTAssertNil(network.lastError)
        XCTAssertTrue(server.showResetNotice)
        XCTAssertTrue(performance.showResetNotice)
        XCTAssertTrue(network.showResetNotice)
        XCTAssertTrue(server.hasPendingDefaults)
        XCTAssertTrue(performance.hasPendingChanges)
        XCTAssertTrue(network.hasPendingChanges)
        XCTAssertEqual(server.host, "127.0.0.1")
        XCTAssertEqual(server.appliedBindAddress, "0.0.0.0")
        XCTAssertEqual(server.basePathText, "/custom/base")
        XCTAssertEqual(server.modelDirTexts, modelDirs)
        XCTAssertEqual(performance.ssdCacheDir, cacheDir)
        XCTAssertEqual(performance.loadedMaxConcurrent, loadedConcurrent)
        XCTAssertEqual(performance.maxConcurrentText, "8")
        XCTAssertEqual(network.caBundle, caBundle)
        XCTAssertEqual(network.loadedHttpProxy, loadedProxy)
        XCTAssertEqual(network.httpProxy, "")

        let logBinding = server.bind(Binding(
            get: { server.logLevel }, set: { server.logLevel = $0 }
        ), save: { XCTFail("Reset drafts must wait for Apply") })
        logBinding.wrappedValue = "debug"
        XCTAssertEqual(server.logLevel, "debug")

        server.cancelReset()
        performance.cancelReset()
        network.cancelReset()
        XCTAssertFalse(server.showResetNotice)
        XCTAssertFalse(performance.showResetNotice)
        XCTAssertFalse(network.showResetNotice)
        XCTAssertFalse(server.hasPendingDefaults)
        XCTAssertEqual(server.host, "0.0.0.0")
        XCTAssertEqual(server.samplingTemperatureText, "0.23")
        XCTAssertEqual(server.logLevel, "info")
        XCTAssertEqual(performance.maxConcurrentText, "42")
        XCTAssertEqual(performance.idleTimeoutText, "900")
        XCTAssertEqual(network.httpProxy, "http://unsaved-proxy:8080")
        XCTAssertEqual(performance.loadedMaxConcurrent, loadedConcurrent)
        XCTAssertEqual(network.loadedHttpProxy, loadedProxy)

        await server.resetDefaults(client: client)
        await performance.resetDefaults(client: client)
        await network.resetDefaults(client: client)
        server.confirmReset()
        performance.confirmReset()
        network.confirmReset()
        server.cancelReset()
        performance.cancelReset()
        network.cancelReset()
        XCTAssertTrue(server.hasPendingDefaults)
        XCTAssertEqual(server.host, "127.0.0.1")
        XCTAssertEqual(performance.maxConcurrentText, "8")
        XCTAssertEqual(network.httpProxy, "")
        server.samplingTemperatureText = "0.45"
        await server.resetDefaults(client: client)
        server.cancelReset()
        XCTAssertTrue(server.hasPendingDefaults)
        XCTAssertEqual(server.samplingTemperatureText, "0.45")
    }

    // MARK: - Decode

    func testSamplingDecodesFromNestedObject() throws {
        // Mirrors `omlx.settings.SamplingSettings.to_dict()` — the read
        // shape is nested under `sampling`, separate from the flat
        // `sampling_*` keys on the patch body.
        let json = """
        {
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "log_level": "info",
                "server_aliases": []
            },
            "sampling": {
                "max_context_window": 32768,
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.05
            }
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(GlobalSettingsDTO.self, from: json)
        XCTAssertEqual(dto.sampling?.maxContextWindow, 32768)
        XCTAssertEqual(dto.sampling?.maxTokens, 4096)
        XCTAssertEqual(dto.sampling?.temperature, 0.7)
        XCTAssertEqual(dto.sampling?.topP, 0.95)
        XCTAssertEqual(dto.sampling?.topK, 20)
        XCTAssertEqual(dto.sampling?.repetitionPenalty, 1.05)
    }

    func testSamplingFieldIsOptional() throws {
        // Older server builds, or a server that hasn't populated sampling
        // yet, omit the key entirely. Decode must succeed with nil.
        let json = """
        {
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "log_level": "info",
                "server_aliases": []
            }
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(GlobalSettingsDTO.self, from: json)
        XCTAssertNil(dto.sampling)
    }

    // MARK: - Patch encode

    func testPatchEncodesEmbeddingBatchSizeAsSnakeCaseFlatKey() throws {
        // Scheduler writes use the flat GlobalSettingsRequest shape, so the
        // Swift camelCase property must encode to embedding_batch_size.
        var patch = GlobalSettingsPatch()
        patch.embeddingBatchSize = 8

        let data = try encoder.encode(patch)
        let str = String(data: data, encoding: .utf8) ?? ""

        XCTAssertTrue(str.contains("\"embedding_batch_size\":8"), "got: \(str)")
    }

    func testPatchEncodesModelDirsAsSnakeCaseFlatKey() throws {
        var patch = GlobalSettingsPatch()
        patch.modelDirs = ["/Users/test/.omlx/models", "/Users/test/.lmstudio/models"]

        let data = try encoder.encode(patch)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["model_dirs"] as? [String], [
            "/Users/test/.omlx/models",
            "/Users/test/.lmstudio/models"
        ])
    }

    func testPatchEncodesHfCacheEnabledAsSnakeCaseFlatKey() throws {
        var patch = GlobalSettingsPatch()
        patch.hfCacheEnabled = false

        let data = try encoder.encode(patch)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["hf_cache_enabled"] as? Bool, false)
    }

    func testPatchEncodesHotCacheMaxSizeAsSnakeCaseFlatKey() throws {
        var patch = GlobalSettingsPatch()
        patch.hotCacheMaxSize = "8GB"

        let data = try encoder.encode(patch)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["hot_cache_max_size"] as? String, "8GB")
    }

    func testPatchDistinguishesIdleTimeoutOmittedNullAndValue() throws {
        let omittedData = try encoder.encode(GlobalSettingsPatch())
        let omitted = try JSONSerialization.jsonObject(
            with: omittedData
        ) as! [String: Any]
        XCTAssertNil(omitted["idle_timeout_seconds"])

        var disabledPatch = GlobalSettingsPatch()
        disabledPatch.idleTimeoutSeconds = .null
        let disabledData = try encoder.encode(disabledPatch)
        let disabled = try JSONSerialization.jsonObject(
            with: disabledData
        ) as! [String: Any]
        XCTAssertTrue(disabled["idle_timeout_seconds"] is NSNull)

        var enabledPatch = GlobalSettingsPatch()
        enabledPatch.idleTimeoutSeconds = .value(120)
        let enabledData = try encoder.encode(enabledPatch)
        let enabled = try JSONSerialization.jsonObject(
            with: enabledData
        ) as! [String: Any]
        XCTAssertEqual(enabled["idle_timeout_seconds"] as? Int, 120)
    }

    func testPatchEncodesSamplingFieldsAsSnakeCaseFlatKeys() throws {
        // The Python `GlobalSettingsRequest` accepts the sampling defaults
        // as flat `sampling_*` keys (omlx/admin/routes.py:229-234), not
        // nested. The .convertToSnakeCase strategy on Swift's encoder must
        // produce exactly that wire shape.
        var patch = GlobalSettingsPatch()
        patch.samplingMaxContextWindow = 32768
        patch.samplingMaxTokens = 4096
        patch.samplingTemperature = 0.5
        patch.samplingTopP = 0.9
        patch.samplingTopK = 40
        patch.samplingRepetitionPenalty = 1.1

        let data = try encoder.encode(patch)
        let str = String(data: data, encoding: .utf8) ?? ""

        XCTAssertTrue(str.contains("\"sampling_max_context_window\":32768"), "got: \(str)")
        XCTAssertTrue(str.contains("\"sampling_max_tokens\":4096"))
        XCTAssertTrue(str.contains("\"sampling_temperature\":0.5"))
        XCTAssertTrue(str.contains("\"sampling_top_p\":0.9"))
        XCTAssertTrue(str.contains("\"sampling_top_k\":40"))
        XCTAssertTrue(str.contains("\"sampling_repetition_penalty\":1.1"))
    }

    func testPatchOmitsNilSamplingFields() throws {
        // `encodeIfPresent` for Optionals means nil fields are skipped —
        // the server's merge semantics treat any present field as an edit.
        // A patch that only touches temperature must not also overwrite
        // top_p / top_k / etc to nil.
        var patch = GlobalSettingsPatch()
        patch.samplingTemperature = 0.42

        let data = try encoder.encode(patch)
        let str = String(data: data, encoding: .utf8) ?? ""

        XCTAssertTrue(str.contains("\"sampling_temperature\":0.42"))
        XCTAssertFalse(str.contains("sampling_max_tokens"))
        XCTAssertFalse(str.contains("sampling_top_p"))
        XCTAssertFalse(str.contains("sampling_top_k"))
        XCTAssertFalse(str.contains("sampling_repetition_penalty"))
        XCTAssertFalse(str.contains("sampling_max_context_window"))
    }

    func testPatchWithNoSamplingFieldsOmitsAllKeys() throws {
        // A purely network-side patch (e.g. updating port) must not carry
        // empty sampling keys, or the server's merge logic would no-op
        // through them but the wire payload bloats.
        var patch = GlobalSettingsPatch()
        patch.port = 9000

        let data = try encoder.encode(patch)
        let str = String(data: data, encoding: .utf8) ?? ""

        XCTAssertTrue(str.contains("\"port\":9000"))
        XCTAssertFalse(str.contains("sampling_"))
    }

    func testUsageDecodesFromNestedObjectAndIsOptional() throws {
        // Mirrors `omlx.settings.UsageSettings.to_dict()` under the `usage`
        // key; servers without the toggle omit the block entirely.
        let json = """
        {
            "server": {"host": "127.0.0.1", "port": 8080, "log_level": "info", "server_aliases": []},
            "usage": {"usage_history": false}
        }
        """.data(using: .utf8)!
        XCTAssertEqual(try decoder.decode(GlobalSettingsDTO.self, from: json).usage?.usageHistory, false)

        let legacy = """
        {
            "server": {"host": "127.0.0.1", "port": 8080, "log_level": "info", "server_aliases": []}
        }
        """.data(using: .utf8)!
        XCTAssertNil(try decoder.decode(GlobalSettingsDTO.self, from: legacy).usage)
    }

    func testPatchEncodesUsageHistoryAsSnakeCaseFlatKey() throws {
        var patch = GlobalSettingsPatch()
        patch.usageHistory = false

        let data = try encoder.encode(patch)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["usage_history"] as? Bool, false)

        let empty = try JSONSerialization.jsonObject(
            with: try encoder.encode(GlobalSettingsPatch())
        ) as! [String: Any]
        XCTAssertNil(empty["usage_history"])
    }

    func testServerDecodesAudioUploadSize() throws {
        let json = """
        {
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "log_level": "info",
                "server_aliases": [],
                "max_audio_upload_size": "500MB"
            }
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(GlobalSettingsDTO.self, from: json)
        XCTAssertEqual(dto.server.maxAudioUploadSize, "500MB")
    }

    func testPatchEncodesAudioUploadSizeAsFlatSnakeCaseKey() throws {
        var patch = GlobalSettingsPatch()
        patch.maxAudioUploadSize = "1GB"

        let data = try encoder.encode(patch)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        XCTAssertEqual(json["max_audio_upload_size"] as? String, "1GB")
    }
}


private final class SettingsDefaultsURLProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "127.0.0.1" && request.url?.port == 18159
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        do {
            XCTAssertEqual(request.httpMethod, "GET", "Reset must not save settings")
            let path = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .appendingPathComponent("Fixtures/global-settings.json")
            var json = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
            if request.url?.path == "/admin/api/global-settings" {
                var server = json["server"] as! [String: Any]
                server["host"] = "0.0.0.0"
                json["server"] = server
                var scheduler = json["scheduler"] as! [String: Any]
                scheduler["max_concurrent_requests"] = 64
                json["scheduler"] = scheduler
                json["network"] = ["http_proxy": "http://proxy:8080", "https_proxy": "",
                                   "no_proxy": "localhost", "ca_bundle": "/custom/ca.pem"]
            }
            let response = HTTPURLResponse(url: request.url!, statusCode: 200,
                                           httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: try JSONSerialization.data(withJSONObject: json))
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
