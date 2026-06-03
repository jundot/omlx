// The server-side default profile lives in `GlobalSettings.sampling`.
// These tests pin the decode shape of the nested `sampling` object on the
// read side and the flat `sampling_*` keys on the patch side, so a future
// rename on either edge breaks the build instead of silently dropping the
// server defaults the Profiles tab and Server screen depend on.

import XCTest
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
}
