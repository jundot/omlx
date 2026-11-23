import XCTest
@testable import oMLX

final class ModelsScreenSortingTests: XCTestCase {

    func testSortModelsByNameIgnoresCase() throws {
        let models = try [
            makeModel("Qwen"),
            makeModel("gpt"),
            makeModel("Llama"),
            makeModel("mistral"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["gpt", "Llama", "mistral", "Qwen"])
    }

    func testSortModelsByNamePreservesInputOrderForCaseOnlyTies() throws {
        let models = try [
            makeModel("qwen"),
            makeModel("Qwen"),
            makeModel("QWEN"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["qwen", "Qwen", "QWEN"])
    }

    func testSortModelsByNameUsesDisplayNameWhenPresent() throws {
        let models = try [
            makeModel("llama", displayName: "Meta/llama"),
            makeModel("Qwen", displayName: "deepsweet/Qwen"),
            makeModel("gemma", displayName: "Google/gemma"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["Qwen", "gemma", "llama"])
    }

    func testAliasIsDisplayedAndCopiedWithoutChangingIdentity() throws {
        let id = "Qwen3.8-27B-oQ6e-mtp"
        let model = try makeModel(
            id, displayName: "Jundot/\(id)", settings: ["model_alias": "test"]
        )

        XCTAssertEqual(model.displayAlias, "test")
        XCTAssertEqual(model.apiModelName, "test")
        XCTAssertEqual(model.id, id)
        XCTAssertEqual(model.displayTitle, "Jundot/\(id)")
    }

    func testAbsentAliasesHideBadgeAndCopyModelID() throws {
        let settingsCases: [Any?] = [
            nil, NSNull(), [String: String](),
            ["model_alias": NSNull()], ["model_alias": ""],
        ]
        for settings in settingsCases {
            let model = try makeModel("model-id", displayName: "Display name", settings: settings)

            XCTAssertNil(model.displayAlias)
            XCTAssertEqual(model.apiModelName, "model-id")
            XCTAssertEqual(model.displayTitle, "Display name")
        }
    }

    func testSortModelsByNameDoesNotUseAliases() throws {
        let models = try [
            makeModel("first", displayName: "Alpha", settings: ["model_alias": "zulu"]),
            makeModel("second", displayName: "Zulu", settings: ["model_alias": "alpha"]),
        ]

        XCTAssertEqual(sortModelsByName(Array(models.reversed())).map(\.id), ["first", "second"])
    }

    private func makeModel(
        _ id: String, displayName: String? = nil, settings: Any? = nil
    ) throws -> ModelDTO {
        var json: [String: Any] = [
            "id": id, "loaded": false, "is_loading": false, "estimated_size": 0,
        ]
        json["display_name"] = displayName
        json["settings"] = settings
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ModelDTO.self, from: JSONSerialization.data(withJSONObject: json))
    }
}
