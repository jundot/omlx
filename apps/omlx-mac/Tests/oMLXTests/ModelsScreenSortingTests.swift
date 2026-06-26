import XCTest
@testable import oMLX

final class ModelsScreenSortingTests: XCTestCase {

    func testSortModelsByNameIgnoresCase() {
        let models = [
            makeModel("Qwen"),
            makeModel("gpt"),
            makeModel("Llama"),
            makeModel("mistral"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["gpt", "Llama", "mistral", "Qwen"])
    }

    func testSortModelsByNamePreservesInputOrderForCaseOnlyTies() {
        let models = [
            makeModel("qwen"),
            makeModel("Qwen"),
            makeModel("QWEN"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["qwen", "Qwen", "QWEN"])
    }

    func testSortModelsByNameUsesDisplayNameWhenPresent() {
        let models = [
            makeModel("llama", displayName: "Meta/llama"),
            makeModel("Qwen", displayName: "deepsweet/Qwen"),
            makeModel("gemma", displayName: "Google/gemma"),
        ]

        let ids = sortModelsByName(models).map(\.id)

        XCTAssertEqual(ids, ["Qwen", "gemma", "llama"])
    }

    private func makeModel(_ id: String, displayName: String? = nil) -> ModelDTO {
        ModelDTO(
            id: id,
            displayName: displayName,
            modelPath: nil,
            loaded: false,
            isLoading: false,
            estimatedSize: 0,
            estimatedSizeFormatted: nil,
            pinned: nil,
            isDefault: nil,
            engineType: nil,
            modelType: nil,
            configModelType: nil,
            thinkingDefault: nil,
            dflashCompatible: nil,
            dflashCompatibilityReason: nil,
            dflashSsdCacheAvailable: nil,
            mtpCompatible: nil,
            mtpCompatibilityReason: nil,
            settings: nil
        )
    }
}
