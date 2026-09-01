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
            actualSize: nil,
            actualSizeFormatted: nil,
            pinned: nil,
            isDefault: nil,
            isFavorite: nil,
            engineType: nil,
            modelType: nil,
            configModelType: nil,
            modelContextLength: nil,
            thinkingDefault: nil,
            dflashCompatible: nil,
            dflashCompatibilityReason: nil,
            dflashSsdCacheAvailable: nil,
            mtpCompatible: nil,
            mtpCompatibilityReason: nil,
            qwen4PleSsdOffloadSupported: nil,
            qwen4PleSsdOffloadForced: nil,
            qwen4PleResidentBytes: nil,
            qwen4PleMmapBytes: nil,
            virtual: nil,
            settings: nil
        )
    }
}

final class DownloadsScreenSortingTests: XCTestCase {

    func testHuggingFaceSortOptionsMatchHubOrderAndQueryValues() {
        XCTAssertEqual(
            SuggestedSort.allCases,
            [.trending, .likes, .downloads, .created, .updated,
             .mostParams, .leastParams, .size]
        )
        XCTAssertEqual(
            SuggestedSort.allCases.map(\.apiValue),
            ["trending", "likes", "downloads", "created", "updated",
             "most_params", "least_params", "largest"]
        )
    }

    func testModelScopeKeepsOnlyItsSupportedSortOptions() {
        XCTAssertEqual(
            SuggestedSort.modelScopeCases,
            [.trending, .downloads, .likes]
        )
        XCTAssertEqual(
            SuggestedSort.modelScopeCases.map(\.modelScopeAPIValue),
            ["trending", "downloads", "likes"]
        )
    }

    @MainActor
    func testDownloadsDefaultsToTrendingAndResetsUnsupportedModelScopeSort() {
        let viewModel = DownloadsScreenVM()

        XCTAssertEqual(viewModel.recommendedSort, .trending)
        XCTAssertEqual(viewModel.suggestedSortOptions, SuggestedSort.allCases)

        viewModel.source = .ms

        XCTAssertEqual(viewModel.recommendedSort, .trending)
        XCTAssertEqual(viewModel.suggestedSortOptions, SuggestedSort.modelScopeCases)
    }

    @MainActor
    func testModelScopeFilterCountAndClear() {
        let viewModel = DownloadsScreenVM()
        viewModel.source = .ms
        viewModel.msExperienceFilters = [.apiInference, .modelDemo]
        viewModel.msSelectedTask = MSTaskOption(
            value: "text-generation",
            label: "Text Generation"
        )

        XCTAssertEqual(viewModel.activeSuggestedFilterCount, 3)

        viewModel.clearSuggestedFilters()

        XCTAssertTrue(viewModel.msExperienceFilters.isEmpty)
        XCTAssertNil(viewModel.msSelectedTask)
        XCTAssertEqual(viewModel.activeSuggestedFilterCount, 0)
        XCTAssertTrue(viewModel.msMLXOnly)
    }
}
