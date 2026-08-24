import XCTest
@testable import oMLX

final class MenubarControllerModelsTests: XCTestCase {

    // MARK: - toggleState

    func testToggleStateUnloadingTakesPrecedenceOverLoaded() {
        let model = makeModel("a", loaded: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: ["a"], loading: []),
            .unloading
        )
    }

    func testToggleStateUnloadingTakesPrecedenceOverLoading() {
        let model = makeModel("a", isLoading: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: ["a"], loading: ["a"]),
            .unloading
        )
    }

    func testToggleStateLoadingWhenModelIsLoading() {
        let model = makeModel("a", isLoading: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: [], loading: []),
            .loading
        )
    }

    func testToggleStateLoadingWhenLoadIsPendingLocally() {
        let model = makeModel("a")
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: [], loading: ["a"]),
            .loading
        )
    }

    func testToggleStateUnloadWhenLoaded() {
        let model = makeModel("a", loaded: true)
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: [], loading: []),
            .unload
        )
    }

    func testToggleStateLoadWhenIdle() {
        let model = makeModel("a")
        XCTAssertEqual(
            MenubarController.toggleState(for: model, unloading: [], loading: []),
            .load
        )
    }

    // MARK: - reconcileUnloading

    func testReconcileUnloadingKeepsOnlyStillLoadedIDs() {
        let models = [makeModel("a", loaded: true), makeModel("b", loaded: false)]
        XCTAssertEqual(
            MenubarController.reconcileUnloading(["a", "b"], against: models),
            ["a"]
        )
    }

    func testReconcileUnloadingDropsUnknownIDs() {
        let models = [makeModel("a", loaded: true)]
        XCTAssertEqual(
            MenubarController.reconcileUnloading(["x"], against: models),
            []
        )
    }

    func testReconcileUnloadingEmptyStaysEmpty() {
        let models = [makeModel("a", loaded: true)]
        XCTAssertEqual(
            MenubarController.reconcileUnloading([], against: models),
            []
        )
    }

    // MARK: - reconcileLoading

    func testReconcileLoadingKeepsUnacknowledgedIDs() {
        let models = [makeModel("a")]
        XCTAssertEqual(
            MenubarController.reconcileLoading(["a"], against: models),
            ["a"]
        )
    }

    func testReconcileLoadingDropsOnceServerReportsLoading() {
        let models = [makeModel("a", isLoading: true)]
        XCTAssertEqual(
            MenubarController.reconcileLoading(["a"], against: models),
            []
        )
    }

    func testReconcileLoadingDropsOnceLoaded() {
        let models = [makeModel("a", loaded: true)]
        XCTAssertEqual(
            MenubarController.reconcileLoading(["a"], against: models),
            []
        )
    }

    func testReconcileLoadingDropsUnknownIDs() {
        XCTAssertEqual(
            MenubarController.reconcileLoading(["x"], against: [makeModel("a")]),
            []
        )
    }

    // MARK: - visibleMenuModels

    func testVisibleMenuModelsDropsVirtualEntries() {
        let models = [
            makeModel("a"),
            makeModel("markitdown", loaded: true, virtual: true),
        ]
        XCTAssertEqual(
            MenubarController.visibleMenuModels(models).map(\.id),
            ["a"]
        )
    }

    // MARK: - partitionForMenu

    func testPartitionForMenuSplitsLoadedFavoritesAndLibrary() {
        let models = [
            makeModel("c"),
            makeModel("b", loaded: true),
            makeModel("a", isFavorite: true),
            makeModel("d", isLoading: true),
        ]
        let (loaded, favorites, library) = MenubarController.partitionForMenu(models)
        XCTAssertEqual(loaded.map(\.id), ["b", "d"])
        XCTAssertEqual(favorites.map(\.id), ["a"])
        XCTAssertEqual(library.map(\.id), ["c"])
    }

    func testPartitionForMenuKeepsLoadedFavoriteOnlyInLoaded() {
        let models = [makeModel("a", loaded: true, isFavorite: true)]
        let (loaded, favorites, library) = MenubarController.partitionForMenu(models)
        XCTAssertEqual(loaded.map(\.id), ["a"])
        XCTAssertTrue(favorites.isEmpty)
        XCTAssertTrue(library.isEmpty)
    }

    // MARK: - modelMenuTitle

    func testModelMenuTitlePrefersDisplayName() {
        XCTAssertEqual(
            MenubarController.modelMenuTitle(for: makeModel("org/model", displayName: "Model")),
            "Model"
        )
    }

    func testModelMenuTitleFallsBackToID() {
        XCTAssertEqual(
            MenubarController.modelMenuTitle(for: makeModel("org/model", loaded: true)),
            "org/model"
        )
    }

    // MARK: - ModelDTO.sizeLabel

    func testSizeLabelWhileLoadingUsesEstimated() {
        let model = makeModel("a", isLoading: true,
                              estimatedSizeFormatted: "10 GB", actualSizeFormatted: "9 GB")
        XCTAssertEqual(model.sizeLabel, "10 GB")
    }

    func testSizeLabelLoadedPrefersActual() {
        let model = makeModel("a", loaded: true,
                              estimatedSizeFormatted: "10 GB", actualSizeFormatted: "9 GB")
        XCTAssertEqual(model.sizeLabel, "9 GB")
    }

    func testSizeLabelLoadedWithoutActualUsesEstimated() {
        let model = makeModel("a", loaded: true, estimatedSizeFormatted: "10 GB")
        XCTAssertEqual(model.sizeLabel, "10 GB")
    }

    func testSizeLabelEmptyWhenNoSizes() {
        XCTAssertEqual(makeModel("a", loaded: true).sizeLabel, "")
    }

    // MARK: - Fixed KV launch preflight

    func testGenerativeEnginesUseFixedKVLaunchPreflight() {
        XCTAssertTrue(makeModel("llm", engineType: "batched").usesFixedKVLaunchPreflight)
        XCTAssertTrue(makeModel("vlm", engineType: "vlm").usesFixedKVLaunchPreflight)
    }

    func testNonTextAndDiffusionEnginesKeepDirectLoadPath() {
        for engineType in ["embedding", "reranker", "audio_stt", "audio_tts"] {
            XCTAssertFalse(
                makeModel(engineType, engineType: engineType).usesFixedKVLaunchPreflight
            )
        }
        XCTAssertFalse(
            makeModel(
                "diffusion",
                engineType: "vlm",
                configModelType: "diffusion_gemma"
            ).usesFixedKVLaunchPreflight
        )
        XCTAssertFalse(makeModel("legacy").usesFixedKVLaunchPreflight)
    }

    // MARK: - Helpers

    private func makeModel(
        _ id: String,
        displayName: String? = nil,
        loaded: Bool = false,
        isLoading: Bool = false,
        isFavorite: Bool = false,
        virtual: Bool = false,
        estimatedSizeFormatted: String? = nil,
        actualSizeFormatted: String? = nil,
        engineType: String? = nil,
        configModelType: String? = nil
    ) -> ModelDTO {
        ModelDTO(
            id: id,
            displayName: displayName,
            modelPath: nil,
            loaded: loaded,
            isLoading: isLoading,
            estimatedSize: 0,
            estimatedSizeFormatted: estimatedSizeFormatted,
            actualSize: nil,
            actualSizeFormatted: actualSizeFormatted,
            pinned: nil,
            isDefault: nil,
            isFavorite: isFavorite,
            engineType: engineType,
            modelType: nil,
            configModelType: configModelType,
            modelContextLength: nil,
            thinkingDefault: nil,
            dflashCompatible: nil,
            dflashCompatibilityReason: nil,
            dflashSsdCacheAvailable: nil,
            mtpCompatible: nil,
            mtpCompatibilityReason: nil,
            virtual: virtual,
            settings: nil,
            memory: nil
        )
    }
}
