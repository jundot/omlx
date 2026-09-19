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

    func testModelLibraryRowsExposeProfileWithoutDuplicatingBaseSize() throws {
        let profile = ProfileDTO(
            name: "thinking",
            displayName: "Thinking",
            description: nil,
            createdAt: nil,
            updatedAt: nil,
            sourceTemplate: nil,
            isBuiltin: false,
            exposeAsModel: true,
            modelId: "qwen:thinking",
            hasEngineFields: false,
            settings: nil
        )
        let base = makeModel(
            "qwen",
            removalKind: "local_cache",
            exposedProfiles: [profile]
        )

        let rows = modelLibraryRows([base])

        XCTAssertEqual(Set(rows.map(\.id)), ["qwen", "qwen:thinking"])
        let profileRow = try XCTUnwrap(rows.first { $0.id == "qwen:thinking" })
        XCTAssertEqual(profileRow.removalKind, "profile")
        XCTAssertEqual(profileRow.sourceModelId, "qwen")
        XCTAssertEqual(profileRow.profileName, "thinking")
        XCTAssertEqual(profileRow.estimatedSize, 0)
        XCTAssertEqual(profileRow.deletable, true)
    }

    private func makeModel(
        _ id: String,
        displayName: String? = nil,
        removalKind: String? = nil,
        exposedProfiles: [ProfileDTO]? = nil
    ) -> ModelDTO {
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
            sourceType: nil,
            sourceRepoId: nil,
            deletable: nil,
            removalKind: removalKind,
            sourceModelId: nil,
            profileName: nil,
            exposedProfiles: exposedProfiles,
            settings: nil
        )
    }
}
