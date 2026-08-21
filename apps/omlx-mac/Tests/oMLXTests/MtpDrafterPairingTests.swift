import XCTest
@testable import oMLX

final class MtpDrafterPairingTests: XCTestCase {

    /// The screenshot scenario: an MTP checkpoint named after its chat
    /// model resolves by stripping the `-MTP` token from the model id.
    func testResolvesTargetByStrippingMtpToken() {
        let drafter = makeDrafter(id: "mlx-community--Qwen3.8-27B-MTP-8bit",
                                  helperKind: "mtp")
        let chat = makeModel(id: "mlx-community--Qwen3.8-27B-8bit")

        let target = MtpDrafterPairing.resolveTarget(drafter: drafter, in: [drafter, chat])

        XCTAssertEqual(target?.id, chat.id)
    }

    func testResolvesTargetByStrippingAssistantToken() {
        let drafter = makeDrafter(id: "gemma-4-26B-A4B-it-assistant",
                                  helperKind: "assistant")
        let chat = makeModel(id: "gemma-4-26B-A4B-it")

        let target = MtpDrafterPairing.resolveTarget(drafter: drafter, in: [drafter, chat])

        XCTAssertEqual(target?.id, chat.id)
    }

    /// A chat model that already references the drafter in its settings
    /// wins over the name heuristic.
    func testSettingsReferenceTakesPriorityOverNameMatch() {
        let drafter = makeDrafter(id: "org--Foo-MTP-8bit", helperKind: "mtp")
        let nameMatch = makeModel(id: "org--Foo-8bit")
        let referenced = makeModel(id: "org--Bar-8bit",
                                   vlmMtpDraftModel: "org--Foo-MTP-8bit")

        let target = MtpDrafterPairing.resolveTarget(
            drafter: drafter, in: [drafter, nameMatch, referenced])

        XCTAssertEqual(target?.id, referenced.id)
    }

    /// Settings references match on the Hub repo id too, not just the
    /// local model id.
    func testSettingsReferenceMatchesSourceRepoId() {
        let drafter = makeDrafter(id: "mlx-community--Qwen3.8-27B-MTP-8bit",
                                  helperKind: "mtp",
                                  sourceRepoId: "mlx-community/Qwen3.8-27B-MTP-8bit")
        let referenced = makeModel(
            id: "local-qwen",
            vlmMtpDraftModel: "mlx-community/Qwen3.8-27B-MTP-8bit")

        let target = MtpDrafterPairing.resolveTarget(drafter: drafter, in: [drafter, referenced])

        XCTAssertEqual(target?.id, referenced.id)
    }

    /// When the local id doesn't name-match, the drafter's repo id is
    /// stripped and compared against other models' repo ids.
    func testFallsBackToSourceRepoIdHeuristic() {
        let drafter = makeDrafter(id: "qwen-mtp-local-copy",
                                  helperKind: "mtp",
                                  sourceRepoId: "mlx-community/Qwen3.8-27B-MTP-8bit")
        let chat = makeModel(id: "qwen-local",
                             sourceRepoId: "mlx-community/Qwen3.8-27B-8bit")

        let target = MtpDrafterPairing.resolveTarget(drafter: drafter, in: [drafter, chat])

        XCTAssertEqual(target?.id, chat.id)
    }

    func testReturnsNilWhenNoCandidateExists() {
        let drafter = makeDrafter(id: "mlx-community--Qwen3.8-27B-MTP-8bit",
                                  helperKind: "mtp")
        let unrelated = makeModel(id: "mlx-community--Llama-3.2-3B-4bit")

        XCTAssertNil(MtpDrafterPairing.resolveTarget(drafter: drafter,
                                                     in: [drafter, unrelated]))
    }

    /// A helper checkpoint is never a valid attach target, even when its
    /// id happens to match the stripped drafter name.
    func testHelperModelsAreNotEligibleTargets() {
        let drafter = makeDrafter(id: "org--Foo-MTP", helperKind: "mtp")
        let helperWithMatchingName = makeDrafter(id: "org--Foo", helperKind: "dflash")

        XCTAssertNil(MtpDrafterPairing.resolveTarget(
            drafter: drafter, in: [drafter, helperWithMatchingName]))
    }

    /// The server payload's helper fields decode through the snake_case
    /// decoder strategy used by OMLXClient.
    func testDecodesHelperFieldsFromServerPayload() throws {
        let json = """
        {
            "id": "mlx-community--Qwen3.8-27B-MTP-8bit",
            "loaded": false,
            "is_loading": false,
            "estimated_size": 451900000,
            "is_helper": true,
            "helper_kind": "mtp",
            "source_repo_id": "mlx-community/Qwen3.8-27B-MTP-8bit"
        }
        """.data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let model = try decoder.decode(ModelDTO.self, from: json)

        XCTAssertEqual(model.isHelper, true)
        XCTAssertEqual(model.helperKind, "mtp")
        XCTAssertEqual(model.sourceRepoId, "mlx-community/Qwen3.8-27B-MTP-8bit")
        XCTAssertTrue(model.isAttachableDrafter)
    }

    func testChatModelIsNotAnAttachableDrafter() {
        XCTAssertFalse(makeModel(id: "org--Foo-8bit").isAttachableDrafter)
    }

    // MARK: - Fixtures

    private func makeDrafter(id: String,
                             helperKind: String,
                             sourceRepoId: String? = nil) -> ModelDTO {
        makeModel(id: id, isHelper: true, helperKind: helperKind,
                  sourceRepoId: sourceRepoId)
    }

    private func makeModel(id: String,
                           isHelper: Bool? = nil,
                           helperKind: String? = nil,
                           sourceRepoId: String? = nil,
                           vlmMtpDraftModel: String? = nil) -> ModelDTO {
        ModelDTO(
            id: id,
            displayName: nil,
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
            isHelper: isHelper,
            helperKind: helperKind,
            sourceRepoId: sourceRepoId,
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
            virtual: nil,
            settings: vlmMtpDraftModel.map(makeSettings(vlmMtpDraftModel:))
        )
    }

    /// `ModelSettingsDTO` has ~60 optional fields; decoding a one-key
    /// JSON payload is far more compact than the memberwise initializer.
    private func makeSettings(vlmMtpDraftModel: String) -> ModelSettingsDTO {
        let json = "{ \"vlm_mtp_draft_model\": \"\(vlmMtpDraftModel)\" }".data(using: .utf8)!
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try! decoder.decode(ModelSettingsDTO.self, from: json)
    }
}
