import XCTest
@testable import oMLX

@MainActor
final class ModelSettingsScreenVMTests: XCTestCase {

    func testVlmMtpDraftModelOptionsIncludeQwenMtpConfigType() {
        let vm = ModelSettingsScreenVM()
        vm.modelID = "Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-Target"
        vm.allModels = [
            makeModel(
                id: "Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-Target",
                configModelType: "qwen3_5_moe"
            ),
            makeModel(
                id: "Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-MTP-Drafter",
                configModelType: "qwen3_5_mtp"
            ),
            makeModel(id: "Qwen3.6-Regular-Model", configModelType: "qwen3_5_moe"),
        ]

        let values = vm.vlmMtpDraftModelOptions().map(\.0)

        XCTAssertTrue(values.contains("Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-MTP-Drafter"))
        XCTAssertFalse(values.contains("Qwopus3.6-35B-A3B-v1-4bit-MLXVLM-Target"))
        XCTAssertFalse(values.contains("Qwen3.6-Regular-Model"))
    }

    func testVlmMtpDraftModelOptionsKeepAssistantAndStandaloneMtpFallbacks() {
        let vm = ModelSettingsScreenVM()
        vm.modelID = "target"
        vm.allModels = [
            makeModel(id: "gemma-assistant-draft", configModelType: nil),
            makeModel(id: "model-MTP-draft", configModelType: nil),
            makeModel(id: "model-MTPLX-runtime", configModelType: nil),
        ]

        let values = vm.vlmMtpDraftModelOptions().map(\.0)

        XCTAssertTrue(values.contains("gemma-assistant-draft"))
        XCTAssertTrue(values.contains("model-MTP-draft"))
        XCTAssertFalse(values.contains("model-MTPLX-runtime"))
    }

    private func makeModel(id: String, configModelType: String?) -> ModelDTO {
        ModelDTO(
            id: id,
            displayName: nil,
            modelPath: nil,
            loaded: false,
            isLoading: false,
            estimatedSize: 0,
            estimatedSizeFormatted: nil,
            pinned: nil,
            isDefault: nil,
            engineType: nil,
            modelType: nil,
            configModelType: configModelType,
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
