import XCTest
@testable import oMLX

@MainActor
final class ModelSettingsScreenVMTests: XCTestCase {

    func testModelTypeOptionsMatchServerValues() {
        let values = ModelSettingsScreenVM.modelTypeOptions.map(\.0)

        XCTAssertEqual(
            values,
            [
                "",
                "llm",
                "vlm",
                "embedding",
                "reranker",
                "audio_stt",
                "audio_tts",
                "audio_sts",
            ]
        )
    }

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

    func testQwenAneControlsUseMeasuredDefaults() {
        let vm = ModelSettingsScreenVM()

        XCTAssertFalse(vm.qwen35AnePrefillEnabled)
        XCTAssertEqual(vm.qwen35AnePrefillSequenceLength, "2048")
        XCTAssertEqual(vm.qwen35AnePrefillFraction, "0.53")
        XCTAssertEqual(vm.qwen35AnePrefillMaxLayers, "64")
        XCTAssertTrue(vm.qwen35AnePrefillDualAne)
        XCTAssertTrue(vm.qwen35AnePrefillGdn)
        XCTAssertEqual(vm.qwen35AnePrefillGdnFraction, "0.5")
        XCTAssertEqual(vm.qwen35AnePrefillGdnMaxLayers, "48")
    }

    func testQwenAneFractionFormatterPreservesSettingsValues() {
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.5), "0.5")
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.53), "0.53")
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.527), "0.527")
    }

    func testQwenAneArbitraryInputValidation() {
        XCTAssertEqual(try? QwenAneSettingsValidator.promptBlock("2112").get(), 2112)
        XCTAssertThrowsError(try QwenAneSettingsValidator.promptBlock("2100").get())
        XCTAssertEqual(try? QwenAneSettingsValidator.mlpFraction("0.467").get(), 0.467)
        XCTAssertThrowsError(try QwenAneSettingsValidator.mlpFraction("0.91").get())
        XCTAssertEqual(try? QwenAneSettingsValidator.gdnFraction("0.527").get(), 0.527)
    }

    func testQwenAneSettingsAreIncludedInWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.qwen35AnePrefillEnabled = true

        let settings = vm.currentSettingsDict()

        XCTAssertEqual(settings["qwen35_ane_prefill_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_sequence_length"]?.value as? Int, 2048)
        XCTAssertEqual(settings["qwen35_ane_prefill_fraction"]?.value as? Double, 0.53)
        XCTAssertEqual(settings["qwen35_ane_prefill_max_layers"]?.value as? Int, 64)
        XCTAssertEqual(settings["qwen35_ane_prefill_dual_ane"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn_fraction"]?.value as? Double, 0.5)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn_max_layers"]?.value as? Int, 48)
    }

    func testQwenAneCompatibilityUsesQwenConfigFamily() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "qwen", configModelType: "qwen3_5_moe")
        XCTAssertTrue(vm.isQwen35AnePrefillModel)

        vm.model = makeModel(id: "qwen", configModelType: "qwen3-6")
        XCTAssertTrue(vm.isQwen35AnePrefillModel)

        vm.model = makeModel(id: "qwen", configModelType: "qwen3_8")
        XCTAssertTrue(vm.isQwen35AnePrefillModel)

        vm.model = makeModel(id: "other", configModelType: "gemma4")
        XCTAssertFalse(vm.isQwen35AnePrefillModel)
    }

    func testQwenAneSettingsDecodeFromServerAndEncodeForPatch() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let json = #"""
        {
            "qwen35_ane_prefill_enabled": true,
            "qwen35_ane_prefill_sequence_length": 2048,
            "qwen35_ane_prefill_fraction": 0.53,
            "qwen35_ane_prefill_max_layers": 64,
            "qwen35_ane_prefill_dual_ane": true,
            "qwen35_ane_prefill_gdn": true,
            "qwen35_ane_prefill_gdn_fraction": 0.5,
            "qwen35_ane_prefill_gdn_max_layers": 48
        }
        """#
        let dto = try decoder.decode(ModelSettingsDTO.self, from: Data(json.utf8))
        XCTAssertEqual(dto.qwen35AnePrefillFraction, 0.53)
        XCTAssertEqual(dto.qwen35AnePrefillGdnFraction, 0.5)

        var patch = ModelSettingsPatch()
        patch.qwen35AnePrefillEnabled = true
        patch.qwen35AnePrefillFraction = 0.53
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(patch)) as? [String: Any]
        XCTAssertEqual(object?["qwen35_ane_prefill_enabled"] as? Bool, true)
        XCTAssertEqual(object?["qwen35_ane_prefill_fraction"] as? Double, 0.53)
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
            actualSize: nil,
            actualSizeFormatted: nil,
            pinned: nil,
            isDefault: nil,
            isFavorite: nil,
            engineType: nil,
            modelType: nil,
            configModelType: configModelType,
            modelContextLength: nil,
            thinkingDefault: nil,
            dflashCompatible: nil,
            dflashCompatibilityReason: nil,
            dflashSsdCacheAvailable: nil,
            mtpCompatible: nil,
            mtpCompatibilityReason: nil,
            virtual: nil,
            settings: nil
        )
    }
}
