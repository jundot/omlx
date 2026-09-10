import XCTest
import SwiftUI
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

    func testLightningMtpAllowsTurboQuantInWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.mtpEnabled = true
        vm.turboquantKvEnabled = true

        XCTAssertNil(vm.mtpConflictReason)

        let settings = vm.currentSettingsDict()
        XCTAssertEqual(settings["mtp_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["turboquant_kv_enabled"]?.value as? Bool, true)
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
        XCTAssertFalse(vm.qwen35AnePrefillCpuEnabled)
        XCTAssertEqual(vm.qwen35AnePrefillCpuFraction, "0.135")
        XCTAssertEqual(vm.qwen35AnePrefillCpuDownFraction, "0")
        XCTAssertEqual(vm.qwen35AnePrefillCpuThreads, "8")
        XCTAssertTrue(vm.qwen35AnePrefillCpuSharedResource)
    }

    func testQwenAneFractionFormatterPreservesSettingsValues() {
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.5), "0.5")
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.53), "0.53")
        XCTAssertEqual(ModelSettingsScreenVM.formatPct(0.527), "0.527")
    }

    func testQwenAneArbitraryInputValidation() {
        XCTAssertEqual(try? QwenAneSettingsValidator.promptBlock("2112").get(), 2112)
        XCTAssertThrowsError(try QwenAneSettingsValidator.promptBlock("2100").get())
        XCTAssertEqual(
            try? QwenAneSettingsValidator.tailPadding("1357", sequenceLength: "2048").get(),
            1357
        )
        XCTAssertThrowsError(
            try QwenAneSettingsValidator.tailPadding("2048", sequenceLength: "2048").get()
        )
        XCTAssertEqual(
            try? QwenAneSettingsValidator.mlpFraction("0.467", cpuFraction: "0.137").get(),
            0.467
        )
        XCTAssertThrowsError(
            try QwenAneSettingsValidator.mlpFraction("0.9", cpuFraction: "0.1").get()
        )
        XCTAssertEqual(
            try? QwenAneSettingsValidator.cpuFraction("0.137", mlpFraction: "0.467").get(),
            0.137
        )
        XCTAssertThrowsError(try QwenAneSettingsValidator.cpuThreads("8.5").get())
        XCTAssertThrowsError(try QwenAneSettingsValidator.cpuThreads("65").get())
        XCTAssertEqual(try? QwenAneSettingsValidator.gdnFraction("0.527").get(), 0.527)
        XCTAssertEqual(
            try? QwenAneSettingsValidator.cpuGdnFraction("0.047", gdnFraction: "0.527").get(),
            0.047
        )
        XCTAssertThrowsError(
            try QwenAneSettingsValidator.cpuGdnFraction("0.5", gdnFraction: "0.5").get()
        )
    }

    func testQwenAneSettingsAreIncludedInWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "qwen", configModelType: "qwen3_5")
        vm.qwen35AnePrefillEnabled = true
        vm.qwen35AnePrefillCpuEnabled = true

        let settings = vm.currentSettingsDict()

        XCTAssertEqual(settings["qwen35_ane_prefill_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_sequence_length"]?.value as? Int, 2048)
        XCTAssertEqual(settings["qwen35_ane_prefill_fraction"]?.value as? Double, 0.53)
        XCTAssertEqual(settings["qwen35_ane_prefill_max_layers"]?.value as? Int, 64)
        XCTAssertEqual(settings["qwen35_ane_prefill_dual_ane"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn_fraction"]?.value as? Double, 0.5)
        XCTAssertEqual(settings["qwen35_ane_prefill_gdn_max_layers"]?.value as? Int, 48)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_fraction"]?.value as? Double, 0.135)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_down_fraction"]?.value as? Double, 0.0)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_gdn_fraction"]?.value as? Double, 0.0)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_threads"]?.value as? Int, 8)
        XCTAssertEqual(settings["qwen35_ane_prefill_cpu_shared_resource"]?.value as? Bool, true)
    }

    func testDisabledQwenAneSettingIsIncludedInWorkingProfile() {
        let settings = ModelSettingsScreenVM().currentSettingsDict()

        XCTAssertEqual(settings["qwen35_ane_prefill_enabled"]?.value as? Bool, false)
        XCTAssertNil(settings["qwen35_ane_prefill_sequence_length"])
    }

    func testQwenAneProfileBindingCreatesWorkingState() {
        let vm = ModelSettingsScreenVM()
        let binding = vm.bindProfile(Binding(
            get: { vm.qwen35AnePrefillEnabled },
            set: { vm.qwen35AnePrefillEnabled = $0 }
        ))

        binding.wrappedValue = true

        XCTAssertTrue(vm.qwen35AnePrefillEnabled)
        XCTAssertTrue(vm.profileDirty)
    }

    func testApplyingQwenAneTunerResultStagesWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.aneTuningStatus = ANETuningStatusResponse(
            tuningId: "tune-1",
            modelId: "qwen",
            status: "complete",
            phase: "complete",
            message: "Done",
            current: 1,
            total: 1,
            results: [],
            recommendation: ANETuningRecommendationDTO(
                enabled: true,
                mlpFraction: 0.467,
                gdnEnabled: true,
                gdnFraction: 0.527,
                cpuEnabled: nil,
                cpuFraction: nil,
                cpuDownFraction: nil,
                cpuGdnFraction: nil,
                fusedDown: true,
                cpuThreads: nil,
                cpuSharedResource: nil,
                processingTps: 123.4,
                speedupPercent: 12.3,
                sequenceLength: 2112,
                tailPaddingMinTokens: 1400
            ),
            error: nil,
            terminationReason: nil
        )

        // The tuner ran in whatever ANE mode the model had; applying its
        // result must not flip dual_ane and invalidate the measurement.
        vm.qwen35AnePrefillDualAne = false

        vm.applyANETuningRecommendation()

        XCTAssertTrue(vm.profileDirty)
        XCTAssertTrue(vm.qwen35AnePrefillEnabled)
        XCTAssertEqual(vm.qwen35AnePrefillSequenceLength, "2112")
        XCTAssertEqual(vm.qwen35AnePrefillTailPaddingMinTokens, "1400")
        XCTAssertEqual(vm.qwen35AnePrefillFraction, "0.467")
        XCTAssertTrue(vm.qwen35AnePrefillFusedDown)
        XCTAssertFalse(vm.qwen35AnePrefillDualAne)
        XCTAssertTrue(vm.qwen35AnePrefillGdn)
        XCTAssertEqual(vm.qwen35AnePrefillGdnFraction, "0.527")
    }

    func testK2AnePresetsKeepOneThirdAndExistingValues() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "mova", configModelType: "k2_horizon")
        vm.qwen35AnePrefillEnabled = true
        let presets = ModelSettingsScreenVM.aneFractionOptions(current: "0.42", presets: [1.0 / 3.0, 0.5])
        XCTAssertEqual(presets.map(\.0), ["0.42", String(1.0 / 3.0), "0.5"])
        XCTAssertEqual(presets.map(\.1), ["42%", "33%", "50%"])
        vm.qwen35AnePrefillFraction = presets[0].0
        XCTAssertEqual(vm.currentSettingsDict()[ProfileSettingsKey.qwen35AnePrefillFraction]?.value as? Double, 0.42)
        vm.qwen35AnePrefillFraction = presets[1].0
        XCTAssertEqual(vm.currentSettingsDict()[ProfileSettingsKey.qwen35AnePrefillFraction]?.value as? Double, 1.0 / 3.0)
        XCTAssertEqual(ModelSettingsScreenVM.aneFractionOptions(current: vm.qwen35AnePrefillFraction, presets: [1.0 / 3.0, 0.5]).count, 2)
    }

    func testK2SharedPresetsKeepLoadedStringSelection() {
        for current in ["0", "0.0", "1", "1.0", String(1.0 / 3.0)] {
            let options = ModelSettingsScreenVM.aneFractionOptions(current: current, presets: [0, 1.0 / 3.0, 1])
            XCTAssertEqual(options.map(\.1), ["0%", "33%", "100%"])
            XCTAssertEqual(options.filter { $0.0 == current }.count, 1)
        }
    }

    func testMovaAneRecommendationUsesSharedProfileFields() throws {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "mova", configModelType: "k2_horizon")
        let data = Data(#"{"tuning_id":"k2","model_id":"mova","status":"completed","phase":"completed","message":"Done","current":1,"total":1,"results":[],"recommendation":{"backend":"k2","enabled":true,"mlp_fraction":0.3333333333333333,"shared_fraction":1,"gdn_enabled":false,"processing_tps":100,"speedup_percent":4,"sequence_length":2048}}"#.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        vm.aneTuningStatus = try decoder.decode(ANETuningStatusResponse.self, from: data)
        XCTAssertEqual(vm.aneTuningStatus?.recommendation?.processingTps, 100)
        XCTAssertEqual(vm.aneTuningStatus?.recommendation?.speedupPercent, 4)
        vm.applyANETuningRecommendation()
        XCTAssertTrue(vm.thinkingForced)
        XCTAssertTrue(vm.qwen35AnePrefillEnabled)
        XCTAssertTrue(vm.qwen35AnePrefillGdn)
        XCTAssertEqual(Double(vm.qwen35AnePrefillFraction), 1.0 / 3.0)
        XCTAssertEqual(Double(vm.qwen35AnePrefillSharedFraction), 1)
        XCTAssertTrue(vm.profileDirty)
        let fields = vm.currentSettingsDict()
        XCTAssertNotNil(fields[ProfileSettingsKey.qwen35AnePrefillEnabled])
    }

    func testControlsUseServerCapabilitiesForAnyModelFamily() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = Data(#"{"id":"future","loaded":false,"is_loading":false,"estimated_size":0,"config_model_type":"future_model","thinking_forced":true,"reasoning_effort_options":["medium","high"],"reasoning_effort_default":"medium","reasoning_effort_custom":false,"ane_prefill_backend":"qwen","ane_prefill_mlp_fractions":[0.25]}"#.utf8)
        let vm = ModelSettingsScreenVM()
        vm.model = try decoder.decode(ModelDTO.self, from: data)
        XCTAssertTrue(vm.thinkingForced)
        XCTAssertTrue(vm.isQwen35AnePrefillModel)
        XCTAssertEqual(vm.reasoningEffortPresets, ["medium", "high"])
        XCTAssertEqual(vm.model?.anePrefillMlpFractions, [0.25])
        vm.addKwarg(.enableThinking)
        XCTAssertTrue(vm.chatTemplateEntries.isEmpty)
        vm.addKwarg(.reasoningEffort)
        XCTAssertEqual(vm.chatTemplateEntries.first?.value, "medium")
        XCTAssertNil(vm.currentSettingsDict()["enable_thinking"])
        vm.model?.anePrefillBackend = nil
        XCTAssertFalse(vm.isQwen35AnePrefillModel)
    }

    func testQwen4SsdOffloadWireKeysAndCompatibility() throws {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "qwen4", configModelType: "qwen4_exp")
        XCTAssertTrue(vm.isQwen4Exp)

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let dto = try decoder.decode(
            ModelSettingsDTO.self,
            from: Data(#"{"qwen4_ple_ssd_offload":true}"#.utf8)
        )
        XCTAssertEqual(dto.qwen4PleSsdOffload, true)

        var patch = ModelSettingsPatch()
        patch.qwen4PleSsdOffload = true
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(
            with: encoder.encode(patch)
        ) as? [String: Any]
        XCTAssertEqual(object?["qwen4_ple_ssd_offload"] as? Bool, true)
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
            "qwen35_ane_prefill_gdn_max_layers": 48,
            "qwen35_ane_prefill_cpu_enabled": true,
            "qwen35_ane_prefill_cpu_fraction": 0.135,
            "qwen35_ane_prefill_cpu_down_fraction": 0.2,
            "qwen35_ane_prefill_cpu_gdn_fraction": 0.05,
            "qwen35_ane_prefill_cpu_threads": 8,
            "qwen35_ane_prefill_cpu_shared_resource": true
        }
        """#
        let dto = try decoder.decode(ModelSettingsDTO.self, from: Data(json.utf8))
        XCTAssertEqual(dto.qwen35AnePrefillFraction, 0.53)
        XCTAssertEqual(dto.qwen35AnePrefillGdnFraction, 0.5)
        XCTAssertEqual(dto.qwen35AnePrefillCpuEnabled, true)
        XCTAssertEqual(dto.qwen35AnePrefillCpuFraction, 0.135)
        XCTAssertEqual(dto.qwen35AnePrefillCpuDownFraction, 0.2)
        XCTAssertEqual(dto.qwen35AnePrefillCpuGdnFraction, 0.05)
        XCTAssertEqual(dto.qwen35AnePrefillCpuThreads, 8)
        XCTAssertEqual(dto.qwen35AnePrefillCpuSharedResource, true)

        var patch = ModelSettingsPatch()
        patch.qwen35AnePrefillEnabled = true
        patch.qwen35AnePrefillFraction = 0.53
        patch.qwen35AnePrefillCpuEnabled = true
        patch.qwen35AnePrefillCpuFraction = 0.135
        patch.qwen35AnePrefillCpuDownFraction = 0.2
        patch.qwen35AnePrefillCpuGdnFraction = 0.05
        patch.qwen35AnePrefillCpuThreads = 8
        patch.qwen35AnePrefillCpuSharedResource = true
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(patch)) as? [String: Any]
        XCTAssertEqual(object?["qwen35_ane_prefill_enabled"] as? Bool, true)
        XCTAssertEqual(object?["qwen35_ane_prefill_fraction"] as? Double, 0.53)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_enabled"] as? Bool, true)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_fraction"] as? Double, 0.135)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_down_fraction"] as? Double, 0.2)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_gdn_fraction"] as? Double, 0.05)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_threads"] as? Int, 8)
        XCTAssertEqual(object?["qwen35_ane_prefill_cpu_shared_resource"] as? Bool, true)
    }

    func testANETunerOverridesEncodeForStartRequest() throws {
        let request = ANETuningStartRequest(
            modelId: "qwen",
            sequenceLength: 2048,
            repeats: 2,
            allowCpu: false,
            allowCpuGate: false,
            allowCpuDown: true,
            allowAneGdn: false,
            allowCpuGdn: false,
            allowCpuSharedResource: false
        )
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase

        let data = try encoder.encode(request)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        XCTAssertEqual(object?["allow_cpu"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_gate"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_down"] as? Bool, true)
        XCTAssertEqual(object?["allow_ane_gdn"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_gdn"] as? Bool, false)
        XCTAssertEqual(object?["allow_cpu_shared_resource"] as? Bool, false)
    }

    func testK2UsesExistingReasoningEffortAndBudgetSettings() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(id: "mova", configModelType: "k2_horizon")
        XCTAssertEqual(vm.reasoningEffortPresets, ["low", "medium", "high"])
        vm.addKwarg(.enableThinking)
        XCTAssertTrue(vm.chatTemplateEntries.isEmpty)
        vm.addKwarg(.reasoningEffort)
        XCTAssertEqual(vm.chatTemplateEntries.first?.value, "high")
        vm.thinkingBudgetEnabled = true
        vm.thinkingBudgetTokens = "1"
        let settings = vm.currentSettingsDict()
        XCTAssertNil(settings["enable_thinking"])
        XCTAssertEqual(settings["thinking_budget_tokens"]?.value as? Int, 1)
        let kwargs = settings["chat_template_kwargs"]?.value as? [String: AnyCodable]
        XCTAssertEqual(kwargs?["reasoning_effort"]?.value as? String, "high")
        vm.model = makeModel(id: "qwen", configModelType: "qwen3")
        XCTAssertTrue(vm.reasoningEffortPresets.contains("xhigh"))
        XCTAssertNotNil(vm.currentSettingsDict()["enable_thinking"])
    }




    func testThinkingPatchCanClearK2SettingWithoutChangingOtherWireValues() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let cases: [(Bool??, String)] = [
            (nil, "{}"),
            (.some(nil), #"{"enable_thinking":null}"#),
            (true, #"{"enable_thinking":true}"#),
            (false, #"{"enable_thinking":false}"#),
        ]
        for (value, expected) in cases {
            var patch = ModelSettingsPatch()
            patch.enableThinking = value
            XCTAssertEqual(String(decoding: try encoder.encode(patch), as: UTF8.self), expected)
        }
    }

    func testSharedAneSettingsPreserveSplitAndExcludeOtherBackendOptions() {
        let vm = ModelSettingsScreenVM()
        vm.qwen35AnePrefillEnabled = true
        vm.qwen35AnePrefillFraction = String(1.0 / 3.0)
        vm.qwen35AnePrefillSharedFraction = "0"
        vm.qwen35AnePrefillCpuEnabled = true
        vm.model = makeModel(id: "k2", configModelType: "k2_horizon")
        let k2 = vm.currentSettingsDict()
        XCTAssertEqual(k2["qwen35_ane_prefill_enabled"]?.value as? Bool, true)
        XCTAssertEqual(k2["qwen35_ane_prefill_fraction"]?.value as? Double, 1.0 / 3.0)
        XCTAssertEqual(k2["qwen35_ane_prefill_shared_fraction"]?.value as? Double, 0)
        XCTAssertNil(k2["qwen35_ane_prefill_cpu_enabled"])
        XCTAssertFalse(k2.keys.contains { $0.hasPrefix("k2_ane_") })
        vm.model = makeModel(id: "qwen", configModelType: "qwen3_5")
        let qwen = vm.currentSettingsDict()
        XCTAssertEqual(qwen["qwen35_ane_prefill_enabled"]?.value as? Bool, true)
        XCTAssertEqual(qwen["qwen35_ane_prefill_cpu_enabled"]?.value as? Bool, true)
        XCTAssertNil(qwen["qwen35_ane_prefill_shared_fraction"])
    }

    func testMtplxImportIsOfferedOnlyForTheServerSideSidecarMarker() {
        let vm = ModelSettingsScreenVM()
        vm.model = makeModel(
            id: "Qwen3.8-27B-MTPLX-Optimized-Speed",
            configModelType: "qwen3_5",
            mtpCompatible: false,
            mtpCompatibilityReason: "MTPLX side-car detected but not imported. Import it to merge the MTP head into the checkpoint index."
        )
        XCTAssertTrue(vm.canImportMtplxSidecar)

        vm.model = makeModel(
            id: "missing-mtp",
            configModelType: "qwen3_5",
            mtpCompatible: false,
            mtpCompatibilityReason: "Config declares MTP layers but the weight files contain neither mtp.* tensors nor native nextn layers."
        )
        XCTAssertFalse(vm.canImportMtplxSidecar)

        vm.model = makeModel(
            id: "already-compatible",
            configModelType: "qwen3_5",
            mtpCompatible: true,
            mtpCompatibilityReason: "MTPLX side-car detected but not imported."
        )
        XCTAssertFalse(vm.canImportMtplxSidecar)
    }

    func testMtplxImportEndpointUsesModelScopedAdminRoute() {
        XCTAssertEqual(
            AdminAPI.importMtplx("Qwen3.8-27B-MTPLX-Optimized-Speed"),
            "/admin/api/models/Qwen3.8-27B-MTPLX-Optimized-Speed/import-mtplx"
        )
    }

    private func makeModel(
        id: String,
        configModelType: String?,
        mtpCompatible: Bool? = nil,
        mtpCompatibilityReason: String? = nil
    ) -> ModelDTO {
        var model = ModelDTO(
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
            mtpCompatible: mtpCompatible,
            mtpCompatibilityReason: mtpCompatibilityReason,
            qwen4PleSsdOffloadSupported: nil,
            qwen4PleSsdOffloadForced: nil,
            qwen4PleResidentBytes: nil,
            qwen4PleMmapBytes: nil,
            virtual: nil,
            settings: nil
        )
        model.thinkingForced = configModelType == "k2_horizon"
        model.reasoningEffortOptions = model.thinkingForced == true ? ["low", "medium", "high"] : ["low", "medium", "high", "xhigh", "max"]
        model.reasoningEffortDefault = model.thinkingForced == true ? "high" : "low"
        model.reasoningEffortCustom = model.thinkingForced != true
        model.anePrefillBackend = configModelType == "k2_horizon" ? "k2" : (configModelType?.hasPrefix("qwen3_5") == true ? "qwen" : nil)
        model.anePrefillDefaultFraction = model.anePrefillBackend == "k2" ? 1.0 / 3.0 : 0.53
        model.anePrefillMlpFractions = [1.0 / 3.0, 0.5]
        model.anePrefillSharedFractions = [0, 1.0 / 3.0, 1]
        return model
    }
}
