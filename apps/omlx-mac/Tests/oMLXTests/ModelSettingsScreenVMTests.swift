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

    func testExpertStreamingDefaultsAndOptions() {
        let vm = ModelSettingsScreenVM()

        XCTAssertFalse(vm.expertStreamingEnabled)
        XCTAssertEqual(vm.expertStreamingMode, "soft_reap")
        XCTAssertEqual(vm.expertStreamingCacheExperts, "32")
        XCTAssertEqual(vm.expertStreamingScratchExperts, "32")
        XCTAssertTrue(vm.expertStreamingFastResourceLoading)
        XCTAssertEqual(vm.expertStreamingCachePolicy, "route_frequency")
        XCTAssertTrue(vm.expertStreamingDirectIo)
        XCTAssertTrue(vm.expertStreamingNativeDemand)
        XCTAssertTrue(vm.expertStreamingDecodeScratchAsCache)
        XCTAssertEqual(vm.expertStreamingIoCoalescingKib, "64")
        XCTAssertEqual(
            ModelSettingsScreenVM.expertStreamingModeOptions.map(\.0),
            ["soft_reap", "cache_only"]
        )
        XCTAssertEqual(
            ModelSettingsScreenVM.expertStreamingCachePolicyOptions.map(\.0),
            ["route_frequency", "lru"]
        )
    }

    func testExpertStreamingWireKeysDecodeAndEncode() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let dto = try decoder.decode(
            ModelSettingsDTO.self,
            from: Data(#"""
            {
              "expert_streaming_enabled": true,
              "expert_streaming_mode": "soft_reap",
              "expert_streaming_manifest": "/tmp/manifest.json",
              "expert_streaming_cache_experts": 48,
              "expert_streaming_scratch_experts": 24,
              "expert_streaming_cache_policy": "lru",
              "expert_streaming_fast_resource_loading": true,
              "expert_streaming_direct_io": true,
              "expert_streaming_native_demand": true,
              "expert_streaming_decode_scratch_as_cache": true,
              "expert_streaming_io_coalescing_kib": 128
            }
            """#.utf8)
        )
        XCTAssertEqual(dto.expertStreamingEnabled, true)
        XCTAssertEqual(dto.expertStreamingMode, "soft_reap")
        XCTAssertEqual(dto.expertStreamingManifest, "/tmp/manifest.json")
        XCTAssertEqual(dto.expertStreamingCacheExperts, 48)
        XCTAssertEqual(dto.expertStreamingScratchExperts, 24)
        XCTAssertEqual(dto.expertStreamingCachePolicy, "lru")
        XCTAssertEqual(dto.expertStreamingFastResourceLoading, true)
        XCTAssertEqual(dto.expertStreamingDirectIo, true)
        XCTAssertEqual(dto.expertStreamingNativeDemand, true)
        XCTAssertEqual(dto.expertStreamingDecodeScratchAsCache, true)
        XCTAssertEqual(dto.expertStreamingIoCoalescingKib, 128)

        var patch = ModelSettingsPatch()
        patch.expertStreamingEnabled = true
        patch.expertStreamingMode = "cache_only"
        patch.expertStreamingCacheExperts = 64
        patch.expertStreamingScratchExperts = 16
        patch.expertStreamingCachePolicy = "route_frequency"
        patch.expertStreamingFastResourceLoading = true
        patch.expertStreamingDirectIo = true
        patch.expertStreamingNativeDemand = true
        patch.expertStreamingDecodeScratchAsCache = true
        patch.expertStreamingIoCoalescingKib = 64
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let object = try JSONSerialization.jsonObject(with: encoder.encode(patch)) as? [String: Any]
        XCTAssertEqual(object?["expert_streaming_enabled"] as? Bool, true)
        XCTAssertEqual(object?["expert_streaming_mode"] as? String, "cache_only")
        XCTAssertEqual(object?["expert_streaming_cache_experts"] as? Int, 64)
        XCTAssertEqual(object?["expert_streaming_scratch_experts"] as? Int, 16)
        XCTAssertEqual(object?["expert_streaming_cache_policy"] as? String, "route_frequency")
        XCTAssertEqual(object?["expert_streaming_fast_resource_loading"] as? Bool, true)
        XCTAssertEqual(object?["expert_streaming_direct_io"] as? Bool, true)
        XCTAssertEqual(object?["expert_streaming_native_demand"] as? Bool, true)
        XCTAssertEqual(object?["expert_streaming_decode_scratch_as_cache"] as? Bool, true)
        XCTAssertEqual(object?["expert_streaming_io_coalescing_kib"] as? Int, 64)
    }

    func testSoftReapSettingsAreIncludedInWorkingProfile() {
        let vm = ModelSettingsScreenVM()
        vm.expertStreamingEnabled = true
        vm.expertStreamingMode = "soft_reap"
        vm.expertStreamingManifest = "/manifests/qwen.json"
        vm.expertStreamingCacheExperts = "40"
        vm.expertStreamingScratchExperts = "20"
        vm.expertStreamingCachePolicy = "lru"
        vm.expertStreamingFastResourceLoading = true
        vm.expertStreamingDirectIo = true
        vm.expertStreamingNativeDemand = true
        vm.expertStreamingDecodeScratchAsCache = true
        vm.expertStreamingIoCoalescingKib = "64"

        let settings = vm.currentSettingsDict()

        XCTAssertEqual(settings["expert_streaming_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["expert_streaming_mode"]?.value as? String, "soft_reap")
        XCTAssertEqual(settings["expert_streaming_manifest"]?.value as? String, "/manifests/qwen.json")
        XCTAssertEqual(settings["expert_streaming_cache_experts"]?.value as? Int, 40)
        XCTAssertEqual(settings["expert_streaming_scratch_experts"]?.value as? Int, 20)
        XCTAssertEqual(settings["expert_streaming_cache_policy"]?.value as? String, "lru")
        XCTAssertEqual(settings["expert_streaming_fast_resource_loading"]?.value as? Bool, true)
        XCTAssertEqual(settings["expert_streaming_direct_io"]?.value as? Bool, true)
        XCTAssertEqual(settings["expert_streaming_native_demand"]?.value as? Bool, true)
        XCTAssertEqual(settings["expert_streaming_decode_scratch_as_cache"]?.value as? Bool, true)
        XCTAssertEqual(settings["expert_streaming_io_coalescing_kib"]?.value as? Int, 64)
    }

    func testCacheOnlyProfileDoesNotPersistManifest() {
        let vm = ModelSettingsScreenVM()
        vm.expertStreamingEnabled = true
        vm.expertStreamingMode = "cache_only"
        vm.expertStreamingManifest = "/manifests/old.json"

        let settings = vm.currentSettingsDict()

        XCTAssertEqual(settings["expert_streaming_enabled"]?.value as? Bool, true)
        XCTAssertEqual(settings["expert_streaming_mode"]?.value as? String, "cache_only")
        XCTAssertNil(settings["expert_streaming_manifest"])
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
            qwen4PleSsdOffloadSupported: nil,
            qwen4PleSsdOffloadForced: nil,
            qwen4PleResidentBytes: nil,
            qwen4PleMmapBytes: nil,
            expertStreamingSupported: nil,
            expertStreamingResidentBytes: nil,
            virtual: nil,
            settings: nil
        )
    }
}
