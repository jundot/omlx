// Storage roofline DTOs — mirrors omlx/admin/routes.py /api/bench/storage/*
// (start, {job_id}/results, predict) and the report shape produced by
// omlx/utils/storage_roofline.py.

import Foundation

// MARK: - Job control

/// Body for `POST /admin/api/bench/storage/start`.
struct StorageBenchStartRequest: Encodable, Sendable {
    let modelId: String?
    let fileGb: Double
    let samples: Int
    let readMb: Int
}

struct StorageBenchStartResponse: Codable, Sendable {
    let jobId: String
    let status: String
}

/// Response from `GET /admin/api/bench/storage/{job_id}/results`.
/// `report` is only present once `status == "completed"`.
struct StorageBenchJobResponse: Codable, Sendable {
    let jobId: String
    let status: String
    let progress: StorageBenchProgressDTO?
    let error: String?
    let report: StorageRooflineReportDTO?

    var isTerminal: Bool { status == "completed" || status == "failed" }
}

struct StorageBenchProgressDTO: Codable, Equatable, Sendable {
    let phase: String
    let done: Int
    let total: Int
}

// MARK: - Report

struct StorageRooflineReportDTO: Codable, Equatable, Sendable {
    let version: Int?
    let timestamp: String?
    let volume: StorageVolumeDTO?
    let measurement: StorageMeasurementDTO?
    let profile: MoEStepProfileDTO?
    let prediction: RooflinePredictionDTO?
    let calibration: RooflineCalibrationDTO?
    let paramsSource: String?
    let paramsAuto: StorageAutoParamsDTO?
    let path: String?
}

struct StorageVolumeDTO: Codable, Equatable, Sendable {
    let path: String?
    let mount: String?
    let filesystem: String?
    let mediaName: String?
    let protocolField: String?
    let location: String?
    let solidState: Bool?
    let totalBytes: Int64?
    let freeBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case path, mount, filesystem, mediaName
        case protocolField = "protocol"
        case location, solidState, totalBytes, freeBytes
    }
}

struct StorageMeasurementDTO: Codable, Equatable, Sendable {
    let volumeMount: String?
    let fileBytes: Int64?
    let seqReadBps: Double?
    let randReadBps: Double?
    let randIops: Double?
    let randLatMsP50: Double?
    let randLatMsP90: Double?
    let randLatMsP99: Double?
    let randLatMsMax: Double?
    let writeBps: Double?
    let samples: Int?
    let readMb: Int?
    let cacheClean: Bool?
    let method: String?
    let warnings: [String]?
}

struct MoEStepProfileDTO: Codable, Equatable, Sendable {
    let modelDir: String?
    let modelType: String?
    let supported: Bool?
    let reason: String?
    let numMoeLayers: Int?
    let routedTotalPerLayer: Int?
    let topK: Int?
    let routedActiveBytesPerLayer: Int64?
    let sharedBytesPerLayer: Int64?
    let bytesPerStep: Int64?
    let checkpointBytes: Int64?
}

struct RooflinePredictionDTO: Codable, Equatable, Sendable {
    let bytesPerStep: Int64?
    let bytesVerify: Int64?
    let verifyByteMult: Double?
    let tokPerCycle: Double?
    let ceilingBaseTokS: Double?
    let ceilingMtpTokS: Double?
    let mtpProfitable: Bool?
    let marginTokPerCycle: Double?
    let explanation: String?
    // F2: measured-bytes/token ceiling + wall-clock verdict.
    let ceilingEffectiveTokS: Double?
    let bytesPerTokenBase: Double?
    let measuredMtpSlowdown: Double?
    let measuredMtpPays: Bool?
}

struct RooflineCalibrationDTO: Codable, Equatable, Sendable {
    let measuredBaseTokS: Double?
    let predictedCeilingBaseTokS: Double?
    let efficiency: Double?
    let predictedCeilingEffectiveTokS: Double?
    let efficiencyEffective: Double?
}


/// Response from `GET /admin/api/bench/storage/auto-params`.
/// `available == false` means defaults are in use (normal first-run state).
struct StorageAutoParamsDTO: Codable, Equatable, Sendable {
    let available: Bool?
    let modelDir: String?
    let derivedAt: String?
    let tokPerCycle: Double?
    let verifyByteMult: Double?
    let bytesPerTokenBase: Double?
    let source: AutoParamsSourceDTO?
}

/// Per-derivation bookkeeping (which bench runs produced the numbers).
struct AutoDerivationDTO: Codable, Equatable, Sendable {
    let tokPerCycle: Double?
    let verifyByteMult: Double?
    let bytesPerToken: Double?
    let cycles: Int?
    let accepted: Int?
    let drafted: Int?
    let decodeBytes: Int64?
    let decodeTokens: Int?
}

struct AutoParamsSourceDTO: Codable, Equatable, Sendable {
    let verifyMult: AutoDerivationDTO?
    let tokPerCycle: AutoDerivationDTO?
    let bytesPerToken: AutoDerivationDTO?
}