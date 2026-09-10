import SwiftUI

@MainActor
protocol ANETuningClientProtocol: AnyObject {
    func startANETuning(_ body: ANETuningStartRequest) async throws -> ANETuningStartResponse
    func getANETuningResults(tuningId: String) async throws -> ANETuningStatusResponse
    func cancelANETuning(tuningId: String) async throws -> ANETuningCancelResponse
}

extension OMLXClient: ANETuningClientProtocol {}

struct ANETuningOverrides: Equatable {
    let allowCPU: Bool
    let allowCPUGate: Bool
    let allowCPUDown: Bool
    let allowANEGDN: Bool
    let allowCPUGDN: Bool
    let allowCPUSharedResource: Bool

    init(
        allowCPU: Bool = true,
        allowCPUGate: Bool = true,
        allowCPUDown: Bool = true,
        allowANEGDN: Bool = true,
        allowCPUGDN: Bool = true,
        allowCPUSharedResource: Bool = true
    ) {
        self.allowCPU = allowCPU
        self.allowCPUGate = allowCPUGate
        self.allowCPUDown = allowCPUDown
        self.allowANEGDN = allowANEGDN
        self.allowCPUGDN = allowCPUGDN
        self.allowCPUSharedResource = allowCPUSharedResource
    }
}

/// App-scoped owner for the server's single ANE tuning run.
///
/// Model Settings screens are transient: leaving the model detail or changing
/// sidebar sections destroys their local view model. The server run is not
/// tied to that view lifetime, so this session lives beside the long-running
/// benchmark view models in `AppServices` and keeps polling while off-screen.
@MainActor
@Observable
final class ANETuningSessionVM {
    private(set) var tuningID: String?
    private(set) var modelID: String?
    private(set) var isRunning: Bool = false
    private(set) var status: ANETuningStatusResponse?
    private(set) var lastError: String?

    @ObservationIgnored
    private var pollTask: Task<Void, Never>?
    @ObservationIgnored
    private var generation: UInt = 0
    @ObservationIgnored
    private let pollInterval: Duration

    init(pollInterval: Duration = .seconds(1)) {
        self.pollInterval = pollInterval
    }

    func isForModel(_ modelID: String) -> Bool {
        self.modelID == modelID
    }

    func status(for modelID: String) -> ANETuningStatusResponse? {
        isForModel(modelID) ? status : nil
    }

    func isRunning(for modelID: String) -> Bool {
        isForModel(modelID) && isRunning
    }

    func error(for modelID: String) -> String? {
        isForModel(modelID) ? lastError : nil
    }

    func start(
        modelID: String,
        sequenceLengthText: String,
        overrides: ANETuningOverrides = ANETuningOverrides(),
        client: any ANETuningClientProtocol
    ) {
        guard !isRunning else { return }
        guard let sequenceLength = Int(sequenceLengthText) else {
            if self.modelID != modelID {
                generation &+= 1
                pollTask?.cancel()
                pollTask = nil
                tuningID = nil
                status = nil
                isRunning = false
            }
            self.modelID = modelID
            lastError = "ANE prompt block must be a number."
            return
        }

        generation &+= 1
        let runGeneration = generation
        pollTask?.cancel()

        self.modelID = modelID
        tuningID = nil
        status = nil
        lastError = nil
        isRunning = true

        let request = ANETuningStartRequest(
            modelId: modelID,
            sequenceLength: sequenceLength,
            repeats: 2,
            allowCpu: overrides.allowCPU,
            allowCpuGate: overrides.allowCPU && overrides.allowCPUGate,
            allowCpuDown: overrides.allowCPU && overrides.allowCPUDown,
            allowAneGdn: overrides.allowANEGDN,
            allowCpuGdn: overrides.allowCPU
                && overrides.allowANEGDN
                && overrides.allowCPUGDN,
            allowCpuSharedResource: overrides.allowCPU
                && overrides.allowCPUSharedResource
        )

        pollTask = Task { [weak self] in
            do {
                let started = try await client.startANETuning(request)
                guard let self,
                      self.generation == runGeneration,
                      self.modelID == modelID,
                      self.isRunning else { return }
                self.tuningID = started.tuningId
                await self.poll(
                    tuningID: started.tuningId,
                    modelID: modelID,
                    generation: runGeneration,
                    client: client
                )
            } catch is CancellationError {
                return
            } catch {
                guard let self,
                      self.generation == runGeneration,
                      self.modelID == modelID else { return }
                self.isRunning = false
                self.lastError = error.omlxDescription
                self.pollTask = nil
            }
        }
    }

    func cancel(client: any ANETuningClientProtocol) {
        guard let tuningID, isRunning else { return }
        let runGeneration = generation
        Task { [weak self] in
            do {
                _ = try await client.cancelANETuning(tuningId: tuningID)
                guard let self,
                      self.generation == runGeneration,
                      self.tuningID == tuningID else { return }
                // The poll loop records the server's terminal cancelled
                // snapshot, including any partial result matrix.
                self.lastError = nil
            } catch {
                guard let self,
                      self.generation == runGeneration,
                      self.tuningID == tuningID else { return }
                self.lastError = error.omlxDescription
            }
        }
    }

    /// Invalidate a process-local run id. Navigation never calls this; server
    /// stop/restart and server-target changes do because the backend's in-memory
    /// run registry no longer describes the same process.
    func reset() {
        generation &+= 1
        pollTask?.cancel()
        pollTask = nil
        tuningID = nil
        modelID = nil
        isRunning = false
        status = nil
        lastError = nil
    }

    private func poll(
        tuningID: String,
        modelID: String,
        generation runGeneration: UInt,
        client: any ANETuningClientProtocol
    ) async {
        while !Task.isCancelled {
            guard generation == runGeneration,
                  self.tuningID == tuningID,
                  self.modelID == modelID,
                  isRunning else { return }

            do {
                let snapshot = try await client.getANETuningResults(tuningId: tuningID)
                guard generation == runGeneration,
                      self.tuningID == tuningID,
                      self.modelID == modelID else { return }

                guard snapshot.modelId == modelID else {
                    isRunning = false
                    lastError = "ANE tuning returned a result for a different model."
                    pollTask = nil
                    return
                }

                status = snapshot
                lastError = nil
                if snapshot.status != "running" {
                    isRunning = false
                    pollTask = nil
                    return
                }
            } catch is CancellationError {
                return
            } catch {
                guard generation == runGeneration,
                      self.tuningID == tuningID,
                      self.modelID == modelID else { return }
                lastError = error.omlxDescription

                // A missing run cannot recover: the server was restarted or
                // its process-local registry expired. Other transport failures
                // are retried so a brief disconnect does not strand the UI.
                if Self.isMissingRun(error) {
                    isRunning = false
                    pollTask = nil
                    return
                }
            }

            do {
                try await Task.sleep(for: pollInterval)
            } catch {
                return
            }
        }
    }

    private static func isMissingRun(_ error: Error) -> Bool {
        guard let clientError = error as? OMLXClientError else { return false }
        if case .http(let status, _) = clientError {
            return status == 404
        }
        return false
    }
}
