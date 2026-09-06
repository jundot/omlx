// Shared model behind the menubar metric items (LIV / AVG / ALL) and their
// popovers. The stats poller feeds one applyTick() per poll cycle; the
// status-item glyphs and any open popover read the latest rates plus a
// fixed-capacity history window for the activity graphs.

import Foundation
import Observation

/// Aggregate throughput reading for one metric item.
struct MetricRates: Equatable, Sendable {
    /// Prompt-processing speed in tok/s. nil = unknown (fetch failed,
    /// admin auth unavailable, or server off) — distinct from an idle 0.
    var promptTps: Double?
    /// Token-generation speed in tok/s.
    var generationTps: Double?
}

/// Presentation-ready serving data for one menubar metric scope. This keeps
/// popovers independent of the poller's transport DTOs.
struct ServingStatsSnapshot: Equatable, Sendable {
    let rates: MetricRates?
    let totalPromptTokens: Int?
    let totalCachedTokens: Int?
    let cacheEfficiency: Double?
    let totalRequests: Int?
    let liveActivity: MenubarStatsPoller.Stats.LiveActivity?

    static let unavailable = ServingStatsSnapshot(
        rates: nil,
        totalPromptTokens: nil,
        totalCachedTokens: nil,
        cacheEfficiency: nil,
        totalRequests: nil,
        liveActivity: nil
    )

    var isUnavailable: Bool {
        rates?.promptTps == nil
            && rates?.generationTps == nil
            && totalPromptTokens == nil
            && totalCachedTokens == nil
            && cacheEfficiency == nil
            && totalRequests == nil
            && liveActivity == nil
    }
}

@MainActor
@Observable
final class MenubarMetricsStore {
    enum Kind: String, CaseIterable, Sendable {
        case live
        case average
        case alltime
    }

    struct Series: Equatable, Sendable {
        var promptTps: [Double] = []
        var generationTps: [Double] = []
    }

    /// Samples kept per series — at the default 1 s cadence the activity
    /// graphs cover the last minute.
    nonisolated static let historyCapacity = 60

    private(set) var snapshots: [Kind: ServingStatsSnapshot] = [:]
    private(set) var history: [Kind: Series] = [:]
    private(set) var serverIsRunning = false

    /// One call per poller tick. Unknown readings surface as nil rates (the
    /// glyph shows "–") but roll a 0 into the history so the graph timeline
    /// stays contiguous.
    func applyTick(
        live: ServingStatsSnapshot?,
        average: ServingStatsSnapshot?,
        alltime: ServingStatsSnapshot?,
        serverRunning: Bool
    ) {
        serverIsRunning = serverRunning
        record(.live, live)
        record(.average, average)
        record(.alltime, alltime)
    }

    /// Server transitioned to stopped/failed: blank the readings and freeze
    /// the history where it was.
    func markServerStopped() {
        serverIsRunning = false
        snapshots = [:]
    }

    func snapshot(for kind: Kind) -> ServingStatsSnapshot {
        snapshots[kind] ?? .unavailable
    }

    private func record(_ kind: Kind, _ snapshot: ServingStatsSnapshot?) {
        snapshots[kind] = snapshot ?? .unavailable
        var series = history[kind] ?? Series()
        Self.append(&series.promptTps, snapshot?.rates?.promptTps ?? 0)
        Self.append(&series.generationTps, snapshot?.rates?.generationTps ?? 0)
        history[kind] = series
    }

    nonisolated static func append(_ series: inout [Double], _ value: Double) {
        series.append(value)
        let overflow = series.count - historyCapacity
        if overflow > 0 {
            series.removeFirst(overflow)
        }
    }
}

// MARK: - Rate aggregation (pure, unit-testable)

extension MenubarMetricsStore {
    /// Maps each polling source into one shared popover model. LIV is
    /// intentionally limited to instantaneous rates and current requests;
    /// AVG and ALL retain the scalar counters from their respective scopes.
    nonisolated static func snapshot(
        for kind: Kind,
        from stats: MenubarStatsPoller.Stats?
    ) -> ServingStatsSnapshot? {
        guard let stats else {
            return nil
        }

        switch kind {
        case .live:
            guard let rates = liveRates(from: stats) else {
                return nil
            }
            return ServingStatsSnapshot(
                rates: rates,
                totalPromptTokens: nil,
                totalCachedTokens: nil,
                cacheEfficiency: nil,
                totalRequests: nil,
                liveActivity: stats.liveActivity
            )
        case .average, .alltime:
            return ServingStatsSnapshot(
                rates: averageRates(from: stats),
                totalPromptTokens: stats.totalPromptTokens,
                totalCachedTokens: stats.totalCachedTokens,
                cacheEfficiency: stats.cacheEfficiency,
                totalRequests: stats.totalRequests,
                liveActivity: nil
            )
        }
    }

    /// Instantaneous rates summed across every in-flight request of every
    /// model. A decoded-but-idle activity payload yields 0/0; a missing
    /// payload (fetch disabled or failed) yields nil.
    nonisolated static func liveRates(
        from stats: MenubarStatsPoller.Stats?
    ) -> MetricRates? {
        guard let models = stats?.activeModels?.models else {
            return nil
        }
        var promptTps = 0.0
        var generationTps = 0.0
        for model in models {
            for prefill in model.prefilling ?? [] {
                promptTps += max(0, prefill.speed ?? 0)
            }
            for generation in model.generating ?? [] {
                generationTps += max(0, generation.tokensPerSecond ?? 0)
            }
        }
        return MetricRates(promptTps: promptTps, generationTps: generationTps)
    }

    /// Cumulative-average rates as reported by the stats endpoints (used for
    /// both the session and the all-time scope).
    nonisolated static func averageRates(
        from stats: MenubarStatsPoller.Stats?
    ) -> MetricRates? {
        guard let stats else {
            return nil
        }
        return MetricRates(
            promptTps: stats.avgPrefillTps,
            generationTps: stats.avgGenerationTps
        )
    }
}
