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

    private(set) var rates: [Kind: MetricRates] = [:]
    private(set) var history: [Kind: Series] = [:]
    private(set) var serverIsRunning = false

    /// One call per poller tick. Unknown readings surface as nil rates (the
    /// glyph shows "–") but roll a 0 into the history so the graph timeline
    /// stays contiguous.
    func applyTick(
        live: MetricRates?,
        average: MetricRates?,
        alltime: MetricRates?,
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
        rates = [:]
    }

    private func record(_ kind: Kind, _ reading: MetricRates?) {
        rates[kind] = reading
        var series = history[kind] ?? Series()
        Self.append(&series.promptTps, reading?.promptTps ?? 0)
        Self.append(&series.generationTps, reading?.generationTps ?? 0)
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
