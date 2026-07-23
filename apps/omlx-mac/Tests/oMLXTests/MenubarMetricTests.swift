// Unit coverage for the menubar metric items' data path: rate aggregation
// from poller payloads, the history ring buffer, and the glyph formatting /
// signature logic that gates status-item re-rasters.

import Foundation
import XCTest
@testable import oMLX

@MainActor
final class MenubarMetricTests: XCTestCase {

    private func decodeStats(_ json: String) throws -> MenubarStatsPoller.Stats {
        try JSONDecoder().decode(
            MenubarStatsPoller.Stats.self,
            from: XCTUnwrap(json.data(using: .utf8))
        )
    }

    // MARK: - Live rate aggregation

    func testLiveRatesSumAcrossModelsAndRequests() throws {
        let stats = try decodeStats(
            """
            {
              "active_models": {
                "models": [
                  {
                    "id": "model-a",
                    "prefilling": [
                      {"processed": 10, "total": 100, "speed": 300.0},
                      {"processed": 20, "total": 100, "speed": 200.5}
                    ],
                    "generating": [
                      {"generated_tokens": 5, "tokens_per_second": 40.0}
                    ]
                  },
                  {
                    "id": "model-b",
                    "generating": [
                      {"generated_tokens": 9, "tokens_per_second": 2.5}
                    ]
                  }
                ]
              }
            }
            """
        )

        let rates = try XCTUnwrap(MenubarMetricsStore.liveRates(from: stats))
        XCTAssertEqual(rates.promptTps, 500.5)
        XCTAssertEqual(rates.generationTps, 42.5)
    }

    func testLiveRatesAreZeroWhenDecodedButIdle() throws {
        let stats = try decodeStats(#"{"active_models": {"models": []}}"#)

        let rates = try XCTUnwrap(MenubarMetricsStore.liveRates(from: stats))
        XCTAssertEqual(rates.promptTps, 0)
        XCTAssertEqual(rates.generationTps, 0)
    }

    func testLiveRatesAreNilWithoutAnActivityPayload() throws {
        XCTAssertNil(MenubarMetricsStore.liveRates(from: nil))

        let statsWithoutActivity = try decodeStats(#"{"total_prompt_tokens": 3}"#)
        XCTAssertNil(MenubarMetricsStore.liveRates(from: statsWithoutActivity))
    }

    func testAverageRatesPassThroughSnapshotFields() throws {
        XCTAssertNil(MenubarMetricsStore.averageRates(from: nil))

        let stats = try decodeStats(
            #"{"avg_prefill_tps": 512.3, "avg_generation_tps": 48.7}"#
        )
        let rates = try XCTUnwrap(MenubarMetricsStore.averageRates(from: stats))
        XCTAssertEqual(rates.promptTps, 512.3)
        XCTAssertEqual(rates.generationTps, 48.7)
    }

    // MARK: - History ring buffer

    func testHistoryAppendCapsAtCapacityDroppingOldest() {
        var series: [Double] = []
        for value in 0..<(MenubarMetricsStore.historyCapacity + 5) {
            MenubarMetricsStore.append(&series, Double(value))
        }

        XCTAssertEqual(series.count, MenubarMetricsStore.historyCapacity)
        XCTAssertEqual(series.first, 5)
        XCTAssertEqual(series.last, Double(MenubarMetricsStore.historyCapacity + 4))
    }

    func testApplyTickRecordsRatesAndRollsHistory() {
        let store = MenubarMetricsStore()
        store.applyTick(
            live: MetricRates(promptTps: 100, generationTps: 10),
            average: MetricRates(promptTps: 200, generationTps: 20),
            alltime: nil,
            serverRunning: true
        )

        XCTAssertTrue(store.serverIsRunning)
        XCTAssertEqual(store.rates[.live]?.promptTps, 100)
        XCTAssertEqual(store.rates[.average]?.generationTps, 20)
        XCTAssertNil(store.rates[.alltime])
        // Unknown readings still roll a 0 so the graph timeline stays contiguous.
        XCTAssertEqual(store.history[.alltime]?.promptTps, [0])
        XCTAssertEqual(store.history[.live]?.promptTps, [100])
        XCTAssertEqual(store.history[.live]?.generationTps, [10])
    }

    func testMarkServerStoppedBlanksRatesButFreezesHistory() {
        let store = MenubarMetricsStore()
        store.applyTick(
            live: MetricRates(promptTps: 100, generationTps: 10),
            average: MetricRates(promptTps: 200, generationTps: 20),
            alltime: MetricRates(promptTps: 300, generationTps: 30),
            serverRunning: true
        )

        store.markServerStopped()

        XCTAssertFalse(store.serverIsRunning)
        XCTAssertNil(store.rates[.live])
        XCTAssertNil(store.rates[.average])
        XCTAssertNil(store.rates[.alltime])
        XCTAssertEqual(store.history[.live]?.promptTps, [100])
        XCTAssertEqual(store.history[.alltime]?.generationTps, [30])
    }

    // MARK: - Model library scope pref

    func testModelLibraryScopeParsesAndFallsBackToAll() {
        let defaults = UserDefaults.standard
        defer { defaults.removeObject(forKey: MenubarMetricPrefs.modelLibraryScopeKey) }

        defaults.removeObject(forKey: MenubarMetricPrefs.modelLibraryScopeKey)
        XCTAssertEqual(MenubarMetricPrefs.modelLibraryScope, .all, "absent → all (legacy behavior)")

        defaults.set("favorites", forKey: MenubarMetricPrefs.modelLibraryScopeKey)
        XCTAssertEqual(MenubarMetricPrefs.modelLibraryScope, .favoritesOnly)

        defaults.set("all", forKey: MenubarMetricPrefs.modelLibraryScopeKey)
        XCTAssertEqual(MenubarMetricPrefs.modelLibraryScope, .all)

        defaults.set("garbage", forKey: MenubarMetricPrefs.modelLibraryScopeKey)
        XCTAssertEqual(MenubarMetricPrefs.modelLibraryScope, .all, "unknown raw values fall back to all")
    }

    // MARK: - Glyph formatting

    func testFormatTpsCoversUnknownWholeAndCompactRanges() {
        XCTAssertEqual(MenubarMetricGlyph.formatTps(nil), "–")
        XCTAssertEqual(MenubarMetricGlyph.formatTps(.infinity), "–")
        XCTAssertEqual(MenubarMetricGlyph.formatTps(-3), "0tk/s")
        XCTAssertEqual(MenubarMetricGlyph.formatTps(0), "0tk/s")
        XCTAssertEqual(MenubarMetricGlyph.formatTps(24.4), "24tk/s")
        XCTAssertEqual(MenubarMetricGlyph.formatTps(999.6), "1000tk/s")
        XCTAssertEqual(MenubarMetricGlyph.formatTps(12_345), "12.3ktk/s")
    }

    func testSignatureIsStableForIdenticalReadingsAndTracksEveryInput() {
        let base = MenubarMetricGlyph.signature(
            tag: "LIV", promptValue: "500t/s", generationValue: "24t/s",
            darkMenubar: false
        )

        XCTAssertEqual(
            base,
            MenubarMetricGlyph.signature(
                tag: "LIV", promptValue: "500t/s", generationValue: "24t/s",
                darkMenubar: false
            ),
            "identical readings must not trigger a re-raster"
        )

        XCTAssertNotEqual(base, MenubarMetricGlyph.signature(
            tag: "AVG", promptValue: "500t/s", generationValue: "24t/s",
            darkMenubar: false
        ))
        XCTAssertNotEqual(base, MenubarMetricGlyph.signature(
            tag: "LIV", promptValue: "501t/s", generationValue: "24t/s",
            darkMenubar: false
        ))
        XCTAssertNotEqual(base, MenubarMetricGlyph.signature(
            tag: "LIV", promptValue: "500t/s", generationValue: "25t/s",
            darkMenubar: false
        ))
        XCTAssertNotEqual(base, MenubarMetricGlyph.signature(
            tag: "LIV", promptValue: "500t/s", generationValue: "24t/s",
            darkMenubar: true
        ))
    }
}
