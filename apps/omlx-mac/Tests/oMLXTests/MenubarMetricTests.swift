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

    func testScopedSnapshotsMapLiveAndAggregateData() throws {
        let stats = try decodeStats(
            """
            {
              "total_prompt_tokens": 12000,
              "total_cached_tokens": 3000,
              "cache_efficiency": 25.0,
              "avg_prefill_tps": 512.3,
              "avg_generation_tps": 48.7,
              "total_requests": 9,
              "active_models": {
                "models": [{
                  "id": "model-a",
                  "prefilling": [{"processed": 3, "total": 10, "speed": 300.0}],
                  "generating": [{"generated_tokens": 5, "tokens_per_second": 40.0}]
                }]
              }
            }
            """
        )

        let live = try XCTUnwrap(MenubarMetricsStore.snapshot(for: .live, from: stats))
        XCTAssertEqual(live.rates, MetricRates(promptTps: 300, generationTps: 40))
        XCTAssertNil(live.totalPromptTokens)
        XCTAssertNotNil(live.liveActivity)

        for kind in [MenubarMetricsStore.Kind.average, .alltime] {
            let aggregate = try XCTUnwrap(MenubarMetricsStore.snapshot(for: kind, from: stats))
            XCTAssertEqual(aggregate.rates, MetricRates(promptTps: 512.3, generationTps: 48.7))
            XCTAssertEqual(aggregate.totalPromptTokens, 12000)
            XCTAssertEqual(aggregate.totalCachedTokens, 3000)
            XCTAssertEqual(aggregate.cacheEfficiency, 25.0)
            XCTAssertEqual(aggregate.totalRequests, 9)
            XCTAssertNil(aggregate.liveActivity)
        }
    }

    func testUnavailableSnapshotKeepsUnknownValuesDistinctFromIdle() {
        XCTAssertNil(MenubarMetricsStore.snapshot(for: .live, from: nil))
        XCTAssertTrue(ServingStatsSnapshot.unavailable.isUnavailable)

        let idle = ServingStatsSnapshot(
            rates: MetricRates(promptTps: 0, generationTps: 0),
            totalPromptTokens: nil,
            totalCachedTokens: nil,
            cacheEfficiency: nil,
            totalRequests: nil,
            liveActivity: nil
        )
        XCTAssertFalse(idle.isUnavailable)
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
            live: ServingStatsSnapshot(
                rates: MetricRates(promptTps: 100, generationTps: 10),
                totalPromptTokens: nil,
                totalCachedTokens: nil,
                cacheEfficiency: nil,
                totalRequests: nil,
                liveActivity: nil
            ),
            average: ServingStatsSnapshot(
                rates: MetricRates(promptTps: 200, generationTps: 20),
                totalPromptTokens: 300,
                totalCachedTokens: 40,
                cacheEfficiency: 13.3,
                totalRequests: 2,
                liveActivity: nil
            ),
            alltime: nil,
            serverRunning: true
        )

        XCTAssertTrue(store.serverIsRunning)
        XCTAssertEqual(store.snapshot(for: .live).rates?.promptTps, 100)
        XCTAssertEqual(store.snapshot(for: .average).rates?.generationTps, 20)
        XCTAssertEqual(store.snapshot(for: .average).totalRequests, 2)
        XCTAssertTrue(store.snapshot(for: .alltime).isUnavailable)
        // Unknown readings still roll a 0 so the graph timeline stays contiguous.
        XCTAssertEqual(store.history[.alltime]?.promptTps, [0])
        XCTAssertEqual(store.history[.live]?.promptTps, [100])
        XCTAssertEqual(store.history[.live]?.generationTps, [10])
    }

    func testMarkServerStoppedBlanksRatesButFreezesHistory() {
        let store = MenubarMetricsStore()
        store.applyTick(
            live: ServingStatsSnapshot(
                rates: MetricRates(promptTps: 100, generationTps: 10),
                totalPromptTokens: nil,
                totalCachedTokens: nil,
                cacheEfficiency: nil,
                totalRequests: nil,
                liveActivity: nil
            ),
            average: ServingStatsSnapshot(
                rates: MetricRates(promptTps: 200, generationTps: 20),
                totalPromptTokens: nil,
                totalCachedTokens: nil,
                cacheEfficiency: nil,
                totalRequests: nil,
                liveActivity: nil
            ),
            alltime: ServingStatsSnapshot(
                rates: MetricRates(promptTps: 300, generationTps: 30),
                totalPromptTokens: nil,
                totalCachedTokens: nil,
                cacheEfficiency: nil,
                totalRequests: nil,
                liveActivity: nil
            ),
            serverRunning: true
        )

        store.markServerStopped()

        XCTAssertFalse(store.serverIsRunning)
        XCTAssertTrue(store.snapshot(for: .live).isUnavailable)
        XCTAssertTrue(store.snapshot(for: .average).isUnavailable)
        XCTAssertTrue(store.snapshot(for: .alltime).isUnavailable)
        XCTAssertEqual(store.history[.live]?.promptTps, [100])
        XCTAssertEqual(store.history[.alltime]?.generationTps, [30])
    }

    func testLiveSnapshotRetainsRequestActivity() throws {
        let stats = try decodeStats(
            """
            {
              "active_models": {
                "models": [{
                  "id": "model-a",
                  "generating": [{
                    "generated_tokens": 128,
                    "tokens_per_second": 42.1,
                    "elapsed_seconds": 3
                  }]
                }],
                "total_waiting_requests": 2
              }
            }
            """
        )

        let live = try XCTUnwrap(MenubarMetricsStore.snapshot(for: .live, from: stats))
        XCTAssertEqual(live.rates, MetricRates(promptTps: 0, generationTps: 42.1))
        XCTAssertEqual(live.liveActivity?.groups.first?.requests.first?.title, "GEN 42.1 tok/s")
        XCTAssertEqual(live.liveActivity?.queuedRequestCount, 2)
    }

    func testPrefillTitlesCompactOnlyProgressCounts() throws {
        let stats = try decodeStats(
            """
            {
              "active_models": {
                "models": [{
                  "id": "model-a",
                  "prefilling": [
                    {"processed": 999, "total": 1000, "speed": 0},
                    {"processed": 1024, "total": 1700, "speed": 0},
                    {"processed": 10000, "total": 12400, "speed": 0}
                  ],
                  "generating": [{"generated_tokens": 9999, "tokens_per_second": 0}]
                }]
              }
            }
            """
        )
        let requests = try XCTUnwrap(
            MenubarMetricsStore.snapshot(for: .live, from: stats)?.liveActivity?.groups.first?.requests
        )

        XCTAssertEqual(requests[0].title, "100% · 999 / 1.0k tk")
        XCTAssertEqual(requests[1].title, "60% · 1.0k / 1.7k tk")
        XCTAssertEqual(requests[2].title, "81% · 10k / 12.4k tk")
        XCTAssertEqual(requests[3].title, "9,999 tk")
    }

    func testDetailedServingStatsPopoversKeepOneActivityBlockAndScopeSpecificTerminalDetail() {
        let live = ServingStatsPresentation.popoverScope(for: .live)
        let average = ServingStatsPresentation.popoverScope(for: .average)
        let alltime = ServingStatsPresentation.popoverScope(for: .alltime)

        XCTAssertEqual([live.kind, average.kind, alltime.kind], [.live, .average, .alltime])
        XCTAssertEqual(
            [live.activityGraphCount, average.activityGraphCount, alltime.activityGraphCount],
            [2, 2, 2]
        )
        XCTAssertTrue(live.showsThroughput)
        XCTAssertTrue(average.showsThroughput)
        XCTAssertTrue(alltime.showsThroughput)
        XCTAssertEqual(live.terminalDetail, .currentRequests)
        XCTAssertEqual(average.terminalDetail, .scalarMetrics)
        XCTAssertEqual(alltime.terminalDetail, .scalarMetrics)
        XCTAssertTrue(live.showsTerminalDetailHeader)
        XCTAssertFalse(average.showsTerminalDetailHeader)
        XCTAssertFalse(alltime.showsTerminalDetailHeader)
    }

    func testServingStatsMenuCompositionKeepsLiveDetailAndHeaderlessScalarSummaries() {
        XCTAssertEqual(
            ServingStatsPresentation.menuComposition(serverIsRunning: false),
            .serverOff
        )

        guard case .sections(let sections) = ServingStatsPresentation.menuComposition(
            serverIsRunning: true
        ) else {
            return XCTFail("running server must compose the serving stats menu")
        }

        XCTAssertEqual(sections.map(\.kind), [.live, .average, .alltime])
        XCTAssertEqual(sections.map(\.showsThroughput), [true, true, true])
        XCTAssertEqual(sections.map(\.activityGraphCount), [2, 0, 0])
        XCTAssertEqual(
            sections.map(\.terminalDetail),
            [.currentRequests, .scalarMetrics, .scalarMetrics]
        )
        XCTAssertEqual(sections.map(\.showsTerminalDetailHeader), [true, false, false])
    }

    func testMenubarStatsPanelLayoutKeepsServingAndSystemContentAligned() {
        XCTAssertEqual(MenubarStatsPanelLayout.width, 270)
        XCTAssertEqual(MenubarStatsPanelLayout.horizontalInset, 14)
        XCTAssertEqual(MenubarStatsPanelLayout.verticalInset, 8)
        XCTAssertEqual(MenubarStatsPanelLayout.contentWidth, 242)
    }

    func testCurrentRequestMarkersFollowRequestKind() {
        XCTAssertEqual(CurrentRequestMarker.marker(for: .prefill), .prefill)
        XCTAssertEqual(CurrentRequestMarker.marker(for: .generating), .generating)
        XCTAssertNil(CurrentRequestMarker.marker(for: .nonStreaming))
        XCTAssertEqual(CurrentRequestMarker.prefill.abbreviation, "PP")
        XCTAssertEqual(CurrentRequestMarker.generating.abbreviation, "TG")
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

    func testRateReadoutsUseSharedFormattingAndSurfaceSpecificUnitSpacing() {
        XCTAssertEqual(MenubarMetricGlyph.formatTps(nil), "–")
        XCTAssertEqual(MenubarMetricGlyph.formatTps(.infinity), "–")
        XCTAssertEqual(ServingStatsPresentation.popoverTps(nil), "–")
        XCTAssertEqual(ServingStatsPresentation.popoverTps(.infinity), "–")

        for value in [1.0, 100, 9_999, 10_000] {
            let expected = ActivityFormat.rate(value)
            XCTAssertEqual(MenubarMetricGlyph.formatTps(value), "\(expected)tk/s")
            XCTAssertEqual(ServingStatsPresentation.popoverTps(value), "\(expected) tk/s")
        }
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

/// Coverage for the menubar diagnostic log the hidden-icon alert's View Log
/// button and `omlx diagnose menubar` both read. Every case redirects
/// `MenubarLog.url` at a temp file so the user's real log is never touched.
@MainActor
final class MenubarLogTests: XCTestCase {

    private var tempDirectory: URL!
    private var originalURL: URL!

    override func setUp() {
        super.setUp()
        originalURL = MenubarLog.url
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MenubarLogTests-\(UUID().uuidString)", isDirectory: true)
        MenubarLog.url = tempDirectory.appendingPathComponent("menubar.log")
    }

    override func tearDown() {
        MenubarLog.url = originalURL
        try? FileManager.default.removeItem(at: tempDirectory)
        super.tearDown()
    }

    private func contents() throws -> String {
        try String(contentsOf: MenubarLog.url, encoding: .utf8)
    }

    func testWriteCreatesTheFileAndAppends() throws {
        MenubarLog.write("first")
        MenubarLog.write("second")

        let lines = try contents().split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 2)
        XCTAssertTrue(lines[0].hasSuffix(" first"), "unexpected line: \(lines[0])")
        XCTAssertTrue(lines[1].hasSuffix(" second"), "unexpected line: \(lines[1])")
        // Timestamp prefix is what makes a shared log readable — assert the
        // shape rather than a literal date.
        XCTAssertNotNil(
            lines[0].range(of: #"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "#, options: .regularExpression)
        )
    }

    func testWriteTrimsRunawayFileAndKeepsWholeLines() throws {
        try FileManager.default.createDirectory(
            at: tempDirectory,
            withIntermediateDirectories: true
        )
        // One long line, then filler past the trim threshold: the tail cut
        // lands mid-line, so the partial fragment has to be dropped.
        let filler = String(repeating: "x", count: 511) + "\n"
        var seed = "STALE-HEAD " + filler
        while seed.utf8.count < MenubarLog.maxBytes + 4096 {
            seed += filler
        }
        try seed.write(to: MenubarLog.url, atomically: true, encoding: .utf8)

        MenubarLog.write("after trim")

        let text = try contents()
        let size = text.utf8.count
        XCTAssertLessThanOrEqual(size, MenubarLog.keepBytes + 256)
        XCTAssertFalse(text.contains("STALE-HEAD"), "trim kept the head instead of the tail")
        XCTAssertTrue(text.hasSuffix("after trim\n"))
        for line in text.split(separator: "\n").dropLast() {
            XCTAssertEqual(line.count, 511, "trim left a partial line: \(line.prefix(32))…")
        }
    }

    func testOpenSeedsTheFileWhenMissing() throws {
        XCTAssertFalse(FileManager.default.fileExists(atPath: MenubarLog.url.path))
        // Only the seeding half is asserted; `open` also hands the URL to
        // NSWorkspace, which is a no-op worth nothing in a test run.
        MenubarLog.write("log opened before any probe ran")
        XCTAssertTrue(FileManager.default.fileExists(atPath: MenubarLog.url.path))
        XCTAssertTrue(try contents().hasSuffix("log opened before any probe ran\n"))
    }
}
