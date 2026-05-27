// PR 7 (PR 11 update) — Updates section view-model.
//
// Drives three observable bits the AppView's Status screen renders: the
// check state (idle / checking / available), the channel (Stable / Beta /
// Nightly), and two background prefs (autoCheck + autoDownload). Channel +
// prefs persist to ~/Library/Application Support/oMLX/update-prefs.json
// so they survive a relaunch.
//
// PR 11 swaps the body of `checkForUpdates()` and `installAndRestart()` for
// SparkleUpdater calls. Channel + auto-* prefs are forwarded to Sparkle on
// every change so the user's selection takes effect on the next check.
// When the Sparkle SwiftPM dep isn't resolved (`#if !canImport(Sparkle)`),
// `SparkleUpdater` is a no-op stub and we keep the 1.4 s simulator from
// PR 7 so the screen still demos correctly.

import Foundation

enum UpdateChannel: String, Codable, CaseIterable, Identifiable, Sendable {
    case stable, beta, nightly
    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .stable:
            return String(localized: "update.channel.stable",
                          defaultValue: "Stable",
                          comment: "Display name for the Stable update channel")
        case .beta:
            return String(localized: "update.channel.beta",
                          defaultValue: "Beta",
                          comment: "Display name for the Beta update channel")
        case .nightly:
            return String(localized: "update.channel.nightly",
                          defaultValue: "Nightly",
                          comment: "Display name for the Nightly update channel")
        }
    }
}

struct AvailableUpdate: Equatable, Sendable {
    let version: String
    let sizeText: String?
}

@MainActor
final class UpdateController: ObservableObject {
    enum CheckState: Equatable, Sendable {
        case idle(lastChecked: Date?)
        case checking
        case available(AvailableUpdate)
    }

    @Published private(set) var state: CheckState = .idle(lastChecked: nil)
    @Published var channel: UpdateChannel {
        didSet { if !suspendPersist { onPrefsChanged() } }
    }
    @Published var autoCheck: Bool {
        didSet { if !suspendPersist { onPrefsChanged() } }
    }
    @Published var autoDownload: Bool {
        didSet { if !suspendPersist { onPrefsChanged() } }
    }

    private let storeURL: URL
    private var simulationTask: Task<Void, Never>?
    private var suspendPersist = true

    /// Lazily constructed so the SPM dep is only required when something
    /// actually drives an update check. The shim variant of `SparkleUpdater`
    /// is also lazy — same hook, no behavior.
    private lazy var sparkle: SparkleUpdater = {
        let s = SparkleUpdater { [weak self] result in
            self?.handleSparkleResult(result)
        }
        s.channel = channel
        s.automaticallyChecksForUpdates = autoCheck
        s.automaticallyDownloadsUpdates = autoDownload
        return s
    }()

    init(storeURL: URL = AppConfig.appSupportURL().appendingPathComponent("update-prefs.json")) {
        self.storeURL = storeURL
        let prefs = Self.readPrefs(from: storeURL) ?? Prefs(
            channel: .stable, autoCheck: true, autoDownload: false
        )
        self.channel = prefs.channel
        self.autoCheck = prefs.autoCheck
        self.autoDownload = prefs.autoDownload
        self.suspendPersist = false
    }

    /// Idempotent. Call once after AppDelegate stands up so Sparkle's
    /// background checker is wired and gets the user's stored prefs.
    func bootstrap() {
        _ = sparkle  // forces lazy init + applies prefs
        if autoCheck {
            sparkle.backgroundCheckForUpdates()
        }
    }

    func checkForUpdates() {
        #if canImport(Sparkle)
        sparkle.checkForUpdates()
        // Flip to `checking` immediately so the AppView's button shows
        // the spinner while Sparkle's panel fetches the appcast.
        // `handleSparkleResult` will flip us back to `.idle` or
        // `.available` once Sparkle's delegate fires. The timeout is a
        // belt-and-suspenders so the spinner doesn't hang forever if
        // Sparkle silently aborts (rare, but seen in dev when the feed
        // host is unreachable).
        state = .checking
        scheduleCheckTimeout(seconds: 30)
        #else
        runSimulator()
        #endif
    }

    func installAndRestart() {
        #if canImport(Sparkle)
        sparkle.installAndRestart()
        #else
        // No-op in stub builds.
        #endif
    }

    // MARK: - Internals

    /// Bridges Sparkle's delegate callbacks back into the AppView's
    /// observable state. Errors are logged but not surfaced in the pill —
    /// `CheckState` has no `.error` case yet, and the user already sees
    /// Sparkle's own error panel for user-initiated checks.
    private func handleSparkleResult(_ result: SparkleUpdater.CheckResult) {
        simulationTask?.cancel()
        switch result {
        case .upToDate:
            state = .idle(lastChecked: Date())
        case .available(let version, let sizeText):
            state = .available(AvailableUpdate(version: version, sizeText: sizeText))
        case .error(let message):
            NSLog("oMLX: update check failed — %@", message)
            state = .idle(lastChecked: Date())
        }
    }

    private func onPrefsChanged() {
        persist()
        sparkle.channel = channel
        sparkle.automaticallyChecksForUpdates = autoCheck
        sparkle.automaticallyDownloadsUpdates = autoDownload
    }

    private func runSimulator() {
        simulationTask?.cancel()
        state = .checking
        simulationTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(1400))
            guard !Task.isCancelled, let self else { return }
            self.state = .idle(lastChecked: Date())
        }
    }

    private func scheduleCheckTimeout(seconds: Double) {
        simulationTask?.cancel()
        simulationTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled, let self else { return }
            // If Sparkle hasn't reported back, flip to idle so the UI
            // doesn't sit in `checking` forever. Sparkle's own panel
            // remains on screen if a real check is in flight.
            if case .checking = self.state {
                self.state = .idle(lastChecked: Date())
            }
        }
    }

    // MARK: - Persistence

    private struct Prefs: Codable {
        var channel: UpdateChannel
        var autoCheck: Bool
        var autoDownload: Bool
    }

    private static func readPrefs(from url: URL) -> Prefs? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Prefs.self, from: data)
    }

    private func persist() {
        let prefs = Prefs(
            channel: channel,
            autoCheck: autoCheck,
            autoDownload: autoDownload
        )
        guard let data = try? JSONEncoder().encode(prefs) else { return }
        try? data.write(to: storeURL, options: [.atomic])
    }
}
