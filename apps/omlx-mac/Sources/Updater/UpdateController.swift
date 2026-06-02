// Updates section view-model.
//
// Drives the AppView's Status screen: the check state (idle / checking /
// available), the channel (Stable / Beta / Nightly), and two background
// prefs (autoCheck + autoDownload). Channel + prefs persist to
// `~/Library/Application Support/oMLX/update-prefs.json` so they survive
// a relaunch.
//
// Update mechanism: GitHub Releases is the single source of truth. The
// PyObjC menubar app shipped this pattern; the Swift app uses the same
// flow via `ReleasesChecker` + `AppUpdater`. No appcast XML, no EdDSA
// keys. Apple's notarization stapled to each .dmg is the trust boundary.

import AppKit
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
    let notes: String
    let htmlURL: URL
    let dmgURL: URL?
}

@MainActor
final class UpdateController: ObservableObject {
    enum CheckState: Equatable, Sendable {
        case idle(lastChecked: Date?)
        case checking
        case downloading(percent: Int)
        case available(AvailableUpdate)
        case ready(AvailableUpdate)
    }

    @Published private(set) var state: CheckState = .idle(lastChecked: nil)
    @Published private(set) var lastError: String?
    @Published var channel: UpdateChannel {
        didSet { if !suspendPersist { persist() } }
    }
    @Published var autoCheck: Bool {
        didSet {
            guard !suspendPersist else { return }
            persist()
            if autoCheck {
                backgroundCheck()
                scheduleBackgroundChecker()
            } else {
                backgroundTimer?.invalidate()
                backgroundTimer = nil
            }
        }
    }
    @Published var autoDownload: Bool {
        didSet { if !suspendPersist { persist() } }
    }

    private let storeURL: URL
    private let currentVersion: String
    private var suspendPersist = true
    private var checkTask: Task<Void, Never>?
    private var updater: AppUpdater?
    private var backgroundTimer: Timer?

    init(
        storeURL: URL = AppConfig.appSupportURL().appendingPathComponent("update-prefs.json"),
        currentVersion: String = Bundle.main.shortVersionString
    ) {
        self.storeURL = storeURL
        self.currentVersion = currentVersion
        let prefs = Self.readPrefs(from: storeURL) ?? Prefs(
            channel: .stable, autoCheck: true, autoDownload: false
        )
        self.channel = prefs.channel
        self.autoCheck = prefs.autoCheck
        self.autoDownload = prefs.autoDownload
        self.suspendPersist = false
    }

    /// Idempotent. Call once after AppDelegate stands up so we clean up any
    /// staged bundle from a prior session and (when enabled) kick off a
    /// background check.
    func bootstrap() {
        AppUpdater.cleanupStaged()
        if autoCheck {
            backgroundCheck()
            scheduleBackgroundChecker()
        }
    }

    /// User-initiated check.
    func checkForUpdates() {
        checkTask?.cancel()
        state = .checking
        lastError = nil
        checkTask = Task { [weak self] in
            guard let self else { return }
            await self.runCheck(userInitiated: true)
        }
    }

    /// One-button "Install & Restart" — matches the PyObjC menubar's
    /// flow. When the state is `.available`, kick off the download and
    /// auto-finish into `.ready`, then swap + terminate from the
    /// `onReady` callback below. When the state is already `.ready`
    /// (auto-download completed in the background), swap immediately.
    func installAndRestart() {
        switch state {
        case .available(let info):
            guard let dmg = info.dmgURL else {
                lastError = String(localized: "update.error.no_dmg",
                                   defaultValue: "No installable DMG was attached to this release.",
                                   comment: "Shown when the release has no matching DMG asset")
                return
            }
            startDownload(info: info, dmgURL: dmg, autoInstall: true)
        case .ready:
            performSwap()
        default:
            break
        }
    }

    private func performSwap() {
        if AppUpdater.performSwapAndRelaunch() {
            NSApp.terminate(nil)
        } else {
            lastError = String(localized: "update.error.swap_failed",
                               defaultValue: "Could not find the staged update. Try downloading again.",
                               comment: "Shown when the swap script can't find the staged bundle")
            state = .idle(lastChecked: Date())
        }
    }

    // MARK: - Internals

    private func backgroundCheck() {
        checkTask?.cancel()
        checkTask = Task { [weak self] in
            guard let self else { return }
            await self.runCheck(userInitiated: false)
        }
    }

    private func scheduleBackgroundChecker() {
        backgroundTimer?.invalidate()
        // Re-check every 24 h while the app is running.
        backgroundTimer = Timer.scheduledTimer(withTimeInterval: 24 * 3600, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.autoCheck { self.backgroundCheck() }
            }
        }
    }

    private func runCheck(userInitiated: Bool) async {
        do {
            let result = try await ReleasesChecker.check(
                currentVersion: currentVersion,
                channel: channel
            )
            await MainActor.run {
                if let release = result {
                    let info = AvailableUpdate(
                        version: release.version,
                        sizeText: release.dmgSize.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) },
                        notes: release.notes,
                        htmlURL: release.htmlURL,
                        dmgURL: release.dmgURL
                    )
                    self.state = .available(info)
                    if self.autoDownload, let dmg = info.dmgURL, !userInitiated {
                        self.startDownload(info: info, dmgURL: dmg, autoInstall: false)
                    }
                } else {
                    self.state = .idle(lastChecked: Date())
                }
            }
        } catch is CancellationError {
            // Quietly drop — a fresh check is in flight or app is shutting down.
        } catch {
            NSLog("oMLX: update check failed — %@", String(describing: error))
            await MainActor.run {
                if userInitiated {
                    self.lastError = String(describing: error)
                }
                self.state = .idle(lastChecked: Date())
            }
        }
    }

    private func startDownload(info: AvailableUpdate, dmgURL: URL, autoInstall: Bool) {
        let updater = AppUpdater(
            dmgURL: dmgURL,
            version: info.version,
            onProgress: { [weak self] progress in
                guard let self else { return }
                switch progress {
                case .starting, .mounting, .staging:
                    if case .downloading = self.state { /* keep showing percent */ } else {
                        self.state = .downloading(percent: 0)
                    }
                case .downloading(let pct, _, _):
                    self.state = .downloading(percent: pct)
                case .ready:
                    self.state = .ready(info)
                }
            },
            onError: { [weak self] err in
                guard let self else { return }
                self.lastError = String(describing: err)
                self.state = .available(info)
                self.updater = nil
            },
            onReady: { [weak self] in
                guard let self else { return }
                self.state = .ready(info)
                self.updater = nil
                if autoInstall {
                    self.performSwap()
                }
            }
        )
        self.updater = updater
        updater.start()
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

private extension Bundle {
    var shortVersionString: String {
        (infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0.0.0"
    }
}
