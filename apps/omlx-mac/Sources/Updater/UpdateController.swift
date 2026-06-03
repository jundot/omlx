// Updates section view-model.
//
// Drives the AppView's Status screen: the check state (idle / checking /
// available), the channel (Stable / Release Candidate / Dev), and two background
// prefs (autoCheck + autoNotify). Channel + prefs persist to
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
    case stable
    case releaseCandidate = "release_candidate"
    case dev

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .stable:
            return String(localized: "update.channel.stable",
                          defaultValue: "Stable",
                          comment: "Display name for the Stable update channel")
        case .releaseCandidate:
            return String(localized: "update.channel.release_candidate",
                          defaultValue: "Release Candidate",
                          comment: "Display name for the Release Candidate update channel")
        case .dev:
            return String(localized: "update.channel.dev",
                          defaultValue: "Dev",
                          comment: "Display name for the Dev update channel")
        }
    }

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        switch raw {
        case "stable":
            self = .stable
        case "release_candidate", "beta":
            self = .releaseCandidate
        case "dev", "nightly":
            self = .dev
        default:
            self = .stable
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

struct AvailableUpdate: Equatable, Identifiable, Sendable {
    let version: String
    let sizeText: String?
    let notes: String
    let htmlURL: URL
    let dmgURL: URL?

    var id: String { version }
}

@MainActor
final class UpdateController: ObservableObject {
    static let stateDidChangeNotification = Notification.Name("OMLXUpdateControllerStateDidChange")

    enum CheckState: Equatable, Sendable {
        case idle(lastChecked: Date?)
        case checking
        case downloading(percent: Int)
        case available(AvailableUpdate)
        case ready(AvailableUpdate)
    }

    @Published private(set) var state: CheckState = .idle(lastChecked: nil) {
        didSet {
            NotificationCenter.default.post(
                name: Self.stateDidChangeNotification,
                object: self
            )
        }
    }
    @Published private(set) var lastError: String?
    @Published private(set) var confirmationUpdate: AvailableUpdate?
    @Published var channel: UpdateChannel {
        didSet {
            guard !suspendPersist else { return }
            persist()
            checkForUpdates()
        }
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
    @Published var autoNotify: Bool {
        didSet { if !suspendPersist { persist() } }
    }

    private let storeURL: URL
    private let currentVersion: String
    private var suspendPersist = true
    private var checkTask: Task<Void, Never>?
    private var updater: AppUpdater?
    private var backgroundTimer: Timer?
    private var terminateForUpdate: (@MainActor () -> Void)?
    private var presentUpdateConfirmation: (@MainActor () -> Void)?
    private var deferredPromptVersion: String?

    init(
        storeURL: URL = AppConfig.appSupportURL().appendingPathComponent("update-prefs.json"),
        currentVersion: String = Bundle.main.shortVersionString
    ) {
        self.storeURL = storeURL
        self.currentVersion = currentVersion
        let prefs = Self.readPrefs(from: storeURL) ?? Prefs(
            channel: .stable, autoCheck: true, autoNotify: false
        )
        self.channel = prefs.channel
        self.autoCheck = prefs.autoCheck
        self.autoNotify = prefs.autoNotify
        self.deferredPromptVersion = prefs.deferredPromptVersion
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

    func setTerminateForUpdate(_ handler: @escaping @MainActor () -> Void) {
        self.terminateForUpdate = handler
    }

    func setPresentUpdateConfirmation(_ handler: @escaping @MainActor () -> Void) {
        self.presentUpdateConfirmation = handler
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

    func requestUpdateConfirmation() {
        switch state {
        case .available(let info):
            guard info.dmgURL != nil else {
                lastError = noInstallableDMGMessage
                return
            }
            presentConfirmation(for: info, automatic: false)
        case .ready(let info):
            presentConfirmation(for: info, automatic: false)
        default:
            break
        }
    }

    func dismissUpdateConfirmation() {
        confirmationUpdate = nil
    }

    func deferUpdate(_ info: AvailableUpdate) {
        deferredPromptVersion = info.version
        persist()
        confirmationUpdate = nil
    }

    func confirmUpdate(_ info: AvailableUpdate) {
        confirmationUpdate = nil
        installAndRestart(matchingVersion: info.version)
    }

    /// One-button "Install & Restart". When the state is `.available`, kick
    /// off the download and auto-finish into `.ready`, then swap + terminate
    /// from the `onReady` callback below. When the state is already `.ready`,
    /// swap immediately.
    func installAndRestart() {
        installAndRestart(matchingVersion: nil)
    }

    private func installAndRestart(matchingVersion: String?) {
        switch state {
        case .available(let info):
            guard matchingVersion == nil || matchingVersion == info.version else { return }
            guard let dmg = info.dmgURL else {
                lastError = noInstallableDMGMessage
                return
            }
            startDownload(info: info, dmgURL: dmg, autoInstall: true)
        case .ready(let info):
            guard matchingVersion == nil || matchingVersion == info.version else { return }
            performSwap()
        default:
            break
        }
    }

    private func performSwap() {
        if AppUpdater.performSwapAndRelaunch() {
            if let terminateForUpdate {
                terminateForUpdate()
            } else {
                NSApp.terminate(nil)
            }
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
                    if userInitiated {
                        if info.dmgURL == nil {
                            self.lastError = self.noInstallableDMGMessage
                        } else {
                            self.presentConfirmation(for: info, automatic: false)
                        }
                    } else if self.autoNotify, info.dmgURL != nil {
                        self.presentConfirmation(for: info, automatic: true)
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

    private func presentConfirmation(for info: AvailableUpdate, automatic: Bool) {
        if automatic, deferredPromptVersion == info.version {
            return
        }
        lastError = nil
        confirmationUpdate = info
        presentUpdateConfirmation?()
    }

    private var noInstallableDMGMessage: String {
        String(localized: "update.error.no_dmg",
               defaultValue: "No installable DMG was attached to this release.",
               comment: "Shown when the release has no matching DMG asset")
    }

    // MARK: - Persistence

    private struct Prefs: Codable {
        var channel: UpdateChannel
        var autoCheck: Bool
        var autoNotify: Bool
        var deferredPromptVersion: String?

        enum CodingKeys: String, CodingKey {
            case channel
            case autoCheck
            case autoNotify
            case autoDownload
            case deferredPromptVersion
        }

        init(
            channel: UpdateChannel,
            autoCheck: Bool,
            autoNotify: Bool,
            deferredPromptVersion: String? = nil
        ) {
            self.channel = channel
            self.autoCheck = autoCheck
            self.autoNotify = autoNotify
            self.deferredPromptVersion = deferredPromptVersion
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.channel = try container.decodeIfPresent(UpdateChannel.self, forKey: .channel) ?? .stable
            self.autoCheck = try container.decodeIfPresent(Bool.self, forKey: .autoCheck) ?? true
            self.autoNotify = try container.decodeIfPresent(Bool.self, forKey: .autoNotify)
                ?? container.decodeIfPresent(Bool.self, forKey: .autoDownload)
                ?? false
            self.deferredPromptVersion = try container.decodeIfPresent(String.self, forKey: .deferredPromptVersion)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(channel, forKey: .channel)
            try container.encode(autoCheck, forKey: .autoCheck)
            try container.encode(autoNotify, forKey: .autoNotify)
            try container.encodeIfPresent(deferredPromptVersion, forKey: .deferredPromptVersion)
        }
    }

    private static func readPrefs(from url: URL) -> Prefs? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Prefs.self, from: data)
    }

    private func persist() {
        let prefs = Prefs(
            channel: channel,
            autoCheck: autoCheck,
            autoNotify: autoNotify,
            deferredPromptVersion: deferredPromptVersion
        )
        guard let data = try? JSONEncoder().encode(prefs) else { return }
        try? FileManager.default.createDirectory(
            at: storeURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? data.write(to: storeURL, options: [.atomic])
    }
}

private extension Bundle {
    var shortVersionString: String {
        (infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0.0.0"
    }
}
