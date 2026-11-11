// Updates section view-model.
//
// Drives the AppView's Status screen: the check state (idle / checking /
// available), the channel (Stable / Release Candidate / Dev), and three background
// prefs (autoCheck + backgroundDownload + autoNotify). Channel + prefs persist to
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
@Observable
final class UpdateController {
    static let stateDidChangeNotification = Notification.Name("OMLXUpdateControllerStateDidChange")

    enum CheckState: Equatable, Sendable {
        case idle(lastChecked: Date?)
        case checking
        case downloading(percent: Int)
        case available(AvailableUpdate)
        case ready(AvailableUpdate)
    }

    private(set) var state: CheckState = .idle(lastChecked: nil) {
        didSet {
            NotificationCenter.default.post(
                name: Self.stateDidChangeNotification,
                object: self
            )
        }
    }
    private(set) var lastError: String?
    private(set) var confirmationUpdate: AvailableUpdate?
    var channel: UpdateChannel {
        didSet {
            guard !suspendPersist else { return }
            persist()
            checkForUpdates()
        }
    }
    var autoCheck: Bool {
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
    var autoNotify: Bool {
        didSet { if !suspendPersist { persist() } }
    }
    /// Stages a found update so installing only needs a relaunch.
    /// It never installs on its own.
    var backgroundDownload: Bool {
        didSet {
            guard !suspendPersist else { return }
            persist()
            if backgroundDownload {
                if case .available(let info) = state, let dmg = info.dmgURL, updater == nil {
                    startDownload(info: info, dmgURL: dmg, autoInstall: false,
                                  notifyWhenDone: autoNotify)
                }
            } else if let info = backgroundDownloadInfo {
                cancelDownload()
                state = .available(info)
            }
        }
    }

    private let storeURL: URL
    private let currentVersion: String
    @ObservationIgnored
    private var suspendPersist = true
    @ObservationIgnored
    private var checkTask: Task<Void, Never>?
    @ObservationIgnored
    private var updater: AppUpdater?
    /// Bumped whenever `updater` is replaced or cancelled.
    /// A cancelled updater still calls `onError`, so older callbacks are dropped.
    @ObservationIgnored
    private var downloadGeneration = 0
    @ObservationIgnored
    private var backgroundDownloadInfo: AvailableUpdate?
    @ObservationIgnored
    private var backgroundTimer: Timer?
    @ObservationIgnored
    private var terminateForUpdate: (@MainActor () -> Void)?
    @ObservationIgnored
    private var presentUpdateConfirmation: (@MainActor () -> Void)?
    @ObservationIgnored
    private var deferredPromptVersion: String?

    init(
        storeURL: URL = AppConfig.appSupportURL().appendingPathComponent("update-prefs.json"),
        currentVersion: String = Bundle.main.shortVersionString
    ) {
        self.storeURL = storeURL
        self.currentVersion = currentVersion
        let prefs = Self.readPrefs(from: storeURL) ?? Prefs(
            channel: .stable, autoCheck: true, autoNotify: false, backgroundDownload: true
        )
        self.channel = prefs.channel
        self.autoCheck = prefs.autoCheck
        self.autoNotify = prefs.autoNotify
        self.backgroundDownload = prefs.backgroundDownload
        self.deferredPromptVersion = prefs.deferredPromptVersion
        self.suspendPersist = false
    }

    /// Idempotent. Call once after AppDelegate stands up so we drop a stale
    /// staged bundle from a prior session and (when enabled) kick off a
    /// background check.
    func bootstrap() {
        // Only a check surfaces a staged bundle, so keep one only when checks run.
        AppUpdater.cleanupStaged(currentVersion: currentVersion, keepPending: autoCheck)
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
        // The check can pick another release, e.g. after a channel change.
        cancelDownload()
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
        lastError = nil
        installAndRestart(matchingVersion: info.version)
        // A failure before the download starts keeps the sheet up with the error.
        if lastError == nil {
            confirmationUpdate = nil
        }
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
            // NSApp.terminate is ignored inside the sheet button's action.
            DispatchQueue.main.async { [weak self] in self?.performSwap(version: info.version) }
        default:
            break
        }
    }

    private func performSwap(version: String) {
        // A check can replace or discard the staged bundle before a deferred swap runs.
        guard case .ready(let ready) = state, ready.version == version else { return }
        if AppUpdater.performSwapAndRelaunch() {
            // NSApp.terminate returns without quitting while a sheet is attached.
            for sheet in NSApp.windows where sheet.sheetParent != nil {
                sheet.sheetParent?.endSheet(sheet)
            }
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
        // Leave a running download alone; the next daily check catches up.
        guard updater == nil else { return }
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
            try Task.checkCancellation()
            // A download confirmed while the check ran owns the state.
            guard updater == nil else { return }
            lastError = nil
            guard let release = result else {
                state = .idle(lastChecked: Date())
                await AppUpdater.discardStaged()
                return
            }
            let info = AvailableUpdate(
                version: release.version,
                sizeText: release.dmgSize.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) },
                notes: release.notes,
                htmlURL: release.htmlURL,
                dmgURL: release.dmgURL
            )
            var stagedReady = false
            if case .ready(let ready) = state, ready.version == info.version {
                // Verified when it was staged in this session.
                stagedReady = true
            } else {
                // Leave `.ready` first so Install cannot swap a bundle that is being checked or removed.
                state = .available(info)
                stagedReady = await AppUpdater.stagedBundleMatches(version: info.version)
                try Task.checkCancellation()
                guard updater == nil else { return }
                if !stagedReady {
                    await AppUpdater.discardStaged()
                    try Task.checkCancellation()
                    guard updater == nil else { return }
                }
            }
            if stagedReady {
                state = .ready(info)
                if userInitiated || autoNotify {
                    presentConfirmation(for: info, automatic: !userInitiated)
                }
                return
            }
            if userInitiated {
                if info.dmgURL == nil {
                    lastError = noInstallableDMGMessage
                } else {
                    presentConfirmation(for: info, automatic: false)
                }
            } else if backgroundDownload, confirmationUpdate == nil, let dmg = info.dmgURL {
                // The notes are shown once the bundle is staged.
                startDownload(info: info, dmgURL: dmg, autoInstall: false, notifyWhenDone: autoNotify)
            } else if autoNotify, info.dmgURL != nil {
                presentConfirmation(for: info, automatic: true)
            }
        } catch is CancellationError {
            // Quietly drop — a fresh check is in flight or app is shutting down.
        } catch {
            NSLog("oMLX: update check failed — %@", String(describing: error))
            if userInitiated {
                lastError = String(describing: error)
            }
            // A failed check says nothing about an update that is already staged.
            if case .ready = state { return }
            state = .idle(lastChecked: Date())
        }
    }

    /// `autoInstall` downloads were confirmed by the user.
    /// They may use a metered network, report errors, and swap when staged.
    /// Background downloads stay quiet and only stage the bundle.
    private func startDownload(
        info: AvailableUpdate,
        dmgURL: URL,
        autoInstall: Bool,
        notifyWhenDone: Bool = false
    ) {
        cancelDownload()
        backgroundDownloadInfo = autoInstall ? nil : info
        let generation = downloadGeneration
        let updater = AppUpdater(
            dmgURL: dmgURL,
            version: info.version,
            allowsMeteredNetwork: autoInstall,
            onProgress: { [weak self] progress in
                guard let self, generation == self.downloadGeneration else { return }
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
                guard let self, generation == self.downloadGeneration else { return }
                if autoInstall {
                    self.lastError = String(describing: err)
                }
                self.state = .available(info)
                self.updater = nil
                self.backgroundDownloadInfo = nil
                if autoInstall {
                    // The sheet closed when the user confirmed; reopen it with the error.
                    self.presentConfirmation(for: info, automatic: false)
                } else if notifyWhenDone {
                    self.presentConfirmation(for: info, automatic: true)
                }
            },
            onReady: { [weak self] in
                guard let self, generation == self.downloadGeneration else { return }
                self.state = .ready(info)
                self.lastError = nil
                self.updater = nil
                self.backgroundDownloadInfo = nil
                if autoInstall {
                    self.performSwap(version: info.version)
                } else if notifyWhenDone {
                    self.presentConfirmation(for: info, automatic: true)
                }
            }
        )
        self.updater = updater
        updater.start()
    }

    private func cancelDownload() {
        updater?.cancel()
        updater = nil
        backgroundDownloadInfo = nil
        downloadGeneration += 1
    }

    private func presentConfirmation(for info: AvailableUpdate, automatic: Bool) {
        if automatic, deferredPromptVersion == info.version {
            return
        }
        // Keep the last install failure so the reopened sheet still shows it.
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
        var backgroundDownload: Bool
        var deferredPromptVersion: String?

        enum CodingKeys: String, CodingKey {
            case channel
            case autoCheck
            case autoNotify
            case autoDownload
            case backgroundDownload
            case deferredPromptVersion
        }

        init(
            channel: UpdateChannel,
            autoCheck: Bool,
            autoNotify: Bool,
            backgroundDownload: Bool,
            deferredPromptVersion: String? = nil
        ) {
            self.channel = channel
            self.autoCheck = autoCheck
            self.autoNotify = autoNotify
            self.backgroundDownload = backgroundDownload
            self.deferredPromptVersion = deferredPromptVersion
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.channel = try container.decodeIfPresent(UpdateChannel.self, forKey: .channel) ?? .stable
            self.autoCheck = try container.decodeIfPresent(Bool.self, forKey: .autoCheck) ?? true
            self.autoNotify = try container.decodeIfPresent(Bool.self, forKey: .autoNotify)
                ?? container.decodeIfPresent(Bool.self, forKey: .autoDownload)
                ?? false
            // The legacy `autoDownload` key was this same background download toggle.
            self.backgroundDownload = try container.decodeIfPresent(Bool.self, forKey: .backgroundDownload)
                ?? container.decodeIfPresent(Bool.self, forKey: .autoDownload)
                ?? true
            self.deferredPromptVersion = try container.decodeIfPresent(String.self, forKey: .deferredPromptVersion)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(channel, forKey: .channel)
            try container.encode(autoCheck, forKey: .autoCheck)
            try container.encode(autoNotify, forKey: .autoNotify)
            try container.encode(backgroundDownload, forKey: .backgroundDownload)
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
            backgroundDownload: backgroundDownload,
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
