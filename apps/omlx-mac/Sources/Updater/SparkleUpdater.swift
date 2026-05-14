// PR 11 — Sparkle wrapper.
//
// Wraps `SPUStandardUpdaterController` so the rest of the app talks to a
// single observable surface (`UpdateController`) regardless of whether the
// Sparkle SwiftPM dep is resolved at build time. The resolution is gated on
// `#if canImport(Sparkle)` to keep the build green if the dep is absent
// (e.g. air-gapped CI cache miss); in that case `UpdateController` falls
// back to its 1.4 s simulator from PR 7.
//
// Channel routing
//   The user's channel selection (UpdateController.channel) is forwarded to
//   Sparkle via `SPUUpdaterDelegate.allowedChannels(for:)`. The appcast XML
//   marks individual `<item>`s with `<sparkle:channel>` and Sparkle filters
//   accordingly. Channel set is also constrained by Info.plist's
//   `SUAllowedChannels` so injected appcasts can't escalate channels.
//
// Server orphan story
//   Sparkle 2.x installs by sending `[NSApp terminate:]`, which fires our
//   `applicationWillTerminate` (AppDelegate.swift). That handler already
//   does SIGTERM → wait ≤8s → SIGKILL on the Python child + a `reapSync`
//   belt-and-suspenders, so no Sparkle-specific glue is required.

import Foundation

#if canImport(Sparkle)
import Sparkle

@MainActor
final class SparkleUpdater: NSObject, ObservableObject {
    enum CheckResult: Sendable {
        case upToDate
        case available(version: String, sizeText: String?)
        case error(String)
    }

    private let updaterController: SPUStandardUpdaterController
    private let driver: SparkleDriver
    private let onResult: @MainActor (CheckResult) -> Void

    /// The most recent feed result, also forwarded synchronously through
    /// `onResult` so `UpdateController` can update its `state`.
    @Published private(set) var lastResult: CheckResult?

    /// Bound at init time. Sparkle reads this on every check so changing
    /// channels takes effect on the next call (no relaunch needed).
    var channel: UpdateChannel = .stable {
        didSet { driver.channel = channel }
    }

    var automaticallyChecksForUpdates: Bool {
        get { updaterController.updater.automaticallyChecksForUpdates }
        set { updaterController.updater.automaticallyChecksForUpdates = newValue }
    }

    var automaticallyDownloadsUpdates: Bool {
        get { updaterController.updater.automaticallyDownloadsUpdates }
        set { updaterController.updater.automaticallyDownloadsUpdates = newValue }
    }

    init(onResult: @escaping @MainActor (CheckResult) -> Void) {
        let driver = SparkleDriver()
        self.driver = driver
        self.onResult = onResult
        self.updaterController = SPUStandardUpdaterController(
            startingUpdater: true,
            updaterDelegate: driver,
            userDriverDelegate: driver
        )
        super.init()
        driver.deliver = { [weak self] result in
            // Sparkle invokes delegate callbacks on the main thread for
            // SPUStandardUpdaterController, so we're already on MainActor.
            MainActor.assumeIsolated {
                guard let self else { return }
                self.lastResult = result
                self.onResult(result)
            }
        }
    }

    /// User-initiated check. Equivalent to clicking "Check Now" — Sparkle
    /// will pop its own confirmation/install UI when an update is found.
    func checkForUpdates() {
        updaterController.checkForUpdates(nil)
    }

    /// Background check that doesn't pop UI unless an update is available.
    func backgroundCheckForUpdates() {
        updaterController.updater.checkForUpdatesInBackground()
    }

    /// Routes through Sparkle's standard panel. Sparkle 2.x dedupes against
    /// any in-flight or staged download, so the user lands on the install
    /// confirmation without re-fetching the feed. Bypassing that one
    /// confirmation requires implementing a custom `SPUUserDriver`; that's
    /// a feature, not a wiring fix.
    func installAndRestart() {
        updaterController.checkForUpdates(nil)
    }
}

/// `SPUUpdaterDelegate` + `SPUStandardUserDriverDelegate` glue. Kept private
/// so SparkleUpdater stays the single public surface.
private final class SparkleDriver: NSObject, SPUUpdaterDelegate, SPUStandardUserDriverDelegate {
    var channel: UpdateChannel = .stable

    /// Set by SparkleUpdater right after construction. We forward feed
    /// results through this so UpdateController.state can react.
    var deliver: ((SparkleUpdater.CheckResult) -> Void)?

    func allowedChannels(for updater: SPUUpdater) -> Set<String> {
        switch channel {
        case .stable:  return ["stable"]
        case .beta:    return ["stable", "beta"]
        case .nightly: return ["stable", "beta", "nightly"]
        }
    }

    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        let size: String? = item.contentLength > 0
            ? ByteCountFormatter.string(fromByteCount: Int64(item.contentLength), countStyle: .file)
            : nil
        deliver?(.available(version: item.displayVersionString, sizeText: size))
    }

    func updaterDidNotFindUpdate(_ updater: SPUUpdater) {
        deliver?(.upToDate)
    }

    func updater(_ updater: SPUUpdater, didAbortWithError error: any Error) {
        let nsError = error as NSError
        // Sparkle reports the user cancelling a check, or the feed
        // returning "no update", as SUError.noUpdateError (1001) inside
        // its own error domain. Treat both as "up to date" — they aren't
        // failures.
        if nsError.domain == SUSparkleErrorDomain,
           nsError.code == SUError.noUpdateError.rawValue {
            deliver?(.upToDate)
        } else {
            deliver?(.error(error.localizedDescription))
        }
    }
}

#else

// Stub shim used when the Sparkle SPM dep failed to resolve (e.g. an air-
// gapped CI cache miss). UpdateController treats absence as a signal to
// stay on its built-in 1.4 s simulator from PR 7.
@MainActor
final class SparkleUpdater: NSObject, ObservableObject {
    enum CheckResult: Sendable {
        case upToDate
        case available(version: String, sizeText: String?)
        case error(String)
    }

    var channel: UpdateChannel = .stable
    var automaticallyChecksForUpdates: Bool = true
    var automaticallyDownloadsUpdates: Bool = false

    init(onResult: @escaping @MainActor (CheckResult) -> Void) {
        _ = onResult
    }

    func checkForUpdates() {}
    func backgroundCheckForUpdates() {}
    func installAndRestart() {}
}

#endif
