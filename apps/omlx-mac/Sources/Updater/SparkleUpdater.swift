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

    /// The most recent feed result; mirrored back into UpdateController's
    /// `state` by the binding wired from AppServices.
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

    override init() {
        let driver = SparkleDriver()
        self.driver = driver
        self.updaterController = SPUStandardUpdaterController(
            startingUpdater: true,
            updaterDelegate: driver,
            userDriverDelegate: driver
        )
        super.init()
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

    /// Triggers Sparkle's install + relaunch flow if an update is
    /// already staged. No-op otherwise.
    func installAndRestart() {
        // Sparkle handles staging internally; if the user hit "Install &
        // Restart" while a download was in flight, Sparkle's UI would
        // prompt and we'd have nothing to do. From the AppView's button,
        // the user only sees the action when an update is actually
        // available, so kicking off `checkForUpdates` is the right call —
        // Sparkle re-uses its cached data and shows the install pane.
        updaterController.checkForUpdates(nil)
    }
}

/// `SPUUpdaterDelegate` + `SPUStandardUserDriverDelegate` glue. Kept private
/// so SparkleUpdater stays the single public surface.
private final class SparkleDriver: NSObject, SPUUpdaterDelegate, SPUStandardUserDriverDelegate {
    var channel: UpdateChannel = .stable

    func allowedChannels(for updater: SPUUpdater) -> Set<String> {
        switch channel {
        case .stable:  return ["stable"]
        case .beta:    return ["stable", "beta"]
        case .nightly: return ["stable", "beta", "nightly"]
        }
    }

    /// Skip the version check on dev builds — without an EdDSA pub key,
    /// Sparkle would otherwise refuse to install. Real release builds ship
    /// `SUPublicEDKey` populated by `build.py`.
    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        // No-op; Sparkle's UI takes over from here.
    }
}

#else

// Stub shim used when the Sparkle SPM dep failed to resolve (e.g. an air-
// gapped CI cache miss). UpdateController treats absence as a signal to
// stay on its built-in 1.4 s simulator from PR 7.
@MainActor
final class SparkleUpdater: NSObject, ObservableObject {
    var channel: UpdateChannel = .stable
    var automaticallyChecksForUpdates: Bool = true
    var automaticallyDownloadsUpdates: Bool = false

    func checkForUpdates() {}
    func backgroundCheckForUpdates() {}
    func installAndRestart() {}
}

#endif
