// Application delegate for the menubar-only thin shell. Sequences the
// activation policy, the server bootstrap, the menubar, and the POSIX
// signal handlers. There is no app window — the NSStatusItem is the whole
// UI.
//
// Boot flow
//   applicationWillFinishLaunching → setActivationPolicy(.regular)
//   applicationDidFinishLaunching  → load AppConfig
//                                    → resolve PythonRuntime (the host venv)
//                                    → spawn ServerProcess
//                                    → create MenubarController
//                                    → install POSIX SignalHandlers
//                                    → flip to .accessory next tick
//   applicationWillTerminate       → await server.stop()
//
// Activation-policy dance: we start .regular and flip to .accessory only
// AFTER the status item registers with WindowServer. Do NOT set
// LSUIElement=true in Info.plist — combining it with the runtime flip makes
// Tahoe ControlCenter silently block the status item (issue #725).

import AppKit
import ServiceManagement

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private(set) var server: ServerProcess?
    private var menubar: MenubarController?

    nonisolated func applicationWillFinishLaunching(_ notification: Notification) {
        // Regular policy until the status item registers; we flip to
        // Accessory after creating the menubar (next runloop tick).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Force the Dock / About-panel / alert icon to the Flyto mark. The
        // standard About panel uses NSApp.applicationIconImage; relying on the
        // compiled AppIcon.icns is fragile (Xcode 26 doesn't route the .icon
        // bundle through actool, and the Tahoe-format .icns doesn't always
        // load as the app icon on macOS 15). Setting it here works in every
        // build configuration.
        if let icon = NSImage(named: "AppLogo") {
            // A vector asset loads with no concrete size; applicationIconImage
            // renders blank unless we give it real pixel dimensions.
            icon.size = NSSize(width: 512, height: 512)
            NSApp.applicationIconImage = icon
        }

        let config = AppConfig.load()
        bootstrapServer(config: config)
        scheduleAccessoryPolicyFlip()

        // Register as a login item on first launch so the app appears in
        // System Settings > Login Items and survives reboot without relying
        // solely on the legacy LaunchAgent. Registered once, then the user owns
        // the toggle from the settings panel.
        LaunchAtLogin.registerOnceOnFirstLaunch()
    }

    private func bootstrapServer(config: AppConfig) {
        do {
            let runtime = try PythonRuntime.resolve()
            let server = ServerProcess(
                runtime: runtime,
                host: config.host,
                port: config.port,
                basePath: URL(fileURLWithPath: config.basePath, isDirectory: true)
            )
            self.server = server
            self.menubar = makeMenubar(server: server, config: config)

            // Install signal handlers BEFORE the spawn so a fast crash of the
            // parent during startup still reaps any child we managed to spawn.
            SignalHandlers.shared.install { [weak server] in
                server?.reapSync()
            }

            switch try server.start() {
            case .started, .alreadyRunning:
                break
            case .portConflict:
                // ServerProcess already posted .portConflictNotification +
                // updated state to .failed; MenubarController surfaces it on
                // the next click.
                break
            }
        } catch {
            // Surface the failure in the menubar header so the user has a
            // recovery affordance without digging through logs.
            self.menubar = makeMenubar(server: nil, config: config, lastError: error)
            NSLog("oMLX: server bootstrap failed — \(error)")
        }
    }

    private func makeMenubar(
        server: ServerProcess?,
        config: AppConfig,
        lastError: Error? = nil
    ) -> MenubarController {
        MenubarController(
            server: server,
            config: config,
            lastError: lastError,
            requestQuit: { [weak self] in self?.requestQuit() }
        )
    }

    private func scheduleAccessoryPolicyFlip() {
        // Defer the policy flip so the status item has time to register with
        // WindowServer before we hide the Dock icon (mirrors the Python
        // menubar's switchToAccessoryPolicy_).
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.accessory)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    func requestQuit() {
        NSApp.terminate(nil)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // Pure menubar app: every terminate is a real quit (the only user
        // path is the status item's "Quit"). Return .terminateNow explicitly
        // so nothing vetoes it; applicationWillTerminate reaps the child.
        return .terminateNow
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Synchronous reap on the main thread: SIGTERM the child, wait, then
        // SIGKILL. No async/await here — applicationWillTerminate runs on the
        // main thread, and blocking it while awaiting a @MainActor task would
        // deadlock the very task we're waiting on.
        NotificationCenter.default.removeObserver(self)
        server?.reapSync(timeout: 5)
    }
}

// MARK: - Launch at login (SMAppService)

/// Wraps `SMAppService.mainApp` so the menubar app registers itself as a
/// macOS login item -- visible and toggleable in System Settings > Login
/// Items. `mainApp` is the least-restrictive SMAppService tier (no Developer
/// ID required), so an ad-hoc signed local build can register; it may land in
/// `.enabled` or `.requiresApproval` (user approves in the panel) -- both are
/// visible there.
enum LaunchAtLogin {
    private static let didInitKey = "didInitialLoginItemRegistration"

    /// Register exactly once, on the first launch that sees this build. The
    /// UserDefaults flag is set only after a throw-free `register()`, so:
    ///   - a register() failure retries on the next launch (never silently
    ///     stuck off and absent from the panel), and
    ///   - a user who later turns the item OFF in System Settings is not
    ///     overridden on subsequent launches.
    static func registerOnceOnFirstLaunch() {
        guard !UserDefaults.standard.bool(forKey: didInitKey) else { return }
        do {
            try SMAppService.mainApp.register()
            UserDefaults.standard.set(true, forKey: didInitKey)
            NSLog("[LaunchAtLogin] registered; status=\(statusDescription)")
        } catch {
            NSLog("[LaunchAtLogin] register() failed, will retry next launch: \(error)")
        }
    }

    /// Human-readable `SMAppService.mainApp.status`, read live (never cached).
    static var statusDescription: String {
        switch SMAppService.mainApp.status {
        case .notRegistered:    return "notRegistered"
        case .enabled:          return "enabled"
        case .requiresApproval: return "requiresApproval"
        case .notFound:         return "notFound"
        @unknown default:       return "unknown"
        }
    }
}
