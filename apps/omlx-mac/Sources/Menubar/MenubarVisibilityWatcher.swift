// PR 4 — Bartender / Tahoe ControlCenter hidden-icon detection.
//
// Ports the three-signal visibility check from app.py:355-410 plus the
// one-shot recreate + escalation alert. Tahoe-aware copy includes a
// deep-link to System Settings → Menu Bar (which 26.x exposed for per-app
// status item visibility). The plist-edit "Auto-Fix" path (Full Disk
// Access + Group Container plist) is deferred to a follow-up; the alert
// gives the user manual recovery steps.

import AppKit

@MainActor
final class MenubarVisibilityWatcher {
    private weak var statusItem: NSStatusItem?
    private let recreate: () -> NSStatusItem
    private var didCheckOnce = false
    private var didRecreate = false
    private var didAlertOnce = false

    init(initial: NSStatusItem, recreate: @escaping () -> NSStatusItem) {
        self.statusItem = initial
        self.recreate = recreate
    }

    /// Schedule the post-launch visibility probe. Mirrors app.py's 3 s
    /// timer in `_doFinishLaunching` — gives ControlCenter time to settle
    /// before we conclude the icon is hidden.
    func scheduleInitialCheck(after delay: TimeInterval = 3.0) {
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            self?.checkOnce()
        }
    }

    func checkOnce() {
        guard !didCheckOnce else { return }
        didCheckOnce = true

        if !isHidden() { return }

        if !didRecreate {
            didRecreate = true
            statusItem = recreate()
            // Re-probe after 1 s to give the new item time to register.
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(1.0))
                guard let self, self.isHidden() else { return }
                self.showHiddenAlert()
            }
            return
        }

        showHiddenAlert()
    }

    /// True when ANY of the three strong "is the icon really shown" signals
    /// say no (api visible, NSWindow visible, occlusion bit set).
    /// See app.py:355-410 for the rationale on each signal.
    private func isHidden() -> Bool {
        guard let item = statusItem,
              let button = item.button,
              let window = button.window else { return true }
        let api = item.isVisible
        let visible = window.isVisible
        let occlusion = window.occlusionState.contains(.visible)
        return !(api && visible && occlusion)
    }

    private func showHiddenAlert() {
        guard !didAlertOnce else { return }
        didAlertOnce = true

        // Bring our process forward so the alert isn't behind another window.
        NSApp.activate(ignoringOtherApps: true)

        let mac = ProcessInfo.processInfo.operatingSystemVersion.majorVersion
        let isTahoeOrNewer = mac >= 26

        let alert = NSAlert()
        alert.messageText = "The oMLX menubar icon isn't showing up."

        if isTahoeOrNewer {
            alert.informativeText = """
            macOS Tahoe added per-app menu-bar visibility controls. Open \
            System Settings → Menu Bar and confirm oMLX is enabled.

            If a menu-bar manager (Bartender, Ice) is filtering items, \
            oMLX may be excluded by its rules. Bartender in particular \
            tends to hide PyObjC-style status items and now Swift apps \
            built the same way.
            """
            alert.addButton(withTitle: "Open Menu Bar Settings…")
            alert.addButton(withTitle: "Quit oMLX")
            alert.addButton(withTitle: "OK")
        } else {
            alert.informativeText = """
            Try quitting and relaunching oMLX. If the icon still \
            doesn't appear, check your menu-bar manager (Bartender, Ice, \
            etc.) — third-party apps sometimes filter status items by \
            category and may exclude oMLX.
            """
            alert.addButton(withTitle: "Quit oMLX")
            alert.addButton(withTitle: "OK")
        }

        alert.window.level = .floating
        let response = alert.runModal()

        if isTahoeOrNewer {
            switch response {
            case .alertFirstButtonReturn:
                if let url = URL(string: "x-apple.systempreferences:com.apple.MenuBar-Settings.extension") {
                    NSWorkspace.shared.open(url)
                }
            case .alertSecondButtonReturn:
                NSApp.terminate(nil)
            default:
                break
            }
        } else if response == .alertFirstButtonReturn {
            NSApp.terminate(nil)
        }
    }
}
