// PR 4 - Bartender / Tahoe ControlCenter hidden-icon detection.
//
// Ports the three-signal visibility check from app.py:355-410 plus the
// one-shot recreate + escalation alert. Tahoe-aware recovery includes
// System Settings plus the StatusKit Auto-Fix flow from the pre-Swift app.

import AppKit

@MainActor
final class MenubarVisibilityWatcher {
    private weak var statusItem: NSStatusItem?
    private let recreate: () -> NSStatusItem
    private var didCheckOnce = false
    private var didRecreate = false
    private var didAlertOnce = false
    private let statusKitPlistURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(
            "Library/Group Containers/group.com.apple.controlcenter/Library/Preferences/group.com.apple.controlcenter.plist"
        )
    private let logURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/oMLX/logs/menubar.log")

    private struct AutoFixOutcome {
        let success: Bool
        let message: String
        let needsFullDiskAccess: Bool

        init(success: Bool, message: String, needsFullDiskAccess: Bool = false) {
            self.success = success
            self.message = message
            self.needsFullDiskAccess = needsFullDiskAccess
        }
    }

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

        // Bring our process forward so the alert isn't behind another window.
        NSApp.activate(ignoringOtherApps: true)

        let mac = ProcessInfo.processInfo.operatingSystemVersion.majorVersion
        let isTahoeOrNewer = mac >= 26

        if isTahoeOrNewer, isKnownMenuBarManagerRunning() {
            return
        }

        didAlertOnce = true

        let alert = NSAlert()
        alert.messageText = "oMLX Menubar Icon Hidden"

        if isTahoeOrNewer {
            alert.informativeText = """
            The oMLX menubar icon isn't showing up.

            On macOS Tahoe this is usually caused by the StatusKit approval \
            flag being false in system preferences. Auto-Fix will approve \
            oMLX and restart ControlCenter. It needs Full Disk Access.

            You can also enable oMLX manually in System Settings > Menu Bar.
            """
            alert.addButton(withTitle: "Auto-Fix")
            alert.addButton(withTitle: "Open Menu Bar Settings…")
            alert.addButton(withTitle: "View Log")
            alert.addButton(withTitle: "Dismiss")
        } else {
            alert.informativeText = """
            The oMLX menubar icon isn't showing up.

            macOS before Tahoe doesn't offer a System Settings toggle for \
            third-party menubar apps. Try quitting and relaunching oMLX, \
            and check menubar manager tools like Bartender or Ice if you \
            use them.
            """
            alert.addButton(withTitle: "View Log")
            alert.addButton(withTitle: "Dismiss")
        }

        alert.window.level = .floating
        let response = alert.runModal()

        if isTahoeOrNewer {
            switch response {
            case .alertFirstButtonReturn:
                runAutofixFlow()
            case .alertSecondButtonReturn:
                openMenuBarSettings()
            case .alertThirdButtonReturn:
                NSWorkspace.shared.open(logURL)
            default:
                break
            }
        } else if response == .alertFirstButtonReturn {
            NSWorkspace.shared.open(logURL)
        }
    }

    // MARK: - StatusKit Auto-Fix

    private func runAutofixFlow() {
        let result = fixStatusKitPermission()
        if result.needsFullDiskAccess {
            showStatusKitAccessDeniedAlert()
            return
        }
        showAutofixResultAlert(success: result.success, message: result.message)
    }

    private func fixStatusKitPermission() -> AutoFixOutcome {
        let fileManager = FileManager.default

        guard fileManager.fileExists(atPath: statusKitPlistURL.path) else {
            return AutoFixOutcome(
                success: false,
                message: """
                The StatusKit preferences file does not exist on this Mac. \
                Your macOS version may not use this approval flow yet, so \
                the issue is likely not auto-fixable.
                """
            )
        }

        let backup = backupStatusKitPlist()

        var format = PropertyListSerialization.PropertyListFormat.binary
        var outer: [String: Any]
        do {
            let data = try Data(contentsOf: statusKitPlistURL)
            guard let plist = try PropertyListSerialization.propertyList(
                from: data,
                options: [.mutableContainersAndLeaves],
                format: &format
            ) as? [String: Any] else {
                return AutoFixOutcome(
                    success: false,
                    message: "StatusKit preferences did not decode to a dictionary."
                )
            }
            outer = plist
        } catch {
            if isPermissionError(error) {
                return AutoFixOutcome(
                    success: false,
                    message: "Auto-Fix needs Full Disk Access to read the StatusKit preferences.",
                    needsFullDiskAccess: true
                )
            }
            return AutoFixOutcome(
                success: false,
                message: "Failed to read the StatusKit preferences: \(error.localizedDescription)"
            )
        }

        let raw = outer["trackedApplications"]
        let nestedAsData = raw is Data
        var entries: [[String: Any]]

        if raw == nil {
            entries = []
        } else if let rawData = raw as? Data {
            var innerFormat = PropertyListSerialization.PropertyListFormat.binary
            do {
                guard let decoded = try PropertyListSerialization.propertyList(
                    from: rawData,
                    options: [.mutableContainersAndLeaves],
                    format: &innerFormat
                ) as? [[String: Any]] else {
                    return AutoFixOutcome(
                        success: false,
                        message: "trackedApplications decoded to a non-list value. Aborting to avoid corrupting the file."
                    )
                }
                entries = decoded
            } catch {
                return AutoFixOutcome(
                    success: false,
                    message: "Failed to decode trackedApplications: \(error.localizedDescription)"
                )
            }
        } else if let rawEntries = raw as? [[String: Any]] {
            entries = rawEntries
        } else {
            let typeName = String(describing: type(of: raw as Any))
            return AutoFixOutcome(
                success: false,
                message: "Unexpected trackedApplications type: \(typeName)."
            )
        }

        let primaryBundleID = Bundle.main.bundleIdentifier ?? "app.omlx"
        let targetBundleIDs = Set([primaryBundleID, "app.omlx", "com.omlx.app"])
        var normalizedEntries: [[String: Any]] = []
        var changed = false
        var foundAllowedApproval = false
        var hasPrimaryMarker = false
        var hasPrimaryApproval = false
        var matchedBundleIDs: [String] = []

        for var entry in entries {
            if let bareBundleID = bareBundleIdentifier(in: entry),
               targetBundleIDs.contains(bareBundleID) {
                if bareBundleID == primaryBundleID {
                    hasPrimaryMarker = true
                }
                normalizedEntries.append(entry)
                continue
            }

            let locationBundleID = locationBundleIdentifier(in: entry)
            let menuBundleIDs = menuItemBundleIdentifiers(in: entry)
            let locationMatches = locationBundleID.map(targetBundleIDs.contains) ?? false
            let menuMatches = menuBundleIDs.contains { targetBundleIDs.contains($0) }
            let referencesOmlx = locationMatches || menuMatches

            guard referencesOmlx else {
                normalizedEntries.append(entry)
                continue
            }

            guard let bundleID = locationBundleID, targetBundleIDs.contains(bundleID) else {
                // Drop stale cross-app rows such as location=iTerm2 with
                // menuItemLocations=oMLX. ControlCenter may keep those around
                // after bundle-id changes, but they do not back the oMLX toggle.
                changed = true
                continue
            }

            matchedBundleIDs.append(bundleID)
            if bundleID == primaryBundleID {
                hasPrimaryApproval = true
            }

            let expectedMenuLocations = [["bundle": ["_0": bundleID]]]
            if !menuBundleIDs.elementsEqual([bundleID]) {
                entry["menuItemLocations"] = expectedMenuLocations
                changed = true
            }

            if entry["isAllowed"] as? Bool == true {
                foundAllowedApproval = true
            } else {
                entry["isAllowed"] = true
                foundAllowedApproval = true
                changed = true
            }

            normalizedEntries.append(entry)
        }
        entries = normalizedEntries

        var appendedNew = false
        if !hasPrimaryMarker {
            entries.append(statusKitBundleMarker(bundleID: primaryBundleID))
            changed = true
            appendedNew = true
        }
        if !hasPrimaryApproval {
            entries.append(statusKitApprovalEntry(bundleID: primaryBundleID))
            changed = true
            appendedNew = true
            foundAllowedApproval = true
        }

        if !changed {
            let knownIDs = matchedBundleIDs.isEmpty ? primaryBundleID : matchedBundleIDs.joined(separator: ", ")
            return AutoFixOutcome(
                success: true,
                message: """
                oMLX is already approved in StatusKit (\(knownIDs)). If the \
                icon still doesn't appear, the root cause is something else. \
                Share the latest menubar.log with the maintainer.
                """
            )
        }

        do {
            prepareStatusKitPreferenceWrite()

            if nestedAsData || raw == nil {
                outer["trackedApplications"] = try PropertyListSerialization.data(
                    fromPropertyList: entries,
                    format: .binary,
                    options: 0
                )
            } else {
                outer["trackedApplications"] = entries
            }

            let serialized = try PropertyListSerialization.data(
                fromPropertyList: outer,
                format: .binary,
                options: 0
            )
            try writeStatusKitPlist(serialized, backup: backup)
            try validateStatusKitPlist()
        } catch {
            restoreStatusKitPlist(from: backup)
            if isPermissionError(error) {
                return AutoFixOutcome(
                    success: false,
                    message: "Auto-Fix needs Full Disk Access to write the StatusKit preferences.",
                    needsFullDiskAccess: true
                )
            }
            return AutoFixOutcome(
                success: false,
                message: "Failed to write the StatusKit preferences: \(error.localizedDescription)"
            )
        }

        if !restartControlCenter() {
            return AutoFixOutcome(
                success: true,
                message: """
                StatusKit was updated but oMLX couldn't restart ControlCenter. \
                Run `killall ControlCenter` manually.
                """
            )
        }

        let detail = appendedNew || !foundAllowedApproval
            ? "appended a new \(primaryBundleID) entry"
            : "approved the existing oMLX entry"
        return AutoFixOutcome(
            success: true,
            message: """
            Auto-Fix \(detail) in StatusKit and restarted ControlCenter. \
            The menubar icon should appear within a few seconds. If it \
            still doesn't, quit and relaunch oMLX.
            """
        )
    }

    private func backupStatusKitPlist() -> URL? {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: statusKitPlistURL.path) else {
            return nil
        }

        let backupDirectory = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/oMLX/backups")
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let backupURL = backupDirectory
            .appendingPathComponent("statuskit-\(formatter.string(from: Date())).plist")

        do {
            try fileManager.createDirectory(
                at: backupDirectory,
                withIntermediateDirectories: true
            )
            try fileManager.copyItem(at: statusKitPlistURL, to: backupURL)
            return backupURL
        } catch {
            return nil
        }
    }

    private func writeStatusKitPlist(_ data: Data, backup: URL?) throws {
        let fileManager = FileManager.default
        let temporaryURL = statusKitPlistURL
            .deletingLastPathComponent()
            .appendingPathComponent(statusKitPlistURL.lastPathComponent + ".omlx-tmp")

        do {
            try data.write(to: temporaryURL, options: [.atomic])
            _ = try fileManager.replaceItemAt(
                statusKitPlistURL,
                withItemAt: temporaryURL,
                backupItemName: nil,
                options: []
            )
        } catch {
            try? fileManager.removeItem(at: temporaryURL)
            restoreStatusKitPlist(from: backup)
            throw error
        }
    }

    private func validateStatusKitPlist() throws {
        var format = PropertyListSerialization.PropertyListFormat.binary
        let data = try Data(contentsOf: statusKitPlistURL)
        _ = try PropertyListSerialization.propertyList(
            from: data,
            options: [],
            format: &format
        )
    }

    private func restoreStatusKitPlist(from backup: URL?) {
        guard let backup else { return }
        let fileManager = FileManager.default
        do {
            if fileManager.fileExists(atPath: statusKitPlistURL.path) {
                try fileManager.removeItem(at: statusKitPlistURL)
            }
            try fileManager.copyItem(at: backup, to: statusKitPlistURL)
        } catch {
            // Best effort rollback; the result alert tells the user the write failed.
        }
    }

    private func restartControlCenter() -> Bool {
        _ = runKillall("cfprefsd")
        return runKillall("ControlCenter")
    }

    private func prepareStatusKitPreferenceWrite() {
        _ = runKillall("ControlCenter")
        _ = runKillall("cfprefsd")
    }

    private func runKillall(_ processName: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/killall")
        process.arguments = [processName]

        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }

    private func bareBundleIdentifier(in entry: [String: Any]) -> String? {
        guard entry["location"] == nil,
              entry["menuItemLocations"] == nil,
              let bundle = entry["bundle"] as? [String: Any] else {
            return nil
        }
        return bundle["_0"] as? String
    }

    private func locationBundleIdentifier(in entry: [String: Any]) -> String? {
        guard let location = entry["location"] as? [String: Any],
              let bundle = location["bundle"] as? [String: Any] else {
            return nil
        }
        return bundle["_0"] as? String
    }

    private func menuItemBundleIdentifiers(in entry: [String: Any]) -> [String] {
        guard let locations = entry["menuItemLocations"] as? [[String: Any]] else {
            return []
        }

        return locations.compactMap { location in
            guard let bundle = location["bundle"] as? [String: Any] else {
                return nil
            }
            return bundle["_0"] as? String
        }
    }

    private func statusKitBundleMarker(bundleID: String) -> [String: Any] {
        ["bundle": ["_0": bundleID]]
    }

    private func statusKitApprovalEntry(bundleID: String) -> [String: Any] {
        [
            "location": ["bundle": ["_0": bundleID]],
            "menuItemLocations": [["bundle": ["_0": bundleID]]],
            "isAllowed": true
        ]
    }

    private func isPermissionError(_ error: Error) -> Bool {
        let nsError = error as NSError
        if nsError.domain == NSCocoaErrorDomain,
           nsError.code == NSFileReadNoPermissionError
            || nsError.code == NSFileWriteNoPermissionError {
            return true
        }
        if nsError.domain == NSPOSIXErrorDomain,
           nsError.code == 1 || nsError.code == 13 {
            return true
        }
        if let underlying = nsError.userInfo[NSUnderlyingErrorKey] as? Error {
            return isPermissionError(underlying)
        }
        return false
    }

    private func isKnownMenuBarManagerRunning() -> Bool {
        let bundleIDs = [
            "com.surteesstudios.Bartender",
            "com.jordanbaird.Ice",
            "com.jordanbaird.ice"
        ]
        return bundleIDs.contains { bundleID in
            !NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).isEmpty
        }
    }

    // MARK: - Recovery Alerts

    private func openMenuBarSettings() {
        if let url = URL(
            string: "x-apple.systempreferences:com.apple.ControlCenter-Settings.extension?MenuBar"
        ) {
            NSWorkspace.shared.open(url)
        }
    }

    private func openFullDiskAccessSettings() {
        if let url = URL(
            string: "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles"
        ) {
            NSWorkspace.shared.open(url)
        }
    }

    private func showStatusKitAccessDeniedAlert() {
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = "Full Disk Access Required"
        alert.informativeText = """
        Auto-Fix needs macOS permission to edit the StatusKit approval file \
        in your Group Containers folder.

        Enable oMLX in System Settings > Privacy & Security > Full Disk \
        Access, then run Auto-Fix again.
        """
        alert.addButton(withTitle: "Open Full Disk Access")
        alert.addButton(withTitle: "Dismiss")
        alert.window.level = .floating

        if alert.runModal() == .alertFirstButtonReturn {
            openFullDiskAccessSettings()
        }
    }

    private func showAutofixResultAlert(success: Bool, message: String) {
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = success ? "Auto-Fix Succeeded" : "Auto-Fix Failed"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.window.level = .floating
        alert.runModal()
    }
}
