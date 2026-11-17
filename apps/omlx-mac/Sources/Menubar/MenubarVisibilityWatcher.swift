// PR 4 - Bartender / Tahoe ControlCenter hidden-icon detection.
//
// Ports the three-signal visibility check from app.py:355-410 plus the
// one-shot recreate + escalation alert. Tahoe-aware recovery includes
// System Settings plus the StatusKit Auto-Fix flow from the pre-Swift app.
//
// The repair itself lives in `MenubarIconRecovery` at the bottom of this
// file so the Appearance screen can drive it on demand. The launch alert is
// one-shot and has three ways to never reach the user (a running menu bar
// manager, a dismissed alert, a probe that reads the item as visible), and
// the main status item carries no autosaveName — so nothing on our side
// records the removal, and a relaunch can't undo it either. On Tahoe the
// removal lives in ControlCenter's own prefs as isAllowed=false. See #2368.
//
// `MenubarLog` is the third type here: the alert's View Log button and
// `omlx diagnose menubar` both point at menubar.log, so the probe has to
// actually write it. server.log is not an option — that file is the raw
// stdout/stderr handle the Python child owns.

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

        MenubarLog.write("visibility probe: \(probeDescription())")
        if !isHidden() { return }

        if !didRecreate {
            didRecreate = true
            statusItem = recreate()
            MenubarLog.write("visibility probe: recreated NSStatusItem, re-probing in 1s")
            // Re-probe after 1 s to give the new item time to register.
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(1.0))
                guard let self else { return }
                MenubarLog.write("visibility probe: after recreate: \(self.probeDescription())")
                guard self.isHidden() else { return }
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

    /// One line carrying every signal the probe decides on, plus the button
    /// window frame. #1497 turned on that frame: the item reported itself
    /// visible while ControlCenter had parked it at y=-17, and nothing in
    /// the app recorded that anywhere the reporter could send back.
    private func probeDescription() -> String {
        guard let item = statusItem, let button = item.button else {
            return "hidden=true (no status item)"
        }
        guard let window = button.window else {
            return "hidden=true api=\(item.isVisible) (no button window)"
        }
        let frame = window.frame
        let frameText = String(
            format: "(%.1f,%.1f %.1fx%.1f)",
            frame.origin.x, frame.origin.y, frame.width, frame.height
        )
        let screenText = window.screen.map {
            String(format: "%.0fx%.0f", $0.frame.width, $0.frame.height)
        } ?? "none"
        return """
        hidden=\(isHidden()) api=\(item.isVisible) window=\(window.isVisible) \
        occlusion=\(window.occlusionState.contains(.visible)) frame=\(frameText) screen=\(screenText)
        """
    }

    private func showHiddenAlert() {
        guard !didAlertOnce else { return }

        // Bring our process forward so the alert isn't behind another window.
        NSApp.activate(ignoringOtherApps: true)

        if MenubarIconRecovery.isTahoeOrNewer, let manager = runningMenuBarManager() {
            MenubarLog.write("hidden alert suppressed: menu bar manager running (\(manager))")
            return
        }

        didAlertOnce = true
        MenubarLog.write("hidden alert shown")

        let alert = NSAlert()
        alert.messageText = "oMLX Menubar Icon Hidden"

        if MenubarIconRecovery.isTahoeOrNewer {
            alert.informativeText = """
            The oMLX menubar icon isn't showing up.

            On macOS Tahoe this is usually caused by the StatusKit approval \
            flag being false in system preferences. Auto-Fix will approve \
            oMLX and restart ControlCenter. It needs Full Disk Access.

            You can also enable oMLX manually in System Settings > Menu Bar, \
            or run this again later from Settings > Appearance > Menu Bar Icon.
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

        if MenubarIconRecovery.isTahoeOrNewer {
            switch response {
            case .alertFirstButtonReturn:
                MenubarIconRecovery.runAutofixFlow()
            case .alertSecondButtonReturn:
                MenubarIconRecovery.openMenuBarSettings()
            case .alertThirdButtonReturn:
                MenubarLog.open()
            default:
                break
            }
        } else if response == .alertFirstButtonReturn {
            MenubarLog.open()
        }
    }

    /// Bundle id of the first known menu bar manager that's running, if any.
    /// Named in the log so a suppressed alert isn't a silent branch.
    private func runningMenuBarManager() -> String? {
        let bundleIDs = [
            "com.surteesstudios.Bartender",
            "com.jordanbaird.Ice",
            "com.jordanbaird.ice",
            "com.stonerl.Thaw"
        ]
        return bundleIDs.first { bundleID in
            !NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).isEmpty
        }
    }
}

/// Append-only diagnostic log for the menu bar item, at
/// `~/Library/Application Support/oMLX/logs/menubar.log`. The pre-Swift app
/// wrote this file and both the hidden-icon alert and `omlx diagnose
/// menubar` still send people to it, so the Swift port owes it the same
/// content. Kept separate from server.log, which is the Python child's own
/// stdout/stderr handle.
@MainActor
enum MenubarLog {
    /// Settable so tests can redirect the writer at a temp file instead of
    /// appending to (and trimming) the user's real log.
    static var url = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/oMLX/logs/menubar.log")

    /// Trim threshold and the tail we keep when we cross it. The file only
    /// takes a handful of lines per launch, so this is a runaway guard
    /// rather than real rotation.
    static let maxBytes = 256 * 1024
    static let keepBytes = 64 * 1024

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter
    }()

    static func write(_ message: String) {
        let line = "\(timestampFormatter.string(from: Date())) \(message)\n"
        guard let data = line.data(using: .utf8) else { return }

        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        } catch {
            return
        }
        if !fileManager.fileExists(atPath: url.path) {
            fileManager.createFile(atPath: url.path, contents: nil)
        }

        trimIfNeeded()

        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
    }

    /// Opens the log for the alert's View Log button. Writes a line first so
    /// a first-ever click never lands on a missing file.
    static func open() {
        if !FileManager.default.fileExists(atPath: url.path) {
            write("log opened before any probe ran")
        }
        NSWorkspace.shared.open(url)
    }

    private static func trimIfNeeded() {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = (attributes?[.size] as? NSNumber)?.intValue, size > maxBytes else {
            return
        }
        guard let handle = try? FileHandle(forReadingFrom: url) else { return }
        try? handle.seek(toOffset: UInt64(max(0, size - keepBytes)))
        var tail = (try? handle.readToEnd()) ?? Data()
        try? handle.close()
        // The cut lands mid-line; drop the fragment so every line in the
        // file stays complete and parseable by `omlx diagnose menubar`.
        if let newline = tail.firstIndex(of: UInt8(ascii: "\n")) {
            tail = tail[tail.index(after: newline)...]
        }
        try? tail.write(to: url, options: [.atomic])
    }
}

/// StatusKit repair shared by the launch-time hidden-icon alert and the
/// Appearance screen's "Restore" button. Everything here is best-effort and
/// user-triggered: on Tahoe the menu bar belongs to ControlCenter, so the
/// only thing the app can do is flip its own approval flag back on and let
/// the user finish in System Settings.
@MainActor
enum MenubarIconRecovery {

    static var isTahoeOrNewer: Bool {
        ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26
    }

    struct AutoFixOutcome {
        let success: Bool
        let message: String
        let needsFullDiskAccess: Bool

        init(success: Bool, message: String, needsFullDiskAccess: Bool = false) {
            self.success = success
            self.message = message
            self.needsFullDiskAccess = needsFullDiskAccess
        }
    }

    private static let statusKitPlistURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(
            "Library/Group Containers/group.com.apple.controlcenter/Library/Preferences/group.com.apple.controlcenter.plist"
        )

    // MARK: - Entry points

    /// Auto-Fix button of the launch-time alert.
    static func runAutofixFlow() {
        MenubarLog.write("auto-fix requested from hidden-icon alert")
        let result = fixStatusKitPermission()
        MenubarLog.write(
            "auto-fix result: success=\(result.success) fda=\(result.needsFullDiskAccess) — \(oneLine(result.message))"
        )
        if result.needsFullDiskAccess {
            showStatusKitAccessDeniedAlert()
            return
        }
        showAutofixResultAlert(success: result.success, message: result.message)
    }

    /// Appearance > Menu Bar Icon > Restore. Repairs the StatusKit approval
    /// first so the rebuilt status item registers against an already-approved
    /// bundle id, then hands the caller the rebuild, then reports back with a
    /// way into System Settings (which owns the final say on Tahoe).
    static func restore(rebuild: () -> Void) {
        NSApp.activate(ignoringOtherApps: true)
        MenubarLog.write("restore requested from Appearance settings")

        guard isTahoeOrNewer else {
            rebuild()
            MenubarLog.write("restore: rebuilt status items (pre-Tahoe, no StatusKit repair)")
            showRestoreResultAlert(
                message: """
                oMLX rebuilt its menu bar item.

                macOS before Tahoe doesn't offer a System Settings toggle for \
                third-party menubar apps. If the icon still doesn't appear, \
                quit and relaunch oMLX, and check menubar manager tools like \
                Bartender or Ice if you use them.
                """,
                offerSettings: false
            )
            return
        }

        let result = fixStatusKitPermission()
        rebuild()
        MenubarLog.write(
            "restore result: success=\(result.success) fda=\(result.needsFullDiskAccess) — \(oneLine(result.message))"
        )

        if result.needsFullDiskAccess {
            showStatusKitAccessDeniedAlert()
            return
        }
        showRestoreResultAlert(
            message: "oMLX rebuilt its menu bar item.\n\n\(result.message)",
            offerSettings: true
        )
    }

    static func openMenuBarSettings() {
        if let url = URL(
            string: "x-apple.systempreferences:com.apple.ControlCenter-Settings.extension?MenuBar"
        ) {
            NSWorkspace.shared.open(url)
        }
    }

    /// Alert copy is written as a wrapped paragraph; the log wants one line.
    private static func oneLine(_ message: String) -> String {
        message
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - StatusKit Auto-Fix

    private static func fixStatusKitPermission() -> AutoFixOutcome {
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

    private static func backupStatusKitPlist() -> URL? {
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

    private static func writeStatusKitPlist(_ data: Data, backup: URL?) throws {
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

    private static func validateStatusKitPlist() throws {
        var format = PropertyListSerialization.PropertyListFormat.binary
        let data = try Data(contentsOf: statusKitPlistURL)
        _ = try PropertyListSerialization.propertyList(
            from: data,
            options: [],
            format: &format
        )
    }

    private static func restoreStatusKitPlist(from backup: URL?) {
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

    private static func restartControlCenter() -> Bool {
        _ = runKillall("cfprefsd")
        return runKillall("ControlCenter")
    }

    private static func prepareStatusKitPreferenceWrite() {
        _ = runKillall("ControlCenter")
        _ = runKillall("cfprefsd")
    }

    private static func runKillall(_ processName: String) -> Bool {
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

    private static func bareBundleIdentifier(in entry: [String: Any]) -> String? {
        guard entry["location"] == nil,
              entry["menuItemLocations"] == nil,
              let bundle = entry["bundle"] as? [String: Any] else {
            return nil
        }
        return bundle["_0"] as? String
    }

    private static func locationBundleIdentifier(in entry: [String: Any]) -> String? {
        guard let location = entry["location"] as? [String: Any],
              let bundle = location["bundle"] as? [String: Any] else {
            return nil
        }
        return bundle["_0"] as? String
    }

    private static func menuItemBundleIdentifiers(in entry: [String: Any]) -> [String] {
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

    private static func statusKitBundleMarker(bundleID: String) -> [String: Any] {
        ["bundle": ["_0": bundleID]]
    }

    private static func statusKitApprovalEntry(bundleID: String) -> [String: Any] {
        [
            "location": ["bundle": ["_0": bundleID]],
            "menuItemLocations": [["bundle": ["_0": bundleID]]],
            "isAllowed": true
        ]
    }

    private static func isPermissionError(_ error: Error) -> Bool {
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

    // MARK: - Recovery Alerts

    private static func openFullDiskAccessSettings() {
        if let url = URL(
            string: "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles"
        ) {
            NSWorkspace.shared.open(url)
        }
    }

    private static func showStatusKitAccessDeniedAlert() {
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = "Full Disk Access Required"
        alert.informativeText = """
        Auto-Fix needs macOS permission to edit the StatusKit approval file \
        in your Group Containers folder.

        Enable oMLX in System Settings > Privacy & Security > Full Disk \
        Access, then run Auto-Fix again. You can also turn oMLX back on \
        yourself in System Settings > Menu Bar.
        """
        alert.addButton(withTitle: "Open Full Disk Access")
        alert.addButton(withTitle: "Open Menu Bar Settings…")
        alert.addButton(withTitle: "Dismiss")
        alert.window.level = .floating

        switch alert.runModal() {
        case .alertFirstButtonReturn:
            openFullDiskAccessSettings()
        case .alertSecondButtonReturn:
            openMenuBarSettings()
        default:
            break
        }
    }

    private static func showAutofixResultAlert(success: Bool, message: String) {
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = success ? "Auto-Fix Succeeded" : "Auto-Fix Failed"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.window.level = .floating
        alert.runModal()
    }

    private static func showRestoreResultAlert(message: String, offerSettings: Bool) {
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = "Restore Menu Bar Icon"
        alert.informativeText = message
        if offerSettings {
            alert.addButton(withTitle: "Open Menu Bar Settings…")
        }
        alert.addButton(withTitle: "Done")
        alert.window.level = .floating

        if alert.runModal() == .alertFirstButtonReturn, offerSettings {
            openMenuBarSettings()
        }
    }
}
