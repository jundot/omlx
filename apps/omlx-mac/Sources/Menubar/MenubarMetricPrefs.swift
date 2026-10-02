// Appearance preferences shared by the menubar metric items, the stats
// poller, and the Appearance settings screen. All values live in
// UserDefaults (settings.json is server-owned and never sees these).

import Foundation

enum MenubarMetricPrefs {
    /// Reused legacy key: users who had the inline live-activity text ON
    /// automatically get the LIV status item (migration by key reuse).
    static let liveKey = "showLiveActivityInMenuBar"
    static let averageKey = "menubar.showAverageActivity"
    static let alltimeKey = "menubar.showAlltimeActivity"
    /// Double seconds. Absent → 1.0; anything outside the picker set is
    /// clamped back to 1.0 so a corrupt write can't stall or spin the poller.
    static let refreshIntervalKey = "menubar.refreshInterval"
    /// Bool, default false = today's auto behavior (Dock icon only while a
    /// window is open). True keeps the Dock icon visible permanently.
    static let showDockIconKey = "showDockIcon"

    static let refreshIntervalChoices: [Double] = [0.5, 1.0, 2.0, 3.0]

    static var refreshInterval: TimeInterval {
        let raw = UserDefaults.standard.object(forKey: refreshIntervalKey) as? Double ?? 1.0
        return refreshIntervalChoices.contains(raw) ? raw : 1.0
    }

    static var showDockIcon: Bool {
        UserDefaults.standard.bool(forKey: showDockIconKey)
    }

    static var enabledMetrics: EnabledMetrics {
        let defaults = UserDefaults.standard
        return EnabledMetrics(
            live: defaults.bool(forKey: liveKey),
            average: defaults.bool(forKey: averageKey),
            alltime: defaults.bool(forKey: alltimeKey)
        )
    }

    /// Which models the menubar Models submenu lists below the loaded
    /// section. Loaded models always show regardless.
    static let modelLibraryScopeKey = "menubar.modelLibraryScope"

    static var modelLibraryScope: MenuBarModelScope {
        MenuBarModelScope(
            rawValue: UserDefaults.standard.string(forKey: modelLibraryScopeKey) ?? ""
        ) ?? .all
    }

    // Host-metric bar items (CPU / GPU / MEM). These drive the system
    // sampler, not the server-stats poller.
    static let cpuItemKey = "menubar.showCPU"
    static let gpuItemKey = "menubar.showGPU"
    static let memoryItemKey = "menubar.showMemory"

    static var enabledSystemItems: EnabledSystemItems {
        let defaults = UserDefaults.standard
        return EnabledSystemItems(
            cpu: defaults.bool(forKey: cpuItemKey),
            gpu: defaults.bool(forKey: gpuItemKey),
            memory: defaults.bool(forKey: memoryItemKey)
        )
    }
}

struct EnabledSystemItems: Equatable, Sendable {
    var cpu: Bool
    var gpu: Bool
    var memory: Bool

    var any: Bool { cpu || gpu || memory }
}

/// Unknown raw values (corrupt writes, future removals) fall back to .all —
/// the behavior the menu always had.
enum MenuBarModelScope: String, CaseIterable, Sendable {
    case favoritesOnly = "favorites"
    case all = "all"
}

struct EnabledMetrics: Equatable, Sendable {
    var live: Bool
    var average: Bool
    var alltime: Bool

    var any: Bool { live || average || alltime }
}
