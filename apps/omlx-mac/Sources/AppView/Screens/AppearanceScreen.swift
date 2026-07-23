// Appearance pane: menubar refresh cadence, the Dock-icon override, and the
// per-metric menubar item toggles. Everything here is app-local UserDefaults
// (settings.json is server-owned); the menubar controller and AppDelegate
// pick changes up through the UserDefaults change notification, so the
// screen needs no view model and no onChange plumbing.

import SwiftUI

struct AppearanceScreen: View {
    @AppStorage(MenubarMetricPrefs.refreshIntervalKey)
    private var refreshInterval = 1.0
    @AppStorage(MenubarMetricPrefs.showDockIconKey)
    private var showDockIcon = false
    @AppStorage(MenubarMetricPrefs.liveKey)
    private var showLiveActivity = false
    @AppStorage(MenubarMetricPrefs.averageKey)
    private var showAverageActivity = false
    @AppStorage(MenubarMetricPrefs.alltimeKey)
    private var showAlltimeActivity = false
    @AppStorage(MenubarMetricPrefs.cpuItemKey)
    private var showCPUItem = false
    @AppStorage(MenubarMetricPrefs.gpuItemKey)
    private var showGPUItem = false
    @AppStorage(MenubarMetricPrefs.memoryItemKey)
    private var showMemoryItem = false
    @AppStorage(MenubarMetricPrefs.modelLibraryScopeKey)
    private var modelLibraryScope = MenuBarModelScope.all.rawValue

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(String(
                localized: "appearance.section.general",
                defaultValue: "General",
                comment: "Appearance screen section header for the basic options"
            ))
            ListGroup {
                Row(
                    label: String(
                        localized: "appearance.row.refresh_interval",
                        defaultValue: "Refresh Interval",
                        comment: "Appearance row label for the menubar refresh cadence picker"
                    ),
                    sublabel: String(
                        localized: "appearance.row.refresh_interval.sub",
                        defaultValue: "How often menu bar items and their graphs update.",
                        comment: "Appearance row sublabel for the menubar refresh cadence picker"
                    )
                ) {
                    Popup(selection: $refreshInterval, width: 110, options: [
                        (0.5, "0.5 s"),
                        (1.0, "1 s"),
                        (2.0, "2 s"),
                        (3.0, "3 s"),
                    ])
                }
                Row(
                    label: String(
                        localized: "appearance.row.show_dock_icon",
                        defaultValue: "Show Dock Icon",
                        comment: "Appearance row label for the permanent Dock icon toggle"
                    ),
                    sublabel: String(
                        localized: "appearance.row.show_dock_icon.sub",
                        defaultValue: "Keep the Dock icon visible even when no window is open.",
                        comment: "Appearance row sublabel for the permanent Dock icon toggle"
                    ),
                    isLast: true
                ) {
                    Toggle("", isOn: $showDockIcon)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
            }

            SectionHeader(String(
                localized: "appearance.section.model_library",
                defaultValue: "Menu Bar Model Library",
                comment: "Appearance screen section header for the Models submenu scope"
            ))
            .padding(.top, 18)
            ListGroup {
                Row(
                    label: String(
                        localized: "appearance.row.visible_models",
                        defaultValue: "Visible Models",
                        comment: "Appearance row label for the Models submenu scope picker"
                    ),
                    sublabel: String(
                        localized: "appearance.row.visible_models.sub",
                        defaultValue: "Which models the menu bar Models menu lists. Loaded models always show.",
                        comment: "Appearance row sublabel for the Models submenu scope picker"
                    ),
                    isLast: true
                ) {
                    Segmented(selection: $modelLibraryScope, options: [
                        (
                            MenuBarModelScope.favoritesOnly.rawValue,
                            String(
                                localized: "appearance.model_scope.favorites",
                                defaultValue: "Favorites Only",
                                comment: "Models submenu scope option that lists only favorite models"
                            )
                        ),
                        (
                            MenuBarModelScope.all.rawValue,
                            String(
                                localized: "appearance.model_scope.all",
                                defaultValue: "All Models",
                                comment: "Models submenu scope option that lists the whole library"
                            )
                        ),
                    ])
                    .frame(width: 200)
                }
            }

            SectionHeader(
                String(
                    localized: "appearance.section.menubar_items",
                    defaultValue: "Menu Bar Items",
                    comment: "Appearance screen section header for the metric item toggles"
                ),
                subtitle: String(
                    localized: "appearance.section.menubar_items.sub",
                    defaultValue: "Each item shows prompt-processing and token-generation speed in the menu bar. Click one for details and a live graph.",
                    comment: "Appearance screen section subtitle explaining the metric items"
                )
            )
            .padding(.top, 18)
            ListGroup {
                Row(
                    label: String(
                        localized: "appearance.row.live_activity",
                        defaultValue: "Live Activity",
                        comment: "Appearance row label for the LIV menubar item toggle"
                    ),
                    sublabel: String(
                        localized: "appearance.row.live_activity.sub",
                        defaultValue: "Current speeds across running requests (LIV).",
                        comment: "Appearance row sublabel for the LIV menubar item toggle"
                    )
                ) {
                    Toggle("", isOn: $showLiveActivity)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
                Row(
                    label: String(
                        localized: "appearance.row.average_activity",
                        defaultValue: "Average Session Activity",
                        comment: "Appearance row label for the AVG menubar item toggle"
                    ),
                    sublabel: String(
                        localized: "appearance.row.average_activity.sub",
                        defaultValue: "Average speeds since the server started (AVG).",
                        comment: "Appearance row sublabel for the AVG menubar item toggle"
                    )
                ) {
                    Toggle("", isOn: $showAverageActivity)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
                Row(
                    label: String(
                        localized: "appearance.row.alltime_activity",
                        defaultValue: "All Time Activity",
                        comment: "Appearance row label for the ALL menubar item toggle"
                    ),
                    sublabel: String(
                        localized: "appearance.row.alltime_activity.sub",
                        defaultValue: "Average speeds across all sessions (ALL).",
                        comment: "Appearance row sublabel for the ALL menubar item toggle"
                    )
                ) {
                    Toggle("", isOn: $showAlltimeActivity)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
                Row(
                    label: String(
                        localized: "appearance.row.cpu_item",
                        defaultValue: "CPU",
                        comment: "Appearance row label for the CPU usage bar menubar item toggle"
                    ),
                    sublabel: String(
                        localized: "appearance.row.cpu_item.sub",
                        defaultValue: "CPU usage as a vertical bar.",
                        comment: "Appearance row sublabel for the CPU usage bar menubar item toggle"
                    )
                ) {
                    Toggle("", isOn: $showCPUItem)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
                Row(
                    label: String(
                        localized: "appearance.row.gpu_item",
                        defaultValue: "GPU",
                        comment: "Appearance row label for the GPU usage bar menubar item toggle"
                    ),
                    sublabel: String(
                        localized: "appearance.row.gpu_item.sub",
                        defaultValue: "GPU usage as a vertical bar.",
                        comment: "Appearance row sublabel for the GPU usage bar menubar item toggle"
                    )
                ) {
                    Toggle("", isOn: $showGPUItem)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
                Row(
                    label: String(
                        localized: "appearance.row.memory_item",
                        defaultValue: "Memory",
                        comment: "Appearance row label for the memory usage bar menubar item toggle"
                    ),
                    sublabel: String(
                        localized: "appearance.row.memory_item.sub",
                        defaultValue: "Memory usage as a vertical bar (MEM).",
                        comment: "Appearance row sublabel for the memory usage bar menubar item toggle"
                    ),
                    isLast: true
                ) {
                    Toggle("", isOn: $showMemoryItem)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
            }
        }
    }
}

#Preview("AppearanceScreen") {
    ScrollView {
        AppearanceScreen()
            .padding(24)
    }
    .frame(width: 640, height: 520)
    .omlxThemed()
}
