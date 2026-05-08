// PR 6 — AppView shell. NavigationSplitView with the custom sidebar from
// `Sidebar.swift` and an empty-stub content per section. Sized to the design
// canvas (1140×760) with a sane minimum so the window survives a resize.
//
// The shell is the entry point for both `Cmd-,` (the SwiftUI `Settings` scene
// declared in `oMLXApp.swift`) and the menubar's `Admin Panel` item. PR 7+
// fills in real screens; for now each tab routes to a placeholder that names
// its landing PR so reviewers can track progress in-app.

import SwiftUI

struct AppView: View {
    @State private var selection: AppSection = .server
    @State private var search: String = ""

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = scheme == .dark ? OMLXTheme.dark : OMLXTheme.light

        NavigationSplitView {
            Sidebar(selection: $selection, search: $search)
                .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 260)
        } detail: {
            ContentScaffold(section: selection) {
                screen(for: selection)
            }
            .navigationSplitViewColumnWidth(min: 640, ideal: 920)
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 880, idealWidth: 1140, minHeight: 600, idealHeight: 760)
        .background(theme.windowBg)
        .environment(\.omlxTheme, theme)
    }

    @ViewBuilder
    private func screen(for section: AppSection) -> some View {
        switch section {
        case .server:       ServerScreen()
        case .status:       StatusScreen()
        case .logs:         LogsScreen()
        case .models:       ModelsScreen()
        case .downloads:    DownloadsScreen()
        case .integrations: IntegrationsScreen()
        case .security:     SecurityScreen()
        case .about:        AboutScreen()
        }
    }
}

// MARK: - Detail scaffold

/// Wraps the per-section view with the design's toolbar title + scroll body.
/// Mirrors `ContentArea` from the design (omlx-components.jsx:250-292):
/// 42 pt toolbar, 720 pt max content width, 20/28/36 pt padding.
private struct ContentScaffold<Content: View>: View {
    let section: AppSection
    @ViewBuilder var content: () -> Content

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                content()
                    .frame(maxWidth: 720, alignment: .topLeading)
                    .frame(maxWidth: .infinity, alignment: .top)
            }
            .padding(.top, 20)
            .padding(.horizontal, 28)
            .padding(.bottom, 36)
        }
        .background(theme.contentBg)
        .navigationTitle(section.title)
    }
}

#Preview("AppView — light") {
    AppView()
        .frame(width: 1140, height: 760)
        .preferredColorScheme(.light)
}

#Preview("AppView — dark") {
    AppView()
        .frame(width: 1140, height: 760)
        .preferredColorScheme(.dark)
}
