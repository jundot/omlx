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

    @Environment(\.colorScheme) private var scheme
    @EnvironmentObject private var services: AppServices

    var body: some View {
        let theme = scheme == .dark ? OMLXTheme.dark : OMLXTheme.light

        NavigationSplitView {
            Sidebar(selection: bindingForSelection())
                .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 260)
        } detail: {
            ContentScaffold(section: selection, detailTitle: detailTitle) {
                screen(for: selection)
            }
            .navigationSplitViewColumnWidth(min: 640, ideal: 920)
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 880, idealWidth: 1140, minHeight: 600, idealHeight: 760)
        .background(theme.windowBg)
        .environment(\.omlxTheme, theme)
        .onChange(of: services.requestedSection) { _, requested in
            // A screen asked us to navigate elsewhere (e.g. "Edit on
            // Server →" from the per-model Profiles tab). Clear the
            // request after applying so the same section can be requested
            // twice in a row.
            if let requested {
                if requested != .models { services.modelDetailID = nil }
                selection = requested
                services.requestedSection = nil
            }
        }
    }

    /// Drilling out of ModelSettingsScreen via the sidebar (changing section)
    /// must clear the per-model detail id so we don't accidentally re-enter
    /// the detail when the user returns to Models.
    private func bindingForSelection() -> Binding<AppSection> {
        Binding(
            get: { selection },
            set: { newValue in
                if newValue != .models { services.modelDetailID = nil }
                selection = newValue
            }
        )
    }

    private var detailTitle: String? {
        if selection == .models, let id = services.modelDetailID, !id.isEmpty {
            return id
        }
        return nil
    }

    @ViewBuilder
    private func screen(for section: AppSection) -> some View {
        switch section {
        case .server:       ServerScreen()
        case .status:       StatusScreen()
        case .logs:         LogsScreen()
        case .models:
            if let id = services.modelDetailID {
                ModelSettingsScreen(modelID: id)
            } else {
                ModelsScreen()
            }
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
    let detailTitle: String?
    @ViewBuilder var content: () -> Content

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Group {
            if section.fillsContentArea {
                // Skip the outer ScrollView so the screen can claim the
                // available height (Logs uses this for its monospace pane).
                VStack(alignment: .leading, spacing: 0) {
                    content()
                        .frame(maxWidth: 720, alignment: .topLeading)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                }
                .padding(.top, 20)
                .padding(.horizontal, 28)
                .padding(.bottom, 18)
            } else {
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
            }
        }
        .background(theme.contentBg)
        .navigationTitle(detailTitle ?? section.title)
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
