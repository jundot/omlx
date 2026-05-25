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
                // Wider default + min so the longest section labels
                // ("Performance", "Throughput", "Quantization") render
                // without truncation at the resting width.
                .navigationSplitViewColumnWidth(min: 250, ideal: 260, max: 320)
                // Lock the sidebar always-visible — kills the show/hide
                // animation that stuttered on macOS 26's NavigationSplitView
                // and lets the sidebar's glass be the dominant top-left
                // surface uninterrupted.
                .toolbar(removing: .sidebarToggle)
        } detail: {
            ContentScaffold(section: selection, detailTitle: detailTitle) {
                screen(for: selection)
            }
            .navigationSplitViewColumnWidth(min: 640, ideal: 920)
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 880, idealWidth: 1140, minHeight: 600, idealHeight: 760)
        // Hide the SwiftUI-injected window toolbar entirely. This is the
        // configuration System Settings.app uses on macOS 26 — no toolbar
        // zone reserved at the window top, sidebar's glass surface extends
        // up to under the traffic lights, and per-screen titles render
        // inside the detail content (handled by ContentScaffold).
        .toolbar(.hidden, for: .windowToolbar)
        // Extend the entire NavSplit content to fill behind the titlebar
        // area so the sidebar's glass (and DesktopWash on the detail) reach
        // the window's top edge. Each column's content is positioned via its
        // own .safeAreaInset(.top) so screen items stay below the traffic
        // lights despite the safe-area being ignored at this level.
        .ignoresSafeArea(.container, edges: .top)
        // DesktopWash backdrop: the design's radial-gradient wash that the
        // glass surfaces (sidebar, hero card) layer over. theme.windowBg
        // stays painted behind the wash as the opaque base so non-glass
        // areas keep a solid background.
        .background(DesktopWash())
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
        case .network:      NetworkScreen()
        case .performance:  PerformanceScreen()
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
        case .quantization: QuantizationScreen()
        case .throughputBench: ThroughputBenchScreen(vm: services.throughputBench)
        case .accuracyBench:   AccuracyBenchScreen(vm: services.accuracyBench)
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
    @EnvironmentObject private var services: AppServices

    /// Resolved section title, rendered as content (not via .navigationTitle)
    /// because the window toolbar is hidden — Settings.app pattern.
    private var titleText: String { detailTitle ?? section.title }

    @ViewBuilder
    private func sectionTitleHeader() -> some View {
        Text(titleText)
            .font(.omlxText(28, weight: .bold))
            .foregroundStyle(theme.text)
            .frame(maxWidth: .infinity, alignment: .leading)
            // Match the 14pt horizontal padding screen cards apply
            // internally so the title's left edge aligns with the
            // cards' left edge inside the 720pt centered frame.
            .padding(.horizontal, 14)
            .padding(.top, 36)
            .padding(.bottom, 6)
    }

    var body: some View {
        Group {
            if section.fillsContentArea {
                // Skip the outer ScrollView so the screen can claim the
                // available height (Logs uses this for its monospace pane).
                VStack(alignment: .leading, spacing: 0) {
                    sectionTitleHeader()
                    content()
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                }
                .frame(maxWidth: 720, alignment: .topLeading)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .padding(.bottom, 18)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 0) {
                            sectionTitleHeader()
                            content()
                                .padding(.top, 8)
                        }
                        // Wrap title + content together in a single max-width
                        // frame so the section title and the cards share the
                        // same left edge (Settings.app pattern: large title
                        // sits flush with content, not offset).
                        .frame(maxWidth: 720, alignment: .topLeading)
                        .frame(maxWidth: .infinity, alignment: .top)
                        .padding(.bottom, 36)
                    }
                    // Deep-link scroll: when another screen (e.g. the
                    // per-model "Edit on Server →" link) requested a
                    // jump to a named anchor *inside the section we just
                    // switched to*, scroll there once the inner view has
                    // had a runloop tick to lay out. The id includes
                    // both section and anchor so re-requesting the same
                    // anchor in the same section still fires.
                    .task(id: ScrollAnchorKey(section: section,
                                              anchor: services.requestedServerAnchor)) {
                        guard let anchor = services.requestedServerAnchor,
                              section == .server else { return }
                        // One render cycle to let ServerScreen mount its
                        // SectionHeader with the `.id()` we're targeting.
                        try? await Task.sleep(nanoseconds: 60_000_000)
                        withAnimation(.easeInOut(duration: 0.25)) {
                            proxy.scrollTo(anchor.rawValue, anchor: .top)
                        }
                        services.requestedServerAnchor = nil
                    }
                }
            }
        }
        // Title is rendered as content via sectionTitleHeader() — no
        // .navigationTitle here because the window toolbar is hidden in
        // AppView (matches the Settings.app pattern of inline titles on
        // floating-glass sidebar layouts).
    }
}

/// Composite identity used by `ContentScaffold`'s deep-link scroll
/// `.task(id:)` so the scroll fires whenever either the section or the
/// anchor changes — and re-fires if the same anchor is requested twice.
private struct ScrollAnchorKey: Equatable {
    let section: AppSection
    let anchor: ServerAnchor?
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
