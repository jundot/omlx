// AppView shell. NavigationSplitView backed by the `AppSection` enum, one
// sidebar row per screen, grouped into Server / Models / Benchmark / General
// sections. Sized close to the minimum comfortable settings window so the
// first open does not feel oversized, while still surviving a resize.
//
// The shell is the entry point for the menubar's `Admin Panel` item and is
// hosted in the SwiftUI `Window` scene declared in `oMLXApp.swift`.

import AppKit
import MarkdownUI
import SwiftUI

struct AppView: View {
    @State private var selection: AppSection? = .status
    @State private var presentedUpdate: AvailableUpdate?

    @Environment(\.colorScheme) private var scheme
    @Environment(AppServices.self) private var services

    var body: some View {
        let theme = scheme == .dark ? OMLXTheme.dark : OMLXTheme.light
        let section = selectedSection

        NavigationSplitView {
            SettingsSidebar(selection: bindingForSelection())
        } detail: {
            ContentScaffold(section: section, detailTitle: detailTitle) {
                screen(for: section)
            }
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 880, idealWidth: 880, minHeight: 600, idealHeight: 600)
        // The theme resolves this through the dynamic macOS window color so
        // the shell tracks System Settings instead of a fixed canvas color.
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
        .onChange(of: services.updates.confirmationUpdate, initial: true) { _, update in
            presentedUpdate = update
        }
        .sheet(item: $presentedUpdate, onDismiss: {
            services.updates.dismissUpdateConfirmation()
        }) { update in
            UpdateConfirmationSheet(
                update: update,
                updates: services.updates,
                onLater: {
                    services.updates.deferUpdate(update)
                    presentedUpdate = nil
                },
                onConfirm: {
                    services.updates.confirmUpdate(update)
                    presentedUpdate = nil
                }
            )
                .environment(\.omlxTheme, theme)
        }
    }

    /// Drilling out of ModelSettingsScreen via the sidebar (changing section)
    /// must clear the per-model detail id so we don't accidentally re-enter
    /// the detail when the user returns to Models.
    private func bindingForSelection() -> Binding<AppSection?> {
        Binding(
            get: { selection },
            set: { newValue in
                guard let newValue else { return }
                if newValue != .models { services.modelDetailID = nil }
                selection = newValue
            }
        )
    }

    private var selectedSection: AppSection {
        selection ?? .status
    }

    private var detailTitle: String? {
        if selectedSection == .models, let id = services.modelDetailID, !id.isEmpty {
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

// MARK: - Update confirmation

@MainActor
private struct UpdateConfirmationSheet: View {
    let update: AvailableUpdate
    let updates: UpdateController
    let onLater: () -> Void
    let onConfirm: () -> Void

    @Environment(\.omlxTheme) private var theme

    private var trimmedNotes: String {
        update.notes.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var noteBlocks: [ReleaseNotesBlock] {
        ReleaseNotesHTML.blocks(from: trimmedNotes)
    }

    private var isStaged: Bool {
        if case .ready(let ready) = updates.state {
            return ready.version == update.version
        }
        return false
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            notesBody
            Divider()
            footer
        }
        .frame(width: 680, height: 560)
        .background(theme.windowBg)
    }

    private var header: some View {
        HStack(spacing: 12) {
            Squircle(systemSymbol: "arrow.down.circle.fill",
                     size: 34,
                     gradient: SquircleGradient.update)
            VStack(alignment: .leading, spacing: 4) {
                Text(String(localized: "update.confirm.title",
                            defaultValue: "oMLX \(update.version) is available",
                            comment: "Update confirmation sheet title; placeholder is the version"))
                    .font(.omlxText(17, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(String(localized: "update.confirm.subtitle",
                            defaultValue: "Review the release notes before downloading and relaunching.",
                            comment: "Subtitle for the update confirmation sheet"))
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }
            Spacer()
            Button {
                NSWorkspace.shared.open(update.htmlURL)
            } label: {
                Image(systemName: "arrow.up.right.square")
                    .font(.system(size: 13, weight: .semibold))
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .help(String(localized: "update.confirm.view_release",
                         defaultValue: "View release on GitHub",
                         comment: "Tooltip for the release link button in the update confirmation sheet"))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }

    @ViewBuilder
    private var notesBody: some View {
        if trimmedNotes.isEmpty {
            VStack(spacing: 10) {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.system(size: 30, weight: .light))
                    .foregroundStyle(theme.textTertiary)
                Text(String(localized: "update.confirm.empty_notes",
                            defaultValue: "This release does not include detailed notes.",
                            comment: "Empty state when a GitHub release has no release notes"))
                    .font(.omlxText(13))
                    .foregroundStyle(theme.textSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(24)
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(noteBlocks) { block in
                        switch block {
                        case .markdown(let text):
                            Markdown(text)
                                .markdownTheme(.docC)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        case .imageGroup(let images):
                            ReleaseNotesImageGroup(images: images)
                        }
                    }
                }
                .padding(20)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(String(localized: "update.confirm.size",
                            defaultValue: "Download size: \(update.sizeText ?? "Unknown")",
                            comment: "Update confirmation download size line; placeholder is a formatted byte size or Unknown"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                Text(String(localized: "update.confirm.restart_notice",
                            defaultValue: "oMLX will quit, install the update, and relaunch.",
                            comment: "Notice explaining what happens after confirming an update"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
            }
            Spacer()
            Button(String(localized: "update.confirm.later",
                          defaultValue: "Later",
                          comment: "Dismiss button in the update confirmation sheet")) {
                onLater()
            }
            .buttonStyle(.omlx(.normal))
            Button(primaryButtonTitle) {
                onConfirm()
            }
            .buttonStyle(.omlx(.primary))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var primaryButtonTitle: String {
        if isStaged {
            return String(localized: "update.confirm.install_ready",
                          defaultValue: "Install & Relaunch",
                          comment: "Primary button when the update is already staged")
        }
        return String(localized: "update.confirm.install",
                      defaultValue: "Download, Install & Relaunch",
                      comment: "Primary button to download, install, and relaunch")
    }
}

private enum ReleaseNotesBlock: Identifiable {
    case markdown(String)
    case imageGroup([ReleaseNotesImage])

    var id: String {
        switch self {
        case .markdown(let text):
            return "markdown:\(text.hashValue)"
        case .imageGroup(let images):
            return "images:\(images.map(\.id).joined(separator: ","))"
        }
    }
}

private struct ReleaseNotesImage: Identifiable, Equatable {
    let url: URL
    let alt: String

    var id: String { "\(url.absoluteString):\(alt)" }
}

private enum ReleaseNotesHTML {
    private static let imageParagraphPattern = #"(?is)<p\b[^>]*>\s*((?:<img\b[^>]*>\s*)+)</p>"#
    private static let imagePattern = #"(?is)<img\b([^>]*)>"#
    private static let attributePattern = #"([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(['"])(.*?)\2"#

    static func blocks(from raw: String) -> [ReleaseNotesBlock] {
        guard let paragraphRegex = try? NSRegularExpression(pattern: imageParagraphPattern) else {
            return markdownBlocks(raw)
        }

        var blocks: [ReleaseNotesBlock] = []
        var cursor = raw.startIndex
        let fullRange = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        let matches = paragraphRegex.matches(in: raw, range: fullRange)

        for match in matches {
            guard let matchRange = Range(match.range, in: raw) else { continue }
            appendMarkdown(String(raw[cursor..<matchRange.lowerBound]), to: &blocks)

            if match.numberOfRanges > 1,
               let bodyRange = Range(match.range(at: 1), in: raw) {
                let images = extractImages(from: String(raw[bodyRange]))
                if images.isEmpty {
                    appendMarkdown(String(raw[matchRange]), to: &blocks)
                } else {
                    blocks.append(.imageGroup(images))
                }
            }

            cursor = matchRange.upperBound
        }

        appendMarkdown(String(raw[cursor..<raw.endIndex]), to: &blocks)
        return blocks.isEmpty ? markdownBlocks(raw) : blocks
    }

    private static func markdownBlocks(_ raw: String) -> [ReleaseNotesBlock] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? [] : [.markdown(trimmed)]
    }

    private static func appendMarkdown(_ raw: String, to blocks: inout [ReleaseNotesBlock]) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            blocks.append(.markdown(trimmed))
        }
    }

    private static func extractImages(from html: String) -> [ReleaseNotesImage] {
        guard let imageRegex = try? NSRegularExpression(pattern: imagePattern) else { return [] }
        let fullRange = NSRange(html.startIndex..<html.endIndex, in: html)
        return imageRegex.matches(in: html, range: fullRange).compactMap { match in
            guard match.numberOfRanges > 1,
                  let attrsRange = Range(match.range(at: 1), in: html)
            else { return nil }

            let attrs = attributes(from: String(html[attrsRange]))
            guard let src = attrs["src"],
                  let url = URL(string: decodeHTMLEntities(src))
            else { return nil }

            return ReleaseNotesImage(
                url: url,
                alt: attrs["alt"].map(decodeHTMLEntities) ?? ""
            )
        }
    }

    private static func attributes(from raw: String) -> [String: String] {
        guard let attrRegex = try? NSRegularExpression(pattern: attributePattern) else { return [:] }
        var attrs: [String: String] = [:]
        let fullRange = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        for match in attrRegex.matches(in: raw, range: fullRange) {
            guard match.numberOfRanges > 3,
                  let keyRange = Range(match.range(at: 1), in: raw),
                  let valueRange = Range(match.range(at: 3), in: raw)
            else { continue }
            attrs[String(raw[keyRange]).lowercased()] = String(raw[valueRange])
        }
        return attrs
    }

    private static func decodeHTMLEntities(_ raw: String) -> String {
        guard let data = raw.data(using: .utf8),
              let decoded = try? NSAttributedString(
                data: data,
                options: [
                    .documentType: NSAttributedString.DocumentType.html,
                    .characterEncoding: String.Encoding.utf8.rawValue,
                ],
                documentAttributes: nil
              ).string
        else { return raw }
        return decoded
    }
}

private struct ReleaseNotesImageGroup: View {
    let images: [ReleaseNotesImage]

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            ForEach(images) { image in
                AsyncImage(url: image.url) { phase in
                    switch phase {
                    case .empty:
                        ProgressView()
                            .frame(maxWidth: .infinity, minHeight: 120)
                    case .success(let rendered):
                        rendered
                            .resizable()
                            .scaledToFit()
                            .accessibilityLabel(image.alt)
                    case .failure:
                        VStack(spacing: 8) {
                            Image(systemName: "photo")
                                .font(.system(size: 24, weight: .light))
                                .foregroundStyle(theme.textTertiary)
                            if !image.alt.isEmpty {
                                Text(image.alt)
                                    .font(.omlxText(11))
                                    .foregroundStyle(theme.textSecondary)
                                    .multilineTextAlignment(.center)
                            }
                        }
                        .frame(maxWidth: .infinity, minHeight: 120)
                    @unknown default:
                        EmptyView()
                    }
                }
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(theme.groupBorder.opacity(0.55), lineWidth: 1)
                )
            }
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Sidebar

private struct SettingsSidebar: View {
    @Binding var selection: AppSection?

    var body: some View {
        List(selection: $selection) {
            Section {
                SidebarRow(section: .status)
                SidebarRow(section: .server)
                SidebarRow(section: .network)
                SidebarRow(section: .performance)
                SidebarRow(section: .logs)
            } header: {
                Text(String(localized: "sidebar.group.server",
                            defaultValue: "Server",
                            comment: "Sidebar group heading for server-related screens"))
            }
            Section {
                SidebarRow(section: .models)
                SidebarRow(section: .downloads)
                SidebarRow(section: .integrations)
                SidebarRow(section: .quantization)
            } header: {
                Text(String(localized: "sidebar.group.models",
                            defaultValue: "Models",
                            comment: "Sidebar group heading for models/downloads/quant screens"))
            }
            Section {
                SidebarRow(section: .throughputBench)
                SidebarRow(section: .accuracyBench)
            } header: {
                Text(String(localized: "sidebar.group.benchmark",
                            defaultValue: "Benchmark",
                            comment: "Sidebar group heading for accuracy + throughput bench screens"))
            }
            Section {
                SidebarRow(section: .security)
                SidebarRow(section: .about)
            } header: {
                Text(String(localized: "sidebar.group.general",
                            defaultValue: "General",
                            comment: "Sidebar group heading for the about/integrations/logs screens"))
            }
        }
        .listStyle(.sidebar)
        .navigationSplitViewColumnWidth(min: 180, ideal: 195, max: 215)
    }
}

private struct SidebarRow: View {
    let section: AppSection

    var body: some View {
        NavigationLink(value: section) {
            Label(section.title, systemImage: section.symbol)
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
    @Environment(AppServices.self) private var services

    private var titleText: String { detailTitle ?? section.title }

    var body: some View {
        Group {
            if section.fillsContentArea {
                content()
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .frame(maxWidth: 720, alignment: .topLeading)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .padding(.bottom, 18)
                    .background(theme.windowBg)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        content()
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
                    .scrollContentBackground(.hidden)
                    .background(theme.windowBg)
                }
            }
        }
        .navigationTitle(titleText)
        .background(theme.windowBg)
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
