// PR — Downloads model card.
//
// Sheet that renders the upstream README for a model in the Downloads
// screen. Wired to two API surfaces:
//   • HF       → OMLXClient.getHFModelCard(repoId:)
//   • MS       → OMLXClient.getMSModelCard(modelId:)
//
// Both surface the same shape (model card + metadata; YAML front-matter
// stripped server-side, empty `model_card` when the upstream has no
// README), so a single sheet handles both — the `target`'s source enum
// just picks which client method to call and which upstream URL the
// "View on …" link points at.
//
// Rendering uses `MarkdownUI` (swift-markdown-ui) for full block support
// (headings, code blocks, tables, lists, images). Native
// `AttributedString(markdown:)` only handles inline formatting which
// would flatten the typical mlx-community README to plain text.

import SwiftUI
import AppKit
import MarkdownUI

// MARK: - Target / source

/// What the sheet is showing. Identifiable so `.sheet(item:)` can
/// drive presentation off the VM's `modelCardTarget` binding.
struct ModelCardTarget: Identifiable, Equatable, Sendable {
    let repoId: String
    let source: ModelCardSource

    /// `.sheet(item:)` uses this to decide when to re-present. Composing
    /// source into the id means switching HF↔MS for the same string
    /// re-fires the sheet's `.task(id:)` and refetches.
    var id: String { "\(source.rawValue):\(repoId)" }
}

enum ModelCardSource: String, Sendable, Equatable {
    case huggingFace = "hf"
    case modelScope  = "ms"

    var displayName: String {
        switch self {
        case .huggingFace: return "Hugging Face"
        case .modelScope:  return "ModelScope"
        }
    }

    /// Canonical upstream URL for the repo — used by the "View on …"
    /// footer link. Always points at the canonical origin even when the
    /// user has a mirror configured for downloads; the mirror is for
    /// transport, not browsing.
    func upstreamURL(repoId: String) -> URL? {
        switch self {
        case .huggingFace: return URL(string: "https://huggingface.co/\(repoId)")
        case .modelScope:  return URL(string: "https://modelscope.cn/models/\(repoId)")
        }
    }
}

/// Tabs available inside the sheet. The Files tab is HF-only — matches
/// the HTML admin which hides it for MS (MS file lists are usually
/// opaque sharded blobs that don't help the user decide).
enum ModelCardTab: Hashable, CaseIterable {
    case card, files, tags

    var label: String {
        switch self {
        case .card:
            return String(localized: "downloads.card.tab.card",
                          defaultValue: "Model Card",
                          comment: "Tab label in the model card sheet for the README content")
        case .files:
            return String(localized: "downloads.card.tab.files",
                          defaultValue: "Files",
                          comment: "Tab label in the model card sheet for the repo file list")
        case .tags:
            return String(localized: "downloads.card.tab.tags",
                          defaultValue: "Tags",
                          comment: "Tab label in the model card sheet for the repo tag list")
        }
    }

    static func available(for source: ModelCardSource) -> [ModelCardTab] {
        switch source {
        case .huggingFace: return [.card, .files, .tags]
        case .modelScope:  return [.card, .tags]
        }
    }
}

// MARK: - Sheet

@MainActor
struct ModelCardSheet: View {
    let target: ModelCardTarget
    let client: OMLXClient
    /// Triggered by the in-sheet Download button. The host wires this
    /// to `vm.startDownload(repo:)` so the action plays nicely with
    /// the existing source-routed downloader plumbing. Sheet dismisses
    /// itself on tap so the user lands back on the row with the task
    /// already moving in the Active section.
    let onDownload: (String) -> Void

    @State private var state: LoadState = .idle
    /// Currently selected tab in the sheet. Reset to `.card` whenever
    /// the target changes (different repo opened) so we never inherit
    /// a stale Files/Tags selection from a previous model.
    @State private var activeTab: ModelCardTab = .card
    @Environment(\.dismiss) private var dismiss
    @Environment(\.omlxTheme) private var theme

    enum LoadState: Equatable {
        case idle
        case loading
        case loaded(ModelCardDTO)
        case failed(String)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            stateRegion
        }
        .frame(width: 720, height: 600)
        .background(theme.windowBg)
        .task(id: target.id) {
            // Reset tab selection on every (re)open so a tab the
            // previous repo had (e.g. Files on HF) doesn't stick when
            // the user opens an MS card next.
            activeTab = .card
            await load()
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(target.repoId)
                    .font(.omlxText(15, weight: .semibold))
                    .foregroundStyle(theme.text)
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(target.source.displayName)
                    .font(.omlxText(11, weight: .medium))
                    .foregroundStyle(theme.textSecondary)
                    .textCase(.uppercase)
                    .kerning(0.6)
            }
            Spacer(minLength: 12)
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(theme.textSecondary)
                    .padding(6)
                    .background(theme.controlBg)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)
            .help(String(localized: "downloads.card.close",
                         defaultValue: "Close",
                         comment: "Tooltip on the model card sheet's close button"))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    // MARK: State-dependent region

    @ViewBuilder
    private var stateRegion: some View {
        switch state {
        case .idle, .loading:
            Divider().overlay(theme.groupBorder)
            loadingView
        case .failed(let message):
            Divider().overlay(theme.groupBorder)
            errorView(message: message)
        case .loaded(let dto):
            metadataRow(dto: dto)
            if dto.isAdapter == true {
                loraBanner
            }
            tabBar
            Divider().overlay(theme.groupBorder)
            tabBody(dto: dto)
            footer(dto: dto)
        }
    }

    // MARK: Tab bar

    private var tabBar: some View {
        let available = ModelCardTab.available(for: target.source)
        return HStack(spacing: 6) {
            ForEach(available, id: \.self) { tab in
                tabButton(tab: tab, isSelected: activeTab == tab)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 6)
    }

    private func tabButton(tab: ModelCardTab, isSelected: Bool) -> some View {
        Button {
            activeTab = tab
        } label: {
            Text(tab.label)
                .font(.omlxText(11.5, weight: .semibold))
                .foregroundStyle(isSelected ? theme.text : theme.textSecondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    Capsule().fill(isSelected ? theme.controlBg : Color.clear)
                )
                .overlay(
                    Capsule().strokeBorder(isSelected ? theme.inputBorder : Color.clear,
                                           lineWidth: 0.5)
                )
        }
        .buttonStyle(.plain)
    }

    // MARK: Tab body switch

    @ViewBuilder
    private func tabBody(dto: ModelCardDTO) -> some View {
        switch activeTab {
        case .card:
            cardBody(markdown: dto.modelCard)
        case .files:
            filesBody(files: dto.files ?? [])
        case .tags:
            tagsBody(tags: dto.tags ?? [])
        }
    }

    // MARK: Metadata badges (gap #1)

    @ViewBuilder
    private func metadataRow(dto: ModelCardDTO) -> some View {
        // Skip rendering the row entirely when none of the fields are
        // populated — keeps the sheet from showing a hollow strip on
        // metadata-poor repos.
        let hasAnyMetadata = dto.paramsFormatted != nil
            || dto.sizeFormatted != nil
            || dto.pipelineTag != nil
            || (dto.downloads ?? 0) > 0
            || (dto.likes ?? 0) > 0
        if hasAnyMetadata {
            HStack(spacing: 8) {
                if let p = dto.paramsFormatted, !p.isEmpty {
                    metaChip(text: p, accent: false)
                }
                if let s = dto.sizeFormatted, !s.isEmpty {
                    metaChip(text: s, accent: false)
                }
                if let tag = dto.pipelineTag, !tag.isEmpty {
                    metaChip(text: tag, accent: true)
                }
                Spacer(minLength: 6)
                if let d = dto.downloads, d > 0 {
                    metaCounter(symbol: "arrow.down.circle", value: d)
                }
                if let l = dto.likes, l > 0 {
                    metaCounter(symbol: "heart", value: l)
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 12)
        }
    }

    private func metaChip(text: String, accent: Bool) -> some View {
        Text(text)
            .font(.omlxText(10.5, weight: .heavy))
            .kerning(0.4)
            .textCase(.uppercase)
            .foregroundStyle(accent ? theme.accent : theme.textSecondary)
            .padding(.horizontal, 8)
            .frame(height: 22)
            .background(
                Capsule().fill(accent ? theme.accent.opacity(0.12) : theme.codeBg)
            )
            .overlay(
                Capsule().strokeBorder(accent ? theme.accent.opacity(0.25) : theme.inputBorder,
                                       lineWidth: 0.5)
            )
    }

    private func metaCounter(symbol: String, value: Int) -> some View {
        HStack(spacing: 3) {
            Image(systemName: symbol)
                .font(.system(size: 10, weight: .medium))
            Text(Self.compactCount(value))
                .font(.omlxMono(11))
        }
        .foregroundStyle(theme.textSecondary)
    }

    // MARK: LoRA warning (gap #4)

    private var loraBanner: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12))
                .foregroundStyle(theme.warningText)
            Text(String(localized: "downloads.card.lora_warning",
                        defaultValue: "This is a LoRA adapter. It needs a compatible base model to run — downloading on its own won't load in oMLX.",
                        comment: "Warning banner shown in the model card sheet when the repo is an adapter rather than a full model"))
                .font(.omlxText(11))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(theme.warningBg)
        .overlay(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(theme.warningText.opacity(0.25), lineWidth: 0.5)
        )
        .padding(.horizontal, 20)
        .padding(.bottom, 10)
    }

    // MARK: Card body

    @ViewBuilder
    private func cardBody(markdown raw: String) -> some View {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            VStack(spacing: 10) {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.system(size: 32, weight: .light))
                    .foregroundStyle(theme.textTertiary)
                Text(String(localized: "downloads.card.empty",
                            defaultValue: "This model doesn't ship a README.",
                            comment: "Empty state shown when the upstream repo has no model card"))
                    .font(.omlxText(13))
                    .foregroundStyle(theme.textSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(24)
        } else {
            ScrollView {
                Markdown(trimmed)
                    .markdownTheme(.docC)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(20)
                    .textSelection(.enabled)
            }
        }
    }

    // MARK: Files tab (gap #5)

    @ViewBuilder
    private func filesBody(files: [ModelCardFile]) -> some View {
        if files.isEmpty {
            emptyStateView(
                symbol: "doc.text.below.ecg",
                message: String(localized: "downloads.card.files.empty",
                                defaultValue: "No files reported.",
                                comment: "Empty state in the Files tab when the upstream API doesn't list any files")
            )
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(files.enumerated()), id: \.element.id) { idx, file in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Text(file.name)
                                .font(.omlxMono(12))
                                .foregroundStyle(theme.text)
                                .textSelection(.enabled)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer(minLength: 8)
                            if let s = file.sizeFormatted, !s.isEmpty {
                                Text(s)
                                    .font(.omlxMono(11))
                                    .foregroundStyle(theme.textSecondary)
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                        if idx < files.count - 1 {
                            Divider().opacity(0.4)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    // MARK: Tags tab (gap #6)

    @ViewBuilder
    private func tagsBody(tags: [String]) -> some View {
        if tags.isEmpty {
            emptyStateView(
                symbol: "tag",
                message: String(localized: "downloads.card.tags.empty",
                                defaultValue: "No tags.",
                                comment: "Empty state in the Tags tab when the upstream API doesn't list any tags")
            )
        } else {
            ScrollView {
                FlowLayout(spacing: 6) {
                    ForEach(tags, id: \.self) { tag in
                        tagPill(tag)
                    }
                }
                .padding(20)
            }
        }
    }

    private func tagPill(_ tag: String) -> some View {
        Text(tag)
            .font(.omlxMono(11))
            .foregroundStyle(theme.textSecondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                Capsule().fill(theme.codeBg)
            )
            .overlay(
                Capsule().strokeBorder(theme.inputBorder, lineWidth: 0.5)
            )
            .textSelection(.enabled)
    }

    private func emptyStateView(symbol: String, message: String) -> some View {
        VStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(theme.textTertiary)
            Text(message)
                .font(.omlxText(13))
                .foregroundStyle(theme.textSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    // MARK: Footer (gaps #2 + #3)

    @ViewBuilder
    private func footer(dto: ModelCardDTO) -> some View {
        Divider().overlay(theme.groupBorder)
        HStack(spacing: 10) {
            if let url = target.source.upstreamURL(repoId: target.repoId) {
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    Label {
                        Text(viewUpstreamLabel)
                    } icon: {
                        Image(systemName: "arrow.up.right.square")
                            .font(.system(size: 11))
                    }
                    .font(.omlxText(11.5, weight: .medium))
                    .foregroundStyle(theme.textSecondary)
                }
                .buttonStyle(.plain)
            }
            Spacer()
            if dto.isAdapter != true {
                // Adapters can't be downloaded as a runnable model;
                // hide the action entirely so the user doesn't kick
                // off a download that won't load.
                Button {
                    onDownload(target.repoId)
                    dismiss()
                } label: {
                    Label(String(localized: "downloads.card.download",
                                 defaultValue: "Download",
                                 comment: "Primary button in the model card sheet that starts downloading the displayed model"),
                          systemImage: "icloud.and.arrow.down")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.primary))
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
    }

    /// Source-specific label for the "View on …" link. Localized via
    /// inline interpolation of the source's display name.
    private var viewUpstreamLabel: String {
        String(localized: "downloads.card.view_upstream",
               defaultValue: "View on \(target.source.displayName)",
               comment: "Link button in the model card sheet that opens the upstream model page; placeholder is 'Hugging Face' or 'ModelScope'")
    }

    // MARK: Error / loading

    private var loadingView: some View {
        VStack(spacing: 10) {
            ProgressView()
            Text(String(localized: "downloads.card.loading",
                        defaultValue: "Loading model card…",
                        comment: "Status text shown while the model README is being fetched"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func errorView(message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(theme.redDot)
            Text(String(localized: "downloads.card.error",
                        defaultValue: "Couldn't load model card",
                        comment: "Title shown in the model card sheet when the fetch failed"))
                .font(.omlxText(13, weight: .semibold))
                .foregroundStyle(theme.text)
            Text(message)
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
            Button(String(localized: "downloads.card.retry",
                          defaultValue: "Retry",
                          comment: "Retry button shown after the model card fetch failed")) {
                Task { await load() }
            }
            .buttonStyle(.omlx(.primary))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    // MARK: Fetch

    private func load() async {
        state = .loading
        do {
            let dto: ModelCardDTO
            switch target.source {
            case .huggingFace:
                dto = try await client.getHFModelCard(repoId: target.repoId)
            case .modelScope:
                dto = try await client.getMSModelCard(modelId: target.repoId)
            }
            state = .loaded(dto)
        } catch {
            state = .failed(error.omlxDescription)
        }
    }

    // MARK: Helpers

    /// Compact-format a count for the metadata counters: 1234 → "1.2K",
    /// 1_500_000 → "1.5M". Mirrors the HTML admin's display so the same
    /// repos show the same numbers on both surfaces.
    private static func compactCount(_ n: Int) -> String {
        switch n {
        case 1_000_000_000...:
            return String(format: "%.1fB", Double(n) / 1_000_000_000)
        case 1_000_000...:
            return String(format: "%.1fM", Double(n) / 1_000_000)
        case 1_000...:
            return String(format: "%.1fK", Double(n) / 1_000)
        default:
            return String(n)
        }
    }
}

// `FlowLayout` (used by the Tags tab) lives in ProfileViews.swift —
// reused here for consistency. If it ever needs to specialize for the
// model-card sheet, prefer extending the shared type over forking.
