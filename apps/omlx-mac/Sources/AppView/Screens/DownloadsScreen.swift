// PR 8 — Downloads screen.
//
// Wires the HF downloader endpoints (POST /admin/api/hf/download,
// GET /admin/api/hf/tasks at 1 Hz, cancel / retry / delete, /hf/recommended).
//
// Phase 2 — adds a source selector (HF / ModelScope) at the top. The MS
// branch mirrors the HF flow 1:1 against /admin/api/ms/*. Switching the
// source swaps the form + mirror editor; the task list, recent tasks, and
// suggested-models sections rebind to whichever source is active (their
// underlying DTOs are identical — see MSTaskDTO.swift).

import SwiftUI

/// Active downloader source. Mirrors the HTML admin's `downloaderSource`
/// state at `omlx/admin/static/js/dashboard.js:266`.
enum DownloadSource: String, CaseIterable, Hashable, Sendable {
    case hf, ms

    var label: String {
        switch self {
        case .hf: return String(localized: "downloads.source.hf",
                                defaultValue: "Hugging Face",
                                comment: "Source selector option label for Hugging Face")
        case .ms: return String(localized: "downloads.source.ms",
                                defaultValue: "ModelScope",
                                comment: "Source selector option label for ModelScope")
        }
    }
}

struct DownloadsScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = DownloadsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SourceSwitcher(
                source: $vm.source,
                msAvailable: vm.msAvailable
            )

            if vm.source == .hf {
                AddFromHFSection(
                    repoText: $vm.repoText,
                    isStarting: vm.isStarting,
                    mirrorHost: vm.mirrorHost,
                    mirrorIsCustom: vm.mirrorIsCustom,
                    isEditingMirror: $vm.isEditingMirror,
                    mirrorDraft: $vm.mirrorDraft,
                    mirrorBusy: vm.mirrorBusy,
                    searchResults: vm.searchResults,
                    searchLoading: vm.searchLoading,
                    searchDismissed: vm.searchDismissed,
                    onSubmit: { vm.startDownload(client: services.client) },
                    onSaveMirror: { vm.saveMirror(client: services.client) },
                    onResetMirror: { vm.resetMirror(client: services.client) },
                    onPickResult: { vm.pickSearchResult($0) },
                    onDismissSearch: { vm.dismissSearch() },
                    onShowCard: { repo in vm.showModelCard(repoId: repo) }
                )
                .onChange(of: vm.repoText) { _, newValue in
                    vm.updateSearch(query: newValue, client: services.client)
                }
            } else {
                AddFromMSSection(
                    repoText: $vm.msRepoText,
                    isStarting: vm.isStarting,
                    mirrorHost: vm.msMirrorHost,
                    mirrorIsCustom: vm.msMirrorIsCustom,
                    isEditingMirror: $vm.isEditingMsMirror,
                    mirrorDraft: $vm.msMirrorDraft,
                    mirrorBusy: vm.msMirrorBusy,
                    searchResults: vm.msSearchResults,
                    searchLoading: vm.msSearchLoading,
                    searchDismissed: vm.msSearchDismissed,
                    onSubmit: { vm.startDownload(client: services.client) },
                    onSaveMirror: { vm.saveMsMirror(client: services.client) },
                    onResetMirror: { vm.resetMsMirror(client: services.client) },
                    onPickResult: { vm.pickMsSearchResult($0) },
                    onDismissSearch: { vm.dismissMsSearch() },
                    onShowCard: { repo in vm.showModelCard(repoId: repo) }
                )
                .onChange(of: vm.msRepoText) { _, newValue in
                    vm.updateMsSearch(query: newValue, client: services.client)
                }
            }

            ActiveDownloadsSection(
                tasks: vm.activeTasks,
                onCancel: { id in vm.cancel(taskId: id, client: services.client) },
                onRemove: { id in vm.remove(taskId: id, client: services.client) }
            )

            CompletedTasksSection(
                tasks: vm.terminalTasks,
                onRetry: { id in vm.retry(taskId: id, client: services.client) },
                onRemove: { id in vm.remove(taskId: id, client: services.client) },
                onShowCard: { repo in vm.showModelCard(repoId: repo) }
            )

            SuggestedSection(
                models: vm.sortedRecommended,
                sort: $vm.recommendedSort,
                isLoading: vm.recommendedLoading,
                onGet: { repo in vm.startDownload(repo: repo, client: services.client) },
                onRefresh: { Task { await vm.loadRecommended(client: services.client) } },
                onShowCard: { repo in vm.showModelCard(repoId: repo) }
            )

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 8)
            }
        }
        .task { await vm.start(client: services.client) }
        .onDisappear { vm.stop() }
        .sheet(item: $vm.modelCardTarget) { target in
            ModelCardSheet(
                target: target,
                client: services.client,
                onDownload: { repo in
                    // Sheet dismisses itself before this fires; the
                    // download lands in the Active section under
                    // whichever source the sheet was opened from.
                    vm.startDownload(repo: repo, client: services.client)
                }
            )
        }
    }
}

// MARK: - Source switcher

/// Segmented HF / ModelScope toggle pinned at the top of Downloads. When
/// the server's modelscope SDK isn't installed (`/admin/api/ms/status`
/// returns `available: false`), the MS option is disabled with a tooltip
/// rather than hidden — so a user looking for it can see why it's not
/// usable.
private struct SourceSwitcher: View {
    @Binding var source: DownloadSource
    let msAvailable: Bool

    var body: some View {
        HStack(spacing: 8) {
            Segmented(
                selection: $source,
                options: DownloadSource.allCases.map { ($0, $0.label) }
            )
            .disabled(!msAvailable)
            if !msAvailable {
                Text(String(localized: "downloads.source.ms_unavailable",
                            defaultValue: "ModelScope SDK unavailable in this build",
                            comment: "Inline note shown beside the source switcher when the ModelScope SDK isn't installed"))
                    .font(.omlxText(10.5))
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.top, 4)
        .padding(.bottom, 8)
    }
}

// MARK: - Add from HF

private struct AddFromHFSection: View {
    @Binding var repoText: String
    let isStarting: Bool
    let mirrorHost: String
    let mirrorIsCustom: Bool
    @Binding var isEditingMirror: Bool
    @Binding var mirrorDraft: String
    let mirrorBusy: Bool
    let searchResults: [HFModelInfo]
    let searchLoading: Bool
    let searchDismissed: Bool
    let onSubmit: () -> Void
    let onSaveMirror: () -> Void
    let onResetMirror: () -> Void
    let onPickResult: (HFModelInfo) -> Void
    let onDismissSearch: () -> Void
    let onShowCard: (String) -> Void

    @Environment(\.omlxTheme) private var theme
    @FocusState private var mirrorFocused: Bool

    private var showsDropdown: Bool {
        !searchDismissed && (searchLoading || !searchResults.isEmpty)
    }

    var body: some View {
        SectionHeader(String(localized: "downloads.hf.section.title",
                              defaultValue: "Add Model from Hugging Face",
                              comment: "Section heading above the Hugging Face download form"))

        ListGroup {
            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 8) {
                        TextInput(
                            text: $repoText,
                            placeholder: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                            mono: true
                        )
                        .frame(maxWidth: .infinity)
                        .onSubmit(onSubmit)
                        if searchLoading {
                            ProgressView()
                                .controlSize(.small)
                                .padding(.trailing, 2)
                        }
                        Button {
                            onSubmit()
                        } label: {
                            Label(String(localized: "downloads.button.download",
                                         defaultValue: "Download",
                                         comment: "Primary button that starts downloading the entered repo"),
                                  systemImage: "icloud.and.arrow.down")
                                .labelStyle(.titleAndIcon)
                        }
                        .buttonStyle(.omlx(.primary))
                        .disabled(repoText.isEmpty || isStarting)
                    }
                    if showsDropdown {
                        SearchDropdown(
                            results: searchResults,
                            isLoading: searchLoading,
                            onPick: onPickResult,
                            onDismiss: onDismissSearch,
                            onShowCard: onShowCard
                        )
                    }
                    if isEditingMirror {
                        mirrorEditor
                    } else {
                        mirrorSummary
                    }
                }
            }
        }
    }

    private var mirrorSummary: some View {
        HStack(spacing: 8) {
            Image(systemName: "globe")
                .font(.system(size: 11))
                .foregroundStyle(theme.textTertiary)
            Text(String(localized: "downloads.mirror.label",
                        defaultValue: "Mirror:",
                        comment: "Inline label preceding the mirror host on the Downloads screen"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
            Text(mirrorHost)
                .font(.omlxMono(11))
                .foregroundStyle(theme.textSecondary)
            if mirrorIsCustom {
                Text(String(localized: "downloads.mirror.custom",
                            defaultValue: "custom",
                            comment: "Badge shown next to the mirror host when the user has configured a custom endpoint"))
                    .font(.omlxText(10, weight: .medium))
                    .foregroundStyle(theme.blueDot)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(theme.blueDot.opacity(0.12))
                    .clipShape(Capsule())
            }
            Spacer(minLength: 8)
            Button(String(localized: "downloads.mirror.configure",
                          defaultValue: "Configure mirror…",
                          comment: "Button that opens the inline mirror editor")) {
                mirrorDraft = mirrorIsCustom ? mirrorHost : ""
                isEditingMirror = true
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .disabled(mirrorBusy)
        }
    }

    private var mirrorEditor: some View {
        HStack(spacing: 8) {
            TextInput(
                text: $mirrorDraft,
                placeholder: String(localized: "downloads.hf.mirror.placeholder",
                                    defaultValue: "https://hf-mirror.com  (empty = huggingface.co)",
                                    comment: "Placeholder for the HF mirror endpoint input"),
                mono: true
            )
            .frame(maxWidth: .infinity)
            .focused($mirrorFocused)
            .onSubmit { onSaveMirror() }
            Button(String(localized: "downloads.mirror.reset",
                          defaultValue: "Reset",
                          comment: "Button that clears the mirror endpoint back to the default")) {
                mirrorDraft = ""
                onResetMirror()
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .disabled(mirrorBusy || (!mirrorIsCustom && mirrorDraft.isEmpty))
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Generic cancel button")) {
                isEditingMirror = false
                mirrorDraft = ""
            }
            .buttonStyle(.omlx(.normal, size: .small))
            .disabled(mirrorBusy)
            Button(String(localized: "common.save",
                          defaultValue: "Save",
                          comment: "Generic save button")) { onSaveMirror() }
                .buttonStyle(.omlx(.primary, size: .small))
                .disabled(mirrorBusy)
        }
        .onAppear { mirrorFocused = true }
    }
}

// MARK: - Add from MS

/// Visual + behavioral parallel of AddFromHFSection for the ModelScope flow.
/// Keeps the two source forms structurally identical so users moving between
/// them aren't relearning the affordances — only labels + placeholders change.
private struct AddFromMSSection: View {
    @Binding var repoText: String
    let isStarting: Bool
    let mirrorHost: String
    let mirrorIsCustom: Bool
    @Binding var isEditingMirror: Bool
    @Binding var mirrorDraft: String
    let mirrorBusy: Bool
    let searchResults: [MSModelInfo]
    let searchLoading: Bool
    let searchDismissed: Bool
    let onSubmit: () -> Void
    let onSaveMirror: () -> Void
    let onResetMirror: () -> Void
    let onPickResult: (MSModelInfo) -> Void
    let onDismissSearch: () -> Void
    let onShowCard: (String) -> Void

    @Environment(\.omlxTheme) private var theme
    @FocusState private var mirrorFocused: Bool

    private var showsDropdown: Bool {
        !searchDismissed && (searchLoading || !searchResults.isEmpty)
    }

    var body: some View {
        SectionHeader(String(localized: "downloads.ms.section.title",
                              defaultValue: "Add Model from ModelScope",
                              comment: "Section heading above the ModelScope download form"))

        ListGroup {
            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 8) {
                        TextInput(
                            text: $repoText,
                            placeholder: "mlx-community/Qwen2.5-7B-Instruct-4bit",
                            mono: true
                        )
                        .frame(maxWidth: .infinity)
                        .onSubmit(onSubmit)
                        if searchLoading {
                            ProgressView()
                                .controlSize(.small)
                                .padding(.trailing, 2)
                        }
                        Button {
                            onSubmit()
                        } label: {
                            Label(String(localized: "downloads.button.download",
                                         defaultValue: "Download",
                                         comment: "Primary button that starts downloading the entered repo"),
                                  systemImage: "icloud.and.arrow.down")
                                .labelStyle(.titleAndIcon)
                        }
                        .buttonStyle(.omlx(.primary))
                        .disabled(repoText.isEmpty || isStarting)
                    }
                    if showsDropdown {
                        SearchDropdown(
                            results: searchResults,
                            isLoading: searchLoading,
                            onPick: onPickResult,
                            onDismiss: onDismissSearch,
                            onShowCard: onShowCard
                        )
                    }
                    if isEditingMirror {
                        mirrorEditor
                    } else {
                        mirrorSummary
                    }
                }
            }
        }
    }

    private var mirrorSummary: some View {
        HStack(spacing: 8) {
            Image(systemName: "globe")
                .font(.system(size: 11))
                .foregroundStyle(theme.textTertiary)
            Text(String(localized: "downloads.mirror.label",
                        defaultValue: "Mirror:",
                        comment: "Inline label preceding the mirror host on the Downloads screen"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
            Text(mirrorHost)
                .font(.omlxMono(11))
                .foregroundStyle(theme.textSecondary)
            if mirrorIsCustom {
                Text(String(localized: "downloads.mirror.custom",
                            defaultValue: "custom",
                            comment: "Badge shown next to the mirror host when the user has configured a custom endpoint"))
                    .font(.omlxText(10, weight: .medium))
                    .foregroundStyle(theme.blueDot)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(theme.blueDot.opacity(0.12))
                    .clipShape(Capsule())
            }
            Spacer(minLength: 8)
            Button(String(localized: "downloads.mirror.configure",
                          defaultValue: "Configure mirror…",
                          comment: "Button that opens the inline mirror editor")) {
                mirrorDraft = mirrorIsCustom ? mirrorHost : ""
                isEditingMirror = true
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .disabled(mirrorBusy)
        }
    }

    private var mirrorEditor: some View {
        HStack(spacing: 8) {
            TextInput(
                text: $mirrorDraft,
                placeholder: String(localized: "downloads.ms.mirror.placeholder",
                                    defaultValue: "https://modelscope.cn  (empty = ModelScope default)",
                                    comment: "Placeholder for the ModelScope mirror endpoint input"),
                mono: true
            )
            .frame(maxWidth: .infinity)
            .focused($mirrorFocused)
            .onSubmit { onSaveMirror() }
            Button(String(localized: "downloads.mirror.reset",
                          defaultValue: "Reset",
                          comment: "Button that clears the mirror endpoint back to the default")) {
                mirrorDraft = ""
                onResetMirror()
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .disabled(mirrorBusy || (!mirrorIsCustom && mirrorDraft.isEmpty))
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Generic cancel button")) {
                isEditingMirror = false
                mirrorDraft = ""
            }
            .buttonStyle(.omlx(.normal, size: .small))
            .disabled(mirrorBusy)
            Button(String(localized: "common.save",
                          defaultValue: "Save",
                          comment: "Generic save button")) { onSaveMirror() }
                .buttonStyle(.omlx(.primary, size: .small))
                .disabled(mirrorBusy)
        }
        .onAppear { mirrorFocused = true }
    }
}

// MARK: - Search dropdown

private struct SearchDropdown: View {
    let results: [HFModelInfo]
    let isLoading: Bool
    let onPick: (HFModelInfo) -> Void
    let onDismiss: () -> Void
    /// Reveal a small `info.circle` action on the hovered row. Skipped
    /// from the always-visible button family because each dropdown row
    /// is itself a Button (`onPick`); a permanent trailing icon would
    /// crowd the dense row and confuse "did the user mean to pick this
    /// or open the card?". Finder-style hover reveal sidesteps both
    /// problems.
    let onShowCard: (String) -> Void

    @Environment(\.omlxTheme) private var theme
    @State private var hoveredId: String?

    /// Cap visible rows at 8 — anything more is noise and the user can keep
    /// typing to narrow further. The HF API returns up to `limit` items
    /// (20 by default in the VM), we just truncate the visible slice.
    private var visible: ArraySlice<HFModelInfo> { results.prefix(8) }

    var body: some View {
        VStack(spacing: 0) {
            if results.isEmpty && isLoading {
                HStack(spacing: 8) {
                    Text(String(localized: "downloads.search.loading",
                                defaultValue: "Searching…",
                                comment: "Placeholder shown inside the autocomplete dropdown while a search is in flight"))
                        .font(.omlxText(11))
                        .foregroundStyle(theme.textTertiary)
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            } else {
                ForEach(Array(visible.enumerated()), id: \.element.repoId) { idx, m in
                    // Two distinct tap targets in a single HStack:
                    //   1. row Button (left) — selects this result for download
                    //   2. info Button (right, hover-only) — opens the model card
                    // On hover we also hide the downloads count so the icon
                    // slots into the trailing area cleanly instead of overlapping.
                    HStack(spacing: 0) {
                        Button {
                            onPick(m)
                        } label: {
                            row(model: m, isHovered: hoveredId == m.repoId)
                        }
                        .buttonStyle(.plain)
                        .frame(maxWidth: .infinity)
                        if hoveredId == m.repoId {
                            Button {
                                onShowCard(m.repoId)
                            } label: {
                                Image(systemName: "info.circle")
                                    .font(.system(size: 12))
                                    .foregroundStyle(theme.textSecondary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .help(String(localized: "downloads.button.show_card",
                                         defaultValue: "View model card",
                                         comment: "Tooltip on the info button that opens a model's README sheet"))
                            .transition(.opacity)
                        }
                    }
                    .onHover { hovering in
                        withAnimation(.easeOut(duration: 0.1)) {
                            hoveredId = hovering ? m.repoId : nil
                        }
                    }
                    if idx < visible.count - 1 {
                        Divider().opacity(0.4)
                    }
                }
            }
        }
        .background(theme.groupBg)
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .onExitCommand(perform: onDismiss)
    }

    private func row(model m: HFModelInfo, isHovered: Bool) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 11))
                .foregroundStyle(theme.textTertiary)
            VStack(alignment: .leading, spacing: 1) {
                Text(m.repoId)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let detail = secondaryLine(m) {
                    Text(detail)
                        .font(.omlxText(10.5))
                        .foregroundStyle(theme.textTertiary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 6)
            // Hide the downloads count on hover so the trailing slot is
            // free for the info button (rendered as a sibling outside
            // this row Button so its tap doesn't fall through to onPick).
            if !isHovered, let downloads = m.downloads, downloads > 0 {
                Text(formatNumber(downloads))
                    .font(.omlxMono(10.5))
                    .foregroundStyle(theme.textTertiary)
                    .padding(.trailing, 12)
            }
        }
        .padding(.leading, 12)
        .padding(.vertical, 6)
        .contentShape(Rectangle())
    }

    private func secondaryLine(_ m: HFModelInfo) -> String? {
        var parts: [String] = []
        if let p = m.paramsFormatted, !p.isEmpty { parts.append(p) }
        if let s = m.sizeFormatted, !s.isEmpty { parts.append(s) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

// MARK: - Active downloads

private struct ActiveDownloadsSection: View {
    let tasks: [HFTaskDTO]
    let onCancel: (String) -> Void
    let onRemove: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "downloads.active.title",
                   defaultValue: "Active Downloads",
                   comment: "Section heading for the list of in-progress downloads"),
            subtitle: tasks.isEmpty
                ? String(localized: "downloads.active.subtitle.empty",
                         defaultValue: "No active tasks",
                         comment: "Subtitle for Active Downloads when there are none")
                : String(localized: "downloads.active.subtitle.running",
                         defaultValue: "\(tasks.count) running",
                         comment: "Subtitle for Active Downloads; placeholder is the count of running tasks")
        )

        if !tasks.isEmpty {
            ListGroup {
                ForEach(Array(tasks.enumerated()), id: \.element.id) { idx, task in
                    FreeRow(isLast: idx == tasks.count - 1) {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack(spacing: 8) {
                                Image(systemName: "icloud.and.arrow.down")
                                    .font(.system(size: 12))
                                    .foregroundStyle(theme.blueDot)
                                Text(task.repoId)
                                    .font(.omlxMono(12))
                                    .foregroundStyle(theme.text)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Spacer(minLength: 4)
                                Text(String(localized: "downloads.progress.bytes",
                                            defaultValue: "\(Int(task.progress))% · \(formatBytes(task.downloadedSize)) of \(formatBytes(task.totalSize))",
                                            comment: "Per-row progress line during downloads. Placeholders: percent, bytes downloaded, total bytes"))
                                    .font(.omlxMono(11))
                                    .foregroundStyle(theme.textSecondary)
                                Button {
                                    if task.statusEnum == .pending || task.statusEnum == .downloading {
                                        onCancel(task.taskId)
                                    } else {
                                        onRemove(task.taskId)
                                    }
                                } label: {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 11))
                                }
                                .buttonStyle(.omlx(.plain, size: .small))
                                .help(String(localized: "downloads.cancel.help",
                                             defaultValue: "Cancel",
                                             comment: "Tooltip on the X button that cancels or removes a download task"))
                            }
                            ProgressBar(progress: task.progress / 100, colors: [Color(rgb24: 0x0A84FF), Color(rgb24: 0x5E5CE6)])
                            HStack(spacing: 12) {
                                StatusChip(task: task)
                                if !task.error.isEmpty {
                                    Text(task.error)
                                        .font(.omlxMono(10.5))
                                        .foregroundStyle(theme.redDot)
                                        .lineLimit(2)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private struct StatusChip: View {
    let task: HFTaskDTO
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let cfg: (Color, String) = {
            switch task.statusEnum {
            case .downloading: return (theme.blueDot,
                                       String(localized: "downloads.status.downloading",
                                              defaultValue: "Downloading",
                                              comment: "Status chip label while a download is actively transferring bytes"))
            case .pending:     return (theme.amberDot,
                                       String(localized: "downloads.status.queued",
                                              defaultValue: "Queued",
                                              comment: "Status chip label for a download waiting to start"))
            case .completed:   return (theme.greenDot,
                                       String(localized: "downloads.status.completed",
                                              defaultValue: "Completed",
                                              comment: "Status chip label for a finished download"))
            case .failed:      return (theme.redDot,
                                       String(localized: "downloads.status.failed",
                                              defaultValue: "Failed",
                                              comment: "Status chip label for a download that errored out"))
            case .cancelled:   return (theme.textTertiary,
                                       String(localized: "downloads.status.cancelled",
                                              defaultValue: "Cancelled",
                                              comment: "Status chip label for a download cancelled by the user"))
            case .paused:      return (theme.amberDot,
                                       String(localized: "downloads.status.paused",
                                              defaultValue: "Paused",
                                              comment: "Status chip label for a paused download"))
            case .none:        return (theme.textTertiary, task.status.capitalized)
            }
        }()
        StatusPill(status: .custom(color: cfg.0, label: cfg.1, fillBg: true))
    }
}

// MARK: - Completed / Failed tasks

private struct CompletedTasksSection: View {
    let tasks: [HFTaskDTO]
    let onRetry: (String) -> Void
    let onRemove: (String) -> Void
    /// Open the model-card sheet for a row. Source is resolved by the VM
    /// from the active downloader tab.
    let onShowCard: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        if tasks.isEmpty { EmptyView() } else {
            SectionHeader(String(localized: "downloads.recent.title",
                                  defaultValue: "Recent Tasks",
                                  comment: "Section heading for recently completed or failed downloads"),
                          subtitle: String(localized: "downloads.recent.subtitle",
                                           defaultValue: "\(tasks.count) recent",
                                           comment: "Subtitle for Recent Tasks; placeholder is the count of recent terminal tasks"))
            ListGroup {
                ForEach(Array(tasks.enumerated()), id: \.element.id) { idx, task in
                    FreeRow(isLast: idx == tasks.count - 1) {
                        HStack(spacing: 8) {
                            StatusChip(task: task)
                            Text(task.repoId)
                                .font(.omlxMono(12))
                                .foregroundStyle(theme.text)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer(minLength: 4)
                            if task.statusEnum == .failed || task.statusEnum == .cancelled {
                                Button(String(localized: "downloads.button.retry",
                                              defaultValue: "Retry",
                                              comment: "Button label that re-runs a failed or cancelled download")) { onRetry(task.taskId) }
                                    .buttonStyle(.omlx(.normal, size: .small))
                            }
                            Button {
                                onShowCard(task.repoId)
                            } label: {
                                Image(systemName: "info.circle")
                                    .font(.system(size: 11))
                            }
                            .buttonStyle(.omlx(.plain, size: .small))
                            .help(String(localized: "downloads.button.show_card",
                                         defaultValue: "View model card",
                                         comment: "Tooltip on the info button that opens a model's README sheet"))
                            Button {
                                onRemove(task.taskId)
                            } label: {
                                Image(systemName: "trash")
                                    .font(.system(size: 11))
                            }
                            .buttonStyle(.omlx(.plain, size: .small))
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Suggested

private struct SuggestedSection: View {
    let models: [HFModelInfo]
    @Binding var sort: SuggestedSort
    let isLoading: Bool
    let onGet: (String) -> Void
    let onRefresh: () -> Void
    /// Open the model-card sheet for a row. Same intent as the equivalent
    /// callback on CompletedTasksSection.
    let onShowCard: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(String(localized: "downloads.suggested.title",
                              defaultValue: "Suggested Models",
                              comment: "Section heading for the recommended-models section"),
                      subtitle: hint) {
            HStack(spacing: 6) {
                Popup(
                    selection: $sort,
                    width: 170,
                    options: SuggestedSort.allCases.map { ($0, $0.label) }
                )
                Button {
                    onRefresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.omlx(.normal, size: .small))
                .disabled(isLoading)
            }
        }

        ListGroup {
            if isLoading && models.isEmpty {
                FreeRow(isLast: true) {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text(String(localized: "downloads.suggested.loading",
                                    defaultValue: "Loading recommendations…",
                                    comment: "Placeholder shown while the recommended-models list is fetching"))
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textSecondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 14)
                }
            } else if models.isEmpty {
                FreeRow(isLast: true) {
                    Text(String(localized: "downloads.suggested.empty",
                                defaultValue: "No suggestions available right now.",
                                comment: "Empty-state message for the Suggested Models section"))
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 14)
                }
            } else {
                ForEach(Array(models.prefix(15).enumerated()), id: \.element.id) { idx, m in
                    let isLast = idx == min(models.count, 15) - 1
                    FreeRow(isLast: isLast) {
                        HStack(spacing: 10) {
                            Squircle(systemSymbol: "cpu",
                                     size: 26,
                                     gradient: SquircleGradient.models)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(m.repoId)
                                    .font(.omlxText(13, weight: .medium))
                                    .foregroundStyle(theme.text)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                                Text(secondaryLine(for: m))
                                    .font(.omlxMono(11))
                                    .foregroundStyle(theme.textSecondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            Spacer(minLength: 8)
                            Button {
                                onShowCard(m.repoId)
                            } label: {
                                Image(systemName: "info.circle")
                                    .font(.system(size: 11))
                            }
                            .buttonStyle(.omlx(.plain, size: .small))
                            .help(String(localized: "downloads.button.show_card",
                                         defaultValue: "View model card",
                                         comment: "Tooltip on the info button that opens a model's README sheet"))
                            Button {
                                onGet(m.repoId)
                            } label: {
                                Label(String(localized: "downloads.suggested.get",
                                             defaultValue: "Get",
                                             comment: "Compact button label that starts downloading a suggested model"),
                                      systemImage: "icloud.and.arrow.down")
                                    .labelStyle(.titleAndIcon)
                            }
                            .buttonStyle(.omlx(.normal, size: .small))
                        }
                    }
                }
            }
        }
    }

    private var hint: String? {
        models.isEmpty
            ? nil
            : String(localized: "downloads.suggested.hint.filtered_by_ram",
                     defaultValue: "Filtered by free RAM",
                     comment: "Subtitle hint on the Suggested Models header explaining the filter")
    }

    private func secondaryLine(for m: HFModelInfo) -> String {
        var bits: [String] = []
        if let p = m.paramsFormatted { bits.append(p) }
        if let s = m.sizeFormatted { bits.append(s) }
        if let dl = m.downloads { bits.append("\(formatNumber(dl)) ↓") }
        return bits.isEmpty ? "—" : bits.joined(separator: " · ")
    }
}

// MARK: - Sort

enum SuggestedSort: String, Hashable, CaseIterable {
    case downloads, params, size

    var label: String {
        switch self {
        case .downloads: return String(localized: "downloads.suggested.sort.downloads",
                                       defaultValue: "Most downloaded",
                                       comment: "Sort option: rank suggested models by download count")
        case .params:    return String(localized: "downloads.suggested.sort.params",
                                       defaultValue: "Parameters: high to low",
                                       comment: "Sort option: rank suggested models by parameter count, descending")
        case .size:      return String(localized: "downloads.suggested.sort.size",
                                       defaultValue: "Size: high to low",
                                       comment: "Sort option: rank suggested models by on-disk size, descending")
        }
    }
}

// MARK: - Helpers

func formatNumber(_ n: Int) -> String {
    let v = Double(n)
    if v >= 1e9 { return String(format: "%.1fB", v / 1e9) }
    if v >= 1e6 { return String(format: "%.1fM", v / 1e6) }
    if v >= 1e3 { return String(format: "%.1fK", v / 1e3) }
    return String(n)
}
