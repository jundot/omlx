// PR 8 — Downloads screen.
//
// Wires the HF downloader endpoints (POST /admin/api/hf/download,
// GET /admin/api/hf/tasks at 1 Hz, cancel / retry / delete, /hf/recommended).
// ModelScope is browser-only (plan §1) and not surfaced here.

import SwiftUI

struct DownloadsScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = DownloadsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            AddFromHFSection(
                repoText: $vm.repoText,
                isStarting: vm.isStarting,
                mirrorHost: vm.mirrorHost,
                mirrorIsCustom: vm.mirrorIsCustom,
                isEditingMirror: $vm.isEditingMirror,
                mirrorDraft: $vm.mirrorDraft,
                mirrorBusy: vm.mirrorBusy,
                onSubmit: { vm.startDownload(client: services.client) },
                onSaveMirror: { vm.saveMirror(client: services.client) },
                onResetMirror: { vm.resetMirror(client: services.client) }
            )

            ActiveDownloadsSection(
                tasks: vm.activeTasks,
                onCancel: { id in vm.cancel(taskId: id, client: services.client) },
                onRemove: { id in vm.remove(taskId: id, client: services.client) }
            )

            CompletedTasksSection(
                tasks: vm.terminalTasks,
                onRetry: { id in vm.retry(taskId: id, client: services.client) },
                onRemove: { id in vm.remove(taskId: id, client: services.client) }
            )

            SuggestedSection(
                models: vm.sortedRecommended,
                sort: $vm.recommendedSort,
                isLoading: vm.recommendedLoading,
                onGet: { repo in vm.startDownload(repo: repo, client: services.client) },
                onRefresh: { Task { await vm.loadRecommended(client: services.client) } }
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
    let onSubmit: () -> Void
    let onSaveMirror: () -> Void
    let onResetMirror: () -> Void

    @Environment(\.omlxTheme) private var theme
    @FocusState private var mirrorFocused: Bool

    var body: some View {
        SectionHeader("Add Model from Hugging Face")

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
                        Button {
                            onSubmit()
                        } label: {
                            Label("Download", systemImage: "icloud.and.arrow.down")
                                .labelStyle(.titleAndIcon)
                        }
                        .buttonStyle(.omlx(.primary))
                        .disabled(repoText.isEmpty || isStarting)
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
            Text("Mirror:")
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
            Text(mirrorHost)
                .font(.omlxMono(11))
                .foregroundStyle(theme.textSecondary)
            if mirrorIsCustom {
                Text("custom")
                    .font(.omlxText(10, weight: .medium))
                    .foregroundStyle(theme.blueDot)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(theme.blueDot.opacity(0.12))
                    .clipShape(Capsule())
            }
            Spacer(minLength: 8)
            Button("Configure mirror…") {
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
                placeholder: "https://hf-mirror.com  (empty = huggingface.co)",
                mono: true
            )
            .frame(maxWidth: .infinity)
            .focused($mirrorFocused)
            .onSubmit { onSaveMirror() }
            Button("Reset") {
                mirrorDraft = ""
                onResetMirror()
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .disabled(mirrorBusy || (!mirrorIsCustom && mirrorDraft.isEmpty))
            Button("Cancel") {
                isEditingMirror = false
                mirrorDraft = ""
            }
            .buttonStyle(.omlx(.normal, size: .small))
            .disabled(mirrorBusy)
            Button("Save") { onSaveMirror() }
                .buttonStyle(.omlx(.primary, size: .small))
                .disabled(mirrorBusy)
        }
        .onAppear { mirrorFocused = true }
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
            "Active Downloads",
            subtitle: tasks.isEmpty ? "No active tasks" : "\(tasks.count) running"
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
                                Text("\(Int(task.progress))% · \(formatBytes(task.downloadedSize)) of \(formatBytes(task.totalSize))")
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
                                .help("Cancel")
                            }
                            ProgressBar(progress: task.progress / 100)
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
            case .downloading: return (theme.blueDot, "Downloading")
            case .pending:     return (theme.amberDot, "Queued")
            case .completed:   return (theme.greenDot, "Completed")
            case .failed:      return (theme.redDot, "Failed")
            case .cancelled:   return (theme.textTertiary, "Cancelled")
            case .paused:      return (theme.amberDot, "Paused")
            case .none:        return (theme.textTertiary, task.status.capitalized)
            }
        }()
        StatusPill(status: .custom(color: cfg.0, label: cfg.1, fillBg: true))
    }
}

private struct ProgressBar: View {
    let progress: Double
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(theme.codeBg)
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(LinearGradient(
                        colors: [
                            Color(rgb24: 0x0A84FF),
                            Color(rgb24: 0x5E5CE6),
                        ],
                        startPoint: .leading, endPoint: .trailing
                    ))
                    .frame(width: geo.size.width * max(0, min(progress, 1)))
                    .animation(.easeOut(duration: 0.4), value: progress)
            }
        }
        .frame(height: 4)
    }
}

// MARK: - Completed / Failed tasks

private struct CompletedTasksSection: View {
    let tasks: [HFTaskDTO]
    let onRetry: (String) -> Void
    let onRemove: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        if tasks.isEmpty { EmptyView() } else {
            SectionHeader("Recent Tasks", subtitle: "\(tasks.count) recent")
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
                                Button("Retry") { onRetry(task.taskId) }
                                    .buttonStyle(.omlx(.normal, size: .small))
                            }
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

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader("Suggested Models", subtitle: hint) {
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
                        Text("Loading recommendations…")
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textSecondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 14)
                }
            } else if models.isEmpty {
                FreeRow(isLast: true) {
                    Text("No suggestions available right now.")
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
                                onGet(m.repoId)
                            } label: {
                                Label("Get", systemImage: "icloud.and.arrow.down")
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
        models.isEmpty ? nil : "Filtered by free RAM"
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
        case .downloads: return "Most downloaded"
        case .params:    return "Parameters: high to low"
        case .size:      return "Size: high to low"
        }
    }
}

// MARK: - View model

@MainActor
final class DownloadsScreenVM: ObservableObject {
    @Published var repoText: String = ""
    @Published private(set) var tasks: [HFTaskDTO] = []
    @Published private(set) var recommended: [HFModelInfo] = []
    @Published private(set) var isStarting: Bool = false
    @Published private(set) var recommendedLoading: Bool = false
    @Published var recommendedSort: SuggestedSort = .downloads
    @Published var lastError: String?

    /// Configured HF mirror endpoint. Empty when using the HF default
    /// (huggingface.co). Loaded once on screen start, kept in sync with
    /// PATCH /admin/api/global-settings (`hf_endpoint`).
    @Published private(set) var mirrorEndpoint: String = ""
    @Published var isEditingMirror: Bool = false
    @Published var mirrorDraft: String = ""
    @Published private(set) var mirrorBusy: Bool = false

    /// Host the user sees on the Downloads screen. Strips scheme so the
    /// inline label reads like `huggingface.co` / `hf-mirror.com` per design.
    var mirrorHost: String {
        let trimmed = mirrorEndpoint.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return "huggingface.co" }
        if let url = URL(string: trimmed), let host = url.host { return host }
        return trimmed
    }

    var mirrorIsCustom: Bool {
        !mirrorEndpoint.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Re-sorted view of `recommended`. Always descending; entries missing
    /// the sort key fall to the bottom so they don't shove valid models out
    /// of the top 15 the section displays.
    var sortedRecommended: [HFModelInfo] {
        switch recommendedSort {
        case .downloads:
            return recommended.sorted { ($0.downloads ?? -1) > ($1.downloads ?? -1) }
        case .params:
            return recommended.sorted { ($0.params ?? -1) > ($1.params ?? -1) }
        case .size:
            return recommended.sorted { ($0.size ?? -1) > ($1.size ?? -1) }
        }
    }

    private weak var client: OMLXClient?
    private var pollTask: Task<Void, Never>?
    private var hasLoadedRecommended = false

    var activeTasks: [HFTaskDTO]   { tasks.filter { $0.isActive } }
    var terminalTasks: [HFTaskDTO] { tasks.filter { !$0.isActive } }

    func start(client: OMLXClient) async {
        self.client = client
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.refreshTasks()
                try? await Task.sleep(for: .seconds(1))
            }
        }
        await refreshMirror(client: client)
        if !hasLoadedRecommended {
            hasLoadedRecommended = true
            await loadRecommended(client: client)
        }
    }

    private func refreshMirror(client: OMLXClient) async {
        do {
            let settings = try await client.getGlobalSettings()
            self.mirrorEndpoint = settings.huggingface?.endpoint ?? ""
        } catch {
            // Non-fatal — leave mirror as default and don't clobber an
            // existing error from a download call.
        }
    }

    func saveMirror(client: OMLXClient) {
        let draft = mirrorDraft.trimmingCharacters(in: .whitespaces)
        // Server treats empty string as "reset to default" — pass it through.
        Task { [weak self] in
            guard let self else { return }
            self.mirrorBusy = true
            defer { Task { @MainActor [weak self] in self?.mirrorBusy = false } }
            do {
                _ = try await client.updateGlobalSettings(
                    GlobalSettingsPatch(hfEndpoint: draft)
                )
                self.mirrorEndpoint = draft
                self.isEditingMirror = false
                self.mirrorDraft = ""
                self.lastError = nil
            } catch {
                self.lastError = self.describe(error)
            }
        }
    }

    func resetMirror(client: OMLXClient) {
        mirrorDraft = ""
        saveMirror(client: client)
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func startDownload(repo: String? = nil, client: OMLXClient) {
        let target = (repo ?? repoText).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { return }
        isStarting = true
        Task { [weak self] in
            defer { Task { @MainActor in self?.isStarting = false } }
            do {
                _ = try await client.startHFDownload(repoId: target)
                if repo == nil { self?.repoText = "" }
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = self.describe(error)
            }
        }
    }

    func cancel(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.cancelHFDownload(taskId: taskId)
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = self.describe(error)
            }
        }
    }

    func retry(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.retryHFDownload(taskId: taskId)
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = self.describe(error)
            }
        }
    }

    func remove(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.removeHFTask(taskId: taskId)
                await self?.refreshTasks()
            } catch {
                guard let self else { return }
                self.lastError = self.describe(error)
            }
        }
    }

    func loadRecommended(client: OMLXClient) async {
        self.recommendedLoading = true
        defer { self.recommendedLoading = false }
        do {
            let resp = try await client.getHFRecommended()
            // Trending-first, then popular, deduped by repoId. Mirrors how
            // the original dashboard surfaces both lists side-by-side; the
            // Swift screen renders one combined feed.
            var seen = Set<String>()
            var merged: [HFModelInfo] = []
            for m in resp.trending + resp.popular where seen.insert(m.repoId).inserted {
                merged.append(m)
            }
            self.recommended = merged
            self.lastError = nil
        } catch {
            // 502/504 are common (HF unreachable, dev offline). Surface but
            // keep UI usable.
            self.lastError = describe(error)
        }
    }

    private func refreshTasks() async {
        guard let client else { return }
        do {
            self.tasks = try await client.listHFTasks().tasks
            self.lastError = nil
        } catch {
            self.lastError = describe(error)
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
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
