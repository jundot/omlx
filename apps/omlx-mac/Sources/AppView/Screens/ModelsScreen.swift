// PR 8 — Models screen.
//
// Reads `/admin/api/models` (GET) and surfaces it as two sections:
//   • Active Models — currently-loaded engines, with an unload affordance
//   • Model Library — every discovered model on disk; load button + drill
//     into ModelSettingsScreen via the chevron.
//
// Polling at 2 s while visible: load/unload responses are eventual (engine
// pool is async) and we want the row state to converge without manual
// refresh. Drilling into a model sets `services.modelDetailID`, which
// AppView swaps the screen content for.

import SwiftUI

struct ModelLaunchTarget: Identifiable, Equatable, Sendable {
    let model: ModelDTO
    var id: String { model.id }
}

struct ModelsScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = ModelsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ActiveModelsSection(
                models: vm.activeModels,
                onUnload: { id in vm.unload(id: id, client: services.client) }
            )

            LibrarySection(
                models: vm.libraryModels,
                isModelLoaded: { id in vm.activeModels.contains(where: { $0.id == id }) },
                deletingID: vm.deletingID,
                onLoad: { id in
                    guard let model = vm.libraryModels.first(where: { $0.id == id }) else { return }
                    if model.usesFixedKVLaunchPreflight {
                        services.modelLaunchTarget = ModelLaunchTarget(model: model)
                    } else {
                        vm.load(id: id, client: services.client)
                    }
                },
                onUnload: { id in vm.unload(id: id, client: services.client) },
                onOpenSettings: { id in services.modelDetailID = id },
                onRequestRemove: { id in vm.pendingRemoveID = id },
                onToggleFavorite: { id, fav in vm.setFavorite(id: id, favorite: fav, client: services.client) }
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
        .confirmationDialog(
            String(localized: "models.delete.confirm_title",
                   defaultValue: "Delete this model from disk?",
                   comment: "Confirmation dialog title shown before deleting a model from disk"),
            isPresented: Binding(
                get: { vm.pendingRemoveID != nil },
                set: { if !$0 { vm.pendingRemoveID = nil } }
            ),
            titleVisibility: .visible,
            presenting: vm.pendingRemoveID
        ) { id in
            Button(String(localized: "models.delete.confirm_button",
                          defaultValue: "Delete \(id)",
                          comment: "Destructive button label inside the delete-model confirmation dialog; placeholder is the model id"),
                   role: .destructive) {
                vm.remove(id: id, client: services.client)
            }
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Generic cancel button"),
                   role: .cancel) { vm.pendingRemoveID = nil }
        } message: { id in
            Text(String(localized: "models.delete.confirm_message",
                        defaultValue: "The model files will be permanently removed from disk and unloaded if currently running.",
                        comment: "Body text inside the delete-model confirmation dialog explaining the impact"))
        }
    }
}

// MARK: - Active section

private struct ActiveModelsSection: View {
    let models: [ModelDTO]
    let onUnload: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    private var memoryFootprint: Int64 {
        models.reduce(0) { $0 + ($1.memory?.estimatedTotalBytes ?? $1.estimatedSize) }
    }

    var body: some View {
        SectionHeader(String(localized: "models.active.title",
                                    defaultValue: "Active Models",
                                    comment: "Section heading for the list of currently-loaded models"),
                      subtitle: String(localized: "models.active.subtitle",
                                       defaultValue: "\(models.count) loaded · \(formatBytes(memoryFootprint))",
                                       comment: "Subtitle for the Active Models section. Placeholders: count of loaded models, total memory footprint"))

        ListGroup {
            if models.isEmpty {
                FreeRow(isLast: true) {
                    Text(String(localized: "models.active.empty",
                                defaultValue: "No models loaded",
                                comment: "Empty-state message shown when no models are currently loaded"))
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 14)
                }
            } else {
                ForEach(Array(models.enumerated()), id: \.element.id) { idx, m in
                    FreeRow(isLast: idx == models.count - 1) {
                        HStack(spacing: 10) {
                            if m.pinned == true {
                                Image(systemName: "pin.fill")
                                    .font(.system(size: 11))
                                    .foregroundStyle(theme.textSecondary)
                            }
                            Text(m.displayTitle)
                                .font(.omlxText(13, weight: .medium))
                                .foregroundStyle(theme.text)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer(minLength: 8)
                            ActiveBadge(model: m)
                            Text(formatBytes(m.memory?.estimatedTotalBytes ?? m.estimatedSize))
                                .font(.omlxMono(11))
                                .foregroundStyle(theme.textSecondary)
                                .frame(minWidth: 60, alignment: .trailing)
                            Button {
                                onUnload(m.id)
                            } label: {
                                Image(systemName: "eject")
                                    .font(.system(size: 12))
                            }
                            .buttonStyle(.omlx(.plain, size: .small))
                            .help(String(localized: "models.active.unload.help",
                                         defaultValue: "Unload model",
                                         comment: "Tooltip on the eject button that unloads an active model"))
                        }
                    }
                }
            }
        }
    }
}

private struct ActiveBadge: View {
    let model: ModelDTO
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        if model.isLoading {
            StatusPill(status: .starting)
        } else if model.loaded {
            StatusPill(status: .custom(color: theme.greenDot,
                                       label: String(localized: "models.active.badge.loaded",
                                                     defaultValue: "Loaded",
                                                     comment: "Status pill label for a model that is currently loaded in memory"),
                                       fillBg: true))
        } else {
            StatusPill(status: .custom(color: theme.textTertiary,
                                       label: String(localized: "models.active.badge.idle",
                                                     defaultValue: "Idle",
                                                     comment: "Status pill label for a model that is not currently loaded"),
                                       fillBg: true))
        }
    }
}

// MARK: - Library section

private struct LibrarySection: View {
    let models: [ModelDTO]
    let isModelLoaded: (String) -> Bool
    let deletingID: String?
    let onLoad: (String) -> Void
    let onUnload: (String) -> Void
    let onOpenSettings: (String) -> Void
    let onRequestRemove: (String) -> Void
    let onToggleFavorite: (String, Bool) -> Void

    @Environment(\.omlxTheme) private var theme

    private var totalSize: Int64 {
        models.reduce(0) { $0 + $1.estimatedSize }
    }

    var body: some View {
        SectionHeader(String(localized: "models.library.title",
                                    defaultValue: "Model Library",
                                    comment: "Section heading for the on-disk model library"),
                      subtitle: String(localized: "models.library.subtitle",
                                       defaultValue: "Models: \(models.count) · \(formatBytes(totalSize)) on disk",
                                       comment: "Subtitle for the Model Library section. Placeholders: model count, total bytes on disk"))

        ListGroup {
            if models.isEmpty {
                FreeRow(isLast: true) {
                    VStack(spacing: 6) {
                        Text(String(localized: "models.library.empty.title",
                                    defaultValue: "No models discovered",
                                    comment: "Empty-state title shown when no models have been discovered on disk"))
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textTertiary)
                        Text(String(localized: "models.library.empty.sub",
                                    defaultValue: "Use the Downloads screen to fetch a model from Hugging Face.",
                                    comment: "Empty-state subtitle directing the user to the Downloads screen"))
                            .font(.omlxText(11))
                            .foregroundStyle(theme.textTertiary)
                    }
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 16)
                }
            } else {
                ForEach(Array(models.enumerated()), id: \.element.id) { idx, m in
                    FreeRow(isLast: idx == models.count - 1) {
                        HStack(spacing: 10) {
                            Button {
                                onToggleFavorite(m.id, !(m.isFavorite ?? false))
                            } label: {
                                Image(systemName: (m.isFavorite ?? false) ? "star.fill" : "star")
                                    .font(.system(size: 12))
                                    .foregroundStyle((m.isFavorite ?? false) ? Color.yellow : theme.textTertiary)
                            }
                            .buttonStyle(.plain)
                            .help((m.isFavorite ?? false)
                                ? String(localized: "models.library.favorite_on.help",
                                         defaultValue: "Favorite — click to remove",
                                         comment: "Tooltip on the filled star that removes a model from favorites")
                                : String(localized: "models.library.favorite_off.help",
                                         defaultValue: "Add to favorites",
                                         comment: "Tooltip on the outlined star that adds a model to favorites"))
                            Squircle(systemSymbol: iconName(for: m),
                                     size: 26,
                                     gradient: gradient(for: m))
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(spacing: 4) {
                                    Text(m.displayTitle)
                                        .font(.omlxText(13, weight: .medium))
                                        .foregroundStyle(theme.text)
                                        .lineLimit(1)
                                        .truncationMode(.tail)
                                    CopyIconButton(value: m.id)
                                }
                                Text("\(m.id) · \(m.estimatedSizeFormatted ?? formatBytes(m.estimatedSize))")
                                    .font(.omlxMono(11))
                                    .foregroundStyle(theme.textSecondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            Spacer(minLength: 8)
                            HStack(spacing: 10) {
                                if isModelLoaded(m.id) {
                                    Button {
                                        onUnload(m.id)
                                    } label: {
                                        Text(String(localized: "models.library.unload",
                                                    defaultValue: "Unload",
                                                    comment: "Button label that unloads a library model from memory"))
                                            .lineLimit(1)
                                            .frame(minWidth: 48)
                                    }
                                    .buttonStyle(.omlx(.plain, size: .small))
                                } else {
                                    Button {
                                        onLoad(m.id)
                                    } label: {
                                        Text(String(localized: "models.library.load",
                                                    defaultValue: "Load",
                                                    comment: "Button label that loads a library model into memory"))
                                            .lineLimit(1)
                                            .frame(minWidth: 48)
                                    }
                                    .buttonStyle(.omlx(.normal, size: .small))
                                    .disabled(m.isLoading)
                                }
                                Button {
                                    onOpenSettings(m.id)
                                } label: {
                                    Image(systemName: "chevron.right")
                                        .font(.system(size: 11))
                                }
                                .buttonStyle(.omlx(.plain, size: .small))
                                .help(String(localized: "models.library.settings.help",
                                             defaultValue: "Settings",
                                             comment: "Tooltip on the chevron that opens a model's settings screen"))
                                Button {
                                    onRequestRemove(m.id)
                                } label: {
                                    if deletingID == m.id {
                                        ProgressView()
                                            .controlSize(.mini)
                                    } else {
                                        Image(systemName: "trash")
                                            .font(.system(size: 11))
                                            .foregroundStyle(theme.redDot)
                                    }
                                }
                                .buttonStyle(.omlx(.plain, size: .small))
                                .disabled(deletingID != nil)
                                .help(String(localized: "models.library.remove.help",
                                             defaultValue: "Remove from disk",
                                             comment: "Tooltip on the trash button that deletes a model from local storage"))
                            }
                            .fixedSize(horizontal: true, vertical: false)
                            .layoutPriority(1)
                        }
                    }
                }
            }
        }
    }

    private func gradient(for m: ModelDTO) -> [Color] {
        switch m.modelType {
        case "embedding", "reranker": return SquircleGradient.downloads
        case "audio_stt", "audio_tts", "audio_sts": return SquircleGradient.integrations
        case "vlm":             return SquircleGradient.update
        default:                return SquircleGradient.models
        }
    }

    private func iconName(for m: ModelDTO) -> String {
        switch m.modelType {
        case "embedding": return "cube.transparent"
        case "reranker":  return "arrow.up.arrow.down"
        case "audio_stt", "audio_tts", "audio_sts": return "waveform"
        case "vlm":     return "eye"
        default:        return "cpu"
        }
    }
}

// MARK: - Fixed-memory launch preflight

@MainActor
struct ModelMemoryLaunchSheet: View {
    let target: ModelLaunchTarget
    let client: OMLXClient
    let onFinished: () -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.omlxTheme) private var theme
    @State private var contextText: String
    @State private var state: EstimateState = .loading
    @State private var isLaunching = false
    @State private var launched = false
    @State private var launchError: String?

    private enum EstimateState: Equatable {
        case loading
        case loaded(ModelMemoryDTO)
        case failed(String)
    }

    init(target: ModelLaunchTarget, client: OMLXClient, onFinished: @escaping () -> Void) {
        self.target = target
        self.client = client
        self.onFinished = onFinished
        let initialContext = target.model.settings?.maxContextWindow
            ?? target.model.modelContextLength
        _contextText = State(initialValue: initialContext.map(String.init) ?? "")
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(theme.groupBorder)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    contextEditor
                    estimateBody
                }
                .padding(20)
            }
            Divider().overlay(theme.groupBorder)
            footer
        }
        .frame(width: 520, height: 570)
        .background(theme.windowBg)
        .task(id: contextText) {
            guard !launched else { return }
            state = .loading
            launchError = nil
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            await fetchEstimate()
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(String(localized: "models.memory.launch_title",
                            defaultValue: "Launch memory",
                            comment: "Title for the fixed KV-cache launch preflight sheet"))
                    .font(.omlxText(16, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(target.model.displayTitle)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Button {
                finish()
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
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var contextEditor: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(String(localized: "models.memory.context",
                            defaultValue: "Context window",
                            comment: "Label for context tokens in the model memory launch sheet"))
                    .font(.omlxText(12, weight: .medium))
                    .foregroundStyle(theme.text)
                Spacer()
                TextField("", text: $contextText)
                    .textFieldStyle(.roundedBorder)
                    .font(.omlxMono(12))
                    .frame(width: 130)
                    .disabled(isLaunching || launched)
                Text(String(localized: "common.tokens",
                            defaultValue: "tokens",
                            comment: "Unit label for token counts"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
            }
            Text(String(localized: "models.memory.context_help",
                        defaultValue: "OMLX reserves this capacity for every active session slot.",
                        comment: "Explanation under context window in the model memory launch sheet"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        }
    }

    @ViewBuilder
    private var estimateBody: some View {
        switch state {
        case .loading:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(String(localized: "models.memory.calculating",
                            defaultValue: "Calculating the fixed allocation...",
                            comment: "Loading text while fetching a model memory estimate"))
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textSecondary)
            }
            .frame(maxWidth: .infinity, minHeight: 230, alignment: .center)
        case .failed(let message):
            messageCard(message, color: theme.redDot)
        case .loaded(let memory):
            memoryBreakdown(memory)
        }
    }

    private func memoryBreakdown(_ memory: ModelMemoryDTO) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(memory.isCommitted
                     ? String(localized: "models.memory.committed",
                              defaultValue: "Fixed KV committed at launch",
                              comment: "Heading shown for committed fixed model memory")
                     : String(localized: "models.memory.estimate",
                              defaultValue: "Estimated launch allocation",
                              comment: "Heading shown for a prelaunch model memory estimate"))
                    .font(.omlxText(12, weight: .semibold))
                    .foregroundStyle(theme.text)
                Spacer()
                Text("\(memory.reservedSessionSlots) / \(memory.requestedSessionSlots) " +
                     String(localized: "models.memory.slots",
                            defaultValue: "session slots",
                            comment: "Suffix for reserved and requested fixed cache slot counts"))
                    .font(.omlxMono(11))
                    .foregroundStyle(memory.configuredConcurrencyCapped ? theme.amberDot : theme.textSecondary)
            }

            MemoryStackedBar(memory: memory)
                .frame(height: 12)

            VStack(spacing: 7) {
                MemoryMetricRow(label: String(localized: "models.memory.weights",
                                              defaultValue: "Model weights (estimated)",
                                              comment: "Memory breakdown row for model weights"),
                                value: formatBytes(memory.weightsBytes), color: theme.blueDot)
                MemoryMetricRow(label: String(localized: "models.memory.kv_pool",
                                              defaultValue: "Fixed KV cache",
                                              comment: "Memory breakdown row for the fixed KV cache pool"),
                                value: formatBytes(memory.fixedKvCacheBytes), color: theme.amberDot)
                MemoryMetricRow(label: String(localized: "models.memory.per_session",
                                              defaultValue: "KV cache per session",
                                              comment: "Memory breakdown row for one fixed session slot"),
                                value: formatBytes(memory.perSessionKvBytes))
                MemoryMetricRow(label: String(localized: "models.memory.other",
                                              defaultValue: "Other known fixed memory",
                                              comment: "Memory breakdown row for other fixed allocations"),
                                value: formatBytes(memory.otherFixedBytes), color: theme.textTertiary)
                Divider().overlay(theme.rowSep)
                MemoryMetricRow(label: String(localized: "models.memory.total",
                                              defaultValue: "Estimated total",
                                              comment: "Memory breakdown row for estimated total allocation"),
                                value: formatBytes(memory.estimatedTotalBytes), emphasized: true)
                if let unified = memory.unifiedMemoryBytes {
                    MemoryMetricRow(label: String(localized: "models.memory.unified",
                                                  defaultValue: "Detected unified memory",
                                                  comment: "Memory breakdown row for detected unified memory"),
                                    value: formatBytes(unified))
                }
                if let available = memory.availableMemoryBytes {
                    MemoryMetricRow(label: String(localized: "models.memory.available",
                                                  defaultValue: "Available now",
                                                  comment: "Memory breakdown row for currently available unified memory"),
                                    value: formatBytes(available))
                }
                if let remaining = memory.projectedRemainingBytes {
                    MemoryMetricRow(label: String(localized: "models.memory.remaining",
                                                  defaultValue: "Remaining headroom",
                                                  comment: "Memory breakdown row for remaining memory headroom"),
                                    value: formatSignedBytes(remaining),
                                    color: remaining >= 0 ? theme.greenDot : theme.redDot)
                }
            }

            if memory.configuredConcurrencyCapped {
                messageCard(
                    String(localized: "models.memory.capped",
                           defaultValue: "Only \(memory.reservedSessionSlots) of \(memory.requestedSessionSlots) configured sessions fit. Extra sessions will wait for a slot.",
                           comment: "Warning shown when memory limits reduce fixed-cache concurrency"),
                    color: theme.amberDot
                )
            } else if memory.fits == false, let reason = memory.fitReason {
                messageCard(reason, color: theme.redDot)
            }

            if let launchError {
                messageCard(launchError, color: theme.redDot)
            }
        }
    }

    private func messageCard(_ message: String, color: Color) -> some View {
        Text(message)
            .font(.omlxText(11))
            .foregroundStyle(theme.text)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(color.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Spacer()
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Generic cancel button")) {
                finish()
            }
            .buttonStyle(.omlx(.plain, size: .regular))
            .disabled(isLaunching)

            Button {
                if launched {
                    finish()
                } else {
                    Task { await launch() }
                }
            } label: {
                if isLaunching {
                    ProgressView().controlSize(.small)
                } else {
                    Text(launched
                         ? String(localized: "common.done",
                                  defaultValue: "Done",
                                  comment: "Generic done button")
                         : String(localized: "models.memory.launch",
                                  defaultValue: "Reserve memory and launch",
                                  comment: "Button that commits fixed model memory and launches the model"))
                }
            }
            .buttonStyle(.omlx(.primary, size: .regular))
            .disabled(!launched && (!currentEstimateCanLaunch || isLaunching))
            .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var currentEstimateCanLaunch: Bool {
        guard case .loaded(let memory) = state,
              Int(contextText) == memory.contextWindow else { return false }
        return memory.canLaunch
    }

    private func fetchEstimate() async {
        let trimmed = contextText.trimmingCharacters(in: .whitespacesAndNewlines)
        let context = trimmed.isEmpty ? nil : Int(trimmed)
        if !trimmed.isEmpty, context == nil {
            state = .failed(String(localized: "models.memory.invalid_context",
                                   defaultValue: "Enter a positive context window.",
                                   comment: "Validation error for a non-positive model context window"))
            return
        }
        if let context, context <= 0 {
            state = .failed(String(localized: "models.memory.invalid_context",
                                   defaultValue: "Enter a positive context window.",
                                   comment: "Validation error for a non-positive model context window"))
            return
        }
        do {
            let estimate = try await client.getModelMemoryEstimate(
                id: target.model.id,
                contextWindow: context
            )
            state = .loaded(estimate)
            if trimmed.isEmpty {
                contextText = String(estimate.contextWindow)
            }
        } catch {
            state = .failed(error.omlxDescription)
        }
    }

    private func launch() async {
        guard case .loaded(let estimate) = state,
              estimate.canLaunch,
              let context = Int(contextText), context == estimate.contextWindow else { return }
        isLaunching = true
        launchError = nil
        defer { isLaunching = false }
        do {
            var patch = ModelSettingsPatch()
            patch.maxContextWindow = context
            _ = try await client.updateModelSettings(id: target.model.id, patch: patch)
            _ = try await client.loadModel(id: target.model.id)
            launched = true
            do {
                state = .loaded(try await client.getModelMemoryEstimate(
                    id: target.model.id,
                    contextWindow: context
                ))
            } catch {
                launchError = String(localized: "models.memory.committed_unavailable",
                                     defaultValue: "The model launched, but its committed memory breakdown could not be refreshed: \(error.omlxDescription)",
                                     comment: "Error shown when a model loads but its committed memory response cannot be fetched")
            }
        } catch {
            launchError = error.omlxDescription
        }
    }

    private func finish() {
        dismiss()
        onFinished()
    }
}

private struct MemoryStackedBar: View {
    let memory: ModelMemoryDTO
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        GeometryReader { proxy in
            let total = max(1, Double(memory.estimatedTotalBytes))
            HStack(spacing: 0) {
                Rectangle()
                    .fill(theme.blueDot)
                    .frame(width: proxy.size.width * Double(memory.weightsBytes) / total)
                Rectangle()
                    .fill(theme.amberDot)
                    .frame(width: proxy.size.width * Double(memory.fixedKvCacheBytes) / total)
                Rectangle()
                    .fill(theme.textTertiary)
            }
            .clipShape(Capsule())
            .overlay(Capsule().stroke(theme.groupBorder, lineWidth: 1))
        }
    }
}

private struct MemoryMetricRow: View {
    let label: String
    let value: String
    var color: Color? = nil
    var emphasized = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 7) {
            if let color {
                Circle().fill(color).frame(width: 7, height: 7)
            }
            Text(label)
                .font(.omlxText(11, weight: emphasized ? .semibold : .regular))
                .foregroundStyle(emphasized ? theme.text : theme.textSecondary)
            Spacer()
            Text(value)
                .font(.omlxMono(11, weight: emphasized ? .semibold : .regular))
                .foregroundStyle(color ?? (emphasized ? theme.text : theme.textSecondary))
        }
    }
}

private func formatSignedBytes(_ bytes: Int64) -> String {
    bytes < 0 ? "-\(formatBytes(-bytes))" : formatBytes(bytes)
}

// MARK: - Helpers

func sortModelsByName(_ models: [ModelDTO]) -> [ModelDTO] {
    models.enumerated().sorted { lhs, rhs in
        // Favorites always sort first; names decide within each group.
        let lf = lhs.element.isFavorite ?? false
        let rf = rhs.element.isFavorite ?? false
        if lf != rf { return lf }
        switch lhs.element.displayTitle.localizedCaseInsensitiveCompare(
            rhs.element.displayTitle
        ) {
        case .orderedAscending:
            return true
        case .orderedDescending:
            return false
        case .orderedSame:
            return lhs.offset < rhs.offset
        }
    }.map(\.element)
}

extension ModelDTO {
    var displayTitle: String {
        displayName ?? settings?.displayName ?? id
    }
}

func formatBytes(_ bytes: Int64) -> String {
    var v = Double(bytes)
    let units = ["B", "KB", "MB", "GB", "TB"]
    var i = 0
    while v >= 1024 && i < units.count - 1 {
        v /= 1024
        i += 1
    }
    return String(format: v < 10 && i > 0 ? "%.2f %@" : "%.1f %@", v, units[i])
}
