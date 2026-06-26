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
                onLoad: { id in vm.load(id: id, client: services.client) },
                onUnload: { id in vm.unload(id: id, client: services.client) },
                onOpenSettings: { id in services.modelDetailID = id },
                onRequestRemove: { id in vm.pendingRemoveID = id }
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
        models.reduce(0) { $0 + $1.estimatedSize }
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
                            Text(m.estimatedSizeFormatted ?? formatBytes(m.estimatedSize))
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

    @Environment(\.omlxTheme) private var theme

    private var totalSize: Int64 {
        models.reduce(0) { $0 + $1.estimatedSize }
    }

    var body: some View {
        SectionHeader(String(localized: "models.library.title",
                                    defaultValue: "Model Library",
                                    comment: "Section heading for the on-disk model library"),
                      subtitle: String(localized: "models.library.subtitle",
                                       defaultValue: "\(models.count) models · \(formatBytes(totalSize)) on disk",
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
                            .layoutPriority(1)
                            Spacer(minLength: 8)
                            if isModelLoaded(m.id) {
                                Button(String(localized: "models.library.unload",
                                              defaultValue: "Unload",
                                              comment: "Button label that unloads a library model from memory")) { onUnload(m.id) }
                                    .buttonStyle(.omlx(.plain, size: .small))
                            } else {
                                Button(String(localized: "models.library.load",
                                              defaultValue: "Load",
                                              comment: "Button label that loads a library model into memory")) { onLoad(m.id) }
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
                    }
                }
            }
        }
    }

    private func gradient(for m: ModelDTO) -> [Color] {
        switch m.modelType {
        case "embed", "rerank": return SquircleGradient.downloads
        case "audio-stt", "audio-tts", "audio-sts": return SquircleGradient.integrations
        case "vlm":             return SquircleGradient.update
        default:                return SquircleGradient.models
        }
    }

    private func iconName(for m: ModelDTO) -> String {
        switch m.modelType {
        case "embed":   return "cube.transparent"
        case "rerank":  return "arrow.up.arrow.down"
        case "audio-stt", "audio-tts", "audio-sts": return "waveform"
        case "vlm":     return "eye"
        default:        return "cpu"
        }
    }
}

// MARK: - Helpers

func sortModelsByName(_ models: [ModelDTO]) -> [ModelDTO] {
    models.enumerated().sorted { lhs, rhs in
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
