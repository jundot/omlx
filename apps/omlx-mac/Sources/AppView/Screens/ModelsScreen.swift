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
    private var vm: ModelsScreenVM { services.models }

    var body: some View {
        @Bindable var vm = vm
        VStack(alignment: .leading, spacing: 0) {
            ActiveModelsSection(
                models: vm.activeModels,
                onUnload: { id in vm.unload(id: id, client: services.client) }
            )

            LibrarySection(
                models: vm.libraryModels,
                searchText: $vm.librarySearch,
                isModelLoaded: { id in vm.activeModels.contains(where: { $0.id == id }) },
                deletingID: vm.deletingID,
                updates: vm,
                onCheckUpdates: { Task { await vm.checkForUpdates(client: services.client) } },
                onLoad: { id in vm.load(id: id, client: services.client) },
                onUnload: { id in vm.unload(id: id, client: services.client) },
                onOpenSettings: { id in services.modelDetailID = id },
                onRequestRemove: { id in vm.pendingRemoveID = id },
                onToggleFavorite: { id, fav in vm.setFavorite(id: id, favorite: fav, client: services.client) }
            )

            FooterBar(error: vm.lastError)
        }
        .task { await vm.start(client: services.client) }
        .onDisappear { vm.stop() }
        .sheet(item: $vm.updateTarget) { target in
            ModelCardSheet(
                target: target,
                client: services.client,
                onDownload: { repo, files in
                    Task {
                        if await vm.startUpdate(repo: repo, files: files, client: services.client) {
                            services.requestedSection = .downloads
                        }
                    }
                }
            )
        }
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
            if vm.allModels.first(where: { $0.id == id })?.isInHFCache == true {
                Text(String(localized: "models.delete.confirm_message_hf_cache",
                            defaultValue: "This model is in the shared Hugging Face cache. Other apps that use the cache will need to download it again.",
                            comment: "Body text inside the delete-model confirmation dialog when the model lives in the shared Hugging Face cache"))
            } else {
                Text(String(localized: "models.delete.confirm_message",
                            defaultValue: "The model files will be permanently removed from disk and unloaded if currently running.",
                            comment: "Body text inside the delete-model confirmation dialog explaining the impact"))
            }
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
                        .font(.omlxText(DesignTokens.FontSize.callout))
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
                                .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                                .foregroundStyle(theme.text)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            if let alias = m.displayAlias {
                                ModelAliasText(alias: alias)
                            }
                            Spacer(minLength: 8)
                            ActiveBadge(model: m)
                            Text(m.estimatedSizeFormatted ?? formatBytes(m.estimatedSize))
                                .font(.omlxMono(DesignTokens.FontSize.subheadline))
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
    @Binding var searchText: String
    let isModelLoaded: (String) -> Bool
    let deletingID: String?
    let updates: ModelsScreenVM
    let onCheckUpdates: () -> Void
    let onLoad: (String) -> Void
    let onUnload: (String) -> Void
    let onOpenSettings: (String) -> Void
    let onRequestRemove: (String) -> Void
    let onToggleFavorite: (String, Bool) -> Void

    @Environment(\.omlxTheme) private var theme
    @FocusState private var searchFocused: Bool

    private var totalSize: Int64 {
        models.reduce(0) { $0 + $1.estimatedSize }
    }

    var body: some View {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        let filteredModels = filterModelsByName(models, query: query)
        let subtitle = query.isEmpty
            ? String(localized: "models.library.subtitle",
                     defaultValue: "Models: \(models.count) · \(formatBytes(totalSize)) on disk",
                     comment: "Subtitle for the Model Library section. Placeholders: model count, total bytes on disk")
            : String(localized: "models.library.search.subtitle",
                     defaultValue: "Models: \(filteredModels.count) of \(models.count) · \(formatBytes(totalSize)) on disk",
                     comment: "Subtitle while filtering the Model Library. Placeholders: matching count, total model count, total bytes on disk")

        SectionHeader(String(localized: "models.library.title",
                                    defaultValue: "Model Library",
                                    comment: "Section heading for the on-disk model library"),
                      subtitle: subtitle) {
            Button(action: onCheckUpdates) {
                HStack(spacing: 5) {
                    if updates.isCheckingUpdates {
                        ProgressView().controlSize(.mini)
                    } else {
                        Image(systemName: "arrow.triangle.2.circlepath")
                    }
                    Text(String(localized: "models.updates.check", defaultValue: "Check for updates"))
                }
            }
            .buttonStyle(.omlx(.normal, size: .small))
            .disabled(updates.isCheckingUpdates || models.isEmpty)
        }

        HStack(spacing: 8) {
            let searchLabel = String(localized: "models.library.search.placeholder",
                                     defaultValue: "Search models…",
                                     comment: "Placeholder and accessibility label for the Model Library search field")
            Image(systemName: "magnifyingglass")
                .foregroundStyle(theme.textTertiary)
                .accessibilityHidden(true)
            TextField(searchLabel, text: $searchText, prompt: Text(searchLabel))
                .textFieldStyle(.plain)
                .font(.omlxText(DesignTokens.FontSize.body))
                .foregroundStyle(theme.text)
                .focused($searchFocused)
                .frame(maxWidth: .infinity)
            Button(String(localized: "models.library.search.clear",
                          defaultValue: "Clear model search",
                          comment: "Accessibility label for the button that clears the Model Library search"),
                   systemImage: "xmark.circle.fill") {
                searchText = ""
                searchFocused = true
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.omlx(.plain, size: .small))
            .opacity(searchText.isEmpty ? 0 : 1)
            .disabled(searchText.isEmpty)
            .accessibilityHidden(searchText.isEmpty)
        }
        .padding(.leading, 8)
        .padding(.vertical, 2)
        .background(theme.inputBg, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(searchFocused ? theme.inputBorderFocus : theme.inputBorder,
                              lineWidth: searchFocused ? 2 : 0.5)
        }
        .padding(.horizontal, 28)
        .padding(.bottom, 8)

        if let status = updates.updateStatus {
            Text(status)
                .font(.omlxText(DesignTokens.FontSize.subheadline))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 28)
                .padding(.bottom, 4)
        }

        ListGroup {
            if models.isEmpty {
                FreeRow(isLast: true) {
                    VStack(spacing: 6) {
                        Text(String(localized: "models.library.empty.title",
                                    defaultValue: "No models discovered",
                                    comment: "Empty-state title shown when no models have been discovered on disk"))
                            .font(.omlxText(DesignTokens.FontSize.callout))
                            .foregroundStyle(theme.textTertiary)
                        Text(String(localized: "models.library.empty.sub",
                                    defaultValue: "Use the Downloads screen to fetch a model from Hugging Face.",
                                    comment: "Empty-state subtitle directing the user to the Downloads screen"))
                            .font(.omlxText(DesignTokens.FontSize.subheadline))
                            .foregroundStyle(theme.textTertiary)
                    }
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 16)
                }
            } else if filteredModels.isEmpty {
                FreeRow(isLast: true) {
                    Text(String(localized: "models.library.search.empty",
                                defaultValue: "No matching models",
                                comment: "Empty-state message when the Model Library search has no matches"))
                        .font(.omlxText(DesignTokens.FontSize.callout))
                        .foregroundStyle(theme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 16)
                }
            } else {
                ForEach(Array(filteredModels.enumerated()), id: \.element.id) { idx, m in
                    LibraryRow(
                        model: m,
                        isLast: idx == filteredModels.count - 1,
                        isLoaded: isModelLoaded(m.id),
                        deletingID: deletingID,
                        updates: updates,
                        onLoad: onLoad,
                        onUnload: onUnload,
                        onOpenSettings: onOpenSettings,
                        onRequestRemove: onRequestRemove,
                        onToggleFavorite: onToggleFavorite
                    )
                }
            }
        }
    }
}

/// Clicking the row opens model settings. Copy and remove show only while the pointer is over the row.
private struct LibraryRow: View {
    let model: ModelDTO
    let isLast: Bool
    let isLoaded: Bool
    let deletingID: String?
    let updates: ModelsScreenVM
    let onLoad: (String) -> Void
    let onUnload: (String) -> Void
    let onOpenSettings: (String) -> Void
    let onRequestRemove: (String) -> Void
    let onToggleFavorite: (String, Bool) -> Void

    @Environment(\.omlxTheme) private var theme
    @State private var isHovered = false

    private var isFavorite: Bool { model.isFavorite ?? false }
    private var showsHoverActions: Bool { isHovered || deletingID == model.id }

    var body: some View {
        FreeRow(isLast: isLast, verticalPadding: 8) {
            content
        }
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
        .onTapGesture { onOpenSettings(model.id) }
        .accessibilityAction(named: Text(Self.settingsLabel)) { onOpenSettings(model.id) }
        .contextMenu {
            Button(Self.copyIDLabel) { copy(model.id) }
            if let alias = model.displayAlias {
                Button(Self.copyAliasLabel) { copy(alias) }
            }
            Button(Self.settingsLabel) { onOpenSettings(model.id) }
            Divider()
            Button(Self.removeLabel, role: .destructive) { onRequestRemove(model.id) }
                .disabled(deletingID != nil)
        }
    }

    private var content: some View {
        HStack(spacing: 10) {
            favoriteButton
            ModelIconTile(modelType: model.modelType)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(model.displayTitle)
                        .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    updateBadge
                }
                details
            }
            Spacer(minLength: 8)
            HStack(spacing: 10) {
                HStack(spacing: 2) {
                    CopyIconButton(value: model.apiModelName,
                                   helpText: model.displayAlias == nil ? Self.copyIDLabel : Self.copyAliasLabel)
                    removeButton
                }
                .opacity(showsHoverActions ? 1 : 0)
                .allowsHitTesting(showsHoverActions)
                LoadToggle(
                    isLoaded: isLoaded,
                    isLoading: model.isLoading,
                    onLoad: { onLoad(model.id) },
                    onUnload: { onUnload(model.id) }
                )
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(theme.textTertiary)
                    .accessibilityHidden(true)
            }
            .fixedSize(horizontal: true, vertical: false)
            .layoutPriority(1)
        }
    }

    private var favoriteButton: some View {
        Button {
            onToggleFavorite(model.id, !isFavorite)
        } label: {
            Image(systemName: isFavorite ? "star.fill" : "star")
                .font(.system(size: 12))
                .foregroundStyle(isFavorite ? Color.yellow : theme.textTertiary)
        }
        .buttonStyle(.plain)
        .help(isFavorite
            ? String(localized: "models.library.favorite_on.help",
                     defaultValue: "Favorite — click to remove",
                     comment: "Tooltip on the filled star that removes a model from favorites")
            : String(localized: "models.library.favorite_off.help",
                     defaultValue: "Add to favorites",
                     comment: "Tooltip on the outlined star that adds a model to favorites"))
    }

    private var details: some View {
        HStack(spacing: 0) {
            if let alias = model.displayAlias {
                ModelAliasText(alias: alias)
                Text(" · ").fixedSize()
            }
            if let id = model.idNotInTitle {
                Text("\(id) · ")
                    .truncationMode(.middle)
            }
            ForEach(Array(detailTags.enumerated()), id: \.offset) { idx, tag in
                let label = idx == 0 ? tag.label : " · \(tag.label)"
                if let help = tag.help {
                    Text(label).fixedSize().help(help)
                } else {
                    Text(label).fixedSize()
                }
            }
        }
        .font(.omlxText(DesignTokens.FontSize.subheadline))
        .foregroundStyle(theme.textSecondary)
        .lineLimit(1)
    }

    private var detailTags: [(label: String, help: String?)] {
        var tags: [(label: String, help: String?)] = []
        if let type = ModelSettingsScreenVM.modelTypeOptions.first(where: { $0.0 == model.modelType && !$0.0.isEmpty }) {
            tags.append((type.1, nil))
        }
        tags.append((model.estimatedSizeFormatted ?? formatBytes(model.estimatedSize), nil))
        if model.isInHFCache {
            tags.append((String(localized: "models.library.hf_cache.badge",
                                defaultValue: "HF cache",
                                comment: "Badge on a library model found in the shared Hugging Face cache"),
                         String(localized: "models.library.hf_cache.help",
                                defaultValue: "Found in the Hugging Face cache. Other apps that use this cache will need to download it again after you delete it.",
                                comment: "Tooltip on the HF cache badge in the model library")))
        }
        for engine in (model.externalEngines ?? []).filter(\.compatible) {
            tags.append((engine.label,
                         String(localized: "models.library.external_engine.help",
                                defaultValue: "\(engine.label) can serve this model. Turn it on in the model's Advanced settings.",
                                comment: "Tooltip on the external engine badge in the model library; placeholder is the engine name")))
        }
        if model.isGGUF {
            tags.append(("GGUF",
                         String(localized: "models.library.gguf.help",
                                defaultValue: "A GGUF file, served by an external engine",
                                comment: "Tooltip on the GGUF badge in the model library")))
        }
        return tags
    }

    private var removeButton: some View {
        Button {
            onRequestRemove(model.id)
        } label: {
            if deletingID == model.id {
                ProgressView()
                    .controlSize(.mini)
            } else {
                Image(systemName: "trash")
                    .font(.system(size: 11))
            }
        }
        .buttonStyle(.omlx(.plain, size: .small))
        .disabled(deletingID != nil)
        .help(Self.removeLabel)
    }

    @ViewBuilder
    private var updateBadge: some View {
        if let update = updates.update(for: model.id) {
            Button { updates.showUpdate(for: model.id) } label: {
                StatusPill(status: .custom(
                    color: .yellow,
                    label: String(localized: "models.updates.available", defaultValue: "Update available"),
                    fillBg: true))
            }
            .buttonStyle(.plain)
            .help(String(localized: "models.updates.details",
                         defaultValue: "\(update.files) files · \(formatBytes(update.size))"))
            .accessibilityLabel(String(localized: "models.updates.preview_label",
                                       defaultValue: "Review update for \(model.displayTitle)"))
        }
    }

    private func copy(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }

    private static var copyAliasLabel: String {
        String(localized: "models.library.copy_alias",
               defaultValue: "Copy alias",
               comment: "Tooltip and context menu item that copies a model's alias")
    }

    private static var copyIDLabel: String {
        String(localized: "common.copy_model_id",
               defaultValue: "Copy model ID",
               comment: "Tooltip and accessibility label for a button that copies a model identifier")
    }

    private static var settingsLabel: String {
        String(localized: "models.library.settings.help",
               defaultValue: "Settings",
               comment: "Tooltip on the chevron that opens a model's settings screen")
    }

    private static var removeLabel: String {
        String(localized: "models.library.remove.help",
               defaultValue: "Remove from disk",
               comment: "Tooltip on the trash button that deletes a model from local storage")
    }
}

/// Load control: load when idle, a loaded mark that turns into unload under the pointer.
private struct LoadToggle: View {
    let isLoaded: Bool
    let isLoading: Bool
    let onLoad: () -> Void
    let onUnload: () -> Void

    @Environment(\.omlxTheme) private var theme
    @State private var isHovered = false

    var body: some View {
        Button(action: isLoaded ? onUnload : onLoad) {
            Group {
                if isLoading {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: symbol)
                        .foregroundStyle(isLoaded || isHovered ? theme.text : theme.textSecondary)
                }
            }
            .font(.system(size: 17))
            .frame(width: 24, height: 24)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
        .onHover { isHovered = $0 }
        .help(label)
        .accessibilityLabel(Text(label))
    }

    private var symbol: String {
        if isLoaded {
            return isHovered ? "eject.circle.fill" : "checkmark.circle.fill"
        }
        return isHovered ? "arrow.up.circle.fill" : "arrow.up.circle"
    }

    private var label: String {
        isLoaded
            ? String(localized: "models.library.unload",
                     defaultValue: "Unload",
                     comment: "Button label that unloads a library model from memory")
            : String(localized: "models.library.load",
                     defaultValue: "Load",
                     comment: "Button label that loads a library model into memory")
    }
}

// MARK: - Helpers

private struct ModelAliasText: View {
    let alias: String
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(alias)
            .font(.omlxText(DesignTokens.FontSize.subheadline))
            .foregroundStyle(theme.text)
            .lineLimit(1)
            .truncationMode(.middle)
            .help(alias)
            .accessibilityLabel(Text(String(localized: "settings.basic.alias.label",
                                            defaultValue: "Model Alias")))
            .accessibilityValue(Text(alias))
    }
}

/// Neutral rounded tile used for model icons in lists and headers.
struct ModelIconTile: View {
    let systemSymbol: String
    let size: CGFloat

    @Environment(\.omlxTheme) private var theme

    init(systemSymbol: String, size: CGFloat = 26) {
        self.systemSymbol = systemSymbol
        self.size = size
    }

    init(modelType: String?, size: CGFloat = 26) {
        self.init(systemSymbol: Self.symbol(for: modelType), size: size)
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: size * 0.27, style: .continuous)
        Image(systemName: systemSymbol)
            .font(.system(size: size * 0.5, weight: .medium))
            .foregroundStyle(theme.textSecondary)
            .frame(width: size, height: size)
            .background(shape.fill(theme.controlBg))
            .overlay(shape.strokeBorder(theme.groupBorder, lineWidth: 0.5))
    }

    static func symbol(for modelType: String?) -> String {
        switch modelType {
        case "vlm": return "text.below.photo"
        case "embedding": return "square.stack.3d.up"
        case "reranker": return "text.line.first.and.arrowtriangle.forward"
        case "decision": return "arrow.triangle.branch"
        case "audio_stt": return "waveform.and.mic"
        case "audio_tts": return "waveform"
        case "audio_sts": return "person.wave.2"
        default: return "text.alignleft"
        }
    }
}

func filterModelsByName(_ models: [ModelDTO], query: String) -> [ModelDTO] {
    let query = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !query.isEmpty else { return models }
    return models.filter {
        $0.id.lowercased().contains(query)
            || $0.displayTitle.lowercased().contains(query)
            || ($0.displayAlias ?? "").lowercased().contains(query)
    }
}

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

    var displayAlias: String? {
        guard let alias = settings?.modelAlias, !alias.isEmpty else { return nil }
        return alias
    }

    var apiModelName: String { displayAlias ?? id }

    /// The API id, or nil when `displayTitle` already ends with it.
    var idNotInTitle: String? {
        let title = displayTitle
        return title == id || title.hasSuffix("/\(id)") ? nil : id
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
