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
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = ModelsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ActiveModelsSection(
                models: vm.activeModels,
                onUnload: { id in vm.unload(id: id, client: services.client) }
            )

            LibrarySection(
                models: vm.libraryModels,
                isModelLoaded: { id in vm.activeModels.contains(where: { $0.id == id }) },
                onLoad: { id in vm.load(id: id, client: services.client) },
                onUnload: { id in vm.unload(id: id, client: services.client) },
                onOpenSettings: { id in services.modelDetailID = id }
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

// MARK: - Active section

private struct ActiveModelsSection: View {
    let models: [ModelDTO]
    let onUnload: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    private var memoryFootprint: Int64 {
        models.reduce(0) { $0 + $1.estimatedSize }
    }

    var body: some View {
        SectionHeader("Active Models",
                      subtitle: "\(models.count) loaded · \(formatBytes(memoryFootprint))")

        ListGroup {
            if models.isEmpty {
                FreeRow(isLast: true) {
                    Text("No models loaded")
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
                            Text(m.id)
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
                            .help("Unload model")
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
            StatusPill(status: .custom(color: theme.greenDot, label: "Loaded", fillBg: true))
        } else {
            StatusPill(status: .custom(color: theme.textTertiary, label: "Idle", fillBg: true))
        }
    }
}

// MARK: - Library section

private struct LibrarySection: View {
    let models: [ModelDTO]
    let isModelLoaded: (String) -> Bool
    let onLoad: (String) -> Void
    let onUnload: (String) -> Void
    let onOpenSettings: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    private var totalSize: Int64 {
        models.reduce(0) { $0 + $1.estimatedSize }
    }

    var body: some View {
        SectionHeader("Model Library",
                      subtitle: "\(models.count) models · \(formatBytes(totalSize)) on disk")

        ListGroup {
            if models.isEmpty {
                FreeRow(isLast: true) {
                    VStack(spacing: 6) {
                        Text("No models discovered")
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textTertiary)
                        Text("Use the Downloads screen to fetch a model from Hugging Face.")
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
                                Text(m.settings?.displayName ?? m.id)
                                    .font(.omlxText(13, weight: .medium))
                                    .foregroundStyle(theme.text)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                                Text("\(m.id) · \(m.estimatedSizeFormatted ?? formatBytes(m.estimatedSize))")
                                    .font(.omlxMono(11))
                                    .foregroundStyle(theme.textSecondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            Spacer(minLength: 8)
                            if isModelLoaded(m.id) {
                                Button("Unload") { onUnload(m.id) }
                                    .buttonStyle(.omlx(.plain, size: .small))
                            } else {
                                Button("Load") { onLoad(m.id) }
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
                            .help("Settings")
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

// MARK: - View model

@MainActor
final class ModelsScreenVM: ObservableObject {
    @Published private(set) var allModels: [ModelDTO] = []
    @Published var lastError: String?

    private weak var client: OMLXClient?
    private var pollTask: Task<Void, Never>?

    var activeModels: [ModelDTO] {
        allModels.filter { $0.loaded || $0.isLoading }
    }
    var libraryModels: [ModelDTO] { allModels }

    func start(client: OMLXClient) async {
        self.client = client
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.refresh()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func load(id: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.loadModel(id: id)
                await self?.refresh()
            } catch {
                guard let self else { return }
                self.lastError = self.describe(error)
            }
        }
    }

    func unload(id: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.unloadModel(id: id)
                await self?.refresh()
            } catch {
                guard let self else { return }
                self.lastError = self.describe(error)
            }
        }
    }

    private func refresh() async {
        guard let client else { return }
        do {
            self.allModels = try await client.listModels().models
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
