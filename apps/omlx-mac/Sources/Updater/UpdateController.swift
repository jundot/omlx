// PR 7 — stub for the Updates section on the Status screen (design v2).
//
// Drives three observable bits: a check state (idle / checking / available),
// the channel (Stable / Beta / Nightly), and two background prefs (autoCheck
// + autoDownload). Channel/prefs persist to AppConfig so they survive a
// relaunch; the actual feed wiring (Sparkle + GitHub releases) lands in PR 11
// behind the same surface — no UI changes needed at that point.
//
// `checkForUpdates()` is a 1.4s simulator: it just bounces idle → checking →
// idle(lastChecked = now). PR 11 swaps that body for `SUUpdater.checkForUpdates`.

import Foundation

enum UpdateChannel: String, Codable, CaseIterable, Identifiable, Sendable {
    case stable, beta, nightly
    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .stable:  return "Stable"
        case .beta:    return "Beta"
        case .nightly: return "Nightly"
        }
    }
}

struct AvailableUpdate: Equatable, Sendable {
    let version: String
    let sizeText: String?
}

@MainActor
final class UpdateController: ObservableObject {
    enum CheckState: Equatable, Sendable {
        case idle(lastChecked: Date?)
        case checking
        case available(AvailableUpdate)
    }

    @Published private(set) var state: CheckState = .idle(lastChecked: nil)
    @Published var channel: UpdateChannel {
        didSet { if !suspendPersist { persist() } }
    }
    @Published var autoCheck: Bool {
        didSet { if !suspendPersist { persist() } }
    }
    @Published var autoDownload: Bool {
        didSet { if !suspendPersist { persist() } }
    }

    private let storeURL: URL
    private var simulationTask: Task<Void, Never>?
    private var suspendPersist = true

    init(storeURL: URL = AppConfig.appSupportURL().appendingPathComponent("update-prefs.json")) {
        self.storeURL = storeURL
        let prefs = Self.readPrefs(from: storeURL) ?? Prefs(
            channel: .stable, autoCheck: true, autoDownload: false
        )
        self.channel = prefs.channel
        self.autoCheck = prefs.autoCheck
        self.autoDownload = prefs.autoDownload
        self.suspendPersist = false
    }

    func checkForUpdates() {
        // Cancel an in-flight simulation so the user isn't stuck in `checking`
        // if they tap twice.
        simulationTask?.cancel()
        state = .checking
        simulationTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(1400))
            guard !Task.isCancelled, let self else { return }
            // Stub always returns "up to date". PR 11 posts a real check.
            self.state = .idle(lastChecked: Date())
        }
    }

    func installAndRestart() {
        // Real implementation lands in PR 11. This stub is a no-op so the
        // button can be wired today without lying to the user.
    }

    // MARK: - Persistence

    private struct Prefs: Codable {
        var channel: UpdateChannel
        var autoCheck: Bool
        var autoDownload: Bool
    }

    private static func readPrefs(from url: URL) -> Prefs? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Prefs.self, from: data)
    }

    private func persist() {
        let prefs = Prefs(
            channel: channel,
            autoCheck: autoCheck,
            autoDownload: autoDownload
        )
        guard let data = try? JSONEncoder().encode(prefs) else { return }
        try? data.write(to: storeURL, options: [.atomic])
    }
}
