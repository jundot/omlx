// PR 7 — Logs screen. Tails `/admin/api/logs` and renders the result in a
// monospaced ScrollView with a Lines popup, file selector, refresh button
// and copy-all. The view auto-refreshes every 5 s while visible.

import SwiftUI
import AppKit

struct LogsScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = LogsScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader("Server Logs", subtitle: vm.subtitle) {
                HStack(spacing: 6) {
                    Popup(
                        selection: $vm.lines,
                        width: 110,
                        options: [
                            (100,   "Last 100"),
                            (500,   "Last 500"),
                            (1000,  "Last 1,000"),
                            (5000,  "Last 5,000"),
                        ]
                    )
                    Button {
                        Task { await vm.reload() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .buttonStyle(.omlx(.normal, size: .small))
                    .disabled(vm.isLoading)
                    Button("Copy") { vm.copyToPasteboard() }
                        .buttonStyle(.omlx(.normal, size: .small))
                        .disabled(vm.lines == 0 || vm.logText.isEmpty)
                }
            }

            if vm.availableFiles.count > 1 {
                ListGroup {
                    Row(label: "Log file", isLast: true) {
                        Popup(
                            selection: $vm.selectedFile,
                            width: 220,
                            options: vm.fileOptions
                        )
                    }
                }
            }

            LogPane(text: vm.logText, isEmpty: vm.logText.isEmpty, isLoading: vm.isLoading)
                .padding(.horizontal, 14)
                .padding(.top, vm.availableFiles.count > 1 ? 0 : 6)
                .padding(.bottom, 8)

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 4)
            }
        }
        .task(id: vm.refreshKey) {
            await vm.start(client: services.client)
        }
        .onChange(of: vm.lines) { _, _ in vm.bumpRefreshKey() }
        .onChange(of: vm.selectedFile) { _, _ in vm.bumpRefreshKey() }
        .onDisappear { vm.stop() }
    }
}

// MARK: - Log pane

private struct LogPane: View {
    let text: String
    let isEmpty: Bool
    let isLoading: Bool

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                if isEmpty && !isLoading {
                    Text("No log entries.")
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 36)
                } else {
                    Text(text)
                        .font(.omlxMono(11.5))
                        .foregroundStyle(theme.text)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                }
            }
        }
        .frame(minHeight: 360, idealHeight: 480)
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }
}

// MARK: - View model

@MainActor
final class LogsScreenVM: ObservableObject {
    @Published var lines: Int = 500
    @Published var selectedFile: String = ""
    @Published var logText: String = ""
    @Published var availableFiles: [String] = []
    @Published var lastError: String?
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var totalLines: Int = 0
    @Published private(set) var refreshKey: Int = 0

    private weak var client: OMLXClient?
    private var pollTask: Task<Void, Never>?

    var subtitle: String {
        guard !logText.isEmpty else { return "" }
        return "\(totalLines) lines"
    }

    var fileOptions: [(String, String)] {
        availableFiles.map { name in
            (name, name == "server.log" ? "server.log (current)" : name)
        }
    }

    func start(client: OMLXClient) async {
        self.client = client
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tick()
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func reload() async {
        await tick()
    }

    func bumpRefreshKey() {
        refreshKey &+= 1
    }

    func copyToPasteboard() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(logText, forType: .string)
    }

    private func tick() async {
        guard let client else { return }
        isLoading = true
        defer { isLoading = false }

        let file = selectedFile.isEmpty ? nil : selectedFile
        do {
            let dto = try await client.getLogs(lines: lines, file: file)
            self.logText = dto.logs
            self.totalLines = dto.totalLines
            self.availableFiles = dto.availableFiles
            if selectedFile.isEmpty {
                self.selectedFile = dto.logFile
            }
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
