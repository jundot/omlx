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
                            (20000, "Last 20,000"),
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
                    if vm.hasRotatedFiles {
                        Button("Clear History") { vm.showClearConfirm = true }
                            .buttonStyle(.omlx(.normal, size: .small))
                            .disabled(vm.isClearing)
                            .help("Delete all rotated log files (server.log stays)")
                    }
                }
            }
            .confirmationDialog(
                "Delete all rotated log files?",
                isPresented: $vm.showClearConfirm,
                titleVisibility: .visible
            ) {
                Button("Delete \(vm.rotatedFileCount) file\(vm.rotatedFileCount == 1 ? "" : "s")", role: .destructive) {
                    Task { await vm.clearHistory(client: services.client) }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Removes every rotated log file (server.log.YYYY-MM-DD). The active server.log is kept.")
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
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
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
        ZStack {
            // NSTextView handles tens of thousands of lines effortlessly,
            // unlike SwiftUI's Text which lays out the entire string up
            // front on every render. The bridge below keeps selection,
            // smooth scrolling, and Cmd+A intact.
            LogTextView(
                text: text,
                fgColor: NSColor(theme.text),
                bgColor: NSColor(theme.codeBg)
            )
            if isEmpty && !isLoading {
                Text("No log entries.")
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 36)
            }
        }
        .frame(minHeight: 360, idealHeight: 480, maxHeight: .infinity)
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.cornerRadius, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }
}

/// AppKit-backed monospaced text view wrapped in a scroll view. Reads its
/// content from `text` and follows the bottom only when the user is already
/// pinned there — preserving the scroll position when they've scrolled up
/// to inspect a specific entry.
private struct LogTextView: NSViewRepresentable {
    let text: String
    let fgColor: NSColor
    let bgColor: NSColor

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        scroll.drawsBackground = false

        guard let textView = scroll.documentView as? NSTextView else {
            return scroll
        }
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = false
        textView.allowsUndo = false
        textView.usesFindBar = true
        textView.isIncrementalSearchingEnabled = true
        textView.drawsBackground = true
        textView.backgroundColor = bgColor
        textView.textContainerInset = NSSize(width: 12, height: 10)
        textView.font = NSFont.monospacedSystemFont(ofSize: 11.5, weight: .regular)
        textView.textColor = fgColor
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        // Log lines are often wider than the view; soft-wrap to width so
        // there's no horizontal scrollbar but long messages still readable.
        if let container = textView.textContainer {
            container.widthTracksTextView = true
            container.lineFragmentPadding = 0
        }
        textView.string = text
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let textView = scroll.documentView as? NSTextView else { return }

        textView.backgroundColor = bgColor
        textView.textColor = fgColor
        if textView.string == text { return }

        // Decide whether to follow the tail before mutating the string.
        let wasPinnedToBottom: Bool = {
            let clip = scroll.contentView
            let visibleMaxY = clip.documentVisibleRect.maxY
            let documentMaxY = clip.documentRect.maxY
            // 40 pt slack — keeps "follow the tail" feel even if the user
            // nudged the scrollwheel a hair.
            return documentMaxY - visibleMaxY < 40
        }()

        textView.string = text

        if wasPinnedToBottom {
            DispatchQueue.main.async {
                textView.scrollToEndOfDocument(nil)
            }
        }
    }
}

// MARK: - View model

@MainActor
final class LogsScreenVM: ObservableObject {
    @Published var lines: Int = 100
    @Published var selectedFile: String = ""
    @Published var logText: String = ""
    @Published var availableFiles: [String] = []
    @Published var lastError: String?
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var totalLines: Int = 0
    @Published private(set) var refreshKey: Int = 0

    /// Drives the "Delete all rotated log files?" confirmation. Bound to
    /// the .confirmationDialog isPresented arg.
    @Published var showClearConfirm: Bool = false
    @Published private(set) var isClearing: Bool = false

    /// True when there's at least one rotated file (server.log.YYYY-MM-DD)
    /// to delete. The active server.log isn't counted.
    var hasRotatedFiles: Bool { rotatedFileCount > 0 }
    var rotatedFileCount: Int {
        availableFiles.filter { $0 != "server.log" }.count
    }

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

    /// Wipe every rotated log file. The server keeps `server.log` untouched
    /// (deleting it would silently truncate the live tail because the
    /// process holds an open handle). After the delete settles we refresh
    /// so the file list and tail content reflect the new state.
    func clearHistory(client: OMLXClient) async {
        isClearing = true
        defer { isClearing = false }
        do {
            let resp = try await client.deleteLogs(file: nil)
            // If the user was viewing a rotated file, drop them back to
            // server.log — the one they were on no longer exists.
            if selectedFile != "server.log",
               resp.deleted.contains(selectedFile) {
                selectedFile = "server.log"
            }
            await tick()
            lastError = nil
        } catch {
            lastError = describe(error)
        }
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
