// PR 12 — Accuracy Benchmark screen.
//
// Mirrors the "Accuracy" tab from the HTML admin panel
// (omlx/admin/templates/dashboard/_bench.html + dashboard.js accBench*).
// Wires the /admin/api/bench/accuracy/* endpoints — queue add / status /
// remove / results / reset / cancel — onto a stack of sections:
//
//   Configuration         — model picker, batch size segmented, extended-
//                           thinking toggle, and a tap-to-toggle benchmark
//                           grid with inline per-benchmark sample-size
//                           dropdowns. Hard-coded catalog mirrors the HTML
//                           dropdown order.
//
//   Queue                 — visible only when the server reports a running
//                           bench or pending queue items. Shows the active
//                           model (spinner + last-progress message + cancel
//                           button) and each queued entry (model + comma-
//                           separated benchmarks + remove button).
//
//   Error banner          — same shape as QuantizationScreen.
//
//   Results               — accumulating cards keyed by `bench::model`.
//                           Big accuracy %, model badge, optional extended-
//                           thinking pill, correct/total · time. Expandable
//                           per-category breakdown when the server emits it.
//
//   Text export           — collapsible one-liner dump with copy-to-clipboard.
//
// v1 strategy: poll. 2 s while a bench is running OR the queue is non-empty,
// 8 s while idle. Per-question SSE progress isn't surfaced — only block-level
// `message + current/total` from `lastProgress`, which is the same level the
// HTML UI exposes.

import SwiftUI
import AppKit

struct AccuracyBenchScreen: View {
    @EnvironmentObject private var services: AppServices
    // VM is owned by AppServices so an in-flight queue (or in-progress
    // benchmark) survives screen unloads. Same rationale as
    // ThroughputBenchScreen — see AppServices.accuracyBench.
    @ObservedObject var vm: AccuracyBenchScreenVM

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HeaderSection()

            ConfigurationSection(
                models: vm.models,
                selectedModelId: $vm.selectedModelId,
                batchSize: $vm.batchSize,
                enableThinking: $vm.enableThinking,
                selectedBenchmarks: $vm.selectedBenchmarks,
                sampleSizes: $vm.sampleSizes,
                isAdding: vm.isAdding,
                canSubmit: vm.canSubmit,
                onSubmit: { vm.addToQueue(client: services.client) }
            )

            QueueSection(
                status: vm.status,
                onCancel: { vm.cancelRunning(client: services.client) },
                onRemove: { idx in vm.removeFromQueue(client: services.client, index: idx) }
            )

            BannerSection(error: vm.lastError)

            ResultsSection(
                results: vm.results,
                onClear: { vm.resetResults(client: services.client) }
            )

            if !vm.results.isEmpty {
                TextExportSection(results: vm.results)
            }
        }
        // `start()` is idempotent: refreshes models + polls once, then
        // restarts the poll loop (which cancels its predecessor). The
        // poll task continues across screen unloads since AppServices
        // owns the VM, so we don't tear it down on disappear.
        .task { await vm.start(client: services.client) }
    }
}

// MARK: - Header

private struct HeaderSection: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Accuracy Benchmark")
                .font(.omlxText(11, weight: .semibold))
                .foregroundStyle(theme.textSecondary)
                .textCase(.uppercase)
                .kerning(0.6)
            Text("Measure model accuracy")
                .font(.omlxText(20, weight: .semibold))
                .foregroundStyle(theme.text)
            Text("Queue benchmarks across models. Results accumulate until you reset them. Resume across app launches via the server-side queue.")
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 14)
        .padding(.top, 18)
        .padding(.bottom, 10)
    }
}

// MARK: - Benchmark catalog (mirrors the HTML dropdown order/labels)

struct BenchmarkCatalogEntry: Hashable, Identifiable {
    let key: String
    let displayName: String
    let category: String
    var id: String { key }
}

private let benchmarkCatalog: [BenchmarkCatalogEntry] = [
    .init(key: "mmlu",          displayName: "MMLU",          category: "Knowledge"),
    .init(key: "mmlu_pro",      displayName: "MMLU-Pro",      category: "Knowledge"),
    .init(key: "kmmlu",         displayName: "KMMLU (Korean)", category: "Knowledge"),
    .init(key: "cmmlu",         displayName: "CMMLU (Chinese)", category: "Knowledge"),
    .init(key: "jmmlu",         displayName: "JMMLU (Japanese)", category: "Knowledge"),
    .init(key: "hellaswag",     displayName: "HellaSwag",     category: "Reasoning"),
    .init(key: "truthfulqa",    displayName: "TruthfulQA",    category: "Reasoning"),
    .init(key: "arc_challenge", displayName: "ARC-Challenge", category: "Reasoning"),
    .init(key: "winogrande",    displayName: "WinoGrande",    category: "Reasoning"),
    .init(key: "gsm8k",         displayName: "GSM8K",         category: "Math"),
    .init(key: "mathqa",        displayName: "MathQA",        category: "Math"),
    .init(key: "humaneval",     displayName: "HumanEval",     category: "Code"),
    .init(key: "mbpp",          displayName: "MBPP",          category: "Code"),
    .init(key: "livecodebench", displayName: "LiveCodeBench", category: "Code"),
    .init(key: "bbq",           displayName: "BBQ",           category: "Safety"),
    .init(key: "safetybench",   displayName: "SafetyBench",   category: "Safety"),
]

private let sampleSizeOptions: [(Int, String)] = [
    (0,    "Full"),
    (50,   "50"),
    (100,  "100"),
    (200,  "200"),
    (500,  "500"),
    (1000, "1000"),
]

private let batchSizeOptions: [(Int, String)] = [
    (1, "1"), (2, "2"), (4, "4"), (8, "8"), (16, "16"), (32, "32"),
]

// MARK: - Configuration

private struct ConfigurationSection: View {
    let models: [ModelDTO]
    @Binding var selectedModelId: String
    @Binding var batchSize: Int
    @Binding var enableThinking: Bool
    @Binding var selectedBenchmarks: Set<String>
    @Binding var sampleSizes: [String: Int]
    let isAdding: Bool
    let canSubmit: Bool
    let onSubmit: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            "Configuration",
            subtitle: subtitleText
        )

        ListGroup {
            Row(label: "Model", sublabel: "Loaded models are listed first") {
                Popup(
                    selection: $selectedModelId,
                    width: 320,
                    options: modelOptions
                )
            }

            Row(
                label: "Batch size",
                sublabel: "Higher batches finish faster but use more memory"
            ) {
                Segmented(selection: $batchSize, options: batchSizeOptions)
                    .frame(width: 260)
            }

            Row(
                label: "Extended thinking",
                sublabel: "Enable per-question reasoning traces (slower)"
            ) {
                Toggle("", isOn: $enableThinking).labelsHidden().toggleStyle(.switch)
            }

            FreeRow {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Text("Benchmarks")
                            .font(.omlxText(13, weight: .medium))
                            .foregroundStyle(theme.text)
                        Text(benchmarksSubtitle)
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.textSecondary)
                        Spacer(minLength: 0)
                    }
                    BenchmarkGrid(
                        selected: $selectedBenchmarks,
                        sampleSizes: $sampleSizes
                    )
                }
            }

            Row(isLast: true) {
                HStack {
                    Spacer()
                    Button {
                        onSubmit()
                    } label: {
                        if isAdding {
                            ProgressView()
                                .controlSize(.small)
                                .padding(.trailing, 2)
                            Text("Adding…")
                        } else {
                            Label("Add to Queue & Run", systemImage: "play.fill")
                                .labelStyle(.titleAndIcon)
                        }
                    }
                    .buttonStyle(.omlx(.primary))
                    .disabled(!canSubmit || isAdding)
                }
            }
        }
    }

    private var modelOptions: [PopupOption<String>] {
        var opts = [PopupOption(value: "", label: "Select a model…")]
        let sorted = models.sorted { (a, b) -> Bool in
            if a.loaded != b.loaded { return a.loaded && !b.loaded }
            return a.id.localizedCaseInsensitiveCompare(b.id) == .orderedAscending
        }
        opts += sorted.map { m in
            let badge = m.loaded ? " • loaded" : ""
            return PopupOption(value: m.id, label: "\(m.id)\(badge)")
        }
        return opts
    }

    private var subtitleText: String? {
        if models.isEmpty { return "Loading models…" }
        let count = selectedBenchmarks.count
        return "\(count) benchmark\(count == 1 ? "" : "s") selected"
    }

    private var benchmarksSubtitle: String {
        let count = selectedBenchmarks.count
        if count == 0 { return "Tap to select. 0 = full dataset." }
        return "\(count) selected"
    }
}

// MARK: - Benchmark grid

private struct BenchmarkGrid: View {
    @Binding var selected: Set<String>
    @Binding var sampleSizes: [String: Int]

    var body: some View {
        // 3-column grid on width > 600, 2-column otherwise. GeometryReader
        // gives us the live width so we don't need a fixed window size.
        GeometryReader { geo in
            let cols = geo.size.width > 600 ? 3 : 2
            let layout = Array(
                repeating: GridItem(.flexible(), spacing: 8),
                count: cols
            )
            LazyVGrid(columns: layout, alignment: .leading, spacing: 8) {
                ForEach(benchmarkCatalog) { entry in
                    BenchmarkCard(
                        entry: entry,
                        isSelected: selected.contains(entry.key),
                        sampleSize: binding(for: entry.key),
                        onToggle: { toggle(entry.key) }
                    )
                }
            }
        }
        .frame(minHeight: gridHeight)
    }

    /// LazyVGrid inside a GeometryReader needs an explicit frame height —
    /// otherwise it collapses to zero. Estimate generously: 8 rows × 64 pt
    /// covers both 2-column (8 rows) and 3-column (6 rows) layouts.
    private var gridHeight: CGFloat {
        let rows = Double(benchmarkCatalog.count) / 2.0
        return CGFloat((rows.rounded(.up)) * 66)
    }

    private func toggle(_ key: String) {
        if selected.contains(key) {
            selected.remove(key)
        } else {
            selected.insert(key)
            if sampleSizes[key] == nil { sampleSizes[key] = 100 }
        }
    }

    private func binding(for key: String) -> Binding<Int> {
        Binding(
            get: { sampleSizes[key] ?? 100 },
            set: { sampleSizes[key] = $0 }
        )
    }
}

private struct BenchmarkCard: View {
    let entry: BenchmarkCatalogEntry
    let isSelected: Bool
    @Binding var sampleSize: Int
    let onToggle: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: onToggle) {
                HStack(alignment: .center, spacing: 8) {
                    Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                        .font(.system(size: 13))
                        .foregroundStyle(isSelected ? theme.accent : theme.textTertiary)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(entry.displayName)
                            .font(.omlxText(12.5, weight: .medium))
                            .foregroundStyle(theme.text)
                        Text(entry.category)
                            .font(.omlxText(10.5))
                            .foregroundStyle(theme.textTertiary)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isSelected {
                HStack(spacing: 6) {
                    Text("Samples:")
                        .font(.omlxText(10.5))
                        .foregroundStyle(theme.textTertiary)
                    Popup(
                        selection: $sampleSize,
                        width: 90,
                        options: sampleSizeOptions
                    )
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(isSelected ? theme.controlBg : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .strokeBorder(
                    isSelected ? theme.inputBorder : theme.groupBorder,
                    lineWidth: 0.5
                )
        )
    }
}

// MARK: - Queue

private struct QueueSection: View {
    let status: AccuracyQueueStatus?
    let onCancel: () -> Void
    let onRemove: (Int) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        // Use `isActivelyEvaluating` instead of bare `running`: the
        // server's `running` stays True while the bench task is alive,
        // including the post-result model-unload window. During that
        // window the result card is already on screen, so showing a
        // "Running" row with the final progress message reads as the
        // task being stuck. See AccuracyQueueStatus.isActivelyEvaluating.
        let activelyRunning = status?.isActivelyEvaluating == true
        let queue = status?.queue ?? []
        let showSection = activelyRunning || !queue.isEmpty

        if showSection {
            SectionHeader("Queue", subtitle: subtitle(running: activelyRunning, queue: queue))

            ListGroup {
                if activelyRunning {
                    let isLast = queue.isEmpty
                    FreeRow(isLast: isLast) {
                        RunningRow(
                            modelId: status?.currentModel ?? "",
                            progress: status?.lastProgress,
                            onCancel: onCancel
                        )
                    }
                }

                ForEach(Array(queue.enumerated()), id: \.offset) { idx, item in
                    FreeRow(isLast: idx == queue.count - 1) {
                        QueuedRow(
                            index: idx,
                            item: item,
                            onRemove: { onRemove(idx) }
                        )
                    }
                }
            }
        }
    }

    private func subtitle(running: Bool, queue: [AccuracyQueueItem]) -> String {
        let queuedCount = queue.count
        let queuedPart = "\(queuedCount) queued"
        if running { return "\(queuedPart) · 1 running" }
        if queuedCount == 0 { return "no active runs" }
        return queuedPart
    }
}

private struct RunningRow: View {
    let modelId: String
    let progress: AccuracyProgressDTO?
    let onCancel: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text(modelId.isEmpty ? "Running…" : modelId)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
                StatusPill(status: .custom(
                    color: theme.blueDot,
                    label: "Running",
                    fillBg: true
                ))
                Spacer(minLength: 6)
                Button(action: onCancel) {
                    Label("Cancel", systemImage: "stop.fill")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.destructive, size: .small))
            }
            if let line = progressLine {
                Text(line)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var progressLine: String? {
        guard let p = progress else { return nil }
        var bits: [String] = []
        if let bench = p.benchmark, !bench.isEmpty { bits.append(bench) }
        if let msg = p.message, !msg.isEmpty { bits.append(msg) }
        if let cur = p.current, let tot = p.total, tot > 0 {
            bits.append("\(cur)/\(tot)")
        }
        if let bCur = p.benchCurrent, let bTot = p.benchTotal, bTot > 0 {
            bits.append("bench \(bCur)/\(bTot)")
        }
        return bits.isEmpty ? nil : bits.joined(separator: " · ")
    }
}

private struct QueuedRow: View {
    let index: Int
    let item: AccuracyQueueItem
    let onRemove: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            Image(systemName: "list.bullet.rectangle")
                .font(.system(size: 12))
                .foregroundStyle(theme.textTertiary)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.modelId)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(item.benchmarks.map(displayName(for:)).joined(separator: ", "))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 6)
            Button {
                onRemove()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 11))
            }
            .buttonStyle(.omlx(.plain, size: .small))
            .help("Remove from queue")
        }
    }

    private func displayName(for key: String) -> String {
        benchmarkCatalog.first(where: { $0.key == key })?.displayName ?? key
    }
}

// MARK: - Error banner

private struct BannerSection: View {
    let error: String?

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        if let error, !error.isEmpty {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(theme.redDot)
                    .font(.system(size: 11))
                    .padding(.top, 1)
                Text(error)
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.text)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(10)
            .background(theme.redDot.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .padding(.horizontal, 18)
            .padding(.top, 6)
        }
    }
}

// MARK: - Results

private struct ResultsSection: View {
    let results: [AccuracyResultDTO]
    let onClear: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        if !results.isEmpty {
            SectionHeader(
                "Results",
                subtitle: "\(results.count) result\(results.count == 1 ? "" : "s")"
            ) {
                Button("Clear all") { onClear() }
                    .buttonStyle(.omlx(.plain, size: .small))
                    .foregroundStyle(theme.redDot)
            }

            ListGroup {
                ForEach(Array(results.enumerated()), id: \.element.id) { idx, result in
                    FreeRow(isLast: idx == results.count - 1) {
                        ResultCard(result: result)
                    }
                }
            }
        }
    }
}

private struct ResultCard: View {
    let result: AccuracyResultDTO

    @State private var categoriesOpen: Bool = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text(percentText)
                    .font(.omlxText(22, weight: .semibold))
                    .foregroundStyle(accuracyColor)
                    .monospacedDigit()
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(benchmarkDisplay)
                            .font(.omlxText(13, weight: .medium))
                            .foregroundStyle(theme.text)
                        Pill(label: result.modelId, color: theme.blueDot)
                        if result.thinkingUsed {
                            Pill(label: "Extended thinking", color: Color(rgb24: 0x5E5CE6))
                        }
                    }
                    Text(subtitleText)
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textSecondary)
                }
                Spacer(minLength: 0)
            }

            if result.categoryScores?.isEmpty == false {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { categoriesOpen.toggle() }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: categoriesOpen ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                        Text("Categories")
                            .font(.omlxText(11, weight: .medium))
                    }
                    .foregroundStyle(theme.textSecondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                if categoriesOpen, let scores = result.categoryScores {
                    CategoriesTable(scores: scores)
                }
            }
        }
    }

    private var percentText: String {
        let pct = result.accuracy * 100
        return String(format: "%.1f%%", pct)
    }

    private var accuracyColor: Color {
        let pct = result.accuracy * 100
        if pct >= 70 { return theme.greenDot }
        if pct >= 40 { return theme.amberDot }
        return theme.redDot
    }

    private var benchmarkDisplay: String {
        benchmarkCatalog.first(where: { $0.key == result.benchmark })?.displayName
            ?? result.benchmark
    }

    private var subtitleText: String {
        let time = String(format: "%.1f s", result.timeS)
        return "\(result.correct) / \(result.total) · \(time)"
    }
}

private struct Pill: View {
    let label: String
    let color: Color

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(label)
            .font(.omlxText(10, weight: .medium))
            .foregroundStyle(color)
            .lineLimit(1)
            .truncationMode(.middle)
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
    }
}

private struct CategoriesTable: View {
    let scores: [String: Double]

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let entries = scores.sorted { $0.value > $1.value }
        VStack(spacing: 0) {
            ForEach(Array(entries.enumerated()), id: \.element.key) { idx, pair in
                HStack(spacing: 10) {
                    Text(pair.key)
                        .font(.omlxText(11))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Spacer(minLength: 8)
                    Text(String(format: "%.1f%%", pair.value * 100))
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.text)
                        .monospacedDigit()
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .overlay(alignment: .bottom) {
                    if idx < entries.count - 1 {
                        Rectangle()
                            .fill(theme.rowSep)
                            .frame(height: 0.5)
                            .padding(.leading, 10)
                    }
                }
            }
        }
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
}

// MARK: - Text export

private struct TextExportSection: View {
    let results: [AccuracyResultDTO]

    @State private var isOpen: Bool = false
    @State private var copied: Bool = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader("Text Export")

        ListGroup {
            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Button {
                            withAnimation(.easeOut(duration: 0.15)) { isOpen.toggle() }
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: isOpen ? "chevron.down" : "chevron.right")
                                    .font(.system(size: 9, weight: .semibold))
                                Text(isOpen ? "Hide text dump" : "Show text dump")
                                    .font(.omlxText(11, weight: .medium))
                            }
                            .foregroundStyle(theme.textSecondary)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        Spacer(minLength: 0)
                        Button {
                            copyToClipboard()
                        } label: {
                            Label(copied ? "Copied" : "Copy",
                                  systemImage: copied ? "checkmark" : "doc.on.doc")
                                .labelStyle(.titleAndIcon)
                        }
                        .buttonStyle(.omlx(.normal, size: .small))
                    }
                    if isOpen {
                        ScrollView {
                            Text(textDump)
                                .font(.omlxMono(11))
                                .foregroundStyle(theme.text)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(8)
                                .textSelection(.enabled)
                        }
                        .frame(maxHeight: 180)
                        .background(theme.codeBg)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                    }
                }
            }
        }
        .padding(.bottom, 18)
    }

    private var textDump: String {
        results.map { r in
            let pct = String(format: "%.1f%%", r.accuracy * 100)
            let time = String(format: "%.1f s", r.timeS)
            return "\(r.benchmark) · \(r.modelId) · \(pct) (\(r.correct)/\(r.total)) · \(time)"
        }.joined(separator: "\n")
    }

    private func copyToClipboard() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(textDump, forType: .string)
        copied = true
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            await MainActor.run { copied = false }
        }
    }
}

// MARK: - View model

@MainActor
final class AccuracyBenchScreenVM: ObservableObject {
    // Form state
    @Published var selectedModelId: String = ""
    @Published var selectedBenchmarks: Set<String> = []
    @Published var sampleSizes: [String: Int] = [:]
    @Published var batchSize: Int = 4
    @Published var enableThinking: Bool = false

    // Server state
    @Published private(set) var models: [ModelDTO] = []
    @Published private(set) var status: AccuracyQueueStatus?
    @Published private(set) var results: [AccuracyResultDTO] = []

    // UI state
    @Published private(set) var isAdding: Bool = false
    @Published var lastError: String?

    private weak var client: OMLXClient?
    private var pollTask: Task<Void, Never>?

    var canSubmit: Bool {
        !selectedModelId.isEmpty && !selectedBenchmarks.isEmpty
    }

    // MARK: Lifecycle

    func start(client: OMLXClient) async {
        self.client = client
        await loadModels()
        await pollOnce()
        startPolling()
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    // MARK: Loaders

    private func loadModels() async {
        guard let client else { return }
        do {
            let resp = try await client.listModels()
            self.models = resp.models
        } catch {
            self.lastError = "Failed to load models: \(describe(error))"
        }
    }

    private func pollOnce() async {
        guard let client else { return }
        // Status and results are independent endpoints — fan them out so a
        // slow one doesn't block the other.
        async let statusFetch = client.getAccuracyQueueStatus()
        async let resultsFetch = client.listAccuracyResults()
        do {
            let s = try await statusFetch
            self.status = s
        } catch {
            // Status failures are transient — keep the previous snapshot so
            // the queue/running row doesn't flicker out during a hiccup.
        }
        do {
            let r = try await resultsFetch
            self.results = r.results
        } catch {
            // Same logic — last-known results stay visible.
        }
    }

    // MARK: Polling

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let active = await MainActor.run { () -> Bool in
                    let running = self.status?.running == true
                    let queued = (self.status?.queue.isEmpty == false)
                    return running || queued
                }
                // Fast 2 s cadence while work is in flight; idle 8 s otherwise.
                try? await Task.sleep(for: .seconds(active ? 2 : 8))
                if Task.isCancelled { return }
                await self.pollOnce()
            }
        }
    }

    // MARK: Actions

    func addToQueue(client: OMLXClient) {
        guard canSubmit, !isAdding else { return }
        // Snapshot form state — the user can keep editing while the request
        // is in flight; we want the version they confirmed.
        let modelId = selectedModelId
        let benchmarks: [String: Int] = Dictionary(
            uniqueKeysWithValues: selectedBenchmarks.map { key in
                (key, sampleSizes[key] ?? 100)
            }
        )
        let body = AccuracyQueueAddRequest(
            modelId: modelId,
            benchmarks: benchmarks,
            batchSize: batchSize,
            enableThinking: enableThinking
        )
        isAdding = true
        lastError = nil
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.isAdding = false } }
            do {
                let s = try await client.addAccuracyQueue(body)
                await MainActor.run {
                    guard let self else { return }
                    self.status = s
                    // Reset selection on success so the user can stage another
                    // run without manually clearing the grid.
                    self.selectedBenchmarks = []
                }
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to add to queue: \(self?.describe(error) ?? "")"
                }
            }
        }
    }

    func removeFromQueue(client: OMLXClient, index: Int) {
        Task { [weak self] in
            do {
                let s = try await client.removeAccuracyQueue(index: index)
                await MainActor.run { self?.status = s }
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to remove: \(self?.describe(error) ?? "")"
                }
            }
        }
    }

    func cancelRunning(client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.cancelAccuracyBench()
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to cancel: \(self?.describe(error) ?? "")"
                }
            }
        }
    }

    func resetResults(client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.resetAccuracyResults()
                await MainActor.run { self?.results = [] }
                await self?.pollOnce()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to clear results: \(self?.describe(error) ?? "")"
                }
            }
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
    }
}
