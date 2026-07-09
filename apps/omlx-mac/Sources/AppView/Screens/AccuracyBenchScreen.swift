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
    @Environment(AppServices.self) private var services
    // VM is owned by AppServices so an in-flight queue (or in-progress
    // benchmark) survives screen unloads. Same rationale as
    // ThroughputBenchScreen — see AppServices.accuracyBench.
    @Bindable var vm: AccuracyBenchScreenVM

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScreenHeader(
                eyebrow: String(localized: "bench.accuracy.header.eyebrow",
                                defaultValue: "Accuracy Benchmark",
                                comment: "Eyebrow label above the Accuracy Bench screen header"),
                title: String(localized: "bench.accuracy.header.title",
                              defaultValue: "Measure model accuracy",
                              comment: "Accuracy Bench screen primary header"),
                subtitle: String(localized: "bench.accuracy.header.subtitle",
                                 defaultValue: "Queue benchmarks across models. Results accumulate until you reset them. Resume across app launches via the server-side queue.",
                                 comment: "Accuracy Bench screen subtitle explaining queueing behavior")
            )

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

            MessageBanner(error: vm.lastError)

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

// MARK: - Benchmark catalog (mirrors the HTML dropdown order/labels)

struct BenchmarkCatalogEntry: Hashable, Identifiable {
    let key: String
    let displayName: String
    let category: String
    var id: String { key }
}

private let benchmarkCatalog: [BenchmarkCatalogEntry] = {
    let knowledge = String(localized: "bench.accuracy.category.knowledge",
                           defaultValue: "Knowledge",
                           comment: "Accuracy bench catalog category: knowledge benchmarks")
    let reasoning = String(localized: "bench.accuracy.category.reasoning",
                           defaultValue: "Reasoning",
                           comment: "Accuracy bench catalog category: reasoning benchmarks")
    let math = String(localized: "bench.accuracy.category.math",
                      defaultValue: "Math",
                      comment: "Accuracy bench catalog category: math benchmarks")
    let code = String(localized: "bench.accuracy.category.code",
                      defaultValue: "Code",
                      comment: "Accuracy bench catalog category: code benchmarks")
    let safety = String(localized: "bench.accuracy.category.safety",
                        defaultValue: "Safety",
                        comment: "Accuracy bench catalog category: safety benchmarks")
    // Benchmark display names are proper nouns (dataset names) shared with
    // the HTML admin; only the language-tag suffixes are translated.
    return [
        .init(key: "mmlu",          displayName: "MMLU",          category: knowledge),
        .init(key: "mmlu_pro",      displayName: "MMLU-Pro",      category: knowledge),
        .init(key: "kmmlu",
              displayName: String(localized: "bench.accuracy.dataset.kmmlu",
                                  defaultValue: "KMMLU (Korean)",
                                  comment: "Accuracy bench display name with language tag"),
              category: knowledge),
        .init(key: "cmmlu",
              displayName: String(localized: "bench.accuracy.dataset.cmmlu",
                                  defaultValue: "CMMLU (Chinese)",
                                  comment: "Accuracy bench display name with language tag"),
              category: knowledge),
        .init(key: "jmmlu",
              displayName: String(localized: "bench.accuracy.dataset.jmmlu",
                                  defaultValue: "JMMLU (Japanese)",
                                  comment: "Accuracy bench display name with language tag"),
              category: knowledge),
        .init(key: "hellaswag",     displayName: "HellaSwag",     category: reasoning),
        .init(key: "truthfulqa",    displayName: "TruthfulQA",    category: reasoning),
        .init(key: "arc_challenge", displayName: "ARC-Challenge", category: reasoning),
        .init(key: "winogrande",    displayName: "WinoGrande",    category: reasoning),
        .init(key: "gsm8k",         displayName: "GSM8K",         category: math),
        .init(key: "mathqa",        displayName: "MathQA",        category: math),
        .init(key: "humaneval",     displayName: "HumanEval",     category: code),
        .init(key: "mbpp",          displayName: "MBPP",          category: code),
        .init(key: "livecodebench", displayName: "LiveCodeBench", category: code),
        .init(key: "bbq",           displayName: "BBQ",           category: safety),
        .init(key: "safetybench",   displayName: "SafetyBench",   category: safety),
    ]
}()

private let sampleSizeOptions: [(Int, String)] = [
    (0,    String(localized: "bench.accuracy.sample_size.full",
                  defaultValue: "Full",
                  comment: "Sample-size dropdown option: full dataset")),
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
            String(localized: "bench.accuracy.section.configuration",
                   defaultValue: "Configuration",
                   comment: "Section header for the Accuracy Bench configuration block"),
            subtitle: subtitleText
        )

        ListGroup {
            Row(label: String(localized: "bench.accuracy.row.model.label",
                              defaultValue: "Model",
                              comment: "Row label for the Accuracy Bench model picker"),
                sublabel: String(localized: "bench.accuracy.row.model.sub",
                                 defaultValue: "Loaded models are listed first",
                                 comment: "Sublabel under the Accuracy Bench model picker")) {
                Popup(
                    selection: $selectedModelId,
                    width: 320,
                    options: modelOptions
                )
            }

            Row(
                label: String(localized: "bench.accuracy.row.batch_size.label",
                              defaultValue: "Batch size",
                              comment: "Row label for the Accuracy Bench batch-size selector"),
                sublabel: String(localized: "bench.accuracy.row.batch_size.sub",
                                 defaultValue: "Higher batches finish faster but use more memory",
                                 comment: "Sublabel under the Accuracy Bench batch-size selector")
            ) {
                Segmented(selection: $batchSize, options: batchSizeOptions)
            }

            Row(
                label: String(localized: "bench.accuracy.row.thinking.label",
                              defaultValue: "Extended thinking",
                              comment: "Row label for the Accuracy Bench extended-thinking toggle"),
                sublabel: String(localized: "bench.accuracy.row.thinking.sub",
                                 defaultValue: "Enable per-question reasoning traces (slower)",
                                 comment: "Sublabel under the Accuracy Bench extended-thinking toggle")
            ) {
                Toggle("", isOn: $enableThinking).labelsHidden().toggleStyle(.switch)
            }

            FreeRow {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Text(String(localized: "bench.accuracy.benchmarks.title",
                                    defaultValue: "Benchmarks",
                                    comment: "Inline label above the Accuracy Bench benchmark grid"))
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
                            Text(String(localized: "bench.accuracy.button.adding",
                                        defaultValue: "Adding…",
                                        comment: "Accuracy Bench add-to-queue button label while the request is in flight"))
                        } else {
                            Label(String(localized: "bench.accuracy.button.add",
                                         defaultValue: "Add to Queue & Run",
                                         comment: "Accuracy Bench primary button that adds the configured run to the queue"),
                                  systemImage: "play.fill")
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
        var opts = [PopupOption(value: "", label: String(localized: "bench.accuracy.model.placeholder",
                                                         defaultValue: "Select a model…",
                                                         comment: "Placeholder option in the Accuracy Bench model picker"))]
        let sorted = models.sorted { (a, b) -> Bool in
            if a.loaded != b.loaded { return a.loaded && !b.loaded }
            return a.id.localizedCaseInsensitiveCompare(b.id) == .orderedAscending
        }
        opts += sorted.map { m in
            let badge = m.loaded
                ? String(localized: "bench.accuracy.model.loaded_badge",
                         defaultValue: " • loaded",
                         comment: "Suffix appended to a loaded model's name in the Accuracy Bench picker")
                : ""
            return PopupOption(value: m.id, label: "\(m.id)\(badge)")
        }
        return opts
    }

    private var subtitleText: String? {
        if models.isEmpty { return String(localized: "bench.accuracy.subtitle.loading_models",
                                          defaultValue: "Loading models…",
                                          comment: "Accuracy Bench section subtitle while models are loading") }
        let count = selectedBenchmarks.count
        return String(localized: "bench.accuracy.subtitle.selected_count",
                      defaultValue: "Benchmarks selected: \(count)",
                      comment: "Accuracy Bench section subtitle showing how many benchmarks the user has picked; placeholder is the count")
    }

    private var benchmarksSubtitle: String {
        let count = selectedBenchmarks.count
        if count == 0 { return String(localized: "bench.accuracy.benchmarks.subtitle.empty",
                                      defaultValue: "Tap to select. 0 = full dataset.",
                                      comment: "Subtitle next to the Benchmarks grid when nothing is selected") }
        return String(localized: "bench.accuracy.benchmarks.subtitle.count",
                      defaultValue: "\(count) selected",
                      comment: "Subtitle next to the Benchmarks grid showing the selected count")
    }
}

// MARK: - Benchmark grid

private struct BenchmarkGrid: View {
    @Binding var selected: Set<String>
    @Binding var sampleSizes: [String: Int]

    @State private var calculatedHeight: CGFloat = 334

    private struct GridHeightKey: PreferenceKey {
        static let defaultValue: CGFloat = 334
        static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
            value = max(value, nextValue())
        }
    }

    var body: some View {
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
                    .frame(maxHeight: .infinity, alignment: .top)
                }
            }
            .background {
                GeometryReader { contentGeo in
                    Color.clear
                        .preference(key: GridHeightKey.self, value: contentGeo.size.height)
                }
            }
        }
        .frame(height: calculatedHeight)
        .onPreferenceChange(GridHeightKey.self) { height in
            withAnimation {
                self.calculatedHeight = height
            }
        }
    }

    private func toggle(_ key: String) {
        withAnimation {
            if selected.contains(key) {
                selected.remove(key)
            } else {
                selected.insert(key)
                if sampleSizes[key] == nil { sampleSizes[key] = 100 }
            }
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
            Toggle(isOn: Binding(
                get: { isSelected },
                set: { _ in onToggle() }
            )) {
                HStack {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(entry.displayName)
                            .font(.omlxText(12.5, weight: .medium))
                            .foregroundStyle(theme.text)
                        Text(entry.category)
                            .font(.omlxText(10.5))
                            .foregroundStyle(theme.textTertiary)
                    }
                    Spacer()
                }
            }

            if isSelected {
                HStack(spacing: 6) {
                    Text(String(localized: "bench.accuracy.card.samples_label",
                                defaultValue: "Samples:",
                                comment: "Inline label next to the per-benchmark sample-size dropdown"))
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
            SectionHeader(String(localized: "bench.accuracy.section.queue",
                                 defaultValue: "Queue",
                                 comment: "Section header for the Accuracy Bench queue/in-progress block"),
                          subtitle: subtitle(running: activelyRunning, queue: queue))

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
        let queuedPart = String(localized: "bench.accuracy.queue.subtitle.queued",
                                defaultValue: "\(queuedCount) queued",
                                comment: "Accuracy Bench queue subtitle fragment showing queued count")
        if running {
            return String(localized: "bench.accuracy.queue.subtitle.with_running",
                          defaultValue: "\(queuedPart) · 1 running",
                          comment: "Accuracy Bench queue subtitle when one bench is running; placeholder is the queued-count fragment")
        }
        if queuedCount == 0 {
            return String(localized: "bench.accuracy.queue.subtitle.empty",
                          defaultValue: "no active runs",
                          comment: "Accuracy Bench queue subtitle when nothing is running or queued")
        }
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
                Text(modelId.isEmpty
                     ? String(localized: "bench.accuracy.queue.running.placeholder",
                              defaultValue: "Running…",
                              comment: "Accuracy Bench running-row title when the server hasn't reported the model id yet")
                     : modelId)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
                StatusPill(status: .custom(
                    color: theme.blueDot,
                    label: String(localized: "bench.accuracy.status.running",
                                  defaultValue: "Running",
                                  comment: "Status pill label on the Accuracy Bench running row"),
                    fillBg: true
                ))
                Spacer(minLength: 6)
                Button(action: onCancel) {
                    Label(String(localized: "common.cancel",
                                 defaultValue: "Cancel",
                                 comment: "Generic Cancel button label"),
                          systemImage: "stop.fill")
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
            .help(String(localized: "bench.accuracy.queue.remove.help",
                         defaultValue: "Remove from queue",
                         comment: "Tooltip on the X button next to a queued Accuracy Bench entry"))
        }
    }

    private func displayName(for key: String) -> String {
        benchmarkCatalog.first(where: { $0.key == key })?.displayName ?? key
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
                String(localized: "bench.accuracy.section.results",
                       defaultValue: "Results",
                       comment: "Section header for the Accuracy Bench results list"),
                subtitle: String(localized: "bench.accuracy.results.subtitle",
                                 defaultValue: "Results: \(results.count)",
                                 comment: "Subtitle showing the number of accumulated Accuracy Bench results")
            ) {
                Button(String(localized: "bench.accuracy.results.clear_all",
                              defaultValue: "Clear all",
                              comment: "Button in the Accuracy Bench results header that resets the result list")) { onClear() }
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
                            Pill(label: String(localized: "bench.accuracy.result.thinking_pill",
                                               defaultValue: "Extended thinking",
                                               comment: "Pill on an Accuracy Bench result card when extended thinking was used"),
                                 color: Color(rgb24: 0x5E5CE6))
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
                        Text(String(localized: "bench.accuracy.result.categories",
                                    defaultValue: "Categories",
                                    comment: "Disclosure label on an Accuracy Bench result card revealing per-category breakdowns"))
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
        SectionHeader(String(localized: "bench.accuracy.section.text_export",
                             defaultValue: "Text Export",
                             comment: "Section header for the Accuracy Bench text-export block"))

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
                                Text(isOpen
                                     ? String(localized: "bench.accuracy.text_export.hide",
                                              defaultValue: "Hide text dump",
                                              comment: "Disclosure label that hides the Accuracy Bench text export")
                                     : String(localized: "bench.accuracy.text_export.show",
                                              defaultValue: "Show text dump",
                                              comment: "Disclosure label that reveals the Accuracy Bench text export"))
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
                            Label(copied
                                  ? String(localized: "bench.accuracy.text_export.copied",
                                           defaultValue: "Copied",
                                           comment: "Transient confirmation label after copying the Accuracy Bench text dump")
                                  : String(localized: "common.copy",
                                           defaultValue: "Copy",
                                           comment: "Generic Copy button label"),
                                  systemImage: copied ? "checkmark" : "document.on.document")
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
