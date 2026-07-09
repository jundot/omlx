// PR 14 — Throughput Bench screen.
//
// Mirrors the "Throughput Bench" tab from the HTML admin panel
// (omlx/admin/templates/dashboard/_bench.html + dashboard.js benchmark
// section). Wires the /api/bench/* endpoints onto a stack of sections:
//
//   Header          — title + one-line description, matches the rest of
//                     the screens in this app.
//
//   Device chip     — chip-row banner showing chip/memory/GPU cores from
//                     GET /api/device-info. Hidden silently on failure.
//
//   Configuration   — model picker (Popup over /api/models), context
//                     length chips (1024…200K), generation length input,
//                     batch size chips, Run / Cancel button.
//
//   Live progress   — spinner / "Running… (n)" caption while polling
//                     getBenchResults at 1 Hz. SSE is not used in v1 —
//                     the poll endpoint exposes the same `results` array.
//
//   Error banner    — red banner if the most recent call failed or the
//                     server reported a terminal error.
//
//   Single Request  — table-style rows: Test, TTFT, TPOT, pp TPS, tg TPS,
//                     E2E, Throughput, Peak Mem. Only when there is at
//                     least one single-result row.
//
//   Batch Results   — Batch, tg TPS, pp TPS, avg TTFT, E2E, Speedup.
//                     Adds a synthetic "1×" baseline row derived from the
//                     first single-request row whose pp == 1024.
//
//   Text Export     — collapsible monospaced dump of both tables with a
//                     Copy button.

import AppKit
import SwiftUI

struct ThroughputBenchScreen: View {
    @Environment(AppServices.self) private var services
    // VM is owned by AppServices so a running bench survives screen
    // unloads — see AppServices.throughputBench for the rationale.
    @Bindable var vm: ThroughputBenchScreenVM

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScreenHeader(
                eyebrow: String(localized: "bench.throughput.header.eyebrow",
                                defaultValue: "Throughput Benchmark",
                                comment: "Eyebrow label above the Throughput Bench screen header"),
                title: String(localized: "bench.throughput.header.title",
                              defaultValue: "Measure inference speed",
                              comment: "Throughput Bench screen primary header"),
                subtitle: String(localized: "bench.throughput.header.subtitle",
                                 defaultValue: "Single-request TTFT/TPOT + continuous-batching TPS, swept across context lengths and batch sizes. Results stay in memory until the screen unloads.",
                                 comment: "Throughput Bench screen subtitle describing the measurement sweep")
            )

            if let device = vm.device {
                DeviceChip(device: device)
            }

            ConfigurationSection(
                models: vm.models,
                selectedModelId: $vm.selectedModelId,
                promptLengths: $vm.promptLengths,
                genLength: $vm.genLength,
                batchSizes: $vm.batchSizes,
                running: vm.running,
                canRun: vm.canRun,
                onRun: { vm.runBenchmark(client: services.client) },
                onCancel: { vm.cancelBenchmark(client: services.client) }
            )

            if vm.running {
                LiveProgressCard(
                    resultCount: vm.singleResults.count + vm.batchResults.count
                )
            }

            MessageBanner(error: vm.lastError)

            if !vm.singleResults.isEmpty {
                SingleResultsTable(results: vm.singleResults)
            }

            if !vm.batchResults.isEmpty {
                BatchResultsTable(
                    results: vm.batchResults,
                    baseline: vm.batchBaseline
                )
            }

            if !vm.singleResults.isEmpty || !vm.batchResults.isEmpty {
                TextExportSection(
                    isOpen: $vm.exportOpen,
                    text: vm.exportText
                )
            }

            if let upload = vm.uploadState, upload.phase != "idle" {
                UploadSection(state: upload)
            }
        }
        // `start()` is idempotent: it refreshes the model/device lists but
        // leaves the running-bench state alone. The poll task stays alive
        // across screen unloads (see VM .stop comment), so navigation
        // doesn't lose progress.
        .task { await vm.start(client: services.client) }
    }
}

// MARK: - Device chip

private struct DeviceChip: View {
    let device: DeviceInfoDTO
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "cpu")
                .font(.system(size: 10.5))
                .foregroundStyle(theme.textTertiary)
            Text(label)
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 18)
        .padding(.top, 2)
        .padding(.bottom, 8)
    }

    private var label: String {
        var parts: [String] = []
        let chip = [device.chipName, device.chipVariant]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        if !chip.isEmpty { parts.append(chip) }
        if let mem = device.memoryGb, mem > 0 {
            parts.append(String(localized: "bench.throughput.device.memory",
                                defaultValue: "\(mem) GB",
                                comment: "Device chip memory fragment; placeholder is the GB count"))
        }
        if let cores = device.gpuCores, cores > 0 {
            parts.append(String(localized: "bench.throughput.device.gpu_cores",
                                defaultValue: "\(cores) GPU cores",
                                comment: "Device chip GPU-cores fragment; placeholder is the core count"))
        }
        return parts.isEmpty ? "—" : parts.joined(separator: " · ")
    }
}

// MARK: - Configuration

private struct ConfigurationSection: View {
    let models: [ModelDTO]
    @Binding var selectedModelId: String
    @Binding var promptLengths: Set<Int>
    @Binding var genLength: String
    @Binding var batchSizes: Set<Int>
    let running: Bool
    let canRun: Bool
    let onRun: () -> Void
    let onCancel: () -> Void

    var body: some View {
        SectionHeader(
            String(localized: "bench.throughput.section.configuration",
                   defaultValue: "Configuration",
                   comment: "Section header for the Throughput Bench configuration block"),
            subtitle: models.isEmpty
                ? String(localized: "bench.throughput.subtitle.loading_models",
                         defaultValue: "Loading models…",
                         comment: "Throughput Bench section subtitle while models are loading")
                : String(localized: "bench.throughput.subtitle.model_count",
                         defaultValue: "Models available: \(models.count)",
                         comment: "Throughput Bench section subtitle showing how many models are available; placeholder is the count")
        )

        ListGroup {
            Row(label: String(localized: "bench.throughput.row.model.label",
                              defaultValue: "Model",
                              comment: "Row label for the Throughput Bench model picker"),
                sublabel: String(localized: "bench.throughput.row.model.sub",
                                 defaultValue: "Loaded or unloaded — server will load on demand",
                                 comment: "Sublabel under the Throughput Bench model picker")) {
                Popup(
                    selection: $selectedModelId,
                    width: 320,
                    options: modelOptions
                )
            }

            FreeRow {
                ChipRow(
                    title: String(localized: "bench.throughput.row.context_lengths.title",
                                  defaultValue: "Context lengths",
                                  comment: "Inline title above the prompt-length chip row"),
                    sublabel: String(localized: "bench.throughput.row.context_lengths.sub",
                                     defaultValue: "Prompt tokens to feed for each single-request trial",
                                     comment: "Sublabel under the prompt-length chip row"),
                    options: Self.promptLengthOptions,
                    selection: $promptLengths,
                    format: Self.formatPromptLength
                )
            }

            Row(label: String(localized: "bench.throughput.row.gen_length.label",
                              defaultValue: "Generation length",
                              comment: "Row label for the Throughput Bench generation-length field"),
                sublabel: String(localized: "bench.throughput.row.gen_length.sub",
                                 defaultValue: "Output tokens per single-request trial",
                                 comment: "Sublabel under the Throughput Bench generation-length field")) {
                TextInput(
                    text: $genLength,
                    placeholder: "128",
                    mono: true,
                    width: 110
                )
            }

            FreeRow {
                ChipRow(
                    title: String(localized: "bench.throughput.row.batch_sizes.title",
                                  defaultValue: "Batch sizes",
                                  comment: "Inline title above the batch-size chip row"),
                    sublabel: String(localized: "bench.throughput.row.batch_sizes.sub",
                                     defaultValue: "Concurrent requests per batch in the continuous-batching phase",
                                     comment: "Sublabel under the batch-size chip row"),
                    options: Self.batchSizeOptions,
                    selection: $batchSizes,
                    format: { "\($0)" }
                )
            }

            Row(isLast: true) {
                HStack {
                    Spacer()
                    if running {
                        Button {
                            onCancel()
                        } label: {
                            Label(String(localized: "common.cancel",
                                         defaultValue: "Cancel",
                                         comment: "Generic Cancel button label"),
                                  systemImage: "stop.fill")
                                .labelStyle(.titleAndIcon)
                        }
                        .buttonStyle(.omlx(.destructive))
                    } else {
                        Button {
                            onRun()
                        } label: {
                            Label(String(localized: "bench.throughput.button.run",
                                         defaultValue: "Run Benchmark",
                                         comment: "Throughput Bench primary button that starts a sweep"),
                                  systemImage: "play.fill")
                                .labelStyle(.titleAndIcon)
                        }
                        .buttonStyle(.omlx(.primary))
                        .disabled(!canRun)
                    }
                }
            }
        }
    }

    private var modelOptions: [PopupOption<String>] {
        var opts = [PopupOption(value: "", label: String(localized: "bench.throughput.model.placeholder",
                                                         defaultValue: "Select a model…",
                                                         comment: "Placeholder option in the Throughput Bench model picker"))]
        opts += models.map { m in
            PopupOption(value: m.id, label: m.id)
        }
        return opts
    }

    // Mirrors the HTML checkbox keys exactly.
    static let promptLengthOptions: [Int] = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000]
    static let batchSizeOptions: [Int]   = [2, 4, 8]

    static func formatPromptLength(_ n: Int) -> String {
        if n >= 1000 && n % 1000 == 0 { return "\(n / 1000)K" }
        if n >= 1024 && n.isMultiple(of: 1024) { return "\(n / 1024)K" }
        if n >= 1000 {
            let k = Double(n) / 1000
            return String(format: "%.1fK", k)
        }
        return "\(n)"
    }
}

// MARK: - Chip row

private struct ChipRow: View {
    let title: String
    let sublabel: String
    let options: [Int]
    @Binding var selection: Set<Int>
    let format: (Int) -> String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.omlxText(13, weight: .medium))
                    .foregroundStyle(theme.text)
                Text(sublabel)
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            }
            HStack(spacing: 6) {
                ForEach(options, id: \.self) { value in
                    Chip(
                        label: format(value),
                        isSelected: selection.contains(value),
                        onTap: { toggle(value) }
                    )
                }
                Spacer(minLength: 0)
            }
        }
    }

    private func toggle(_ value: Int) {
        if selection.contains(value) {
            selection.remove(value)
        } else {
            selection.insert(value)
        }
    }
}

private struct Chip: View {
    let label: String
    let isSelected: Bool
    let onTap: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Button(action: onTap) {
            Text(label)
                .font(.omlxText(11.5, weight: .medium))
                .foregroundStyle(isSelected ? theme.accentText : theme.textSecondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(isSelected ? theme.accent : theme.controlBg)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .strokeBorder(
                                    isSelected ? Color.clear : theme.inputBorder,
                                    lineWidth: 0.5
                                )
                        )
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Live progress

private struct LiveProgressCard: View {
    let resultCount: Int

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ListGroup {
            FreeRow(isLast: true) {
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)
                    if resultCount == 0 {
                        Text(String(localized: "bench.throughput.progress.warming_up",
                                    defaultValue: "Warming up…",
                                    comment: "Throughput Bench progress label before the first result arrives"))
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textSecondary)
                    } else {
                        Text(String(localized: "bench.throughput.progress.running",
                                    defaultValue: "Running… results so far: \(resultCount)",
                                    comment: "Throughput Bench progress label with how many results have arrived; placeholder is the count"))
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textSecondary)
                    }
                    Spacer(minLength: 0)
                }
            }
        }
    }
}

// MARK: - Banner

// MARK: - Single-request results

private struct SingleResultsTable: View {
    let results: [BenchResultDTO]

    @Environment(\.omlxTheme) private var theme

    private let columnHeaders: [String] = [
        String(localized: "bench.throughput.single.col.test",
               defaultValue: "Test",
               comment: "Single-request results column header: test descriptor"),
        String(localized: "bench.throughput.single.col.ttft",
               defaultValue: "TTFT (ms)",
               comment: "Single-request results column header: time-to-first-token in ms"),
        String(localized: "bench.throughput.single.col.tpot",
               defaultValue: "TPOT (ms)",
               comment: "Single-request results column header: time-per-output-token in ms"),
        String(localized: "bench.throughput.single.col.pp_tps",
               defaultValue: "pp TPS",
               comment: "Single-request results column header: prompt-processing tokens-per-second"),
        String(localized: "bench.throughput.single.col.tg_tps",
               defaultValue: "tg TPS",
               comment: "Single-request results column header: token-generation tokens-per-second"),
        String(localized: "bench.throughput.single.col.e2e",
               defaultValue: "E2E (s)",
               comment: "Single-request results column header: end-to-end latency in seconds"),
        String(localized: "bench.throughput.single.col.throughput",
               defaultValue: "Throughput",
               comment: "Single-request results column header: total throughput"),
        String(localized: "bench.throughput.single.col.peak_mem",
               defaultValue: "Peak Mem",
               comment: "Single-request results column header: peak memory usage"),
    ]

    var body: some View {
        SectionHeader(
            String(localized: "bench.throughput.section.single_results",
                   defaultValue: "Single Request Results",
                   comment: "Section header for the Throughput Bench single-request results table"),
            subtitle: String(localized: "bench.throughput.single.subtitle",
                             defaultValue: "Trials: \(results.count)",
                             comment: "Subtitle for the single-request results table; placeholder is the count")
        )

        ListGroup {
            FreeRow {
                HStack(spacing: 10) {
                    ForEach(Array(columnHeaders.enumerated()), id: \.offset) { _, h in
                        Text(h)
                            .font(.omlxText(10.5, weight: .semibold))
                            .foregroundStyle(theme.textTertiary)
                            .textCase(.uppercase)
                            .kerning(0.4)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            ForEach(Array(results.enumerated()), id: \.offset) { idx, r in
                FreeRow(isLast: idx == results.count - 1) {
                    HStack(spacing: 10) {
                        cell(testLabel(r), mono: true)
                        cell(format1(r.ttftMs))
                        cell(format1(r.tpotMs))
                        cell(format1(r.processingTps))
                        cell(format1(r.genTps))
                        cell(format1(r.e2eLatencyS))
                        cell(format1(r.totalThroughput))
                        cell(formatPeakMem(r.peakMemoryBytes))
                    }
                }
            }
        }
    }

    private func cell(_ text: String, mono: Bool = false) -> some View {
        Text(text)
            .font(mono ? .omlxMono(11.5) : .omlxText(11.5))
            .foregroundStyle(theme.text)
            .lineLimit(1)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func testLabel(_ r: BenchResultDTO) -> String {
        let pp = r.pp ?? 0
        let tg = r.tg ?? 0
        return String(localized: "bench.throughput.single.test_label",
                      defaultValue: "pp \(pp) / tg \(tg)",
                      comment: "Single-request row identifier showing prompt-processing and token-generation counts; placeholders are pp and tg integer counts")
    }
}

// MARK: - Batch results

private struct BatchResultsTable: View {
    let results: [BenchResultDTO]
    let baseline: BenchResultDTO?

    @Environment(\.omlxTheme) private var theme

    private let columnHeaders: [String] = [
        String(localized: "bench.throughput.batch.col.batch",
               defaultValue: "Batch",
               comment: "Batch results column header: batch size"),
        String(localized: "bench.throughput.batch.col.tg_tps",
               defaultValue: "tg TPS",
               comment: "Batch results column header: token-generation tokens-per-second"),
        String(localized: "bench.throughput.batch.col.pp_tps",
               defaultValue: "pp TPS",
               comment: "Batch results column header: prompt-processing tokens-per-second"),
        String(localized: "bench.throughput.batch.col.avg_ttft",
               defaultValue: "avg TTFT (ms)",
               comment: "Batch results column header: average time-to-first-token in ms"),
        String(localized: "bench.throughput.batch.col.e2e",
               defaultValue: "E2E (s)",
               comment: "Batch results column header: end-to-end latency in seconds"),
        String(localized: "bench.throughput.batch.col.speedup",
               defaultValue: "Speedup",
               comment: "Batch results column header: speedup vs 1x baseline"),
    ]

    var body: some View {
        SectionHeader(
            String(localized: "bench.throughput.section.batch_results",
                   defaultValue: "Batch Results",
                   comment: "Section header for the Throughput Bench continuous-batching results table"),
            subtitle: String(localized: "bench.throughput.batch.subtitle",
                             defaultValue: "Continuous batching vs 1× baseline",
                             comment: "Subtitle under the Batch Results section header")
        )

        ListGroup {
            FreeRow {
                HStack(spacing: 10) {
                    ForEach(Array(columnHeaders.enumerated()), id: \.offset) { _, h in
                        Text(h)
                            .font(.omlxText(10.5, weight: .semibold))
                            .foregroundStyle(theme.textTertiary)
                            .textCase(.uppercase)
                            .kerning(0.4)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            if let baseline {
                FreeRow {
                    HStack(spacing: 10) {
                        cell(String(localized: "bench.throughput.batch.baseline_label",
                                    defaultValue: "1× baseline",
                                    comment: "Row label for the synthetic 1x baseline in the batch results table"),
                             mono: true)
                        cell(format1(baseline.genTps))
                        cell(format1(baseline.processingTps))
                        cell(format1(baseline.ttftMs))
                        cell(format1(baseline.e2eLatencyS))
                        cell("1.0×")
                    }
                }
            }
            ForEach(Array(results.enumerated()), id: \.offset) { idx, r in
                FreeRow(isLast: idx == results.count - 1) {
                    HStack(spacing: 10) {
                        cell("\(r.batchSize ?? 0)×", mono: true)
                        cell(format1(r.tgTps))
                        cell(format1(r.ppTps))
                        cell(format1(r.avgTtftMs))
                        cell(format1(r.e2eLatencyS))
                        cell(speedupLabel(for: r))
                    }
                }
            }
        }
    }

    private func cell(_ text: String, mono: Bool = false) -> some View {
        Text(text)
            .font(mono ? .omlxMono(11.5) : .omlxText(11.5))
            .foregroundStyle(theme.text)
            .lineLimit(1)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func speedupLabel(for r: BenchResultDTO) -> String {
        guard let baseline,
              let baseTps = baseline.genTps, baseTps > 0,
              let tgTps = r.tgTps else { return "—" }
        return String(format: "%.2f×", tgTps / baseTps)
    }
}

// MARK: - Text export

private struct TextExportSection: View {
    @Binding var isOpen: Bool
    let text: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { isOpen.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: isOpen ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(theme.textSecondary)
                        Text(String(localized: "bench.throughput.text_export.title",
                                    defaultValue: "Text export",
                                    comment: "Disclosure header for the Throughput Bench text export block"))
                            .font(.omlxText(11, weight: .semibold))
                            .foregroundStyle(theme.textSecondary)
                            .textCase(.uppercase)
                            .kerning(0.6)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                Spacer(minLength: 0)

                if isOpen {
                    Button(String(localized: "common.copy",
                                  defaultValue: "Copy",
                                  comment: "Generic Copy button label")) { copyToPasteboard() }
                        .buttonStyle(.omlx(.normal, size: .small))
                }
            }
            .padding(.horizontal, 14)
            .padding(.top, 10)
            .padding(.bottom, 6)

            if isOpen {
                ListGroup {
                    FreeRow(isLast: true) {
                        ScrollView(.horizontal, showsIndicators: true) {
                            Text(text)
                                .font(.omlxMono(11))
                                .foregroundStyle(theme.text)
                                .fixedSize(horizontal: true, vertical: true)
                                .textSelection(.enabled)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.bottom, 18)
            }
        }
    }

    private func copyToPasteboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

// MARK: - Community leaderboard upload

/// Renders the result of the post-bench upload to the public omlx.ai
/// leaderboard. The upload happens server-side automatically after every
/// run (omlx/admin/benchmark.py:_upload_to_omlx_ai); we just surface the
/// state. Three modes:
///   • uploading: progress row + spinner
///   • skipped:   amber banner explaining why (experimental features)
///   • done:      per-context-length rows with link or error, plus a
///                summary footer showing the owner hash
private struct UploadSection: View {
    let state: BenchUploadStateDTO

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "bench.throughput.section.leaderboard",
                   defaultValue: "Community Leaderboard",
                   comment: "Section header for the Throughput Bench community-upload block"),
            subtitle: subtitle
        )

        switch state.phase {
        case "uploading":
            ListGroup {
                FreeRow(isLast: true) {
                    HStack(spacing: 10) {
                        ProgressView().controlSize(.small)
                        Text(String(localized: "bench.throughput.upload.uploading",
                                    defaultValue: "Uploading to omlx.ai…",
                                    comment: "Status label while bench results are being uploaded to the community leaderboard"))
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.textSecondary)
                        Spacer(minLength: 0)
                        if !state.results.isEmpty {
                            Text(String(localized: "bench.throughput.upload.count_submitted",
                                        defaultValue: "\(state.results.count) submitted",
                                        comment: "Inline counter shown during community upload; placeholder is the submitted count"))
                                .font(.omlxMono(11))
                                .foregroundStyle(theme.textTertiary)
                        }
                    }
                }
            }

        case "skipped":
            SkippedBanner(reason: state.skippedReason, features: state.skippedFeatures)

        case "done":
            ListGroup {
                let rows = state.results
                ForEach(Array(rows.enumerated()), id: \.element.id) { idx, r in
                    FreeRow(isLast: idx == rows.count - 1) {
                        UploadRow(result: r)
                    }
                }
            }
            if let hash = state.ownerHash, !hash.isEmpty {
                OwnerHashRow(ownerHash: hash, success: state.successCount, total: state.total)
            }

        default:
            EmptyView()
        }
    }

    private var subtitle: String? {
        switch state.phase {
        case "uploading": return String(localized: "bench.throughput.upload.subtitle.uploading",
                                        defaultValue: "Submitting results…",
                                        comment: "Community-leaderboard subtitle while results are uploading")
        case "skipped":   return String(localized: "bench.throughput.upload.subtitle.skipped",
                                        defaultValue: "Skipped",
                                        comment: "Community-leaderboard subtitle when the server skipped uploading")
        case "done":
            if state.failedCount == 0 {
                return String(localized: "bench.throughput.upload.subtitle.done_all",
                              defaultValue: "\(state.successCount) of \(state.total) submitted",
                              comment: "Community-leaderboard subtitle on success; placeholders are success count and total")
            }
            return String(localized: "bench.throughput.upload.subtitle.done_partial",
                          defaultValue: "\(state.successCount) ok · \(state.failedCount) failed",
                          comment: "Community-leaderboard subtitle when some uploads failed; placeholders are success and failure counts")
        default: return nil
        }
    }
}

private struct UploadRow: View {
    let result: BenchUploadResultDTO

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            Text(String(localized: "bench.throughput.upload.row.context_length",
                        defaultValue: "pp \(result.contextLength)",
                        comment: "Per-row label showing prompt-processing context length in the upload list; placeholder is the integer length"))
                .font(.omlxMono(12))
                .foregroundStyle(theme.text)
                .frame(width: 80, alignment: .leading)

            if let err = result.error, !err.isEmpty {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(theme.redDot)
                Text(err)
                    .font(.omlxText(11))
                    .foregroundStyle(theme.redDot)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 0)
            } else if let urlString = result.url, let url = URL(string: urlString) {
                Image(systemName: result.duplicate == true
                      ? "document.on.document"
                      : "checkmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(result.duplicate == true ? theme.textTertiary : theme.greenDot)
                Text(result.duplicate == true
                     ? String(localized: "bench.throughput.upload.row.already",
                              defaultValue: "Already submitted",
                              comment: "Upload row label when the leaderboard already has this result")
                     : String(localized: "bench.throughput.upload.row.submitted",
                              defaultValue: "Submitted",
                              comment: "Upload row label after a successful submission"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                Spacer(minLength: 0)
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    Label(String(localized: "common.open",
                                 defaultValue: "Open",
                                 comment: "Generic Open button label"),
                          systemImage: "arrow.up.right.square")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.plain, size: .small))
            } else {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(theme.textTertiary)
                Text(String(localized: "bench.throughput.upload.row.no_url",
                            defaultValue: "No URL returned",
                            comment: "Upload row label when the server returned no leaderboard URL"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textTertiary)
                Spacer(minLength: 0)
            }
        }
    }
}

private struct OwnerHashRow: View {
    let ownerHash: String
    let success: Int
    let total: Int

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "person.crop.circle.badge.checkmark")
                .font(.system(size: 11))
                .foregroundStyle(theme.textTertiary)
            Text(String(localized: "bench.throughput.upload.owner_hash.label",
                        defaultValue: "Owner hash",
                        comment: "Label next to the leaderboard owner-hash identifier"))
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
            Text(ownerHash)
                .font(.omlxMono(11))
                .foregroundStyle(theme.textSecondary)
                .textSelection(.enabled)
            Spacer(minLength: 0)
            Button(String(localized: "common.copy",
                          defaultValue: "Copy",
                          comment: "Generic Copy button label")) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(ownerHash, forType: .string)
            }
            .buttonStyle(.omlx(.plain, size: .small))
        }
        .padding(.horizontal, 18)
        .padding(.top, 4)
        .padding(.bottom, 10)
    }
}

private struct SkippedBanner: View {
    let reason: String?
    let features: [String]

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.bubble")
                .foregroundStyle(theme.amberDot)
                .font(.system(size: 11))
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text(String(localized: "bench.throughput.upload.skipped.title",
                            defaultValue: "Skipped community upload",
                            comment: "Banner heading shown when the server skipped uploading bench results"))
                    .font(.omlxText(12, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(body(reason: reason, features: features))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(theme.amberDot.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .padding(.horizontal, 14)
        .padding(.bottom, 10)
    }

    private func body(reason: String?, features: [String]) -> String {
        switch reason {
        case "experimental_features":
            let list = features.isEmpty
                ? String(localized: "bench.throughput.upload.skipped.experimental_default",
                         defaultValue: "experimental features",
                         comment: "Fallback noun phrase listed in the skipped-upload reason when the server returned no feature names")
                : features.joined(separator: ", ")
            return String(localized: "bench.throughput.upload.skipped.experimental",
                          defaultValue: "Results were not submitted because \(list) were active during the run. These features skew throughput and would pollute the leaderboard.",
                          comment: "Skipped-upload reason when experimental features were active; placeholder is the comma-joined feature list")
        default:
            return String(localized: "bench.throughput.upload.skipped.default",
                          defaultValue: "The server skipped uploading these results.",
                          comment: "Fallback skipped-upload reason when the server didn't provide one")
        }
    }
}

// MARK: - Local helpers

private func format1(_ value: Double?) -> String {
    guard let v = value else { return "—" }
    return String(format: "%.1f", v)
}

private func formatPeakMem(_ bytes: Int64?) -> String {
    guard let b = bytes, b > 0 else { return "—" }
    return formatBytes(b)
}
