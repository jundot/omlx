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
    @EnvironmentObject private var services: AppServices
    // VM is owned by AppServices so a running bench survives screen
    // unloads — see AppServices.throughputBench for the rationale.
    @ObservedObject var vm: ThroughputBenchScreenVM

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HeaderSection()

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

            BannerSection(error: vm.lastError)

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

// MARK: - Header

private struct HeaderSection: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Throughput Benchmark")
                .font(.omlxText(11, weight: .semibold))
                .foregroundStyle(theme.textSecondary)
                .textCase(.uppercase)
                .kerning(0.6)
            Text("Measure inference speed")
                .font(.omlxText(20, weight: .semibold))
                .foregroundStyle(theme.text)
            Text("Single-request TTFT/TPOT + continuous-batching TPS, swept across context lengths and batch sizes. Results stay in memory until the screen unloads.")
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 14)
        .padding(.top, 18)
        .padding(.bottom, 10)
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
        if let mem = device.memoryGb, mem > 0 { parts.append("\(mem) GB") }
        if let cores = device.gpuCores, cores > 0 { parts.append("\(cores) GPU cores") }
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
            "Configuration",
            subtitle: models.isEmpty ? "Loading models…" : "\(models.count) model\(models.count == 1 ? "" : "s") available"
        )

        ListGroup {
            Row(label: "Model", sublabel: "Loaded or unloaded — server will load on demand") {
                Popup(
                    selection: $selectedModelId,
                    width: 320,
                    options: modelOptions
                )
            }

            FreeRow {
                ChipRow(
                    title: "Context lengths",
                    sublabel: "Prompt tokens to feed for each single-request trial",
                    options: Self.promptLengthOptions,
                    selection: $promptLengths,
                    format: Self.formatPromptLength
                )
            }

            Row(label: "Generation length", sublabel: "Output tokens per single-request trial") {
                TextInput(
                    text: $genLength,
                    placeholder: "128",
                    mono: true,
                    width: 110
                )
            }

            FreeRow {
                ChipRow(
                    title: "Batch sizes",
                    sublabel: "Concurrent requests per batch in the continuous-batching phase",
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
                            Label("Cancel", systemImage: "stop.fill")
                                .labelStyle(.titleAndIcon)
                        }
                        .buttonStyle(.omlx(.destructive))
                    } else {
                        Button {
                            onRun()
                        } label: {
                            Label("Run Benchmark", systemImage: "play.fill")
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
        var opts = [PopupOption(value: "", label: "Select a model…")]
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
                        Text("Warming up…")
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textSecondary)
                    } else {
                        Text("Running… (\(resultCount) result\(resultCount == 1 ? "" : "s") so far)")
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

// MARK: - Single-request results

private struct SingleResultsTable: View {
    let results: [BenchResultDTO]

    @Environment(\.omlxTheme) private var theme

    private let columnHeaders: [String] = [
        "Test", "TTFT (ms)", "TPOT (ms)", "pp TPS", "tg TPS",
        "E2E (s)", "Throughput", "Peak Mem",
    ]

    var body: some View {
        SectionHeader(
            "Single Request Results",
            subtitle: "\(results.count) trial\(results.count == 1 ? "" : "s")"
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
        return "pp \(pp) / tg \(tg)"
    }
}

// MARK: - Batch results

private struct BatchResultsTable: View {
    let results: [BenchResultDTO]
    let baseline: BenchResultDTO?

    @Environment(\.omlxTheme) private var theme

    private let columnHeaders: [String] = [
        "Batch", "tg TPS", "pp TPS", "avg TTFT (ms)", "E2E (s)", "Speedup",
    ]

    var body: some View {
        SectionHeader(
            "Batch Results",
            subtitle: "Continuous batching vs 1× baseline"
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
                        cell("1× baseline", mono: true)
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
                        Text("Text export")
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
                    Button("Copy") { copyToPasteboard() }
                        .buttonStyle(.omlx(.normal, size: .small))
                }
            }
            .padding(.horizontal, 14)
            .padding(.top, 10)
            .padding(.bottom, 6)

            if isOpen {
                ListGroup {
                    FreeRow(isLast: true) {
                        Text(text)
                            .font(.omlxMono(11))
                            .foregroundStyle(theme.text)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
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
            "Community Leaderboard",
            subtitle: subtitle
        )

        switch state.phase {
        case "uploading":
            ListGroup {
                FreeRow(isLast: true) {
                    HStack(spacing: 10) {
                        ProgressView().controlSize(.small)
                        Text("Uploading to omlx.ai…")
                            .font(.omlxText(11.5))
                            .foregroundStyle(theme.textSecondary)
                        Spacer(minLength: 0)
                        if !state.results.isEmpty {
                            Text("\(state.results.count) submitted")
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
        case "uploading": return "Submitting results…"
        case "skipped":   return "Skipped"
        case "done":
            if state.failedCount == 0 { return "\(state.successCount) of \(state.total) submitted" }
            return "\(state.successCount) ok · \(state.failedCount) failed"
        default: return nil
        }
    }
}

private struct UploadRow: View {
    let result: BenchUploadResultDTO

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            Text("pp \(result.contextLength)")
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
                      ? "doc.on.doc"
                      : "checkmark.circle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(result.duplicate == true ? theme.textTertiary : theme.greenDot)
                Text(result.duplicate == true ? "Already submitted" : "Submitted")
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                Spacer(minLength: 0)
                Button {
                    NSWorkspace.shared.open(url)
                } label: {
                    Label("Open", systemImage: "arrow.up.right.square")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.omlx(.plain, size: .small))
            } else {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 11))
                    .foregroundStyle(theme.textTertiary)
                Text("No URL returned")
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
            Text("Owner hash")
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
            Text(ownerHash)
                .font(.omlxMono(11))
                .foregroundStyle(theme.textSecondary)
                .textSelection(.enabled)
            Spacer(minLength: 0)
            Button("Copy") {
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
                Text("Skipped community upload")
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
            let list = features.isEmpty ? "experimental features" : features.joined(separator: ", ")
            return "Results were not submitted because \(list) were active during the run. These features skew throughput and would pollute the leaderboard."
        default:
            return "The server skipped uploading these results."
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

// MARK: - View model

@MainActor
final class ThroughputBenchScreenVM: ObservableObject {
    // Form state — defaults mirror the HTML admin panel's pre-ticked options.
    @Published var selectedModelId: String = ""
    @Published var promptLengths: Set<Int> = [4096, 16384]
    @Published var genLength: String = "128"
    @Published var batchSizes: Set<Int> = [2, 4]
    @Published var exportOpen: Bool = false

    // Server state
    @Published private(set) var models: [ModelDTO] = []
    @Published private(set) var device: DeviceInfoDTO?
    @Published private(set) var running: Bool = false
    @Published private(set) var singleResults: [BenchResultDTO] = []
    @Published private(set) var batchResults: [BenchResultDTO] = []
    @Published private(set) var currentBenchId: String?
    /// Server-side upload-to-leaderboard state, populated after the
    /// bench completes. Phases: "idle" (not yet started, or no upload
    /// because of experimental features detected later in the run) →
    /// "uploading" → "done" | "skipped". The poll loop keeps running
    /// past `status=completed` until this reaches a terminal phase so
    /// the user sees the leaderboard URL light up without manually
    /// refreshing.
    @Published private(set) var uploadState: BenchUploadStateDTO?
    @Published var lastError: String?

    private weak var client: OMLXClient?
    private var pollTask: Task<Void, Never>?
    /// Counts poll iterations spent waiting for the upload phase to
    /// terminate after the bench itself completes. Reset on each new
    /// run; capped at 120 (i.e. 2 min at 1 Hz) so a wedged upload
    /// doesn't hold the poll loop hostage forever.
    private var postCompleteTicks: Int = 0

    // MARK: Derived

    var canRun: Bool {
        !selectedModelId.isEmpty
            && !running
            && !promptLengths.isEmpty
            && !batchSizes.isEmpty
            && (Int(genLength) ?? 0) > 0
    }

    /// Synthetic 1× baseline for the Batch Results table: the first single
    /// trial whose pp == 1024 (matches the JS admin panel's behaviour).
    var batchBaseline: BenchResultDTO? {
        singleResults.first(where: { $0.pp == 1024 })
            ?? singleResults.first
    }

    /// Monospaced two-table dump used by the Text export card.
    var exportText: String {
        var lines: [String] = []
        if !singleResults.isEmpty {
            lines.append("# Single request results")
            lines.append(
                ["Test", "TTFT(ms)", "TPOT(ms)", "ppTPS", "tgTPS",
                 "E2E(s)", "Throughput", "PeakMem"]
                    .joined(separator: "\t")
            )
            for r in singleResults {
                lines.append([
                    "pp \(r.pp ?? 0) / tg \(r.tg ?? 0)",
                    format1(r.ttftMs),
                    format1(r.tpotMs),
                    format1(r.processingTps),
                    format1(r.genTps),
                    format1(r.e2eLatencyS),
                    format1(r.totalThroughput),
                    formatPeakMem(r.peakMemoryBytes),
                ].joined(separator: "\t"))
            }
            lines.append("")
        }
        if !batchResults.isEmpty {
            lines.append("# Batch results")
            lines.append(
                ["Batch", "tgTPS", "ppTPS", "avgTTFT(ms)", "E2E(s)", "Speedup"]
                    .joined(separator: "\t")
            )
            let baselineTps = batchBaseline?.genTps ?? 0
            if let baseline = batchBaseline {
                lines.append([
                    "1x baseline",
                    format1(baseline.genTps),
                    format1(baseline.processingTps),
                    format1(baseline.ttftMs),
                    format1(baseline.e2eLatencyS),
                    "1.00x",
                ].joined(separator: "\t"))
            }
            for r in batchResults {
                let speedup: String = {
                    guard baselineTps > 0, let tg = r.tgTps else { return "—" }
                    return String(format: "%.2fx", tg / baselineTps)
                }()
                lines.append([
                    "\(r.batchSize ?? 0)x",
                    format1(r.tgTps),
                    format1(r.ppTps),
                    format1(r.avgTtftMs),
                    format1(r.e2eLatencyS),
                    speedup,
                ].joined(separator: "\t"))
            }
        }
        return lines.isEmpty ? "No results yet." : lines.joined(separator: "\n")
    }

    // MARK: Lifecycle

    /// Idempotent: called every time the screen appears. Refreshes the
    /// model + device lists (cheap, ~ms) but never touches the
    /// running-bench state, results table, or poll task. If the user
    /// navigated away during a run, the same poll task is still alive
    /// updating these `@Published` properties — coming back just
    /// re-subscribes via SwiftUI's diffing.
    func start(client: OMLXClient) async {
        self.client = client
        await loadModels()
        await loadDevice()
    }

    /// Manually tear down the poll task. Not wired to the screen's
    /// `.onDisappear` — the bench survives screen unloads. Kept around
    /// for future "logout / disconnect" flows where the long-lived VM
    /// needs to be reset.
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
            // Surface so the user can recover; polling does not depend on this.
            self.lastError = describe(error)
        }
    }

    private func loadDevice() async {
        guard let client else { return }
        do {
            self.device = try await client.getDeviceInfo()
        } catch {
            // Device chip is a "nice to have" — hide silently on failure
            // so a missing /api/device-info doesn't block running the bench.
            self.device = nil
        }
    }

    // MARK: Actions

    func runBenchmark(client: OMLXClient) {
        guard canRun else { return }
        let body = BenchStartRequest(
            modelId: selectedModelId,
            promptLengths: promptLengths.sorted(),
            generationLength: Int(genLength) ?? 128,
            batchSizes: batchSizes.sorted()
        )
        // Wipe the previous run's tables so a new run doesn't accumulate
        // across unrelated configurations.
        singleResults = []
        batchResults = []
        uploadState = nil
        postCompleteTicks = 0
        lastError = nil
        running = true

        Task { [weak self] in
            do {
                let resp = try await client.startThroughputBench(body)
                await MainActor.run {
                    guard let self else { return }
                    self.currentBenchId = resp.benchId
                    self.pollResults(client: client)
                }
            } catch {
                await MainActor.run {
                    guard let self else { return }
                    self.running = false
                    self.lastError = self.describe(error)
                }
            }
        }
    }

    func cancelBenchmark(client: OMLXClient) {
        guard let benchId = currentBenchId else {
            // Nothing to cancel server-side — flip the UI back regardless
            // so we don't strand the screen in "Running…" forever.
            running = false
            return
        }
        Task { [weak self] in
            do {
                _ = try await client.cancelBench(benchId: benchId)
            } catch {
                await MainActor.run { self?.lastError = self?.describe(error) }
            }
            await MainActor.run {
                self?.running = false
                self?.pollTask?.cancel()
                self?.pollTask = nil
            }
        }
    }

    // MARK: Polling

    /// 1 Hz poll of GET /api/bench/{id}/results while running. Server
    /// returns the full `results` array — we append-dedupe per call so
    /// the in-progress tables don't flicker as new rows arrive.
    private func pollResults(client: OMLXClient) {
        pollTask?.cancel()
        guard let benchId = currentBenchId else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                do {
                    let resp = try await client.getBenchResults(benchId: benchId)
                    await MainActor.run {
                        self.absorb(results: resp.results)
                        if let err = resp.error, !err.isEmpty {
                            self.lastError = err
                        }
                        if let upload = resp.uploadState {
                            self.uploadState = upload
                        }
                        let status = resp.status.lowercased()
                        let terminal = (status == "completed"
                                        || status == "failed"
                                        || status == "cancelled")
                        if terminal {
                            self.running = false
                        }
                    }
                    // Keep polling past `status=completed` until the upload
                    // phase also terminates ("done" | "skipped"). The
                    // backend writes upload state on the same BenchmarkRun
                    // (benchmark.py:_upload_to_omlx_ai) and surfaces it
                    // via /results, so this is just one more tick or two.
                    // Cap with a 120 s safety net so a stuck upload
                    // doesn't keep the poll alive forever.
                    let (stillRunning, uploadDone, hitCap) = await MainActor.run {
                        () -> (Bool, Bool, Bool) in
                        let phase = self.uploadState?.phase ?? "idle"
                        let isTerminal = (phase == "done" || phase == "skipped")
                        self.postCompleteTicks += self.running ? 0 : 1
                        return (self.running, isTerminal,
                                self.postCompleteTicks >= 120)
                    }
                    if !stillRunning && (uploadDone || hitCap) {
                        await MainActor.run { self.postCompleteTicks = 0 }
                        return
                    }
                } catch {
                    // Transient failures (server restart, dropped socket)
                    // shouldn't kill the poll — log and try again.
                    await MainActor.run {
                        self.lastError = self.describe(error)
                    }
                }
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    /// Split the server's flat `results` array into single / batch buckets
    /// and merge against what we already have. Dedupe key:
    ///   • single  → "single::pp::tg"
    ///   • batch   → "batch::batchSize"
    /// Mirrors the JS panel: rows are unique per (testType, key).
    private func absorb(results: [BenchResultDTO]) {
        var singles: [BenchResultDTO] = []
        var batches: [BenchResultDTO] = []
        var seen = Set<String>()
        for r in results {
            switch r.testType {
            case "single":
                let key = "single::\(r.pp ?? 0)::\(r.tg ?? 0)"
                if seen.insert(key).inserted { singles.append(r) }
            case "batch":
                let key = "batch::\(r.batchSize ?? 0)"
                if seen.insert(key).inserted { batches.append(r) }
            default:
                continue
            }
        }
        // Sort for stable presentation regardless of arrival order.
        self.singleResults = singles.sorted { ($0.pp ?? 0, $0.tg ?? 0) < ($1.pp ?? 0, $1.tg ?? 0) }
        self.batchResults = batches.sorted { ($0.batchSize ?? 0) < ($1.batchSize ?? 0) }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
    }
}
