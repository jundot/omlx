// Storage roofline screen — predicts the MoE-streaming decode ceiling
// from a cold SSD measurement plus the model's stored expert bytes per step.
// Mirrors omlx/admin/routes.py /api/bench/storage/* (start, {job_id}/results,
// predict). Two actions:
//
//   Run Measurement — 2 GiB uncached scratch (F_NOCACHE on macOS),
//     sequential + random-2MB reads with latency percentiles, then the
//     model profile + prediction. ~1-2 min; run with inference idle.
//
//   Predict Only — recompute the prediction from the latest stored
//     measurement (headers-only profile, milliseconds).
//
// The verdict row renders tok/cycle vs verify-byte-mult as the structural
// MTP inequality; the calibration row compares measured bench tok/s with
// the cold ceiling (efficiency > 100% means temporal locality + prefetch
// beat the cold-miss assumption, not that the math is wrong).

import SwiftUI

struct StorageRooflineScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = StorageRooflineScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScreenHeader(
                eyebrow: String(localized: "bench.storage.header.eyebrow",
                                defaultValue: "Storage Roofline",
                                comment: "Eyebrow label above the Storage Roofline screen header"),
                title: String(localized: "bench.storage.header.title",
                              defaultValue: "Predict the streaming ceiling",
                              comment: "Storage Roofline screen primary header"),
                subtitle: String(localized: "bench.storage.header.subtitle",
                                 defaultValue: "Cold-measure the volume holding the checkpoint, then derive the theoretical max tok/s for MoE expert streaming and whether MTP can structurally pay. Run with inference idle — concurrent IO steals bandwidth.",
                                 comment: "Storage Roofline screen subtitle"),
            )

            ConfigurationSection(
                models: vm.models,
                selectedModelId: $vm.selectedModelId,
                tokPerCycleText: $vm.tokPerCycleText,
                verifyMultText: $vm.verifyMultText,
                measuredBaseText: $vm.measuredBaseText,
                autoParams: vm.autoParams,
                running: vm.running,
                canRun: vm.canRun,
                canPredict: vm.canPredict,
                onRun: { vm.runBenchmark(client: services.client) },
                onPredict: { Task { await vm.predict() } }
            )

            if vm.running {
                ProgressCard(
                    phase: vm.progressPhase,
                    done: vm.progressDone,
                    total: vm.progressTotal
                )
            }

            MessageBanner(error: vm.lastError)

            if let report = vm.report {
                ReportSections(report: report)
            }

            if !vm.history.isEmpty {
                HistorySection(entries: vm.history)
            }
        }
        .task { await vm.start(client: services.client) }
        .onDisappear { vm.stop() }
    }
}

// MARK: - Configuration

private struct ConfigurationSection: View {
    let models: [ModelDTO]
    @Binding var selectedModelId: String
    @Binding var tokPerCycleText: String
    @Binding var verifyMultText: String
    @Binding var measuredBaseText: String
    let autoParams: StorageAutoParamsDTO?
    let running: Bool
    let canRun: Bool
    let canPredict: Bool
    let onRun: () -> Void
    let onPredict: () -> Void

    var body: some View {
        SectionHeader(
            String(localized: "bench.storage.section.configuration",
                   defaultValue: "Configuration",
                   comment: "Section header for the roofline configuration rows"),
            subtitle: models.isEmpty
                ? String(localized: "bench.storage.config.loading",
                         defaultValue: "Loading models…",
                         comment: "Section subtitle while the model list loads")
                : String(localized: "bench.storage.config.model_count",
                         defaultValue: "Models available: \(models.count)",
                         comment: "Section subtitle showing how many models are available; placeholder is the count")
        )
        ListGroup {
            Row(
                label: String(localized: "bench.storage.config.model",
                              defaultValue: "Model",
                              comment: "Row label for the model picker"),
                sublabel: String(localized: "bench.storage.config.model.sub",
                                 defaultValue: "Volume measured = where this checkpoint lives.",
                                 comment: "Sublabel for the model picker"),
                isLast: false
            ) {
                Popup(
                    selection: $selectedModelId,
                    width: 320,
                    options: models.map { PopupOption(value: $0.id, label: $0.displayName ?? $0.id) }
                )
            }
            Row(
                label: String(localized: "bench.storage.config.tok_per_cycle",
                              defaultValue: "MTP tok/cycle",
                              comment: "Row label for measured MTP tok per cycle"),
                sublabel: String(localized: "bench.storage.config.tok_per_cycle.sub",
                                 defaultValue: "1 + accept rate for depth-1 (Gap-2 bench: 1.76–1.79).",
                                 comment: "Sublabel for tok per cycle input"),
                isLast: false
            ) {
                TextInput(text: $tokPerCycleText, mono: true, width: 80)
            }
            Row(
                label: String(localized: "bench.storage.config.verify_mult",
                              defaultValue: "Verify byte mult",
                              comment: "Row label for verify byte multiplier"),
                sublabel: String(localized: "bench.storage.config.verify_mult.sub",
                                 defaultValue: "Verify reads this many × the base step's bytes (Gap-2 measured 2.3).",
                                 comment: "Sublabel for verify byte multiplier"),
                isLast: false
            ) {
                TextInput(text: $verifyMultText, mono: true, width: 80)
            }
            if let auto = autoParams {
                Row(
                    label: String(localized: "bench.storage.config.params_source",
                                  defaultValue: "Parameters",
                                  comment: "Row label for where verdict params come from"),
                    sublabel: paramsSourceText(auto),
                    isLast: false
                ) {
                    Text(paramsSourceBadge(auto))
                        .font(.caption2.monospaced())
                        .foregroundStyle(auto.available == true ? Color.green : Color.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(
                            (auto.available == true ? Color.green : Color.secondary)
                                .opacity(0.12),
                            in: Capsule()
                        )
                }
            }
            Row(
                label: String(localized: "bench.storage.config.measured_base",
                              defaultValue: "Measured base tok/s",
                              comment: "Row label for measured base tok/s (optional)"),
                sublabel: String(localized: "bench.storage.config.measured_base.sub",
                                 defaultValue: "Optional — fills the calibration row with your bench result.",
                                 comment: "Sublabel for measured base input"),
                isLast: true
            ) {
                TextInput(text: $measuredBaseText, mono: true, width: 80)
            }
        }

        HStack {
            Spacer()
            Button(String(localized: "bench.storage.button.predict",
                         defaultValue: "Predict Only",
                         comment: "Predict-only button (reuses latest measurement)"), action: onPredict)
                .buttonStyle(.omlx(.plain, size: .small))
                .disabled(!canPredict)
            Button(String(localized: "bench.storage.button.run",
                         defaultValue: "Run Measurement",
                         comment: "Run button for the full storage measurement"), action: onRun)
                .buttonStyle(.omlx(.primary))
                .disabled(!canRun)
        }
        .padding(.horizontal, 18)
        .padding(.top, 6)
    }
}


// MARK: - Auto-params badge helpers

/// Short badge text: "measured <date>" or "default".
private func paramsSourceBadge(_ auto: StorageAutoParamsDTO) -> String {
    if auto.available == true, let at = auto.derivedAt {
        let day = String(at.prefix(10))
        return String(localized: "bench.storage.badge.measured",
                      defaultValue: "measured \(day)",
                      comment: "Badge: params derived on this date")
    }
    return String(localized: "bench.storage.badge.default",
                  defaultValue: "default",
                  comment: "Badge: default params in use")
}

/// Longer sublabel explaining where each number came from.
private func paramsSourceText(_ auto: StorageAutoParamsDTO) -> String {
    if auto.available == true {
        var bits: [String] = []
        if let t = auto.tokPerCycle {
            bits.append(String(format: "tok/cycle %.2f", t))
        }
        if let v = auto.verifyByteMult {
            bits.append(String(format: "mult %.2f", v))
        }
        return String(localized: "bench.storage.config.params_source.auto",
                      defaultValue: "Derived from this machine's bench pairs (\(bits.joined(separator: ", "))).",
                      comment: "Sublabel when params are auto-derived")
    }
    return String(localized: "bench.storage.config.params_source.none",
                  defaultValue: "No bench pair with telemetry yet — defaults in use (1.0 / 2.3).",
                  comment: "Sublabel when no auto params exist")
}

// MARK: - History

private struct HistorySection: View {
    let entries: [OMLXClient.StorageHistoryEntry]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(
                String(localized: "bench.storage.section.history",
                       defaultValue: "Measurement history",
                       comment: "Section header for past storage measurements"),
                subtitle: String(localized: "bench.storage.history.sub",
                                 defaultValue: "Last 10 runs — deltas vs the previous run.",
                                 comment: "History section subtitle")
            )
            ListGroup {
                ForEach(Array(entries.enumerated()), id: \.offset) { pair in
                    historyRow(pair.offset, pair.element)
                }
            }
        }
    }

    private func historyRow(_ idx: Int, _ e: OMLXClient.StorageHistoryEntry) -> some View {
        let rand: Double = e.randReadGBps ?? 0
        let deltaText: String? = deltaLabel(idx: idx, e)
        let deltaCol: Color = deltaColor(idx: idx, e)
        let isLast: Bool = idx == entries.count - 1
        let sub: String = rowSublabel(idx: idx, e)
        return Row(
            label: timestampLabel(e),
            sublabel: sub,
            isLast: isLast
        ) {
            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "%.2f GB/s", rand))
                    .font(.omlxMono(12))
                if let d = deltaText {
                    Text(d)
                        .font(.caption2.monospaced())
                        .foregroundStyle(deltaCol)
                }
            }
        }
    }

    private func timestampLabel(_ e: OMLXClient.StorageHistoryEntry) -> String {
        let ts = e.timestamp ?? ""
        return String(ts.prefix(16))
    }

    /// Δ rand MB/s vs the previous (older) entry — the before/after
    /// comparator for a drive or cable swap.
    private func delta(idx: Int, _ e: OMLXClient.StorageHistoryEntry) -> Double? {
        guard idx + 1 < entries.count,
              let cur = e.randReadGBps,
              let prev = entries[idx + 1].randReadGBps else { return nil }
        return cur - prev
    }

    private func deltaLabel(idx: Int, _ e: OMLXClient.StorageHistoryEntry) -> String? {
        guard let d = delta(idx: idx, e) else { return nil }
        return String(format: "%+.2f", d)
    }

    private func deltaColor(idx: Int, _ e: OMLXClient.StorageHistoryEntry) -> Color {
        guard let d = delta(idx: idx, e) else { return .secondary }
        return d >= 0 ? .green : .orange
    }

    private func rowSublabel(idx: Int, _ e: OMLXClient.StorageHistoryEntry) -> String {
        var parts: [String] = []
        if let m = e.volumeMedia, !m.isEmpty { parts.append(m) }
        if let ceilBase = e.ceilingBaseTokS {
            parts.append(String(format: "teto %.2f tok/s", ceilBase))
        }
        if e.cacheClean != true {
            parts.append(String(localized: "bench.storage.history.uncached_flag",
                                 defaultValue: "uncached",
                                 comment: "Flag: measurement may include page-cache reads"))
        }
        return parts.joined(separator: " · ")
    }
}

// MARK: - Progress

private struct ProgressCard: View {
    let phase: String
    let done: Int
    let total: Int

    private var phaseLabel: String {
        switch phase {
        case "write":
            return String(localized: "bench.storage.progress.write",
                          defaultValue: "Writing scratch file…",
                          comment: "Progress label for the write phase")
        case "seq_read":
            return String(localized: "bench.storage.progress.seq_read",
                          defaultValue: "Sequential read (spill/load predictor)…",
                          comment: "Progress label for the sequential read phase")
        case "rand_read":
            return String(localized: "bench.storage.progress.rand_read",
                          defaultValue: "Random 2MB reads (decode predictor)…",
                          comment: "Progress label for the random read phase")
        case "done":
            return String(localized: "bench.storage.progress.done",
                          defaultValue: "Done",
                          comment: "Progress label for the completed phase")
        default:
            return String(localized: "bench.storage.progress.queued",
                          defaultValue: "Starting…",
                          comment: "Progress label for the queued phase")
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            ProgressView()
                .controlSize(.small)
            VStack(alignment: .leading, spacing: 2) {
                Text(phaseLabel)
                    .font(.omlxText(12))
                if total > 0 {
                    Text("\(formatBytes(Int64(done))) / \(formatBytes(Int64(total)))")
                        .font(.omlxMono(10.5))
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 8)
    }
}

// MARK: - Report

private struct ReportSections: View {
    let report: StorageRooflineReportDTO

    var body: some View {
        if let volume = report.volume, let meas = report.measurement {
            SectionHeader(
                String(localized: "bench.storage.section.measurement",
                       defaultValue: "Measurement",
                       comment: "Section header for the storage measurement rows"),
                subtitle: volumeSubtitle(volume)
            )
            ListGroup {
                Row(label: String(localized: "bench.storage.meas.media",
                                  defaultValue: "Media")) {
                    Text(mediaLabel(volume))
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(localized: "bench.storage.meas.seq",
                                  defaultValue: "Sequential read"),
                    sublabel: String(localized: "bench.storage.meas.seq.sub",
                                  defaultValue: "Uncached; spill-convert / load predictor")
                ) {
                    Text(formatBps(meas.seqReadBps))
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(format: String(localized: "bench.storage.meas.rand",
                                                 defaultValue: "Random %dMB read",
                                                 comment: "Row label for the random-read measurement; placeholder is the block size"),
                                   meas.readMb ?? 2),
                    sublabel: String(localized: "bench.storage.meas.rand.sub",
                                  defaultValue: "Decode predictor · one quantized expert")
                ) {
                    Text(formatBps(meas.randReadBps))
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(localized: "bench.storage.meas.latency",
                                  defaultValue: "Latency p50 / p99"),
                    sublabel: String(format: String(localized: "bench.storage.meas.latency.sub",
                                                     defaultValue: "%d IOPS · method %@",
                                                     comment: "Sublabel for latency row; placeholders are IOPS and method"),
                                      Int(meas.randIops ?? 0), meas.method ?? "?")
                ) {
                    Text(String(format: "%.2f / %.2f ms",
                                meas.randLatMsP50 ?? 0, meas.randLatMsP99 ?? 0))
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(localized: "bench.storage.meas.write",
                                  defaultValue: "Write"),
                    isLast: true
                ) {
                    Text(formatBps(meas.writeBps))
                        .font(.omlxMono(12))
                }
            }
            if meas.cacheClean == false {
                HintLine(text: String(localized: "bench.storage.hint.not_cold",
                                defaultValue: "No page-cache bypass on this platform — numbers are an upper bound (RAM speed), not a storage ceiling.",
                                comment: "Hint when the measurement could not bypass the page cache"))
            }
            ForEach(meas.warnings ?? [], id: \.self) { warning in
                HintLine(text: warning)
            }
        }

        if let profile = report.profile {
            SectionHeader(
                String(localized: "bench.storage.section.profile",
                       defaultValue: "Model profile",
                       comment: "Section header for the per-model MoE step profile"),
                subtitle: String(localized: "bench.storage.section.profile.sub",
                                 defaultValue: "Stored expert bytes per decode step, from checkpoint headers.",
                                 comment: "Subheader for the model profile section")
            )
            ListGroup {
                Row(label: String(localized: "bench.storage.profile.arch",
                                  defaultValue: "Architecture")) {
                    Text("\(profile.modelType ?? "?") · \(profile.numMoeLayers ?? 0) MoE layers")
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(localized: "bench.storage.profile.routing",
                                  defaultValue: "Routing"),
                    sublabel: String(localized: "bench.storage.profile.routing.sub",
                                  defaultValue: "Top-\(profile.topK ?? 0) of \(profile.routedTotalPerLayer ?? 0) routed experts per layer")
                ) {
                    Text("\(profile.topK ?? 0) / \(profile.routedTotalPerLayer ?? 0)")
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(localized: "bench.storage.profile.bytes_step",
                                  defaultValue: "Bytes per step"),
                    sublabel: String(localized: "bench.storage.profile.bytes_step.sub",
                                  defaultValue: "Routed-active experts paged off SSD each token (+\(formatBytes(profile.sharedBytesPerLayer ?? 0)) resident shared/layer)")
                ) {
                    Text(formatBytes(profile.bytesPerStep ?? 0))
                        .font(.omlxMono(12))
                }
            }
        }

        if let prediction = report.prediction {
            SectionHeader(
                String(localized: "bench.storage.section.verdict",
                       defaultValue: "Ceiling & MTP verdict",
                       comment: "Section header for the roofline prediction")
            )
            ListGroup {
                Row(
                    label: String(localized: "bench.storage.pred.ceiling_base",
                                  defaultValue: "Base ceiling (cold)"),
                    sublabel: String(localized: "bench.storage.pred.ceiling_base.sub",
                                  defaultValue: "Max tok/s if every expert byte comes off SSD cold")
                ) {
                    Text(String(format: "%.2f tok/s", prediction.ceilingBaseTokS ?? 0))
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(localized: "bench.storage.pred.ceiling_mtp",
                                  defaultValue: "MTP ceiling (cold)"),
                    sublabel: String(format: String(localized: "bench.storage.pred.ceiling_mtp.sub",
                                                     defaultValue: "tok/cycle %.2f vs %.2f× verify bytes",
                                                     comment: "Sublabel for MTP ceiling row; placeholders are tok/cycle and byte multiplier"),
                                      prediction.tokPerCycle ?? 0, prediction.verifyByteMult ?? 0)
                ) {
                    Text(String(format: "%.2f tok/s", prediction.ceilingMtpTokS ?? 0))
                        .font(.omlxMono(12))
                }
                if let eff = prediction.ceilingEffectiveTokS, eff > 0 {
                    Row(
                        label: String(localized: "bench.storage.pred.ceiling_effective",
                                      defaultValue: "Effective ceiling",
                                      comment: "Row label for the measured-bytes/token ceiling"),
                        sublabel: String(localized: "bench.storage.pred.ceiling_effective.sub",
                                         defaultValue: "Measured bytes/token ÷ this volume's random-read bandwidth",
                                         comment: "Sublabel: cold ceiling already nets out locality/prefetch"),
                        isLast: false
                    ) {
                        Text(String(format: "%.2f tok/s", eff))
                            .font(.omlxMono(12))
                    }
                }
                if let slow = prediction.measuredMtpSlowdown, slow > 0 {
                    Row(
                        label: String(localized: "bench.storage.pred.measured_wallclock",
                                      defaultValue: "Measured wall-clock",
                                      comment: "Row label for the measured MTP slowdown"),
                        sublabel: String(localized: "bench.storage.pred.measured_wallclock.sub",
                                         defaultValue: "tok/s base ÷ tok/s MTP from this machine's bench pair — below 1× pays",
                                         comment: "Sublabel for the measured slowdown row"),
                        isLast: false
                    ) {
                        Text(String(format: "%.2f×", slow))
                            .font(.omlxMono(12))
                            .foregroundStyle((prediction.measuredMtpPays == true)
                                             ? Color.green : Color.orange)
                    }
                }
                Row(
                    label: String(localized: "bench.storage.pred.verdict",
                                  defaultValue: "MTP verdict"),
                    isLast: report.calibration == nil
                ) {
                    // Wall-clock measurement wins over the byte model when
                    // a real bench pair exists (the byte model can miss
                    // non-I/O costs like verify-batch compute).
                    let effectivePays = prediction.measuredMtpPays ?? prediction.mtpProfitable
                    Text(effectivePays == true
                         ? String(localized: "bench.storage.pred.profitable",
                                  defaultValue: "Pays — enable",
                                  comment: "Verdict when MTP is structurally profitable")
                         : String(localized: "bench.storage.pred.not_profitable",
                                  defaultValue: "Loses structurally — keep OFF",
                                  comment: "Verdict when MTP is structurally unprofitable"))
                        .font(.omlxText(12, weight: .semibold))
                        .foregroundStyle(effectivePays == true ? Color.green : Color.orange)
                }
            }
            HintLine(text: prediction.explanation ?? "")
            if let src = report.paramsSource, !src.isEmpty {
                HintLine(text: String(
                    localized: "bench.storage.pred.params_source",
                    defaultValue: "Verdict parameters: \(src)",
                    comment: "Which params produced this verdict (explicit/auto/default)"
                ))
            }
        }

        if let calibration = report.calibration {
            SectionHeader(
                String(localized: "bench.storage.section.calibration",
                       defaultValue: "Calibration",
                       comment: "Section header for measured-vs-ceiling calibration")
            )
            ListGroup {
                Row(
                    label: String(localized: "bench.storage.calib.measured",
                                  defaultValue: "Measured base"),
                    sublabel: String(localized: "bench.storage.calib.measured.sub",
                                  defaultValue: "Your bench result"),
                    isLast: false
                ) {
                    Text(String(format: "%.2f tok/s", calibration.measuredBaseTokS ?? 0))
                        .font(.omlxMono(12))
                }
                Row(
                    label: String(localized: "bench.storage.calib.efficiency",
                                  defaultValue: "Ceiling efficiency"),
                    sublabel: calibrationSublabel(calibration),
                    isLast: calibration.efficiencyEffective == nil
                ) {
                    Text(String(format: "%.0f%%", (calibration.efficiency ?? 0) * 100))
                        .font(.omlxMono(12))
                }
                if let effEff = calibration.efficiencyEffective, effEff > 0 {
                    Row(
                        label: String(localized: "bench.storage.calib.efficiency_effective",
                                      defaultValue: "Effective-ceiling efficiency",
                                      comment: "Row label: measured tok/s over the effective (measured bytes/token) ceiling"),
                        sublabel: String(localized: "bench.storage.calib.efficiency_effective.sub",
                                         defaultValue: "Should be near 100% — the effective ceiling already contains the locality/prefetch dividend",
                                         comment: "Sublabel for the effective-ceiling efficiency row"),
                        isLast: true
                    ) {
                        Text(String(format: "%.0f%%", effEff * 100))
                            .font(.omlxMono(12))
                    }
                }
            }
        }
    }

    private func volumeSubtitle(_ volume: StorageVolumeDTO) -> String {
        var parts: [String] = []
        if let fs = volume.filesystem, !fs.isEmpty { parts.append(fs) }
        if let free = volume.freeBytes, free > 0 {
            parts.append(String(localized: "bench.storage.volume.free",
                                defaultValue: "\(formatBytes(free)) free",
                                comment: "Volume subtitle fragment; placeholder is free space"))
        }
        return parts.joined(separator: " · ")
    }

    private func mediaLabel(_ volume: StorageVolumeDTO) -> String {
        var parts: [String] = []
        if let name = volume.mediaName, !name.isEmpty { parts.append(name) }
        if let proto = volume.protocolField, !proto.isEmpty { parts.append(proto) }
        if !parts.isEmpty { parts.append(volume.solidState == true ? "SSD" : "HDD") }
        return parts.isEmpty ? (volume.mount ?? "?") : parts.joined(separator: " · ")
    }

    private func calibrationSublabel(_ c: RooflineCalibrationDTO) -> String {
        let eff = (c.efficiency ?? 0) * 100
        if eff > 100 {
            return String(localized: "bench.storage.calib.above",
                           defaultValue: "Above the cold ceiling: temporal locality + prefetch dividend",
                           comment: "Calibration note when efficiency exceeds 100%")
        }
        if eff >= 70 {
            return String(localized: "bench.storage.calib.near",
                           defaultValue: "Near the cold ceiling: decode is SSD-bound",
                           comment: "Calibration note when efficiency is near the ceiling")
        }
        return String(localized: "bench.storage.calib.below",
                      defaultValue: "Well under ceiling: bottleneck is elsewhere (CPU/Metal/scheduler)",
                      comment: "Calibration note when efficiency is well under the ceiling")
    }

    private func formatBps(_ bps: Double?) -> String {
        guard let v = bps, v > 0 else { return "—" }
        if v >= 1024 * 1024 * 1024 {
            return String(format: "%.2f GiB/s", v / 1024 / 1024 / 1024)
        }
        return String(format: "%.0f MB/s", v / 1024 / 1024)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        guard bytes > 0 else { return "0 B" }
        let gb = Double(bytes) / 1024 / 1024 / 1024
        if gb >= 1 { return String(format: "%.1f GiB", gb) }
        let mb = Double(bytes) / 1024 / 1024
        if mb >= 1 { return String(format: "%.1f MiB", mb) }
        return String(format: "%.0f KB", Double(bytes) / 1024)
    }
}

