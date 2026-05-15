// PR 12 — Quantization screen (oQ universal dynamic quantization).
//
// Mirrors the "Quantizer" tab from the HTML admin panel
// (omlx/admin/templates/dashboard/_models.html:1025-1280 + dashboard.js:3437-
// 3680). Wires the /admin/api/oq/* endpoints — list / estimate / start /
// tasks / cancel / remove — onto a stack of sections:
//
//   Source Model section  — model picker, sensitivity picker (conditional
//                            on the source model offering candidates), oQ
//                            level picker, Start button, status banner.
//
//   Estimate strip        — memory / effective bpw / output size pills
//                            (live from /api/oq/estimate, debounced at
//                            300 ms to match the JS dashboard).
//
//   Advanced settings     — collapsible block with text-only toggle (VLM
//                            only), preserve-MTP toggle (only when the
//                            source model exposes MTP heads), and the
//                            non-quant dtype segmented control.
//
//   Queue                 — every task `_oq_manager` returns. Polls at 2 Hz
//                            while any task is active, idles otherwise.
//
//   About                 — static documentation card (matches the marketing
//                            copy in the HTML so users get the same context
//                            in either UI).

import SwiftUI

struct QuantizationScreen: View {
    @EnvironmentObject private var services: AppServices
    @StateObject private var vm = QuantizationScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HeaderSection()

            SourceModelSection(
                models: vm.models,
                sensitivityCandidates: vm.sensitivityCandidates,
                selectedModelPath: $vm.selectedModelPath,
                sensitivityModelPath: $vm.sensitivityModelPath,
                oqLevel: $vm.oqLevel,
                isStarting: vm.isStarting,
                modelsLoaded: vm.modelsLoaded,
                onStart: { vm.startQuantization(client: services.client) }
            )

            if vm.selectedModelPath.isEmpty == false {
                EstimateStrip(
                    memoryText: vm.memoryText,
                    bpwText: vm.bpwText,
                    outputSizeText: vm.outputSizeText
                )
            }

            AdvancedSection(
                isOpen: $vm.advancedOpen,
                selectedIsVLM: vm.selectedIsVLM,
                selectedHasMTP: vm.selectedHasMTP,
                textOnly: $vm.textOnly,
                preserveMtp: $vm.preserveMtp,
                dtype: $vm.dtype
            )

            BannerSection(error: vm.lastError, success: vm.lastSuccess)

            if vm.modelsLoaded && vm.models.isEmpty {
                EmptyModelsBanner()
            }

            QueueSection(
                tasks: vm.tasks,
                onCancel: { id in vm.cancelTask(taskId: id, client: services.client) },
                onRemove: { id in vm.removeTask(taskId: id, client: services.client) }
            )

            AboutSection()
        }
        .task { await vm.start(client: services.client) }
        .onDisappear { vm.stop() }
        .onChange(of: vm.selectedModelPath) { _, _ in
            // Sensitivity choice is per-source-model; reset when source changes
            // so the dropdown can't dangle at a stale path.
            vm.sensitivityModelPath = ""
            vm.scheduleEstimateRefresh(client: services.client)
        }
        .onChange(of: vm.oqLevel) { _, _ in
            vm.scheduleEstimateRefresh(client: services.client)
        }
        .onChange(of: vm.preserveMtp) { _, _ in
            vm.scheduleEstimateRefresh(client: services.client)
        }
    }
}

// MARK: - Header

private struct HeaderSection: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("oQ Quantization")
                .font(.omlxText(11, weight: .semibold))
                .foregroundStyle(theme.textSecondary)
                .textCase(.uppercase)
                .kerning(0.6)
            Text("Quantize on device")
                .font(.omlxText(20, weight: .semibold))
                .foregroundStyle(theme.text)
            Text("Pick a full-precision model, choose an oQ level, and oMLX builds a mixed-precision plan tuned to that model's per-layer sensitivity. Output is standard mlx-lm safetensors — usable in any MLX runtime.")
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 14)
        .padding(.top, 18)
        .padding(.bottom, 10)
    }
}

// MARK: - Source model + start

private struct SourceModelSection: View {
    let models: [OQModelInfo]
    let sensitivityCandidates: [OQModelInfo]
    @Binding var selectedModelPath: String
    @Binding var sensitivityModelPath: String
    @Binding var oqLevel: Double
    let isStarting: Bool
    let modelsLoaded: Bool
    let onStart: () -> Void

    var body: some View {
        SectionHeader(
            "Source Model",
            subtitle: modelsLoaded ? "\(models.count) full-precision model\(models.count == 1 ? "" : "s") available" : "Loading…"
        )

        ListGroup {
            Row(
                label: "Source",
                sublabel: "Only full-precision models can be quantized"
            ) {
                Popup(
                    selection: $selectedModelPath,
                    width: 320,
                    options: modelOptions
                )
            }

            if !sensitivityCandidates.isEmpty && !selectedModelPath.isEmpty {
                Row(
                    label: "Sensitivity model",
                    sublabel: "Use a quantized variant to analyze layer sensitivity with ~4× less memory"
                ) {
                    Popup(
                        selection: $sensitivityModelPath,
                        width: 320,
                        options: sensitivityOptions
                    )
                }
            }

            Row(label: "oQ level", sublabel: "Lower bits = smaller, faster, less accurate") {
                Popup(
                    selection: $oqLevel,
                    width: 120,
                    options: Self.levelOptions
                )
            }

            Row(isLast: true) {
                HStack {
                    Spacer()
                    Button {
                        onStart()
                    } label: {
                        if isStarting {
                            ProgressView()
                                .controlSize(.small)
                                .padding(.trailing, 2)
                            Text("Starting…")
                        } else {
                            Label("Start Quantization", systemImage: "sparkles")
                                .labelStyle(.titleAndIcon)
                        }
                    }
                    .buttonStyle(.omlx(.primary))
                    .disabled(isStarting || selectedModelPath.isEmpty)
                }
            }
        }
    }

    private var modelOptions: [PopupOption<String>] {
        var opts = [PopupOption(value: "", label: "Select a model…")]
        opts += models.map { m in
            PopupOption(value: m.path, label: "\(m.name) (\(m.sizeFormatted))")
        }
        return opts
    }

    private var sensitivityOptions: [PopupOption<String>] {
        var opts = [PopupOption(value: "", label: "None (use source model)")]
        opts += sensitivityCandidates.map { m in
            PopupOption(value: m.path, label: "\(m.name) (\(m.sizeFormatted))")
        }
        return opts
    }

    // 2 / 3 / 3.5 / 4 / 5 / 6 / 8 — mirrors the HTML <option>s.
    static let levelOptions: [PopupOption<Double>] = [
        PopupOption(value: 2,   label: "oQ2"),
        PopupOption(value: 3,   label: "oQ3"),
        PopupOption(value: 3.5, label: "oQ3.5"),
        PopupOption(value: 4,   label: "oQ4"),
        PopupOption(value: 5,   label: "oQ5"),
        PopupOption(value: 6,   label: "oQ6"),
        PopupOption(value: 8,   label: "oQ8"),
    ]
}

// MARK: - Estimate strip

private struct EstimateStrip: View {
    let memoryText: String
    let bpwText: String
    let outputSizeText: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 18) {
            pill(icon: "memorychip", text: "Est. memory: ~\(memoryText.isEmpty ? "—" : memoryText)")
            pill(icon: "gauge.with.dots.needle.50percent",
                 text: bpwText.isEmpty ? "Calculating…" : "Effective \(bpwText) bpw")
            pill(icon: "shippingbox",
                 text: outputSizeText.isEmpty ? "Calculating…" : "Output size: ~\(outputSizeText)")
        }
        .padding(.horizontal, 18)
        .padding(.top, 4)
        .padding(.bottom, 10)
    }

    private func pill(icon: String, text: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10.5))
                .foregroundStyle(theme.textTertiary)
            Text(text)
                .font(.omlxText(11))
                .foregroundStyle(theme.textSecondary)
        }
    }
}

// MARK: - Advanced

private struct AdvancedSection: View {
    @Binding var isOpen: Bool
    let selectedIsVLM: Bool
    let selectedHasMTP: Bool
    @Binding var textOnly: Bool
    @Binding var preserveMtp: Bool
    @Binding var dtype: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isOpen.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isOpen ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(theme.textSecondary)
                    Text("Advanced settings")
                        .font(.omlxText(11, weight: .semibold))
                        .foregroundStyle(theme.textSecondary)
                        .textCase(.uppercase)
                        .kerning(0.6)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 14)
                .padding(.top, 10)
                .padding(.bottom, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isOpen {
                ListGroup {
                    if selectedIsVLM {
                        Row(
                            label: "Text only",
                            sublabel: "Exclude vision encoder weights (~2-3% smaller, text-only output)"
                        ) {
                            Toggle("", isOn: $textOnly).labelsHidden().toggleStyle(.switch)
                        }
                    }

                    Row(
                        label: "Preserve MTP",
                        sublabel: selectedHasMTP
                            ? "Keep multi-token prediction heads in the quantized output"
                            : "Unavailable — source model has no MTP heads"
                    ) {
                        Toggle("", isOn: $preserveMtp)
                            .labelsHidden()
                            .toggleStyle(.switch)
                            .disabled(!selectedHasMTP)
                    }

                    Row(
                        label: "Non-quant dtype",
                        sublabel: "Precision for tensors that stay un-quantized (norms, scales)",
                        isLast: true
                    ) {
                        Segmented(selection: $dtype, options: [
                            ("bfloat16", "bfloat16"),
                            ("float16",  "float16"),
                        ])
                        .frame(width: 180)
                    }
                }
            }
        }
    }
}

// MARK: - Status banners

private struct BannerSection: View {
    let error: String?
    let success: String?

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let error, !error.isEmpty {
                banner(icon: "exclamationmark.triangle.fill", text: error, color: theme.redDot)
            }
            if let success, !success.isEmpty {
                banner(icon: "checkmark.circle.fill", text: success, color: theme.greenDot)
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, error == nil && success == nil ? 0 : 6)
    }

    private func banner(icon: String, text: String, color: Color) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .font(.system(size: 11))
                .padding(.top, 1)
            Text(text)
                .font(.omlxText(11.5))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

private struct EmptyModelsBanner: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
                .foregroundStyle(theme.textTertiary)
            Text("No full-precision models found on disk. Download one from the Downloads tab first.")
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .padding(.horizontal, 18)
        .padding(.top, 6)
    }
}

// MARK: - Queue

private struct QueueSection: View {
    let tasks: [OQTaskDTO]
    let onCancel: (String) -> Void
    let onRemove: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        if tasks.isEmpty {
            EmptyView()
        } else {
            SectionHeader("Queue", subtitle: "\(tasks.count) task\(tasks.count == 1 ? "" : "s")")

            ListGroup {
                ForEach(Array(tasks.enumerated()), id: \.element.id) { idx, task in
                    FreeRow(isLast: idx == tasks.count - 1) {
                        QueueRow(
                            task: task,
                            onCancel: { onCancel(task.taskId) },
                            onRemove: { onRemove(task.taskId) }
                        )
                    }
                }
            }
        }
    }
}

private struct QueueRow: View {
    let task: OQTaskDTO
    let onCancel: () -> Void
    let onRemove: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .font(.system(size: 12))
                    .foregroundStyle(theme.blueDot)
                Text(task.outputName)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
                StatusChip(status: task.statusEnum)
                Spacer(minLength: 4)
                Text(elapsedText)
                    .font(.omlxMono(11))
                    .foregroundStyle(theme.textTertiary)
                Button {
                    if task.isActive { onCancel() } else { onRemove() }
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 11))
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .help(task.isActive ? "Cancel" : "Remove")
            }
            if task.isActive {
                ProgressBar(progress: max(0, min(task.progress / 100, 1)))
                HStack(spacing: 8) {
                    if !task.phase.isEmpty {
                        Text(task.phase)
                            .font(.omlxText(11))
                            .foregroundStyle(theme.textSecondary)
                    }
                    Spacer(minLength: 0)
                    Text(progressText)
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textTertiary)
                }
            }
            if !task.error.isEmpty {
                Text(task.error)
                    .font(.omlxMono(10.5))
                    .foregroundStyle(theme.redDot)
                    .lineLimit(3)
            }
        }
    }

    private var progressText: String {
        // While running, show "67%". When complete, server emits 100 anyway.
        "\(Int(task.progress.rounded()))%"
    }

    private var elapsedText: String {
        let now = Date().timeIntervalSince1970
        let start = task.startedAt > 0 ? task.startedAt : task.createdAt
        let end = task.completedAt > 0 ? task.completedAt : now
        let secs = max(0, end - start)
        if secs < 60 { return "\(Int(secs))s" }
        let m = Int(secs / 60)
        let s = Int(secs.truncatingRemainder(dividingBy: 60))
        return "\(m)m \(s)s"
    }
}

private struct StatusChip: View {
    let status: OQTaskDTO.Status?
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let cfg: (Color, String) = {
            switch status {
            case .pending:    return (theme.textTertiary, "Pending")
            case .loading:    return (theme.blueDot, "Loading")
            case .quantizing: return (theme.blueDot, "Quantizing")
            case .saving:     return (theme.blueDot, "Saving")
            case .completed:  return (theme.greenDot, "Completed")
            case .failed:     return (theme.redDot, "Failed")
            case .cancelled:  return (theme.textTertiary, "Cancelled")
            case .none:       return (theme.textTertiary, "—")
            }
        }()
        Text(cfg.1)
            .font(.omlxText(10, weight: .semibold))
            .foregroundStyle(cfg.0)
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(cfg.0.opacity(0.12))
            .clipShape(Capsule())
    }
}

private struct ProgressBar: View {
    let progress: Double
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(theme.codeBg)
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(LinearGradient(
                        colors: [Color(rgb24: 0xFF2D55), Color(rgb24: 0xAF52DE)],
                        startPoint: .leading, endPoint: .trailing
                    ))
                    .frame(width: geo.size.width * progress)
                    .animation(.easeOut(duration: 0.4), value: progress)
            }
        }
        .frame(height: 4)
    }
}

// MARK: - About

private struct AboutSection: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader("About oQ Quantization")

        ListGroup {
            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("oMLX Universal Dynamic Quantization")
                        .font(.omlxText(13, weight: .semibold))
                        .foregroundStyle(theme.text)
                    Text("Quantization should not be exclusive to any particular inference server. oQ produces standard mlx-lm models that work everywhere — oMLX, mlx-lm, LM Studio, and any app that supports MLX safetensors format. No custom loader required.")
                        .font(.omlxText(11.5))
                        .foregroundStyle(theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("oQ measures each layer's quantization sensitivity through calibration (relative MSE vs float16) and builds a byte-budgeted mixed-precision plan that allocates bits where the data says they matter most. Every model gets a unique bit allocation tuned to its architecture.")
                        .font(.omlxText(11.5))
                        .foregroundStyle(theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(.bottom, 18)
    }
}

// MARK: - View model

@MainActor
final class QuantizationScreenVM: ObservableObject {
    // Form state
    @Published var selectedModelPath: String = ""
    @Published var sensitivityModelPath: String = ""
    @Published var oqLevel: Double = 4
    @Published var textOnly: Bool = false
    @Published var preserveMtp: Bool = false
    @Published var dtype: String = "bfloat16"
    @Published var advancedOpen: Bool = false

    // Server state
    @Published private(set) var models: [OQModelInfo] = []
    @Published private(set) var allModels: [OQModelInfo] = []
    @Published private(set) var modelsLoaded: Bool = false
    @Published private(set) var tasks: [OQTaskDTO] = []
    @Published private(set) var estimate: OQEstimateResponse?

    // UI state
    @Published private(set) var isStarting: Bool = false
    @Published var lastError: String?
    @Published var lastSuccess: String?

    private weak var client: OMLXClient?
    private var pollTask: Task<Void, Never>?
    private var estimateDebounceTask: Task<Void, Never>?
    private var successClearTask: Task<Void, Never>?

    // Settings (no Codable persistence — form lives only while screen is open).
    private static let groupSize = 64

    // MARK: Derived

    /// True iff the source model offers sensible sensitivity candidates
    /// (same model family at lower precision, etc.). The HTML hides the
    /// dropdown entirely when this is empty.
    var sensitivityCandidates: [OQModelInfo] {
        guard let source = models.first(where: { $0.path == selectedModelPath })
        else { return [] }
        let prefix = source.name.split(separator: "-").prefix(2).joined(separator: "-")
        return allModels.filter { m in
            m.path != selectedModelPath
            && m.isQuantized
            && m.name.hasPrefix(prefix)
        }
    }

    var selectedIsVLM: Bool {
        models.first(where: { $0.path == selectedModelPath })?.isVlm ?? false
    }

    var selectedHasMTP: Bool {
        models.first(where: { $0.path == selectedModelPath })?.hasMtpHeads ?? false
    }

    /// Estimate strip — memory pill. Mirrors `oqEstimatedMemory` in JS:
    /// if a sensitivity model is picked memory ≈ sens.size × 1.5 + 5 GB,
    /// else the `memory_streaming_formatted` from the API, else the source
    /// model's static `memory_streaming.peak_formatted`.
    var memoryText: String {
        if let est = estimate {
            if !sensitivityModelPath.isEmpty,
               let sens = allModels.first(where: { $0.path == sensitivityModelPath }) {
                let bytes = Int64(Double(sens.size) * 1.5) + 5 * 1024 * 1024 * 1024
                return formatBytes(bytes)
            }
            if let m = est.memoryStreamingFormatted, !m.isEmpty { return m }
        }
        return models.first(where: { $0.path == selectedModelPath })?
            .memoryStreaming?.peakFormatted ?? ""
    }

    var bpwText: String {
        guard let est = estimate else { return "" }
        return String(format: "%.1f", est.effectiveBpw)
    }

    var outputSizeText: String {
        estimate?.outputSizeFormatted ?? ""
    }

    // MARK: Lifecycle

    func start(client: OMLXClient) async {
        self.client = client
        await loadModels()
        await loadTasks()
        startPollingIfNeeded()
    }

    func stop() {
        pollTask?.cancel(); pollTask = nil
        estimateDebounceTask?.cancel(); estimateDebounceTask = nil
        successClearTask?.cancel(); successClearTask = nil
    }

    // MARK: Loaders

    private func loadModels() async {
        guard let client else { return }
        do {
            let resp = try await client.listOQModels()
            self.models = resp.models
            self.allModels = resp.allModels
            self.modelsLoaded = true
        } catch {
            self.modelsLoaded = true
            self.lastError = "Failed to load models: \(error)"
        }
    }

    private func loadTasks() async {
        guard let client else { return }
        do {
            let resp = try await client.listOQTasks()
            // If a task just transitioned from active → completed, refresh
            // the model list so the new quantized model shows up as a
            // sensitivity candidate without a manual reload.
            let hadActive = self.tasks.contains(where: { $0.isActive })
            let hasActiveNow = resp.tasks.contains(where: { $0.isActive })
            self.tasks = resp.tasks
            if hadActive && !hasActiveNow {
                await loadModels()
            }
        } catch {
            // Polling failure is expected during server restarts — don't
            // clobber the user-facing banner with transient errors.
        }
    }

    // MARK: Polling

    private func startPollingIfNeeded() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let hasActive = await MainActor.run {
                    self.tasks.contains(where: { $0.isActive })
                }
                if hasActive {
                    try? await Task.sleep(for: .seconds(2))
                    if Task.isCancelled { return }
                    await self.loadTasks()
                } else {
                    // Idle poll cadence — 6 s while no work is queued.
                    try? await Task.sleep(for: .seconds(6))
                    if Task.isCancelled { return }
                    await self.loadTasks()
                }
            }
        }
    }

    // MARK: Estimate (debounced)

    /// Schedules a 300 ms debounced fetch — matches the JS dashboard. Each
    /// call cancels the previous timer so rapid changes (typing in a select,
    /// keyboard arrows) collapse to a single network round-trip.
    func scheduleEstimateRefresh(client: OMLXClient) {
        estimateDebounceTask?.cancel()
        if selectedModelPath.isEmpty {
            estimate = nil
            return
        }
        let path = selectedModelPath
        let level = oqLevel
        let preserve = selectedHasMTP && preserveMtp
        estimateDebounceTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            if Task.isCancelled { return }
            do {
                let est = try await client.estimateOQ(
                    modelPath: path,
                    oqLevel: level,
                    preserveMtp: preserve
                )
                await MainActor.run {
                    guard let self else { return }
                    // Drop the result if the user has moved on to a different
                    // model since this request was kicked off.
                    if self.selectedModelPath == path { self.estimate = est }
                }
            } catch {
                // Silent — the strip will read "Calculating…" which is fine
                // for a transient estimate failure.
            }
        }
    }

    // MARK: Actions

    func startQuantization(client: OMLXClient) {
        guard !selectedModelPath.isEmpty, !isStarting else { return }
        isStarting = true
        lastError = nil
        lastSuccess = nil
        let body = OQStartRequest(
            modelPath: selectedModelPath,
            oqLevel: oqLevel,
            groupSize: Self.groupSize,
            sensitivityModelPath: sensitivityModelPath,
            textOnly: textOnly,
            dtype: dtype,
            preserveMtp: selectedHasMTP && preserveMtp
        )
        let displayName = models.first(where: { $0.path == selectedModelPath })?.name
            ?? selectedModelPath
        let levelLabel = (oqLevel.rounded() == oqLevel)
            ? "oQ\(Int(oqLevel))" : "oQ\(oqLevel)"
        Task { [weak self] in
            defer { Task { @MainActor [weak self] in self?.isStarting = false } }
            do {
                let resp = try await client.startOQQuantization(body)
                await MainActor.run {
                    guard let self else { return }
                    if resp.success {
                        self.lastSuccess = "Quantization started: \(displayName) → \(levelLabel)"
                        self.scheduleSuccessClear()
                    } else {
                        self.lastError = "Server refused the request"
                    }
                }
                await self?.loadTasks()
            } catch {
                await MainActor.run {
                    self?.lastError = "Failed to start: \(error)"
                }
            }
        }
    }

    func cancelTask(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.cancelOQTask(taskId: taskId)
                await self?.loadTasks()
            } catch {
                await MainActor.run { self?.lastError = "Cancel failed: \(error)" }
            }
        }
    }

    func removeTask(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.removeOQTask(taskId: taskId)
                await self?.loadTasks()
            } catch {
                await MainActor.run { self?.lastError = "Remove failed: \(error)" }
            }
        }
    }

    private func scheduleSuccessClear() {
        successClearTask?.cancel()
        successClearTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(5))
            if Task.isCancelled { return }
            await MainActor.run { self?.lastSuccess = nil }
        }
    }
}
