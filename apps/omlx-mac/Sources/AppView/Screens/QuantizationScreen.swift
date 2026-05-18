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
//                            Completed quant tasks expose an "Upload to HF"
//                            button that opens the upload sheet.
//
//   Upload sheet          — credentials + repo + README configuration, then
//                            POST /admin/api/upload/start. The HF token
//                            lives only in the macOS Keychain (service
//                            "app.omlx-next.hf-upload"); never persisted to
//                            UserDefaults or files.
//
//   Upload Tasks          — mirror of the Queue section for upload jobs.
//                            Polled in the same iteration as quant tasks
//                            (2 s while either side is active, 6 s idle).
//
//   About                 — static documentation card (matches the marketing
//                            copy in the HTML so users get the same context
//                            in either UI).

import SwiftUI
import Security

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
                onRemove: { id in vm.removeTask(taskId: id, client: services.client) },
                onUpload: { task in vm.uploadTarget = task }
            )

            UploadTasksSection(
                tasks: vm.uploadTasks,
                onCancel: { id in vm.cancelUpload(taskId: id, client: services.client) },
                onRemove: { id in vm.removeUpload(taskId: id, client: services.client) }
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
        .sheet(item: $vm.uploadTarget) { task in
            UploadModalView(task: task, vm: vm, client: services.client)
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
    let onUpload: (OQTaskDTO) -> Void

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
                            onRemove: { onRemove(task.taskId) },
                            onUpload: { onUpload(task) }
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
    let onUpload: () -> Void

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
                if task.statusEnum == .completed {
                    Button {
                        onUpload()
                    } label: {
                        Label("Upload to HF", systemImage: "arrow.up.circle")
                            .labelStyle(.titleAndIcon)
                    }
                    .buttonStyle(.omlx(.normal, size: .small))
                    .help("Upload to Hugging Face Hub")
                }
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

// MARK: - Upload tasks

// Renders the HF upload queue. Mirrors `QueueSection` for visual parity but
// reads from `vm.uploadTasks` and exposes an "Open" link for completed jobs
// so the user can jump to the freshly published repo on huggingface.co.
private struct UploadTasksSection: View {
    let tasks: [HFUploadTaskDTO]
    let onCancel: (String) -> Void
    let onRemove: (String) -> Void

    @Environment(\.omlxTheme) private var theme

    private var activeCount: Int { tasks.filter { $0.isActive }.count }
    private var completedCount: Int { tasks.filter { $0.statusEnum == .completed }.count }

    var body: some View {
        if tasks.isEmpty {
            EmptyView()
        } else {
            SectionHeader(
                "Uploads",
                subtitle: "\(activeCount) active / \(completedCount) completed"
            )

            ListGroup {
                ForEach(Array(tasks.enumerated()), id: \.element.id) { idx, task in
                    FreeRow(isLast: idx == tasks.count - 1) {
                        UploadRow(
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

private struct UploadRow: View {
    let task: HFUploadTaskDTO
    let onCancel: () -> Void
    let onRemove: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: "arrow.up.circle")
                    .font(.system(size: 12))
                    .foregroundStyle(theme.blueDot)
                Text(task.modelName)
                    .font(.omlxMono(12))
                    .foregroundStyle(theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
                UploadStatusChip(status: task.statusEnum)
                Spacer(minLength: 4)
                if task.statusEnum == .completed, !task.repoUrl.isEmpty,
                   let url = URL(string: task.repoUrl) {
                    Button {
                        NSWorkspace.shared.open(url)
                    } label: {
                        Label("Open", systemImage: "arrow.up.right.square")
                            .labelStyle(.titleAndIcon)
                    }
                    .buttonStyle(.omlx(.plain, size: .small))
                    .help(task.repoUrl)
                }
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
                    Text(task.repoId)
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 0)
                    Text("\(Int(task.progress.rounded()))%")
                        .font(.omlxMono(11))
                        .foregroundStyle(theme.textTertiary)
                }
            } else if !task.repoId.isEmpty {
                Text(task.repoId)
                    .font(.omlxMono(11))
                    .foregroundStyle(theme.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if !task.error.isEmpty {
                Text(task.error)
                    .font(.omlxMono(10.5))
                    .foregroundStyle(theme.redDot)
                    .lineLimit(3)
            }
        }
    }
}

private struct UploadStatusChip: View {
    let status: HFUploadTaskDTO.Status?
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let cfg: (Color, String) = {
            switch status {
            case .pending:   return (theme.textTertiary, "Pending")
            case .uploading: return (theme.blueDot, "Uploading")
            case .completed: return (theme.greenDot, "Completed")
            case .failed:    return (theme.redDot, "Failed")
            case .cancelled: return (theme.textTertiary, "Cancelled")
            case .none:      return (theme.textTertiary, "—")
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

// MARK: - Upload modal

// Sheet presented when the user taps "Upload to HF" on a completed quant
// task. Three sections (Credentials / Repository / README) feed
// `POST /admin/api/upload/start` once the token has been validated and a
// repo name entered. Body submission closes the sheet via uploadTarget=nil.
private struct UploadModalView: View {
    let task: OQTaskDTO
    @ObservedObject var vm: QuantizationScreenVM
    let client: OMLXClient

    @Environment(\.omlxTheme) private var theme

    /// Editable repo name. Pre-filled with the quant task's output name —
    /// users are free to rename before publish.
    @State private var repoName: String = ""
    @State private var isPrivate: Bool = false
    /// "" sentinel triggers the auto-generated README path. Any other value
    /// is the absolute path of another local model whose README will be
    /// copied verbatim.
    @State private var readmeSourcePath: String = ""
    @State private var addRedownloadNotice: Bool = true
    @State private var isStarting: Bool = false
    @State private var localError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Sheet header — explicit since `SectionHeader` styles map to a
            // scrollable screen, not a modal.
            VStack(alignment: .leading, spacing: 4) {
                Text("Upload to Hugging Face")
                    .font(.omlxText(15, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(task.outputName)
                    .font(.omlxMono(11.5))
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .padding(.horizontal, 14)
            .padding(.top, 16)
            .padding(.bottom, 8)

            credentialsSection
            repositorySection
            readmeSection

            if let err = localError, !err.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(theme.redDot)
                        .font(.system(size: 11))
                        .padding(.top, 1)
                    Text(err)
                        .font(.omlxText(11.5))
                        .foregroundStyle(theme.text)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                }
                .padding(10)
                .background(theme.redDot.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .padding(.horizontal, 14)
                .padding(.top, 6)
            }

            footer
        }
        .frame(width: 560)
        .frame(minHeight: 480)
        .background(theme.windowBg)
        .onAppear {
            repoName = task.outputName
            readmeSourcePath = ""
            addRedownloadNotice = true
            localError = nil
        }
    }

    // MARK: Credentials

    private var credentialsSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(
                "Credentials",
                subtitle: vm.uploadValidatedUsername.map { "Logged in as @\($0)" }
                    ?? "Validate a token to enable upload"
            )

            ListGroup {
                Row(
                    label: "HF token",
                    sublabel: "Stored in macOS Keychain. Needs write access to your account."
                ) {
                    HStack(spacing: 6) {
                        TextInput(
                            text: $vm.uploadToken,
                            placeholder: "hf_…",
                            isSecure: true,
                            mono: true,
                            width: 220
                        )
                        Button {
                            Task { await vm.validateUploadToken(client: client) }
                        } label: {
                            if vm.isValidatingToken {
                                ProgressView().controlSize(.small)
                                Text("Validating…")
                            } else {
                                Text(vm.uploadValidatedUsername == nil ? "Validate" : "Re-validate")
                            }
                        }
                        .buttonStyle(.omlx(.normal, size: .small))
                        .disabled(vm.isValidatingToken || vm.uploadToken.isEmpty)
                    }
                }

                if vm.uploadValidatedUsername != nil && !vm.uploadOrgs.isEmpty {
                    Row(
                        label: "Target namespace",
                        sublabel: "Publish under your account or one of your orgs",
                        isLast: true
                    ) {
                        Popup(
                            selection: $vm.uploadNamespace,
                            width: 220,
                            options: namespaceOptions
                        )
                    }
                } else {
                    // Make the last visible row in the group flush with the
                    // bottom rounded edge by toggling `isLast` on it.
                    Row(
                        label: "Target namespace",
                        sublabel: vm.uploadValidatedUsername == nil
                            ? "Available after validation"
                            : "Your account is the only available namespace",
                        isLast: true
                    ) {
                        Text(vm.uploadNamespace.isEmpty ? "—" : "@\(vm.uploadNamespace)")
                            .font(.omlxText(13, weight: .medium))
                            .foregroundStyle(theme.textSecondary)
                    }
                }
            }
        }
    }

    // MARK: Repository

    private var repositorySection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader("Repository")
            ListGroup {
                Row(
                    label: "Repo name",
                    sublabel: "Full repo id will be \(vm.uploadNamespace.isEmpty ? "<namespace>" : vm.uploadNamespace)/<repo-name>"
                ) {
                    HStack(spacing: 6) {
                        Text(vm.uploadNamespace.isEmpty ? "<namespace>/" : "\(vm.uploadNamespace)/")
                            .font(.omlxMono(11.5))
                            .foregroundStyle(theme.textSecondary)
                        TextInput(
                            text: $repoName,
                            placeholder: "model-id",
                            mono: true,
                            width: 240
                        )
                    }
                }

                Row(
                    label: "Private repo",
                    sublabel: "Only you and your org will see this model",
                    isLast: true
                ) {
                    Toggle("", isOn: $isPrivate).labelsHidden().toggleStyle(.switch)
                }
            }
        }
    }

    // MARK: README

    private var readmeSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader("README")
            ListGroup {
                Row(
                    label: "Source",
                    sublabel: "Generate a default card or copy from another local model"
                ) {
                    Popup(
                        selection: $readmeSourcePath,
                        width: 260,
                        options: readmeOptions
                    )
                }
                if readmeSourcePath.isEmpty {
                    Row(
                        label: "Add re-download notice",
                        sublabel: "Append a banner reminding downstream users to re-pull",
                        isLast: true
                    ) {
                        Toggle("", isOn: $addRedownloadNotice).labelsHidden().toggleStyle(.switch)
                    }
                } else {
                    // Trailing row stays flush even when the toggle is hidden.
                    Row(
                        label: "Re-download notice",
                        sublabel: "Disabled when copying an existing README",
                        isLast: true
                    ) {
                        Text("Off")
                            .font(.omlxText(13, weight: .medium))
                            .foregroundStyle(theme.textTertiary)
                    }
                }
            }
        }
    }

    // MARK: Footer

    private var footer: some View {
        HStack(spacing: 8) {
            Spacer()
            Button("Cancel") { vm.uploadTarget = nil }
                .buttonStyle(.omlx(.normal, size: .regular))
                .keyboardShortcut(.cancelAction)
            Button {
                submit()
            } label: {
                if isStarting {
                    ProgressView().controlSize(.small)
                    Text("Uploading…")
                } else {
                    Label("Upload", systemImage: "arrow.up.circle.fill")
                        .labelStyle(.titleAndIcon)
                }
            }
            .buttonStyle(.omlx(.primary))
            .disabled(!canSubmit || isStarting)
            .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    // MARK: Helpers

    private var canSubmit: Bool {
        vm.uploadValidatedUsername != nil
        && !vm.uploadNamespace.isEmpty
        && !repoName.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private var namespaceOptions: [PopupOption<String>] {
        var opts: [PopupOption<String>] = []
        if let user = vm.uploadValidatedUsername {
            opts.append(PopupOption(value: user, label: "@\(user)"))
        }
        for org in vm.uploadOrgs {
            opts.append(PopupOption(value: org.name, label: org.name))
        }
        return opts
    }

    private var readmeOptions: [PopupOption<String>] {
        var opts = [PopupOption(value: "", label: "Auto-generate")]
        opts += vm.uploadCandidateModels.map { m in
            PopupOption(value: m.path, label: "Copy from \(m.name)")
        }
        return opts
    }

    private func submit() {
        let trimmed = repoName.trimmingCharacters(in: .whitespaces)
        guard canSubmit else { return }
        isStarting = true
        localError = nil
        let body = HFUploadStartRequest(
            modelPath: task.outputPath,
            repoId: "\(vm.uploadNamespace)/\(trimmed)",
            hfToken: vm.uploadToken,
            readmeSourcePath: readmeSourcePath,
            autoReadme: readmeSourcePath.isEmpty,
            redownloadNotice: readmeSourcePath.isEmpty && addRedownloadNotice,
            private: isPrivate
        )
        Task { @MainActor in
            await vm.startUpload(body: body, client: client)
            isStarting = false
            if let err = vm.lastUploadError, !err.isEmpty {
                localError = err
            } else {
                vm.uploadTarget = nil
            }
        }
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

    // Upload state — covers the sheet + the Upload Tasks section. The token
    // is hydrated from Keychain on `start()` and re-written after a
    // successful `validateHFUploadToken` round-trip. We hold it in plain
    // memory while the screen is mounted so the sheet's SecureField stays
    // bound; it never gets persisted anywhere except the Keychain.
    @Published var uploadTasks: [HFUploadTaskDTO] = []
    @Published var uploadTarget: OQTaskDTO?
    @Published var uploadCandidateModels: [HFUploadModelInfo] = []
    @Published var uploadToken: String = ""
    @Published var uploadValidatedUsername: String?
    @Published var uploadOrgs: [HFOrgInfo] = []
    @Published var uploadNamespace: String = ""
    @Published var isValidatingToken: Bool = false
    @Published var lastUploadError: String?

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
        // Hydrate the HF token from Keychain. Silent on miss — the sheet
        // shows an empty SecureField and the user can paste a new token.
        if let stored = Keychain.read(), !stored.isEmpty {
            self.uploadToken = stored
        }
        await loadModels()
        await loadUploadCandidates()
        await loadTasks()
        await loadUploadTasks()
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
            // the model list (so the new quantized model shows up as a
            // sensitivity candidate) and the upload candidate list (so the
            // README picker can copy from it). No manual reload required.
            let hadActive = self.tasks.contains(where: { $0.isActive })
            let hasActiveNow = resp.tasks.contains(where: { $0.isActive })
            self.tasks = resp.tasks
            if hadActive && !hasActiveNow {
                await loadModels()
                await loadUploadCandidates()
            }
        } catch {
            // Polling failure is expected during server restarts — don't
            // clobber the user-facing banner with transient errors.
        }
    }

    /// Loads local oQ models that can serve as a README source when the user
    /// picks "Copy from <model>" in the upload sheet. Filtered to oQ output
    /// (matching the HTML panel's `oq_models` slot) so the dropdown stays
    /// short.
    func loadUploadCandidates() async {
        guard let client else { return }
        do {
            let resp = try await client.listHFUploadModels()
            self.uploadCandidateModels = resp.oqModels
        } catch {
            // Soft-fail — the auto-generate path still works without
            // candidates, so we don't block the sheet on this.
        }
    }

    private func loadUploadTasks() async {
        guard let client else { return }
        do {
            let resp = try await client.listHFUploadTasks()
            self.uploadTasks = resp.tasks
        } catch {
            // Polling failure: stay quiet (same rationale as loadTasks).
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
                    || self.uploadTasks.contains(where: { $0.isActive })
                }
                if hasActive {
                    try? await Task.sleep(for: .seconds(2))
                    if Task.isCancelled { return }
                    await self.loadTasks()
                    await self.loadUploadTasks()
                } else {
                    // Idle poll cadence — 6 s while no work is queued.
                    try? await Task.sleep(for: .seconds(6))
                    if Task.isCancelled { return }
                    await self.loadTasks()
                    await self.loadUploadTasks()
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

    // MARK: Upload actions

    /// Validates the current `uploadToken` against `/api/upload/validate-token`.
    /// On success the token is persisted to the Keychain so the next session
    /// skips this round-trip, and the namespace defaults to the returned
    /// username (with orgs available via the Popup in the sheet).
    func validateUploadToken(client: OMLXClient) async {
        let token = uploadToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            lastUploadError = "Token is empty"
            return
        }
        isValidatingToken = true
        lastUploadError = nil
        defer { isValidatingToken = false }
        do {
            let resp = try await client.validateHFUploadToken(hfToken: token)
            self.uploadValidatedUsername = resp.username
            self.uploadOrgs = resp.orgs
            self.uploadNamespace = resp.username
            Keychain.write(token)
        } catch {
            self.uploadValidatedUsername = nil
            self.uploadOrgs = []
            self.uploadNamespace = ""
            self.lastUploadError = "Validate failed: \(describe(error))"
        }
    }

    /// Submits a configured upload job. The caller (the sheet) clears
    /// `uploadTarget` on success; on failure we surface the message via
    /// `lastUploadError` and leave the sheet open so the user can correct
    /// the body and retry without losing their inputs.
    func startUpload(body: HFUploadStartRequest, client: OMLXClient) async {
        lastUploadError = nil
        do {
            let resp = try await client.startHFUpload(body)
            if resp.success == false {
                lastUploadError = "Server refused the request"
            }
            await loadUploadTasks()
            // Make sure the polling loop picks up the new active task even
            // if nothing else was running before this submission.
            startPollingIfNeeded()
        } catch {
            lastUploadError = "Upload failed: \(describe(error))"
        }
    }

    func cancelUpload(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.cancelHFUploadTask(taskId: taskId)
                await self?.loadUploadTasks()
            } catch {
                await MainActor.run {
                    self?.lastUploadError = "Cancel failed: \(error)"
                }
            }
        }
    }

    func removeUpload(taskId: String, client: OMLXClient) {
        Task { [weak self] in
            do {
                _ = try await client.removeHFUploadTask(taskId: taskId)
                await self?.loadUploadTasks()
            } catch {
                await MainActor.run {
                    self?.lastUploadError = "Remove failed: \(error)"
                }
            }
        }
    }

    private func describe(_ error: Error) -> String {
        if let omlx = error as? OMLXClientError { return String(describing: omlx) }
        return error.localizedDescription
    }
}

// MARK: - Keychain helper

// Thin SecItem wrapper. Single account/service pair — the upload screen is
// the only consumer right now, so we keep the surface small. All accesses
// happen on the main actor (called from the VM); the SecItem APIs are
// thread-safe so we don't need additional locking.
enum Keychain {
    private static let service = "app.omlx-next.hf-upload"
    private static let account = "huggingface-token"

    /// Returns the stored token or `nil` if no item exists / the read fails.
    static func read() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecReturnData as String: true,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let str = String(data: data, encoding: .utf8) else {
            return nil
        }
        return str
    }

    /// Writes (or updates) the stored token. No-op on empty input so we
    /// don't accidentally clobber an existing entry with an empty string.
    @discardableResult
    static func write(_ value: String) -> Bool {
        guard let data = value.data(using: .utf8), !value.isEmpty else { return false }
        let baseQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let updateStatus = SecItemUpdate(
            baseQuery as CFDictionary,
            [kSecValueData as String: data] as CFDictionary
        )
        if updateStatus == errSecSuccess { return true }
        if updateStatus == errSecItemNotFound {
            var addQuery = baseQuery
            addQuery[kSecValueData as String] = data
            return SecItemAdd(addQuery as CFDictionary, nil) == errSecSuccess
        }
        return false
    }

    @discardableResult
    static func delete() -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
}
