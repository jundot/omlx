// Context Bench screen.
//
// Mirrors the "Context" tab from the HTML admin panel
// (omlx/admin/templates/dashboard/_bench_context.html + dashboard.js
// context bench section). Wires the /api/bench/context/* endpoints onto
// a stack of sections:
//
//   Header          — title + one-line description.
//
//   Configuration   — model picker (Popup over /api/models, loaded
//                     first), target context selector (16k…512k,
//                     default 128k), amber warning callout (long
//                     runtime, unload-all, auto-apply), Run / Cancel.
//
//   Progress        — phase message + percent bar while polling
//                     getContextBenchStatus at 1.5 Hz. The server
//                     mirrors its SSE progress onto the poll endpoint.
//
//   Error banner    — red banner on start failure or a terminal error.
//
//   Result          — applied context window headline + measurement
//                     details, plus the auto-applied / snapshot notes.

import SwiftUI

struct ContextBenchScreen: View {
    @Environment(AppServices.self) private var services
    // VM is owned by AppServices so a running bench survives screen
    // unloads — same pattern as ThroughputBenchScreenVM.
    @Bindable var vm: ContextBenchScreenVM

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScreenHeader(
                eyebrow: String(localized: "bench.context.header.eyebrow",
                                defaultValue: "Context Benchmark",
                                comment: "Eyebrow label above the Context Bench screen header"),
                title: String(localized: "bench.context.header.title",
                              defaultValue: "Measure usable context",
                              comment: "Context Bench screen primary header"),
                subtitle: String(localized: "bench.context.header.subtitle",
                                 defaultValue: "Finds the largest context window this machine can actually prefill for a model, then applies it to the model's Context Window setting.",
                                 comment: "Context Bench screen subtitle describing what the benchmark measures")
            )

            ConfigurationSection(
                models: vm.models,
                selectedModelId: $vm.selectedModelId,
                targetTokens: $vm.targetTokens,
                targetOptions: vm.availableTargetOptions,
                prefillPriority: vm.prefillPriority,
                running: vm.running,
                canRun: vm.canRun,
                onRun: { vm.runBenchmark(client: services.client) },
                onCancel: { vm.cancelBenchmark(client: services.client) },
                onPriorityChange: { vm.setPrefillPriority($0, client: services.client) }
            )

            if vm.running {
                ProgressCard(
                    message: vm.message,
                    progress: vm.progress
                )
            }

            MessageBanner(error: vm.lastError)

            if let result = vm.result {
                ResultSection(result: result)
            }
        }
        // `start()` is idempotent: it refreshes the model list but leaves
        // the running-bench state alone, so navigation doesn't lose an
        // in-flight measurement.
        .task { await vm.start(client: services.client) }
        // Picking a model with a smaller native context hides the larger
        // target presets — snap the selection back into range.
        .onChange(of: vm.selectedModelId) { _, _ in
            vm.clampTargetToModel()
        }
    }
}

// MARK: - Configuration

private struct ConfigurationSection: View {
    let models: [ModelDTO]
    @Binding var selectedModelId: String
    @Binding var targetTokens: Int
    let targetOptions: [Int]
    let prefillPriority: String
    let running: Bool
    let canRun: Bool
    let onRun: () -> Void
    let onCancel: () -> Void
    let onPriorityChange: (String) -> Void

    var body: some View {
        SectionHeader(
            String(localized: "bench.context.section.configuration",
                   defaultValue: "Configuration",
                   comment: "Section header for the Context Bench configuration block"),
            subtitle: models.isEmpty
                ? String(localized: "bench.context.subtitle.loading_models",
                         defaultValue: "Loading models…",
                         comment: "Context Bench section subtitle while models are loading")
                : String(localized: "bench.context.subtitle.model_count",
                         defaultValue: "Models available: \(models.count)",
                         comment: "Context Bench section subtitle showing how many models are available; placeholder is the count")
        )

        ListGroup {
            Row(label: String(localized: "bench.context.row.model.label",
                              defaultValue: "Model",
                              comment: "Row label for the Context Bench model picker"),
                sublabel: String(localized: "bench.context.row.model.sub",
                                 defaultValue: "Loaded or unloaded — server will load on demand",
                                 comment: "Sublabel under the Context Bench model picker")) {
                Popup(
                    selection: $selectedModelId,
                    width: .controlWide,
                    options: modelOptions
                )
            }

            Row(label: String(localized: "bench.context.row.target.label",
                              defaultValue: "Maximum context to test",
                              comment: "Row label for the Context Bench target selector"),
                sublabel: String(localized: "bench.context.row.target.sub",
                                 defaultValue: "The search stops at this size; larger targets take longer to verify",
                                 comment: "Sublabel under the Context Bench target selector")) {
                Segmented(
                    selection: $targetTokens,
                    options: targetOptions.map {
                        (value: $0, label: "\($0 / 1024)k")
                    }
                )
                .disabled(running)
            }

            Row(label: String(localized: "bench.context.row.priority.label",
                              defaultValue: "Prefill Priority",
                              comment: "Row label for the prefill priority segmented control on the Context Bench screen"),
                sublabel: String(localized: "bench.context.row.priority.sub",
                                 defaultValue: "Mirrors the global setting and saves immediately — serving uses the same mode, so the measured value stays valid",
                                 comment: "Sublabel explaining the Context Bench prefill priority control mirrors the global setting")) {
                Segmented(
                    selection: Binding(
                        get: { prefillPriority },
                        set: { onPriorityChange($0) }
                    ),
                    options: [
                        (value: "context",
                         label: String(localized: "prefill_priority.option.max_context",
                                       defaultValue: "Max Context",
                                       comment: "Prefill priority option that favors the largest context")),
                        (value: "speed",
                         label: String(localized: "prefill_priority.option.speed",
                                       defaultValue: "Speed",
                                       comment: "Prefill priority option that favors prefill speed")),
                    ],
                    icons: ["arrow.up.left.and.arrow.down.right", "speedometer"]
                )
                .disabled(running)
            }

            FreeRow {
                WarningCallout()
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
                            Label(String(localized: "bench.context.button.run",
                                         defaultValue: "Start Benchmark",
                                         comment: "Context Bench primary button that starts the measurement"),
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
        var opts = [PopupOption(value: "", label: String(localized: "bench.context.model.placeholder",
                                                         defaultValue: "Select a model…",
                                                         comment: "Placeholder option in the Context Bench model picker"))]
        opts += models.map { m in
            PopupOption(
                value: m.id,
                label: m.loaded
                    ? String(localized: "bench.context.model.loaded_badge",
                             defaultValue: "\(m.id) • loaded",
                             comment: "Model picker entry for a loaded model; placeholder is the model id")
                    : m.id
            )
        }
        return opts
    }
}

// MARK: - Warning callout

private struct WarningCallout: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(theme.amberDot)
                .font(.system(size: 11))
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text(String(localized: "bench.context.warning.title",
                            defaultValue: "Before you start",
                            comment: "Heading of the warning callout above the Context Bench start button"))
                    .font(.omlxText(12, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(String(localized: "bench.context.warning.body",
                            defaultValue: "This benchmark can take a long time — verification prefills a real prompt at the measured size, which may take many minutes for large models. All loaded models are unloaded when it starts, interrupting active requests. When it finishes, the result is automatically applied to the model's Context Window setting.",
                            comment: "Body of the warning callout above the Context Bench start button"))
                    .font(.omlxText(11))
                    .foregroundStyle(theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(theme.amberDot.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

// MARK: - Progress

private struct ProgressCard: View {
    let message: String
    let progress: Double

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ListGroup {
            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.small)
                        Text(message.isEmpty
                             ? String(localized: "bench.context.progress.starting",
                                      defaultValue: "Starting…",
                                      comment: "Context Bench progress label before the first server update arrives")
                             : message)
                            .font(.omlxText(12))
                            .foregroundStyle(theme.textSecondary)
                            .lineLimit(2)
                        Spacer(minLength: 0)
                        Text("\(Int(progress.rounded()))%")
                            .font(.omlxText(12, weight: .medium))
                            .foregroundStyle(theme.text)
                            .monospacedDigit()
                    }
                    ProgressBar(progress: max(0, min(progress / 100, 1)))
                }
            }
        }
    }
}

// MARK: - Result

private struct ResultSection: View {
    let result: ContextBenchResultDTO

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "bench.context.section.result",
                   defaultValue: "Result",
                   comment: "Section header for the Context Bench result block"),
            subtitle: result.modelId
        )

        ListGroup {
            FreeRow {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(result.appliedTokens.formatted())
                        .font(.omlxText(26, weight: .bold))
                        .foregroundStyle(theme.text)
                        .monospacedDigit()
                    Text(String(localized: "bench.context.result.tokens_label",
                                defaultValue: "tokens applied to Context Window",
                                comment: "Unit caption next to the applied context window headline number"))
                        .font(.omlxText(12))
                        .foregroundStyle(theme.textSecondary)
                    Spacer(minLength: 0)
                }
            }

            Row(label: String(localized: "bench.context.result.measured",
                              defaultValue: "Admission boundary",
                              comment: "Result row label for the raw measured admission boundary")) {
                Text(result.measuredTokens.formatted())
                    .font(.omlxText(12, weight: .medium))
                    .monospacedDigit()
            }

            Row(label: String(localized: "bench.context.result.verified",
                              defaultValue: "Verified prefill",
                              comment: "Result row label for the prompt size the verification prefill completed")) {
                Text(result.verifiedTokens.formatted())
                    .font(.omlxText(12, weight: .medium))
                    .monospacedDigit()
            }

            Row(label: String(localized: "bench.context.result.prefill_tps",
                              defaultValue: "Prefill speed",
                              comment: "Result row label for the verify prefill's tokens per second")) {
                Text(prefillTpsLabel)
                    .font(.omlxText(12, weight: .medium))
                    .monospacedDigit()
            }

            Row(label: String(localized: "bench.context.result.capped_by",
                              defaultValue: "Limited by",
                              comment: "Result row label for what bounded the measurement")) {
                Text(cappedByLabel)
                    .font(.omlxText(12, weight: .medium))
            }

            Row(label: String(localized: "bench.context.result.duration",
                              defaultValue: "Duration",
                              comment: "Result row label for how long the benchmark took")) {
                Text(durationLabel)
                    .font(.omlxText(12, weight: .medium))
                    .monospacedDigit()
            }

            FreeRow(isLast: true) {
                VStack(alignment: .leading, spacing: 4) {
                    if result.applied {
                        HintLine(text: String(localized: "bench.context.result.applied_note",
                                              defaultValue: "The value has been applied to this model's Context Window setting.",
                                              comment: "Hint under the Context Bench result confirming the setting was written"))
                    }
                    HintLine(text: String(localized: "bench.context.result.snapshot_note",
                                          defaultValue: "The result reflects free memory and the Memory Guard tier at benchmark time; rerun after major changes to either.",
                                          comment: "Hint under the Context Bench result explaining the measurement is a snapshot"))
                }
            }
        }
    }

    private var cappedByLabel: String {
        switch result.cappedBy {
        case "target":
            return String(localized: "bench.context.capped.target",
                          defaultValue: "Selected target",
                          comment: "Limited-by value when the selected target bounded the result")
        case "native":
            return String(localized: "bench.context.capped.native",
                          defaultValue: "Model's native context length",
                          comment: "Limited-by value when the model's own context length bounded the result")
        default:
            return String(localized: "bench.context.capped.memory",
                          defaultValue: "Available memory",
                          comment: "Limited-by value when free memory bounded the result")
        }
    }

    private var prefillTpsLabel: String {
        guard let tps = result.prefillTps, tps > 0 else { return "—" }
        return "\(Int(tps.rounded()).formatted()) tok/s"
    }

    private var durationLabel: String {
        let seconds = Int(result.durationS.rounded())
        if seconds >= 60 {
            return "\(seconds / 60)m \(seconds % 60)s"
        }
        return "\(seconds)s"
    }
}
