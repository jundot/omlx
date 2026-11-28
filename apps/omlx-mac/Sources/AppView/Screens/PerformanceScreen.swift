// Phase 3 — Performance.
//
// One tab for every "how the engine runs" knob. Three sections:
//   • Scheduler — max_concurrent_requests (moved from ServerScreen for
//     scheduler coherence), embedding_batch_size, and chunked_prefill.
//   • Memory & Lifecycle — prefill memory guard tier, server-wide idle
//     timeout, model fallback routing.
//   • Cache — master enable toggle gates a hot-cache toggle + size, a
//     cold-cache directory + size, and an advanced initial-blocks tuning
//     knob (requires restart).
//
// All fields are server-side already (`omlx/admin/routes.py:198-235`)
// — Phase 3 is pure UI. Single Apply button at the bottom, Storage /
// Network pattern: disabled until at least one trimmed draft diverges
// from its loaded value, and only changed fields are sent in the PATCH
// so out-of-band edits to siblings stay intact.

import SwiftUI

struct PerformanceScreen: View {
    @Environment(AppServices.self) private var services
    @State private var vm = PerformanceScreenVM()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SchedulerSection(vm: vm)
            MemoryLifecycleSection(vm: vm)
            CacheSection(vm: vm)
            AdvancedSection(vm: vm)

            FooterBar(error: vm.lastError) {
                Button(String(localized: "settings.button.reset_defaults",
                              defaultValue: "Reset Defaults",
                              comment: "Fill this screen with defaults before applying")) {
                    Task { await vm.resetDefaults(client: services.client) }
                }
                .buttonStyle(.omlx(.normal))
                .disabled(vm.isSaving || vm.isLoading || vm.isResetting)
                .help(String(localized: "settings.reset_defaults.help",
                             defaultValue: "Restore default values. Paths and API keys are kept. Click Apply to save."))
                Button(String(localized: "performance.button.apply",
                              defaultValue: "Apply",
                              comment: "Apply button at the bottom of the Performance screen")) {
                    Task { await vm.save(client: services.client) }
                }
                .buttonStyle(.omlx(.primary))
                .disabled(!vm.hasPendingChanges || vm.isSaving || vm.isResetting)
            }
        }
        .alert(String(localized: "settings.reset_defaults.title",
                      defaultValue: "Settings Reset"), isPresented: $vm.showResetNotice) {
            Button(String(localized: "common.cancel", defaultValue: "Cancel"), role: .cancel) {
                vm.cancelReset()
            }
            Button(String(localized: "common.ok", defaultValue: "OK")) {
                vm.confirmReset()
            }
        } message: {
            Text(String(localized: "settings.reset_defaults.message",
                        defaultValue: "Settings have been reset to defaults. Click Apply to save the changes."))
        }
        .task { await vm.load(client: services.client) }
    }
}

// MARK: - Scheduler

private struct SchedulerSection: View {
    @Bindable var vm: PerformanceScreenVM

    var body: some View {
        SectionHeader(
            String(localized: "performance.section.scheduler",
                   defaultValue: "Scheduler",
                   comment: "Section header for the scheduler rows"),
            subtitle: String(localized: "performance.section.scheduler.sub",
                             defaultValue: "How many requests run at once and how the engine batches them.",
                             comment: "Subtitle for the Scheduler section")
        )

        ListGroup {
            Row(
                label: String(localized: "performance.scheduler.max_concurrent",
                              defaultValue: "Max Concurrent Requests",
                              comment: "Row label for max concurrent requests"),
                sublabel: String(localized: "performance.scheduler.max_concurrent.sub",
                                 defaultValue: "Cap on simultaneous /v1 requests.",
                                 comment: "Sublabel for max concurrent requests")
            ) {
                TextInput(text: $vm.maxConcurrentText, mono: true, width: .controlNarrow)
            }
            Row(
                label: String(localized: "performance.scheduler.embedding_batch_size",
                              defaultValue: "Embedding Batch Size",
                              comment: "Row label for embedding batch size"),
                sublabel: String(localized: "performance.scheduler.embedding_batch_size.sub",
                                 defaultValue: "Max input texts per embedding forward pass.",
                                 comment: "Sublabel for embedding batch size")
            ) {
                TextInput(text: $vm.embeddingBatchSizeText, mono: true, width: .controlNarrow)
            }
            Row(
                label: String(localized: "performance.scheduler.chunked_prefill",
                              defaultValue: "Chunked Prefill",
                              comment: "Row label for chunked prefill toggle"),
                sublabel: String(localized: "performance.scheduler.chunked_prefill.sub",
                                 defaultValue: "Split long prompts across scheduler ticks so other requests can interleave.",
                                 comment: "Sublabel for chunked prefill toggle")
            ) {
                RowSwitch(isOn: $vm.chunkedPrefill)
            }
            Row(
                label: String(localized: "performance.scheduler.concurrent_priority",
                              defaultValue: "Concurrent Request Priority",
                              comment: "Row label for the concurrent request priority popup"),
                sublabel: String(localized: "performance.scheduler.concurrent_priority.sub",
                                 defaultValue: "What runs first when a new prompt arrives while other responses are generating. Prefill First can pause them during a long prompt.",
                                 comment: "Sublabel for the concurrent request priority popup")
            ) {
                Popup(
                    selection: $vm.concurrentPriority,
                    width: .controlMedium,
                    options: [
                        ("prefill",
                         String(localized: "concurrent_priority.option.prefill",
                                defaultValue: "Prefill First",
                                comment: "Concurrent request priority option that prefills new prompts first")),
                        ("balanced",
                         String(localized: "concurrent_priority.option.balanced",
                                defaultValue: "Balanced",
                                comment: "Concurrent request priority option that adapts how much GPU time running responses keep")),
                        ("decode",
                         String(localized: "concurrent_priority.option.decode",
                                defaultValue: "Decode First",
                                comment: "Concurrent request priority option that keeps running responses decoding first")),
                    ]
                )
            }
            Row(
                label: String(localized: "performance.scheduler.prefill_priority",
                              defaultValue: "Long Prompt Handling",
                              comment: "Row label for the prefill priority segmented control"),
                sublabel: String(localized: "performance.scheduler.prefill_priority.sub",
                                 defaultValue: "Under memory pressure, Max Context accepts longer prompts at a slower prefill speed. Full Speed keeps prefill at full speed and rejects prompts that would not fit.",
                                 comment: "Sublabel for the prefill priority segmented control"),
                isLast: true
            ) {
                Segmented(
                    selection: $vm.prefillPriority,
                    options: [
                        (value: "context",
                         label: String(localized: "prefill_priority.option.max_context",
                                       defaultValue: "Max Context",
                                       comment: "Prefill priority option that favors the largest context")),
                        (value: "speed",
                         label: String(localized: "prefill_priority.option.speed",
                                       defaultValue: "Full Speed",
                                       comment: "Prefill priority option that favors prefill speed")),
                    ],
                    icons: ["arrow.up.left.and.arrow.down.right", "speedometer"]
                )
            }
        }
    }
}

// MARK: - Memory & Lifecycle

private struct MemoryLifecycleSection: View {
    @Bindable var vm: PerformanceScreenVM
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        SectionHeader(
            String(localized: "performance.section.memory",
                   defaultValue: "Memory & Lifecycle",
                   comment: "Section header for memory and lifecycle settings"),
            subtitle: String(localized: "performance.section.memory.sub",
                             defaultValue: "Memory admission control and auto-unload behavior.",
                             comment: "Subtitle explaining memory and lifecycle settings")
        )

        ListGroup {
            Row(
                label: String(localized: "performance.memory.prefill_guard",
                              defaultValue: "Prefill Memory Guard",
                              comment: "Row label for prefill memory guard toggle"),
                sublabel: String(localized: "performance.memory.prefill_guard.sub",
                                 defaultValue: "Preflight prefill memory before kicking the engine and defer generation scheduling near the ceiling.",
                                 comment: "Sublabel for prefill memory guard")
            ) {
                RowSwitch(isOn: $vm.prefillMemoryGuard)
            }
            Row(
                label: String(localized: "performance.memory.guard_tier",
                              defaultValue: "Memory Guard Tier",
                              comment: "Row label for memory guard tier popup"),
                sublabel: vm.memoryGuardTierDescription
            ) {
                Popup(
                    selection: $vm.memoryGuardTier,
                    width: .controlMedium,
                    options: [
                        ("safe",
                         String(localized: "performance.memory.guard_tier.safe",
                                defaultValue: "Safe",
                                comment: "Memory guard tier option: safe")),
                        ("balanced",
                         String(localized: "performance.memory.guard_tier.balanced",
                                defaultValue: "Balanced",
                                comment: "Memory guard tier option: balanced")),
                        ("aggressive",
                         String(localized: "performance.memory.guard_tier.aggressive",
                                defaultValue: "Aggressive",
                                comment: "Memory guard tier option: aggressive")),
                        ("custom",
                         String(localized: "performance.memory.guard_tier.custom",
                                defaultValue: "Custom",
                                comment: "Memory guard tier option: custom")),
                    ]
                )
                .disabled(!vm.prefillMemoryGuard)
            }
            if vm.prefillMemoryGuard && vm.memoryGuardTier == "custom" {
                Row(
                    label: String(localized: "performance.memory.custom_ceiling",
                                  defaultValue: "Custom Ceiling",
                                  comment: "Row label for memory guard custom ceiling"),
                    sublabel: String(localized: "performance.memory.custom_ceiling.sub",
                                     defaultValue: "Fixed process memory ceiling in GB. Used only with the Custom tier.",
                                     comment: "Sublabel for memory guard custom ceiling")
                ) {
                    TextInput(
                        text: $vm.memoryGuardCustomCeilingText,
                        placeholder: String(localized: "performance.memory.custom_ceiling.placeholder",
                                            defaultValue: "GB",
                                            comment: "Placeholder for custom memory guard ceiling"),
                        mono: true,
                        suffix: "GB",
                        width: .controlCompact
                    )
                }
            }
            if vm.memoryGuardBreakdown != nil || vm.wiredLimitWarningText != nil {
                ceilingPreviewRow
            }
            Row(
                label: String(localized: "performance.memory.idle_timeout",
                              defaultValue: "Idle Timeout",
                              comment: "Row label for idle timeout field"),
                sublabel: String(localized: "performance.memory.idle_timeout.sub",
                                  defaultValue: "Server-wide auto-unload after N seconds idle. Empty or 0 = disabled. Minimum 60.",
                                  comment: "Sublabel for idle timeout")
            ) {
                TextInput(
                    text: $vm.idleTimeoutText,
                    placeholder: String(localized: "performance.memory.idle_timeout.placeholder",
                                        defaultValue: "off",
                                        comment: "Placeholder text for the idle timeout field when disabled"),
                    mono: true,
                    suffix: "s",
                    width: .controlCompact
                )
            }
            Row(
                label: String(localized: "performance.memory.model_fallback",
                              defaultValue: "Model Fallback",
                              comment: "Row label for model fallback toggle"),
                sublabel: String(localized: "performance.memory.model_fallback.sub",
                                 defaultValue: "When the requested model isn't loaded, route to any loaded model instead of 404.",
                                 comment: "Sublabel for model fallback toggle"),
                isLast: true
            ) {
                RowSwitch(isOn: $vm.modelFallback)
            }
        }
    }

    /// Effective-ceiling preview + kernel Metal limit warning. Mirrors the
    /// web dashboard's breakdown under the guard tier dropdown: without it
    /// a Custom ceiling above the Metal cap looks accepted while the guard
    /// silently enforces the clamped value (#1463).
    private var ceilingPreviewRow: some View {
        FreeRow {
            VStack(alignment: .leading, spacing: 8) {
                if let breakdown = vm.memoryGuardBreakdown {
                    Text(breakdown)
                        .font(.omlxText(11.5))
                        .foregroundStyle(theme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let warning = vm.wiredLimitWarningText {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(theme.warningText)
                        VStack(alignment: .leading, spacing: 6) {
                            Text(warning)
                                .font(.omlxText(DesignTokens.FontSize.subheadline))
                                .foregroundStyle(theme.text)
                                .fixedSize(horizontal: false, vertical: true)
                            CodeChip(value: vm.wiredLimitCommand)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(theme.warningBg)
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .strokeBorder(theme.warningText.opacity(0.25), lineWidth: 0.5)
                    )
                }
            }
        }
    }
}

// MARK: - Cache

private struct CacheSection: View {
    @Bindable var vm: PerformanceScreenVM

    var body: some View {
        SectionHeader(
            String(localized: "performance.section.cache",
                   defaultValue: "Cache",
                   comment: "Section header for KV cache settings"),
            subtitle: String(localized: "performance.section.cache.sub",
                             defaultValue: "KV cache spillover. The master switch gates everything below.",
                             comment: "Subtitle for the Cache section")
        )

        ListGroup {
            Row(
                label: String(localized: "performance.cache.enabled",
                              defaultValue: "Cache Enabled",
                              comment: "Row label for the master cache enable toggle"),
                sublabel: String(localized: "performance.cache.enabled.sub",
                                 defaultValue: "Master switch for the engine's KV cache subsystem.",
                                 comment: "Sublabel for the master cache enable toggle")
            ) {
                RowSwitch(isOn: $vm.cacheEnabled)
            }
            Row(
                label: String(localized: "performance.cache.hot_only",
                              defaultValue: "Hot Cache Only",
                              comment: "Row label for the hot cache only toggle"),
                sublabel: String(localized: "performance.cache.hot_only.sub",
                                 defaultValue: "Skip SSD spillover. Useful on fast machines with abundant RAM.",
                                 comment: "Sublabel for hot cache only toggle")
            ) {
                RowSwitch(isOn: $vm.hotCacheOnly)
                    .disabled(!vm.cacheEnabled)
            }
            Row(
                label: String(localized: "performance.cache.hot_size",
                              defaultValue: "Hot Cache Size",
                              comment: "Row label for the hot cache size field"),
                sublabel: String(localized: "performance.cache.hot_size.sub",
                                 defaultValue: "RAM ceiling for hot cache. \"0\" disables, sizes like \"8GB\" are accepted.",
                                 comment: "Sublabel describing accepted hot cache size values")
            ) {
                TextInput(
                    text: $vm.hotCacheMaxSize,
                    placeholder: "0",
                    mono: true,
                    width: .controlCompact
                )
                .disabled(!vm.cacheEnabled)
            }
            Row(
                label: String(localized: "performance.cache.ssd_dir",
                              defaultValue: "SSD Cache Directory",
                              comment: "Row label for the SSD cache directory field"),
                sublabel: String(localized: "performance.cache.ssd_dir.sub",
                                 defaultValue: "Where cold-spillover blocks live. Empty = base_path/cache.",
                                 comment: "Sublabel for the SSD cache directory"),
                warning: vm.ssdCacheUnavailable
                    ? String(localized: "performance.cache.ssd_dir.unavailable",
                             defaultValue: "Not available when the server started. SSD cache is off until the server restarts.",
                             comment: "Warning under the SSD cache directory when it was unavailable at server start")
                    : nil
            ) {
                TextInput(
                    text: $vm.ssdCacheDir,
                    placeholder: "<base_path>/cache",
                    mono: true,
                    width: .controlWide
                )
                .disabled(!vm.cacheEnabled || vm.hotCacheOnly)
            }
            Row(
                label: String(localized: "performance.cache.ssd_size",
                              defaultValue: "SSD Cache Size",
                              comment: "Row label for the SSD cache size field"),
                sublabel: String(localized: "performance.cache.ssd_size.sub",
                                 defaultValue: "Cold-spillover ceiling. \"auto\" = 50% of the sum of free space and existing SSD cache.",
                                 comment: "Sublabel describing accepted SSD cache size values")
            ) {
                TextInput(
                    text: $vm.ssdCacheMaxSize,
                    placeholder: String(localized: "performance.memory.placeholder_auto",
                                        defaultValue: "auto",
                                        comment: "Memory field placeholder meaning automatic"),
                    mono: true,
                    width: .controlCompact
                )
                .disabled(!vm.cacheEnabled || vm.hotCacheOnly)
            }
            Row(
                label: String(localized: "performance.cache.initial_blocks",
                              defaultValue: "Initial Cache Blocks",
                              comment: "Row label for the initial cache blocks field"),
                sublabel: String(localized: "performance.cache.initial_blocks.sub",
                                 defaultValue: "Pre-allocated cache blocks at server start. Requires restart to apply.",
                                 comment: "Sublabel for the initial cache blocks field"),
                isLast: true
            ) {
                TextInput(
                    text: $vm.initialCacheBlocksText,
                    placeholder: String(localized: "performance.memory.placeholder_auto",
                                        defaultValue: "auto",
                                        comment: "Memory field placeholder meaning automatic"),
                    mono: true,
                    width: .controlCompact
                )
                .disabled(!vm.cacheEnabled)
            }
        }
    }
}

// MARK: - Advanced
//
// Rarely changed cache knobs, collapsed by default (same set as the web
// dashboard's Settings → Advanced → Cache / GDN Cache Policy). Applied with
// the screen's Apply button like every other row.

private struct AdvancedSection: View {
    @Bindable var vm: PerformanceScreenVM
    @State private var expanded = false

    /// Mirrors the web dashboard: buffer size and precision only matter when
    /// GDN snapshots go to dedicated SSD files.
    private var gdnSidecarControlsDisabled: Bool {
        !vm.cacheEnabled || vm.hotCacheOnly || vm.gdnSnapshotStorage == "embedded"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            DisclosureSectionHeader(
                title: String(localized: "performance.section.advanced",
                              defaultValue: "Advanced",
                              comment: "Disclosure header for the advanced performance settings"),
                subtitle: String(localized: "performance.section.advanced.sub",
                                 defaultValue: "Applying cache changes unloads loaded models; they reload on the next request.",
                                 comment: "Subtitle for the advanced performance settings"),
                isExpanded: $expanded
            )

            if expanded {
                ListGroup {
                    Row(
                        label: String(localized: "performance.advanced.hot_cache_write_through",
                                      defaultValue: "Write-Through Hot Cache",
                                      comment: "Row label for the hot cache write-through toggle"),
                        sublabel: String(localized: "performance.advanced.hot_cache_write_through.sub",
                                         defaultValue: "Keep blocks in RAM and write them to SSD immediately. Old sessions survive restarts and crashes; recent sessions resume at RAM speed.",
                                         comment: "Sublabel for the hot cache write-through toggle")
                    ) {
                        RowSwitch(isOn: $vm.hotCacheWriteThrough)
                    }
                    Row(
                        label: String(localized: "performance.advanced.ane_compile_cache",
                                      defaultValue: "Reuse Compiled ANE Programs",
                                      comment: "Row label for the ANE compile cache toggle"),
                        sublabel: String(localized: "performance.advanced.ane_compile_cache.sub",
                                         defaultValue: "Keep the ANE programs macOS compiles for Qwen prefill so later loads reuse them. Cuts repeat load time. May require a server restart to take effect.",
                                         comment: "Sublabel for the ANE compile cache toggle"),
                        isLast: true
                    ) {
                        RowSwitch(isOn: $vm.aneCompileCache)
                    }
                }

                SectionHeader(
                    String(localized: "performance.section.gdn_cache",
                           defaultValue: "GDN Cache Policy",
                           comment: "Section header for the GDN cache policy rows"),
                    subtitle: String(localized: "performance.section.gdn_cache.sub",
                                     defaultValue: "Models with Gated DeltaNet (GDN) layers need an extra state snapshot to resume cached prompts without recomputing them.",
                                     comment: "Subtitle for the GDN cache policy section")
                )

                ListGroup {
                    Row(
                        label: String(localized: "performance.gdn.snapshot_storage",
                                      defaultValue: "Snapshot Storage",
                                      comment: "Row label for the GDN snapshot storage picker"),
                        sublabel: String(localized: "performance.gdn.snapshot_storage.sub",
                                         defaultValue: "Auto uses separate SSD files when SSD cache is enabled, otherwise stores snapshots with the KV cache. Dedicated SSD Sidecar requires Hot Cache Only to be off.",
                                         comment: "Sublabel for the GDN snapshot storage picker")
                    ) {
                        Popup(
                            selection: $vm.gdnSnapshotStorage,
                            width: .controlMedium,
                            options: [
                                ("auto", String(localized: "performance.gdn.snapshot_storage.auto",
                                                defaultValue: "Auto (recommended)",
                                                comment: "GDN snapshot storage option: automatic")),
                                ("ssd_sidecar", String(localized: "performance.gdn.snapshot_storage.ssd_sidecar",
                                                       defaultValue: "Dedicated SSD Sidecar",
                                                       comment: "GDN snapshot storage option: separate SSD files")),
                                ("embedded", String(localized: "performance.gdn.snapshot_storage.embedded",
                                                    defaultValue: "With KV Cache",
                                                    comment: "GDN snapshot storage option: stored with the KV cache blocks")),
                            ]
                        )
                    }
                    Row(
                        label: String(localized: "performance.gdn.pending_write_limit",
                                      defaultValue: "SSD Write Buffer",
                                      comment: "Row label for the GDN SSD write buffer size"),
                        sublabel: String(localized: "performance.gdn.pending_write_limit.sub",
                                         defaultValue: "Maximum RAM for snapshots waiting to be written to SSD. When full, cache writes pause until space frees up.",
                                         comment: "Sublabel for the GDN SSD write buffer size")
                    ) {
                        TextInput(
                            text: $vm.gdnSsdPendingMaxSize,
                            placeholder: "512MB",
                            mono: true,
                            width: .controlCompact
                        )
                        .disabled(gdnSidecarControlsDisabled)
                    }
                    Row(
                        label: String(localized: "performance.gdn.sidecar_precision",
                                      defaultValue: "SSD Snapshot Precision",
                                      comment: "Row label for the GDN SSD snapshot precision picker"),
                        sublabel: String(localized: "performance.gdn.sidecar_precision.sub",
                                         defaultValue: "Precision of GDN state stored in SSD files. FP32 is exact; lower precision uses less space but adds reconstruction error.",
                                         comment: "Sublabel for the GDN SSD snapshot precision picker"),
                        warning: vm.gdnSidecarPrecision == "fp32"
                            ? nil
                            : String(localized: "performance.gdn.sidecar_precision.warning",
                                     defaultValue: "Reduced precision can change greedy outputs, break bit-level reproducibility, and reduce output quality. Use FP32 unless you accept this trade-off.",
                                     comment: "Warning under the GDN snapshot precision picker when a reduced precision is selected"),
                        isLast: true
                    ) {
                        Popup(
                            selection: $vm.gdnSidecarPrecision,
                            width: .controlMedium,
                            options: [
                                ("fp32", String(localized: "performance.gdn.sidecar_precision.fp32",
                                                defaultValue: "FP32 (exact, recommended)",
                                                comment: "GDN snapshot precision option: FP32")),
                                ("rht_int16", String(localized: "performance.gdn.sidecar_precision.rht_int16",
                                                     defaultValue: "RHT + int16",
                                                     comment: "GDN snapshot precision option: RHT + int16")),
                                ("bf16", String(localized: "performance.gdn.sidecar_precision.bf16",
                                                defaultValue: "BF16",
                                                comment: "GDN snapshot precision option: BF16")),
                                ("int8", String(localized: "performance.gdn.sidecar_precision.int8",
                                                defaultValue: "Int8 (row-wise)",
                                                comment: "GDN snapshot precision option: int8")),
                                ("rht_int8", String(localized: "performance.gdn.sidecar_precision.rht_int8",
                                                    defaultValue: "RHT + int8 (row-wise)",
                                                    comment: "GDN snapshot precision option: RHT + int8")),
                            ]
                        )
                        .disabled(gdnSidecarControlsDisabled)
                    }
                }
            }
        }
    }
}
