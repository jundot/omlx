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

            HStack {
                Spacer()
                Button(String(localized: "performance.button.apply",
                              defaultValue: "Apply",
                              comment: "Apply button at the bottom of the Performance screen")) {
                    Task { await vm.save(client: services.client) }
                }
                .buttonStyle(.omlx(.primary))
                .disabled(!vm.hasPendingChanges || vm.isSaving)
            }
            .padding(.horizontal, 18)
            .padding(.top, 6)

            if let error = vm.lastError {
                Text(error)
                    .font(.omlxText(11))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 18)
                    .padding(.top, 8)
            }
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
                TextInput(text: $vm.maxConcurrentText, mono: true, width: 90)
            }
            Row(
                label: String(localized: "performance.scheduler.embedding_batch_size",
                              defaultValue: "Embedding Batch Size",
                              comment: "Row label for embedding batch size"),
                sublabel: String(localized: "performance.scheduler.embedding_batch_size.sub",
                                 defaultValue: "Max input texts per embedding forward pass.",
                                 comment: "Sublabel for embedding batch size")
            ) {
                TextInput(text: $vm.embeddingBatchSizeText, mono: true, width: 90)
            }
            Row(
                label: String(localized: "performance.scheduler.chunked_prefill",
                              defaultValue: "Chunked Prefill",
                              comment: "Row label for chunked prefill toggle"),
                sublabel: String(localized: "performance.scheduler.chunked_prefill.sub",
                                 defaultValue: "Split long prompts across scheduler ticks so other requests can interleave.",
                                 comment: "Sublabel for chunked prefill toggle"),
                isLast: true
            ) {
                Toggle("", isOn: $vm.chunkedPrefill)
                    .labelsHidden().toggleStyle(.switch)
            }
        }
    }
}

// MARK: - Memory & Lifecycle

private struct MemoryLifecycleSection: View {
    @Bindable var vm: PerformanceScreenVM

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
                Toggle("", isOn: $vm.prefillMemoryGuard)
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(
                label: String(localized: "performance.memory.guard_tier",
                              defaultValue: "Memory Guard Tier",
                              comment: "Row label for memory guard tier popup"),
                sublabel: vm.memoryGuardTierDescription
            ) {
                Popup(
                    selection: $vm.memoryGuardTier,
                    width: 150,
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
                        width: 110
                    )
                }
            }
            Row(
                label: String(localized: "performance.memory.idle_timeout",
                              defaultValue: "Idle Timeout",
                              comment: "Row label for idle timeout field"),
                sublabel: String(localized: "performance.memory.idle_timeout.sub",
                                 defaultValue: "Server-wide auto-unload after N seconds idle. Empty = disabled. Minimum 60.",
                                 comment: "Sublabel for idle timeout")
            ) {
                TextInput(
                    text: $vm.idleTimeoutText,
                    placeholder: String(localized: "performance.memory.idle_timeout.placeholder",
                                        defaultValue: "off",
                                        comment: "Placeholder text for the idle timeout field when disabled"),
                    mono: true,
                    suffix: "s",
                    width: 110
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
                Toggle("", isOn: $vm.modelFallback)
                    .labelsHidden().toggleStyle(.switch)
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
                Toggle("", isOn: $vm.cacheEnabled)
                    .labelsHidden().toggleStyle(.switch)
            }
            Row(
                label: String(localized: "performance.cache.hot_only",
                              defaultValue: "Hot Cache Only",
                              comment: "Row label for the hot cache only toggle"),
                sublabel: String(localized: "performance.cache.hot_only.sub",
                                 defaultValue: "Skip SSD spillover. Useful on fast machines with abundant RAM.",
                                 comment: "Sublabel for hot cache only toggle")
            ) {
                Toggle("", isOn: $vm.hotCacheOnly)
                    .labelsHidden().toggleStyle(.switch)
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
                    width: 140
                )
                .disabled(!vm.cacheEnabled)
            }
            Row(
                label: String(localized: "performance.cache.ssd_dir",
                              defaultValue: "SSD Cache Directory",
                              comment: "Row label for the SSD cache directory field"),
                sublabel: String(localized: "performance.cache.ssd_dir.sub",
                                 defaultValue: "Where cold-spillover blocks live. Empty = base_path/cache.",
                                 comment: "Sublabel for the SSD cache directory")
            ) {
                TextInput(
                    text: $vm.ssdCacheDir,
                    placeholder: "<base_path>/cache",
                    mono: true,
                    width: 280
                )
                .disabled(!vm.cacheEnabled || vm.hotCacheOnly)
            }
            Row(
                label: String(localized: "performance.cache.ssd_size",
                              defaultValue: "SSD Cache Size",
                              comment: "Row label for the SSD cache size field"),
                sublabel: String(localized: "performance.cache.ssd_size.sub",
                                 defaultValue: "Cold-spillover ceiling. \"auto\" = 10% of SSD capacity.",
                                 comment: "Sublabel describing accepted SSD cache size values")
            ) {
                TextInput(
                    text: $vm.ssdCacheMaxSize,
                    placeholder: String(localized: "performance.memory.placeholder_auto",
                                        defaultValue: "auto",
                                        comment: "Memory field placeholder meaning automatic"),
                    mono: true,
                    width: 140
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
                    width: 110
                )
                .disabled(!vm.cacheEnabled)
            }
        }
    }
}
