// Owns the optional per-metric status items (LIV / AVG / ALL). sync()
// reconciles the items against the Appearance toggles and repaints their
// glyphs — but only when a cheap value signature says the pixels actually
// changed, because every button.image assignment makes the menubar
// re-composite the item even if nothing is visibly different. Each item
// opens a transient popover whose SwiftUI content exists only while shown.

import AppKit
import SwiftUI

@MainActor
final class MenubarMetricItemsController: NSObject, NSPopoverDelegate {

    private struct ItemSpec {
        let kind: MenubarMetricsStore.Kind
        let defaultsKey: String
        /// Fixed Latin tag drawn vertically in the glyph — deliberately not
        /// localized, like the rest of the menubar readout.
        let tag: String
    }

    private static let specs: [ItemSpec] = [
        ItemSpec(kind: .live, defaultsKey: MenubarMetricPrefs.liveKey, tag: "LIV"),
        ItemSpec(kind: .average, defaultsKey: MenubarMetricPrefs.averageKey, tag: "AVG"),
        ItemSpec(kind: .alltime, defaultsKey: MenubarMetricPrefs.alltimeKey, tag: "ALL"),
    ]

    private struct ItemEntry {
        let item: NSStatusItem
        let popover: NSPopover
        var lastSignature: String?
    }

    private var entries: [MenubarMetricsStore.Kind: ItemEntry] = [:]
    private let store: MenubarMetricsStore
    private let openAppearanceSettings: () -> Void
    private let openDashboard: () -> Void
    /// Set by MenubarController to close the other controller's popovers so
    /// menubar dropdowns stay mutually exclusive.
    var willShowPopover: (() -> Void)?

    init(
        store: MenubarMetricsStore,
        openAppearanceSettings: @escaping () -> Void,
        openDashboard: @escaping () -> Void
    ) {
        self.store = store
        self.openAppearanceSettings = openAppearanceSettings
        self.openDashboard = openDashboard
        super.init()
    }

    /// Reconciles status items with the Appearance toggles and refreshes
    /// their glyphs. Called on every poller tick, on UserDefaults changes,
    /// and on server-state transitions; cheap and idempotent when nothing
    /// changed.
    func sync() {
        for spec in Self.specs {
            guard UserDefaults.standard.bool(forKey: spec.defaultsKey) else {
                if let entry = entries.removeValue(forKey: spec.kind) {
                    entry.popover.performClose(nil)
                    NSStatusBar.system.removeStatusItem(entry.item)
                }
                continue
            }

            if entries[spec.kind] == nil {
                entries[spec.kind] = makeEntry(for: spec)
            }
            guard let button = entries[spec.kind]?.item.button else {
                continue
            }

            // Ink follows the status button's own appearance — the menubar
            // can be dark over a dark wallpaper while the app is light.
            let darkMenubar = button.effectiveAppearance
                .bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            let rates = store.rates[spec.kind]
            let promptValue = MenubarMetricGlyph.formatTps(rates?.promptTps)
            let generationValue = MenubarMetricGlyph.formatTps(rates?.generationTps)
            let signature = MenubarMetricGlyph.signature(
                tag: spec.tag,
                promptValue: promptValue,
                generationValue: generationValue,
                darkMenubar: darkMenubar
            )
            if entries[spec.kind]?.lastSignature != signature {
                button.image = MenubarMetricGlyph.image(
                    tag: spec.tag,
                    promptValue: promptValue,
                    generationValue: generationValue,
                    darkMenubar: darkMenubar
                )
                button.toolTip = "oMLX · \(spec.kind.displayName)"
                entries[spec.kind]?.lastSignature = signature
            }
        }
    }

    /// Closes every open metric popover, optionally keeping one. Also called
    /// when the main oMLX menu opens so dropdowns never stack.
    func closeAllPopovers(except kept: MenubarMetricsStore.Kind? = nil) {
        for (kind, entry) in entries where kind != kept && entry.popover.isShown {
            entry.popover.performClose(nil)
        }
    }

    private func makeEntry(for spec: ItemSpec) -> ItemEntry {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.autosaveName = "omlx.metric.\(spec.kind.rawValue)"

        let popover = NSPopover()
        popover.behavior = .transient
        popover.delegate = self

        if let button = item.button {
            button.target = self
            button.action = #selector(metricButtonClicked(_:))
            button.identifier = NSUserInterfaceItemIdentifier(spec.kind.rawValue)
            button.setAccessibilityLabel("oMLX \(spec.tag)")
        }
        return ItemEntry(item: item, popover: popover, lastSignature: nil)
    }

    @objc private func metricButtonClicked(_ sender: NSStatusBarButton) {
        guard
            let raw = sender.identifier?.rawValue,
            let kind = MenubarMetricsStore.Kind(rawValue: raw),
            let entry = entries[kind]
        else {
            return
        }

        if entry.popover.isShown {
            entry.popover.performClose(nil)
            return
        }

        // Transient popovers don't dismiss reliably when the click lands on
        // another of our own status buttons, so enforce one-at-a-time here.
        willShowPopover?()
        closeAllPopovers(except: kind)

        // Content is built on open and torn down on close (popoverDidClose):
        // while the popover is closed no SwiftUI view observes the store, so
        // ticks cost nothing beyond the data update itself.
        let hosting = NSHostingController(
            rootView: MetricPopoverView(
                kind: kind,
                store: store,
                openSettings: { [weak self] in
                    self?.closeAllPopovers()
                    self?.openAppearanceSettings()
                },
                openDashboard: { [weak self] in
                    self?.closeAllPopovers()
                    self?.openDashboard()
                }
            )
            .omlxThemed()
        )
        hosting.sizingOptions = [.preferredContentSize]
        entry.popover.contentViewController = hosting
        // Seed the real fitting size so the popover never opens on (or
        // sticks at) its default frame.
        entry.popover.contentSize = hosting.sizeThatFits(
            in: NSSize(width: 1_000, height: 2_000)
        )
        entry.popover.show(relativeTo: sender.bounds, of: sender, preferredEdge: .minY)
        entry.popover.contentViewController?.view.window?.makeKey()
    }

    func popoverDidClose(_ notification: Notification) {
        (notification.object as? NSPopover)?.contentViewController = nil
    }
}
