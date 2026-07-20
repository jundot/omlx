// Owns the combined host-metric status item. Every enabled metric
// (CPU / GPU / MEM) renders as one tag+bar segment inside a single status
// item, so enabling several doesn't scatter them across separate menubar
// squares with system-managed gaps between. Snapshot-driven —
// MenubarController's sampling timer pushes a fresh SystemStatsSnapshot via
// apply(_:) while any metric is enabled; the glyph re-rasters only when a
// pixel-quantized bar level (or the enabled set / appearance) changes.
// Clicking opens one transient popover stacking the enabled panels.

import AppKit
import SwiftUI

@MainActor
final class SystemMenubarItemsController: NSObject, NSPopoverDelegate {

    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private var lastSignature: String?
    private var lastSnapshot = SystemStatsSnapshot()
    /// Set by MenubarController to close the other controller's popovers so
    /// menubar dropdowns stay mutually exclusive.
    var willShowPopover: (() -> Void)?

    private var enabledKinds: [SystemStatKind] {
        let items = MenubarMetricPrefs.enabledSystemItems
        var kinds: [SystemStatKind] = []
        if items.cpu { kinds.append(.cpu) }
        if items.gpu { kinds.append(.gpu) }
        if items.memory { kinds.append(.memory) }
        return kinds
    }

    /// Stores the snapshot and reconciles + repaints. Called on every
    /// sampling tick.
    func apply(_ snapshot: SystemStatsSnapshot) {
        lastSnapshot = snapshot
        sync()
    }

    /// Reconciles the combined item with the Appearance toggles using the
    /// last snapshot. Also called on UserDefaults changes.
    func sync() {
        let kinds = enabledKinds
        guard !kinds.isEmpty else {
            tearDownItem()
            return
        }

        if statusItem == nil {
            makeItem()
        }
        guard let button = statusItem?.button else {
            return
        }

        let darkMenubar = button.effectiveAppearance
            .bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        let segments: [MenubarMetricGlyph.BarSegment] = kinds.map {
            (tag: $0.tag, fraction: fraction(for: $0))
        }
        let signature = MenubarMetricGlyph.barsSignature(
            segments: segments, darkMenubar: darkMenubar
        )
        if lastSignature != signature {
            button.image = MenubarMetricGlyph.barsImage(
                segments: segments, darkMenubar: darkMenubar
            )
            button.toolTip = "oMLX · " + segments
                .map { "\($0.tag) \(StatsFormat.percent($0.fraction))" }
                .joined(separator: " · ")
            lastSignature = signature
        }

        if let popover, popover.isShown,
           let hosting = popover.contentViewController as? NSHostingController<AnyView> {
            hosting.rootView = panelStackRoot(for: kinds)
        }
    }

    func closeAllPopovers() {
        if popover?.isShown == true {
            popover?.performClose(nil)
        }
    }

    private func fraction(for kind: SystemStatKind) -> Double? {
        switch kind {
        case .cpu:
            return lastSnapshot.cpuTotalUsage
        case .gpu:
            return lastSnapshot.gpuUsage
        case .memory:
            return lastSnapshot.memory.totalBytes > 0
                ? lastSnapshot.memory.usedFraction
                : nil
        }
    }

    private func panelStackRoot(for kinds: [SystemStatKind]) -> AnyView {
        AnyView(
            SystemStatsPanelStack(
                kinds: kinds,
                snapshot: lastSnapshot,
                refreshInterval: MenubarMetricPrefs.refreshInterval
            )
            .padding(.vertical, 4)
            .omlxThemed()
        )
    }

    private func hostingController(for kinds: [SystemStatKind]) -> NSHostingController<AnyView> {
        let hosting = NSHostingController(rootView: panelStackRoot(for: kinds))
        // Keep preferredContentSize tracking the SwiftUI fitting size so the
        // popover follows the fixed panel width instead of freezing on the
        // default frame it opens with.
        hosting.sizingOptions = [.preferredContentSize]
        return hosting
    }

    private func makeItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.autosaveName = "omlx.system.combined"

        let popover = NSPopover()
        popover.behavior = .transient
        popover.delegate = self

        if let button = item.button {
            button.target = self
            button.action = #selector(systemButtonClicked(_:))
            button.setAccessibilityLabel("oMLX system stats")
        }
        self.statusItem = item
        self.popover = popover
        self.lastSignature = nil
    }

    private func tearDownItem() {
        guard let item = statusItem else { return }
        popover?.performClose(nil)
        NSStatusBar.system.removeStatusItem(item)
        statusItem = nil
        popover = nil
        lastSignature = nil
    }

    @objc private func systemButtonClicked(_ sender: NSStatusBarButton) {
        guard let popover else { return }
        if popover.isShown {
            popover.performClose(nil)
            return
        }
        willShowPopover?()
        let hosting = hostingController(for: enabledKinds)
        popover.contentViewController = hosting
        // Seed the popover with the real fitting size so it never flashes
        // (or sticks at) the default popover frame on first open.
        popover.contentSize = hosting.sizeThatFits(
            in: NSSize(width: 1_000, height: 2_000)
        )
        popover.show(relativeTo: sender.bounds, of: sender, preferredEdge: .minY)
        popover.contentViewController?.view.window?.makeKey()
    }

    func popoverDidClose(_ notification: Notification) {
        (notification.object as? NSPopover)?.contentViewController = nil
    }
}
