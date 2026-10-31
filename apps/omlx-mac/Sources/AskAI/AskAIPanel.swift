// Floating Ask AI bar. A non-activating panel takes keyboard focus without
// activating oMLX, so the app the user was in stays frontmost and gets
// focus back as soon as the panel hides.

import AppKit
import SwiftUI

private final class AskAIPanel: NSPanel {
    init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: AskAIPrefs.defaultPanelWidth, height: 64),
            styleMask: [.borderless, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: true
        )
        isFloatingPanel = true
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        hidesOnDeactivate = false
        isReleasedWhenClosed = false
        // Keeps the panel out of AppDelegate's window bookkeeping (Dock icon
        // toggle, Cmd-Q close sweep, main window lookup by "main").
        isExcludedFromWindowsMenu = true
        identifier = NSUserInterfaceItemIdentifier("askai.panel")
        animationBehavior = .utilityWindow
    }

    /// Escape that no view handled, e.g. while answer text has focus.
    var onCancel: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    override func cancelOperation(_ sender: Any?) {
        onCancel?()
    }
}

/// Takes image drops anywhere on the panel. The prompt field handles drops
/// over itself, so this covers the transcript, footer, and margins.
private final class AskAIHostingView: NSHostingView<AnyView> {
    var onDropImages: ((NSPasteboard) -> Bool)?

    required init(rootView: AnyView) {
        super.init(rootView: rootView)
        registerForDraggedTypes(AskAIImage.dropTypes)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not used")
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        carriesImages(sender) ? .copy : []
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        carriesImages(sender) ? .copy : []
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard carriesImages(sender), onDropImages?(sender.draggingPasteboard) == true else { return false }
        window?.makeKey()
        return true
    }

    private func carriesImages(_ sender: NSDraggingInfo) -> Bool {
        !AskAIImage.sources(on: sender.draggingPasteboard).isEmpty
    }
}

/// Callbacks from the panel's SwiftUI content to the window controller.
struct AskAIPanelChrome {
    let heightChanged: (CGFloat) -> Void
    let beginResize: () -> Void
    let resize: (AskAIResizeEdge) -> Void
    let endResize: (AskAIResizeEdge) -> Void
    let resetPosition: () -> Void
    let resetWidth: () -> Void
}

enum AskAIResizeEdge {
    case leading, trailing
}

@MainActor
final class AskAIPanelController: NSObject, NSWindowDelegate {
    static let widthRange: ClosedRange<CGFloat> = 480...1200
    static let cornerRadius: CGFloat = 16

    private let panel = AskAIPanel()
    private weak var controller: AskAIController?
    private var hosting: AskAIHostingView?
    private var contentHeight: CGFloat = 64
    private var width = CGFloat(AskAIPrefs.defaultPanelWidth)
    /// Top-left corner of the panel in screen coordinates.
    private var anchorTop: NSPoint?
    private var visibleFrame = NSRect.zero
    /// Set while this controller moves the window, so `windowDidMove` only
    /// records moves the user made.
    private var isSettingFrame = false
    private var resizeStart: (frame: NSRect, mouseX: CGFloat)?
    private var shadowUpdatePending = false
    /// Bumped by each show and fade-out, so a stale fade does not hide the panel.
    private var fadeID = 0
    private var isFading = false

    init(controller: AskAIController) {
        self.controller = controller
        super.init()
        let chrome = AskAIPanelChrome(
            heightChanged: { [weak self] in self?.contentHeightChanged($0) },
            beginResize: { [weak self] in self?.beginResize() },
            resize: { [weak self] in self?.resize($0) },
            endResize: { [weak self] in self?.endResize($0) },
            resetPosition: { [weak self] in self?.resetPosition() },
            resetWidth: { [weak self] in self?.resetWidth() }
        )
        let root = AskAIPanelView(controller: controller, chrome: chrome)
            .omlxThemed()
        let hosting = AskAIHostingView(rootView: AnyView(root))
        hosting.onDropImages = { [weak controller] pasteboard in
            guard let controller, controller.attachImages(from: pasteboard) else { return false }
            controller.inputFocus.focus()
            return true
        }
        // The controller sizes the window; SwiftUI only reports its height.
        hosting.sizingOptions = []
        // Round the window itself, not only the SwiftUI shape, so the system
        // shadow and window rim follow the corners instead of the square bounds.
        hosting.wantsLayer = true
        hosting.layer?.cornerRadius = Self.cornerRadius
        hosting.layer?.cornerCurve = .continuous
        hosting.layer?.masksToBounds = true
        panel.contentView = hosting
        panel.delegate = self
        panel.onCancel = { [weak controller] in controller?.handle(.escape) }
        self.hosting = hosting
    }

    var isVisible: Bool { panel.isVisible }

    /// Shows the panel at the saved spot, or centered near the top of the
    /// screen under the pointer when none is saved or it is off screen.
    func show(animated: Bool = false) {
        fadeID += 1
        isFading = false
        panel.alphaValue = animated ? 0 : 1
        placeForShow()
        if let hosting {
            hosting.layoutSubtreeIfNeeded()
            let fitting = hosting.fittingSize.height
            if fitting > 0 { contentHeight = min(fitting, visibleFrame.height) }
        }
        applyFrame()
        // Draw before ordering front so the first frame is never a stale size.
        panel.contentView?.layoutSubtreeIfNeeded()
        panel.displayIfNeeded()
        panel.orderFrontRegardless()
        panel.makeKey()
        scheduleShadowUpdate()
        if animated { fade(to: 1) {} }
    }

    /// Animated when the voice card takes the bar's place.
    func hide(animated: Bool = false) {
        guard panel.isVisible, !isFading else { return }
        guard animated else {
            panel.orderOut(nil)
            controller?.panelDidHide()
            return
        }
        isFading = true
        fadeID += 1
        let id = fadeID
        fade(to: 0) { [weak self] in
            guard let self, self.fadeID == id else { return }
            self.isFading = false
            self.panel.orderOut(nil)
            self.panel.alphaValue = 1
            self.controller?.panelDidHide()
        }
    }

    private func fade(to alpha: CGFloat, completion: @escaping @MainActor () -> Void) {
        NSAnimationContext.runAnimationGroup { context in
            context.duration = TalkAIPanelController.fadeDuration
            panel.animator().alphaValue = alpha
        } completionHandler: {
            Task { @MainActor in completion() }
        }
    }

    private func placeForShow() {
        let prefs = controller?.store.prefs
        let saved = prefs?.panelPosition.map { NSPoint(x: $0.x, y: $0.y) }
        // Probe just inside the corner; the exact corner sits on the edge.
        let savedScreen = saved.flatMap { point in
            NSScreen.screens.first {
                NSMouseInRect(NSPoint(x: point.x + 1, y: point.y - 1), $0.frame, false)
            }
        }
        let screen = savedScreen ?? Self.screenUnderPointer()
        visibleFrame = screen.visibleFrame
        width = clampedWidth(CGFloat(prefs?.panelWidth ?? AskAIPrefs.defaultPanelWidth))
        if let saved, savedScreen != nil {
            anchorTop = clampedAnchor(saved)
        } else {
            anchorTop = defaultAnchor()
        }
        publishLayout()
    }

    private func defaultAnchor() -> NSPoint {
        NSPoint(x: visibleFrame.midX - width / 2, y: visibleFrame.maxY - visibleFrame.height * 0.2)
    }

    private func clampedWidth(_ value: CGFloat) -> CGFloat {
        let upper = max(Self.widthRange.lowerBound,
                        min(Self.widthRange.upperBound, visibleFrame.width - 24))
        return min(max(value, Self.widthRange.lowerBound), upper)
    }

    // Keeps the whole width and at least the input row on screen.
    private func clampedAnchor(_ point: NSPoint) -> NSPoint {
        let x = min(max(point.x, visibleFrame.minX), visibleFrame.maxX - width)
        let y = min(max(point.y, visibleFrame.minY + 160), visibleFrame.maxY)
        return NSPoint(x: x, y: y)
    }

    private func publishLayout() {
        guard let anchorTop else { return }
        let below = anchorTop.y - visibleFrame.minY - 12
        let maxHeight = max(160, min(below, visibleFrame.height * 0.6))
        controller?.updatePanelLayout(width: width, maxHeight: maxHeight)
    }

    private func contentHeightChanged(_ height: CGFloat) {
        guard height > 0, abs(height - contentHeight) > 0.5 else { return }
        contentHeight = height
        if panel.isVisible { applyFrame() }
    }

    // Grow downward from a fixed top edge.
    private func applyFrame() {
        guard let anchorTop else { return }
        let height = ceil(contentHeight)
        // Near the bottom of the screen, lift the panel instead of letting
        // it run under the Dock. The saved anchor stays where it was.
        let bottom = max(anchorTop.y - height, visibleFrame.minY)
        isSettingFrame = true
        panel.setFrame(
            NSRect(x: anchorTop.x, y: bottom, width: width, height: height),
            display: true
        )
        isSettingFrame = false
        scheduleShadowUpdate()
    }

    // The shadow is traced from drawn pixels; tracing right after setFrame
    // uses the previous frame and leaves a square rim. Wait one run loop pass.
    private func scheduleShadowUpdate() {
        guard !shadowUpdatePending else { return }
        shadowUpdatePending = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.shadowUpdatePending = false
            self.panel.invalidateShadow()
        }
    }

    // MARK: - Move and resize

    private func beginResize() {
        resizeStart = (panel.frame, NSEvent.mouseLocation.x)
    }

    private func resize(_ edge: AskAIResizeEdge) {
        guard let start = resizeStart, var anchor = anchorTop else { return }
        let dx = NSEvent.mouseLocation.x - start.mouseX
        switch edge {
        case .trailing:
            width = clampedWidth(start.frame.width + dx)
        case .leading:
            width = clampedWidth(start.frame.width - dx)
            anchor.x = start.frame.maxX - width
        }
        anchorTop = anchor
        publishLayout()
        applyFrame()
    }

    private func endResize(_ edge: AskAIResizeEdge) {
        guard resizeStart != nil else { return }
        resizeStart = nil
        controller?.savePanelWidth(width)
        // A leading-edge drag also moves a custom position.
        if edge == .leading, controller?.store.prefs.panelPosition != nil, let anchorTop {
            controller?.savePanelPosition(anchorTop)
        }
    }

    private func resetPosition() {
        controller?.savePanelPosition(nil)
        anchorTop = defaultAnchor()
        publishLayout()
        applyFrame()
    }

    private func resetWidth() {
        let oldWidth = width
        width = clampedWidth(CGFloat(AskAIPrefs.defaultPanelWidth))
        controller?.savePanelWidth(width)
        if controller?.store.prefs.panelPosition == nil {
            anchorTop = defaultAnchor()
        } else if var anchor = anchorTop {
            // Keep the panel centered on where it was.
            anchor.x += (oldWidth - width) / 2
            anchorTop = clampedAnchor(anchor)
            if let anchorTop { controller?.savePanelPosition(anchorTop) }
        }
        publishLayout()
        applyFrame()
    }

    private static func screenUnderPointer() -> NSScreen {
        let point = NSEvent.mouseLocation
        return NSScreen.screens.first { NSMouseInRect(point, $0.frame, false) }
            ?? NSScreen.main
            ?? NSScreen.screens[0]
    }

    nonisolated func windowDidResignKey(_ notification: Notification) {
        MainActor.assumeIsolated {
            hide()
        }
    }

    nonisolated func windowDidMove(_ notification: Notification) {
        MainActor.assumeIsolated {
            guard !isSettingFrame, panel.isVisible else { return }
            let frame = panel.frame
            let top = NSPoint(x: frame.minX, y: frame.maxY)
            anchorTop = top
            if let screen = panel.screen {
                visibleFrame = screen.visibleFrame
            }
            publishLayout()
            controller?.savePanelPosition(top)
        }
    }
}

// MARK: - Handles

/// Grab area over the panel icon: drag moves the panel, right-click offers
/// the reset actions.
struct AskAIPanelDragHandle: NSViewRepresentable {
    let chrome: AskAIPanelChrome

    func makeNSView(context: Context) -> AskAIDragHandleView {
        let view = AskAIDragHandleView()
        view.toolTip = String(localized: "askai.panel.drag.help",
                              defaultValue: "Drag to move. Right-click to reset.",
                              comment: "Tooltip on the Ask AI panel icon that moves the panel")
        return view
    }

    func updateNSView(_ view: AskAIDragHandleView, context: Context) {
        view.chrome = chrome
    }
}

/// Sets a cursor while the pointer is over the view. Uses a tracking area,
/// not a cursor rect: cursor rects only update while the app is active, and
/// the panel is used with oMLX inactive.
class AskAIHoverCursorView: NSView {
    var hoverCursor: NSCursor { .arrow }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(
            rect: .zero,
            options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
            owner: self
        ))
    }

    override func mouseEntered(with event: NSEvent) { hoverCursor.set() }
    override func mouseExited(with event: NSEvent) { NSCursor.arrow.set() }
}

final class AskAIDragHandleView: AskAIHoverCursorView {
    var chrome: AskAIPanelChrome?

    override var hoverCursor: NSCursor { .openHand }

    override func mouseDown(with event: NSEvent) {
        window?.performDrag(with: event)
    }

    override func menu(for event: NSEvent) -> NSMenu? {
        let menu = NSMenu()
        let position = menu.addItem(
            withTitle: String(localized: "askai.panel.reset_position",
                              defaultValue: "Reset Position",
                              comment: "Context menu item that moves the Ask AI panel back to its default spot"),
            action: #selector(resetPosition), keyEquivalent: ""
        )
        position.target = self
        let width = menu.addItem(
            withTitle: String(localized: "askai.panel.reset_width",
                              defaultValue: "Reset Width",
                              comment: "Context menu item that restores the default Ask AI panel width"),
            action: #selector(resetWidth), keyEquivalent: ""
        )
        width.target = self
        return menu
    }

    @objc private func resetPosition() { chrome?.resetPosition() }
    @objc private func resetWidth() { chrome?.resetWidth() }
}

/// Invisible strip on a panel side edge that drags the width.
struct AskAIPanelResizeHandle: NSViewRepresentable {
    let edge: AskAIResizeEdge
    let chrome: AskAIPanelChrome

    func makeNSView(context: Context) -> AskAIResizeHandleView {
        AskAIResizeHandleView()
    }

    func updateNSView(_ view: AskAIResizeHandleView, context: Context) {
        view.edge = edge
        view.chrome = chrome
    }
}

final class AskAIResizeHandleView: AskAIHoverCursorView {
    var edge = AskAIResizeEdge.trailing
    var chrome: AskAIPanelChrome?

    override var hoverCursor: NSCursor { .resizeLeftRight }

    override func mouseDown(with event: NSEvent) { chrome?.beginResize() }
    override func mouseDragged(with event: NSEvent) { chrome?.resize(edge) }
    override func mouseUp(with event: NSEvent) { chrome?.endResize(edge) }
}
