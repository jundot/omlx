// Floating Ask AI voice card: an orb that follows the voice, the current
// state, the reply transcript, and mute and end buttons. It stays up when
// another app takes focus, since the conversation runs hands-free.

import AppKit
import Carbon.HIToolbox
import SwiftUI

private final class TalkAIPanel: NSPanel {
    init() {
        super.init(
            contentRect: NSRect(origin: .zero, size: TalkAIPanelController.size),
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
        isMovableByWindowBackground = true
        // Keeps the panel out of AppDelegate's window bookkeeping.
        isExcludedFromWindowsMenu = true
        identifier = NSUserInterfaceItemIdentifier("talkai.panel")
        animationBehavior = .utilityWindow
    }

    var onCancel: (() -> Void)?
    var onSpace: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    override func cancelOperation(_ sender: Any?) {
        onCancel?()
    }

    // Space ends the user's turn; it never reaches the buttons.
    override func sendEvent(_ event: NSEvent) {
        if event.type == .keyDown, event.keyCode == UInt16(kVK_Space),
           event.modifierFlags.intersection(.deviceIndependentFlagsMask).isEmpty {
            if !event.isARepeat { onSpace?() }
            return
        }
        super.sendEvent(event)
    }
}

@MainActor
final class TalkAIPanelController: NSObject, NSWindowDelegate {
    static let size = NSSize(width: 300, height: 344)
    static let cornerRadius: CGFloat = 24
    /// Cross-fade between the card and the Ask AI bar.
    static let fadeDuration: TimeInterval = 0.15

    private let panel = TalkAIPanel()
    private weak var controller: TalkAIController?
    /// Set while this controller moves the window, so only user drags are saved.
    private var isPlacing = false

    init(controller: TalkAIController) {
        self.controller = controller
        super.init()
        let hosting = NSHostingView(rootView: TalkAIPanelView(controller: controller).omlxThemed())
        hosting.sizingOptions = []
        hosting.wantsLayer = true
        hosting.layer?.cornerRadius = Self.cornerRadius
        hosting.layer?.cornerCurve = .continuous
        hosting.layer?.masksToBounds = true
        panel.contentView = hosting
        panel.delegate = self
        panel.onCancel = { [weak controller] in controller?.endConversation() }
        panel.onSpace = { [weak controller] in controller?.finishTurn() }
    }

    func show() {
        place()
        panel.alphaValue = 0
        panel.orderFrontRegardless()
        panel.makeKey()
        NSAnimationContext.runAnimationGroup { context in
            context.duration = Self.fadeDuration
            panel.animator().alphaValue = 1
        }
    }

    func hide(animated: Bool = false) {
        guard animated else {
            panel.orderOut(nil)
            return
        }
        NSAnimationContext.runAnimationGroup { context in
            context.duration = Self.fadeDuration
            panel.animator().alphaValue = 0
        } completionHandler: { [weak self] in
            Task { @MainActor in
                // A conversation reopened during the fade keeps its panel.
                guard let self, self.controller?.isActive != true else { return }
                self.panel.orderOut(nil)
            }
        }
    }

    /// The card opens under the top center of the Ask AI bar's spot, so the
    /// two read as one panel; the bar's default spot is used when none is saved.
    private func place() {
        let size = Self.size
        let prefs = controller?.store.prefs
        let barWidth = prefs?.panelWidth ?? AskAIPrefs.defaultPanelWidth
        let saved = prefs?.panelPosition.map { NSPoint(x: $0.x + barWidth / 2, y: $0.y) }
        let savedScreen = saved.flatMap { point in
            NSScreen.screens.first { NSMouseInRect(NSPoint(x: point.x, y: point.y - 1), $0.frame, false) }
        }
        let mouse = NSEvent.mouseLocation
        let screen = savedScreen
            ?? NSScreen.screens.first { NSMouseInRect(mouse, $0.frame, false) }
            ?? NSScreen.main ?? NSScreen.screens[0]
        let visible = screen.visibleFrame
        var top = NSPoint(x: visible.midX, y: visible.maxY - visible.height * 0.2)
        if let saved, savedScreen != nil { top = saved }
        let origin = NSPoint(x: min(max(top.x - size.width / 2, visible.minX), visible.maxX - size.width),
                             y: min(max(top.y - size.height, visible.minY), visible.maxY - size.height))
        isPlacing = true
        panel.setFrame(NSRect(origin: origin, size: size), display: true)
        isPlacing = false
    }

    /// Dragging the card moves the shared spot, so the bar follows it.
    func windowDidMove(_ notification: Notification) {
        guard !isPlacing, let controller else { return }
        let frame = panel.frame
        let barWidth = controller.store.prefs.panelWidth
        controller.store.prefs.panelPosition = AskAIPanelPosition(x: frame.midX - barWidth / 2, y: frame.maxY)
    }
}

struct TalkAIPanelView: View {
    let controller: TalkAIController

    @Environment(\.omlxTheme) private var theme
    @State private var copied = false

    private var failure: String? {
        if case .failed(let message) = controller.phase { return message }
        return nil
    }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: TalkAIPanelController.cornerRadius, style: .continuous)
        VStack(spacing: 14) {
            orb
                .frame(width: 140, height: 140)
                .padding(.top, 26)
            Text(statusText)
                .font(.omlxText(DesignTokens.FontSize.title3, weight: .semibold))
                .foregroundStyle(theme.text)
            Group {
                if let failure {
                    Text(failure)
                        .foregroundStyle(theme.redDot)
                        .multilineTextAlignment(.center)
                        .lineLimit(3)
                } else if let notice = controller.notice {
                    Text(notice)
                        .foregroundStyle(theme.amberDot)
                        .multilineTextAlignment(.center)
                        .lineLimit(3)
                } else if let activity = controller.toolActivity, controller.phase == .thinking {
                    Text(activity.text)
                        .foregroundStyle(theme.textSecondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                } else {
                    captionLines
                }
            }
            .font(.omlxText(DesignTokens.FontSize.callout))
            .frame(maxWidth: .infinity, minHeight: 54, alignment: .top)
            .padding(.horizontal, 20)
            HStack(spacing: controller.capabilities.inputImage ? 12 : 16) {
                if failure != nil, controller.failureAction != .none {
                    Button(TalkAIPanelStrings.openSettings) { controller.performFailureAction() }
                        .buttonStyle(.omlx(.normal, size: .small))
                } else {
                    roundButton(controller.isMuted ? "mic.slash.fill" : "mic.fill",
                                tint: controller.isMuted ? theme.redDot : theme.text,
                                help: controller.isMuted ? TalkAIPanelStrings.unmute : TalkAIPanelStrings.mute) {
                        controller.toggleMute()
                    }
                    .disabled(failure != nil)
                }
                if controller.capabilities.inputImage {
                    // Same symbol as the text panel's screen capture line; the tint shows sharing.
                    roundButton("camera.viewfinder",
                                tint: controller.isSharingScreen ? theme.accent : theme.text,
                                help: controller.isSharingScreen
                                    ? TalkAIPanelStrings.stopSharing : TalkAIPanelStrings.shareScreen) {
                        controller.toggleScreenShare()
                    }
                    .disabled(failure != nil)
                }
                if controller.store.prefs.enabled {
                    roundButton("keyboard", tint: theme.text, help: TalkAIPanelStrings.typeInstead) {
                        controller.switchToText()
                    }
                }
                roundButton(copied ? "checkmark" : "doc.on.doc", tint: theme.text,
                            help: TalkAIPanelStrings.copy) {
                    controller.copyReplies()
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.2))
                        copied = false
                    }
                }
                .disabled(!controller.canCopyReplies && controller.transcript.isEmpty)
                roundButton("xmark", tint: theme.redDot, help: TalkAIPanelStrings.end) {
                    controller.endConversation()
                }
            }
            .padding(.bottom, 20)
        }
        .frame(width: TalkAIPanelController.size.width, height: TalkAIPanelController.size.height)
        .appGlass(.regular, in: shape)
        .overlay(shape.strokeBorder(theme.separator.opacity(0.6), lineWidth: 0.5))
    }

    /// One sentence per line; the last line grows with the speech and keeps
    /// its newest words in view.
    private var captionLines: some View {
        VStack(spacing: 2) {
            ForEach(Array(TalkAIController.captionLines(controller.transcript).enumerated()), id: \.offset) { _, line in
                Text(line)
                    .foregroundStyle(theme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.head)
            }
        }
    }

    private var statusText: String {
        switch controller.phase {
        case .idle, .connecting: TalkAIPanelStrings.connecting
        case .listening: controller.isMuted ? TalkAIPanelStrings.muted : TalkAIPanelStrings.listening
        case .userSpeaking: TalkAIPanelStrings.listening
        case .thinking: TalkAIPanelStrings.thinking
        case .speaking: TalkAIPanelStrings.speaking
        case .failed: TalkAIPanelStrings.failed
        }
    }

    /// Green while the user holds the turn, so it reads apart from the reply.
    private var orbColor: Color {
        if failure != nil { return theme.redDot }
        if controller.isMuted { return theme.textTertiary }
        return controller.phase == .userSpeaking ? theme.greenDot : theme.accent
    }

    private var orb: some View {
        let color = orbColor
        let waiting = controller.phase == .thinking || controller.phase == .connecting
        return TimelineView(.animation(paused: !waiting)) { timeline in
            let pulse = waiting
                ? (sin(timeline.date.timeIntervalSinceReferenceDate * 3) + 1) / 2 * 0.08
                : 0
            let scale = 0.8 + controller.level * 0.3 + pulse
            ZStack {
                Circle()
                    .stroke(color.opacity(0.22), lineWidth: 10)
                    .scaleEffect(scale * 1.1)
                Circle()
                    .fill(RadialGradient(colors: [color.opacity(0.95), color.opacity(0.4)],
                                         center: .center, startRadius: 6, endRadius: 80))
                    .scaleEffect(scale)
            }
            .animation(.easeOut(duration: 0.12), value: controller.level)
            .animation(.easeOut(duration: 0.2), value: controller.phase)
            // Outside the scale effect, so the hint keeps its size.
            .overlay {
                if controller.phase == .userSpeaking { spaceHint }
            }
        }
    }

    private var spaceHint: some View {
        VStack(spacing: 6) {
            Text(TalkAIPanelStrings.spaceKey)
                .font(.omlxText(DesignTokens.FontSize.callout, weight: .semibold))
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(.white.opacity(0.18)))
                .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(.white.opacity(0.9), lineWidth: 1.5))
            Text(TalkAIPanelStrings.toSend)
                .font(.omlxText(DesignTokens.FontSize.callout, weight: .medium))
        }
        .foregroundStyle(.white)
        .shadow(color: .black.opacity(0.2), radius: 2)
        .transition(.opacity)
    }

    private func roundButton(_ symbol: String, tint: Color, help: String,
                             action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 44, height: 44)
                .background(Circle().fill(theme.controlBg))
        }
        .buttonStyle(.plain)
        .help(help)
    }
}

private enum TalkAIPanelStrings {
    static var connecting: String {
        String(localized: "talkai.status.connecting", defaultValue: "Connecting…",
               comment: "Ask AI voice status while the voice session starts")
    }

    static var listening: String {
        String(localized: "talkai.status.listening", defaultValue: "Listening",
               comment: "Ask AI voice status while waiting for or hearing the user")
    }

    static var muted: String {
        String(localized: "talkai.status.muted", defaultValue: "Muted",
               comment: "Ask AI voice status while the microphone is muted")
    }

    static var thinking: String {
        String(localized: "talkai.status.thinking", defaultValue: "Thinking",
               comment: "Ask AI voice status after the user's turn, before the reply plays")
    }

    static var speaking: String {
        String(localized: "talkai.status.speaking", defaultValue: "Speaking",
               comment: "Ask AI voice status while the reply plays")
    }

    static var failed: String {
        String(localized: "talkai.status.failed", defaultValue: "Can't Talk Right Now",
               comment: "Ask AI voice status when the session cannot continue")
    }

    static var mute: String {
        String(localized: "talkai.button.mute", defaultValue: "Mute",
               comment: "Help text for the Ask AI voice mute button")
    }

    static var unmute: String {
        String(localized: "talkai.button.unmute", defaultValue: "Unmute",
               comment: "Help text for the Ask AI voice unmute button")
    }

    static var end: String {
        String(localized: "talkai.button.end", defaultValue: "End",
               comment: "Help text for the button that ends the Ask AI voice conversation")
    }

    static var shareScreen: String {
        String(localized: "talkai.button.share_screen", defaultValue: "Share Screen",
               comment: "Help text for the voice card button that sends a screenshot with each turn")
    }

    static var stopSharing: String {
        String(localized: "talkai.button.stop_sharing", defaultValue: "Stop Sharing Screen",
               comment: "Help text for the voice card button while the screen is being shared")
    }

    static var typeInstead: String {
        String(localized: "talkai.button.type_instead", defaultValue: "Type Instead",
               comment: "Help text for the button that switches from the voice card to the Ask AI text bar")
    }

    static var copy: String {
        String(localized: "talkai.button.copy", defaultValue: "Copy Replies",
               comment: "Help text for the button that copies the AI replies of the conversation")
    }

    static var spaceKey: String {
        String(localized: "talkai.orb.space", defaultValue: "Space",
               comment: "Key cap label in the Ask AI voice orb while the user speaks")
    }

    static var toSend: String {
        String(localized: "talkai.orb.to_send", defaultValue: "to send",
               comment: "Shown under the Space key cap: pressing Space sends the user's turn")
    }

    static var openSettings: String {
        String(localized: "talkai.button.open_settings", defaultValue: "Open Settings",
               comment: "Button in the Ask AI voice panel that opens the relevant settings")
    }
}
