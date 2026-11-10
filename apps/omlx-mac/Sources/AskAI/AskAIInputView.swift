// Multi-line prompt field for the Ask AI panel. An NSTextView instead of a
// SwiftUI TextEditor, so the field can commit an in-progress IME composition
// (Korean, Japanese, Chinese) before Return or Command-Return acts, and the
// panel can take Return, Tab, arrows, and Escape while suggestions are open.
// Command-Return always sends. With `sendOnEnter`, Return sends too and
// Shift-Return inserts the newline.

import AppKit
import SwiftUI

/// Key commands the input forwards to the panel.
enum AskAIInputCommand {
    case submit
    case stop
    case newConversation
    case copyLastAnswer
    case escape
    case suggestionUp
    case suggestionDown
    case acceptSuggestion
}

/// Lets the controller focus the field without holding SwiftUI state.
@MainActor
final class AskAIInputFocus {
    weak var textView: NSTextView?

    func focus() {
        guard let textView, let window = textView.window else { return }
        window.makeFirstResponder(textView)
        textView.setSelectedRange(NSRange(location: (textView.string as NSString).length, length: 0))
    }
}

struct AskAIInputView: NSViewRepresentable {
    @Binding var text: String
    @Binding var height: CGFloat
    let suggestionsVisible: Bool
    let sendOnEnter: Bool
    let focus: AskAIInputFocus
    let onCommand: (AskAIInputCommand) -> Void
    /// Returns true when the paste was taken as image attachments.
    let onPasteImages: (NSPasteboard) -> Bool

    static let font = NSFont.systemFont(ofSize: 15)
    static let maxLines: CGFloat = 8

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        // TextKit 1: `measure` reads the layout manager directly.
        let textView = AskAITextView(usingTextLayoutManager: false)
        textView.delegate = context.coordinator
        textView.coordinator = context.coordinator
        textView.font = Self.font
        textView.textColor = .labelColor
        textView.insertionPointColor = .controlAccentColor
        textView.drawsBackground = false
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.textContainerInset = NSSize(width: 0, height: 2)
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = true
        textView.minSize = .zero
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: .greatestFiniteMagnitude)
        textView.autoresizingMask = [.width]
        textView.string = text

        let scroll = NSScrollView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.documentView = textView
        focus.textView = textView
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scroll.documentView as? NSTextView else { return }
        // Never overwrite while the IME holds marked text.
        if textView.string != text, !textView.hasMarkedText() {
            textView.string = text
            textView.setSelectedRange(NSRange(location: (text as NSString).length, length: 0))
            // Typing undo still holds ranges in the replaced text.
            textView.undoManager?.removeAllActions()
        }
        context.coordinator.measure(textView)
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: AskAIInputView

        init(_ parent: AskAIInputView) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
            measure(textView)
        }

        func measure(_ textView: NSTextView) {
            guard let layout = textView.layoutManager, let container = textView.textContainer else { return }
            layout.ensureLayout(for: container)
            let lineHeight = layout.defaultLineHeight(for: AskAIInputView.font)
            let used = max(layout.usedRect(for: container).height, lineHeight)
            let inset = textView.textContainerInset.height * 2
            let next = min(used, lineHeight * AskAIInputView.maxLines) + inset
            if abs(next - parent.height) > 0.5 {
                DispatchQueue.main.async { self.parent.height = next }
            }
        }

        func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            let suggestions = parent.suggestionsVisible
            switch selector {
            case #selector(NSResponder.cancelOperation(_:)):
                parent.onCommand(.escape)
                return true
            case #selector(NSResponder.insertNewline(_:)):
                if suggestions {
                    parent.onCommand(.acceptSuggestion)
                    return true
                }
                // Shift-Return arrives as insertNewline: too and keeps the newline.
                let shifted = NSApp.currentEvent?.modifierFlags.contains(.shift) == true
                guard parent.sendOnEnter, !shifted else { return false }
                // A live IME composition owns Return; commit it, don't send.
                if textView.hasMarkedText() {
                    textView.unmarkText()
                    parent.text = textView.string
                    return true
                }
                parent.onCommand(.submit)
                return true
            case #selector(NSResponder.insertTab(_:)) where suggestions:
                parent.onCommand(.acceptSuggestion)
                return true
            case #selector(NSResponder.moveUp(_:)) where suggestions:
                parent.onCommand(.suggestionUp)
                return true
            case #selector(NSResponder.moveDown(_:)) where suggestions:
                parent.onCommand(.suggestionDown)
                return true
            default:
                return false
            }
        }

        /// Command shortcuts. Returns true when handled.
        func handleKeyEquivalent(_ event: NSEvent, in textView: NSTextView) -> Bool {
            let flags = event.modifierFlags.intersection([.command, .shift, .option, .control])
            let key = event.charactersIgnoringModifiers?.lowercased() ?? ""
            let isReturn = event.keyCode == 36 || event.keyCode == 76
            switch (flags, key) {
            case (.command, _) where isReturn:
                // Commit the IME composition so the send includes it.
                if textView.hasMarkedText() {
                    textView.unmarkText()
                    parent.text = textView.string
                }
                parent.onCommand(.submit)
                return true
            case (.command, "."):
                parent.onCommand(.stop)
                return true
            case (.command, "n"):
                parent.onCommand(.newConversation)
                return true
            case ([.command, .shift], "c"):
                parent.onCommand(.copyLastAnswer)
                return true
            // The panel is often key while oMLX is not the active app, so the
            // Edit menu may not see these. Route them to the text view.
            case (.command, "c"): return NSApp.sendAction(#selector(NSText.copy(_:)), to: textView, from: nil)
            case (.command, "v"): return NSApp.sendAction(#selector(NSText.paste(_:)), to: textView, from: nil)
            case (.command, "x"): return NSApp.sendAction(#selector(NSText.cut(_:)), to: textView, from: nil)
            case (.command, "a"): return NSApp.sendAction(#selector(NSText.selectAll(_:)), to: textView, from: nil)
            case (.command, "z"):
                textView.undoManager?.undo()
                return true
            case ([.command, .shift], "z"):
                textView.undoManager?.redo()
                return true
            default:
                return false
            }
        }
    }
}

private final class AskAITextView: NSTextView {
    weak var coordinator: AskAIInputView.Coordinator?

    override func paste(_ sender: Any?) {
        if coordinator?.parent.onPasteImages(.general) == true { return }
        super.paste(sender)
    }

    // An image drop attaches like Command-V instead of inserting the path.
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        carriesImages(sender) ? .copy : super.draggingEntered(sender)
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        carriesImages(sender) ? .copy : super.draggingUpdated(sender)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard carriesImages(sender),
              coordinator?.parent.onPasteImages(sender.draggingPasteboard) == true
        else { return super.performDragOperation(sender) }
        window?.makeKey()
        window?.makeFirstResponder(self)
        return true
    }

    private func carriesImages(_ sender: NSDraggingInfo) -> Bool {
        !AskAIImage.sources(on: sender.draggingPasteboard).isEmpty
    }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        guard window?.firstResponder === self,
              let coordinator,
              coordinator.handleKeyEquivalent(event, in: self)
        else { return super.performKeyEquivalent(with: event) }
        return true
    }
}
