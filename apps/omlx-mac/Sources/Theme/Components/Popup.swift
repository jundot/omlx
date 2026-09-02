// PR 3 — dropdown picker styled to match the JSX `Popup`.

import AppKit
import SwiftUI

private let fixedWidthPopupHeight: CGFloat = 24

struct PopupOption<Value: Hashable>: Identifiable {
    let value: Value
    let label: String
    var id: Value { value }
}

struct Popup<Value: Hashable>: View {
    @Binding var selection: Value
    var titleKey: LocalizedStringKey
    let options: [PopupOption<Value>]
    let width: CGFloat?
    let fillsWidth: Bool

    @Environment(\.omlxTheme) private var theme

    init(_ titleKey: LocalizedStringKey = "", selection: Binding<Value>, width: CGFloat? = nil, fillsWidth: Bool = false, options: [PopupOption<Value>]) {
        self.titleKey = titleKey
        self._selection = selection
        self.options = options
        self.width = width
        self.fillsWidth = fillsWidth
    }

    init(_ titleKey: LocalizedStringKey = "", selection: Binding<Value>, width: CGFloat? = nil, fillsWidth: Bool = false, options: [(Value, String)]) {
        self.titleKey = titleKey
        self._selection = selection
        self.options = options.map { PopupOption(value: $0.0, label: $0.1) }
        self.width = width
        self.fillsWidth = fillsWidth
    }

    var body: some View {
        if fillsWidth, let width {
            FixedWidthPopup(selection: $selection, options: options, width: width)
                .frame(width: width, height: fixedWidthPopupHeight)
                .background(theme.inputBg)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(theme.inputBorder, lineWidth: 0.5)
                }
        } else {
            Picker(titleKey, selection: $selection) {
                ForEach(options) { opt in
                    Text(opt.label)
                        .tag(opt.value)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: width)
        }
    }
}

private struct FixedWidthPopup<Value: Hashable>: NSViewRepresentable {
    @Binding var selection: Value
    let options: [PopupOption<Value>]
    let width: CGFloat

    func makeCoordinator() -> Coordinator {
        Coordinator(selection: $selection, options: options)
    }

    func makeNSView(context: Context) -> FixedWidthMenuContainer {
        let container = FixedWidthMenuContainer(
            frame: NSRect(x: 0, y: 0, width: width, height: fixedWidthPopupHeight),
        )
        let button = container.button
        button.isBordered = false
        button.controlSize = .regular
        button.alignment = .left
        button.font = .systemFont(ofSize: 13, weight: .medium)
        button.target = context.coordinator
        button.action = #selector(Coordinator.showMenu(_:))
        configure(button)
        return container
    }

    func updateNSView(_ container: FixedWidthMenuContainer, context: Context) {
        context.coordinator.selection = $selection
        context.coordinator.options = options
        configure(container.button)
    }

    private func configure(_ button: NSButton) {
        button.title = options.first(where: { $0.value == selection })?.label ?? ""
    }

    @MainActor
    final class Coordinator: NSObject {
        var selection: Binding<Value>
        var options: [PopupOption<Value>]

        init(selection: Binding<Value>, options: [PopupOption<Value>]) {
            self.selection = selection
            self.options = options
        }

        @objc func showMenu(_ sender: NSButton) {
            let menu = NSMenu()
            for (index, option) in options.enumerated() {
                let item = NSMenuItem(
                    title: option.label,
                    action: #selector(didSelectOption(_:)),
                    keyEquivalent: ""
                )
                item.target = self
                item.representedObject = index
                item.state = option.value == selection.wrappedValue ? .on : .off
                menu.addItem(item)
            }
            menu.popUp(
                positioning: menu.item(withTitle: sender.title),
                at: NSPoint(x: 0, y: sender.bounds.height),
                in: sender
            )
        }

        @objc func didSelectOption(_ sender: NSMenuItem) {
            guard let index = sender.representedObject as? Int,
                  options.indices.contains(index) else { return }
            selection.wrappedValue = options[index].value
        }
    }
}

private final class FixedWidthMenuContainer: NSView {
    let button = NSButton()
    private let chevron = NSImageView()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)

        button.frame = NSRect(
            x: 6,
            y: 0,
            width: max(0, bounds.width - 34),
            height: bounds.height
        )
        button.autoresizingMask = [.width, .height]
        addSubview(button)

        chevron.frame = NSRect(x: bounds.width - 22, y: 6, width: 12, height: 12)
        chevron.autoresizingMask = [.minXMargin]
        chevron.image = NSImage(
            systemSymbolName: "chevron.up.chevron.down",
            accessibilityDescription: nil
        )?.withSymbolConfiguration(.init(pointSize: 10, weight: .semibold))
        chevron.contentTintColor = .secondaryLabelColor
        chevron.imageScaling = .scaleProportionallyDown
        addSubview(chevron)
    }

    required init?(coder: NSCoder) {
        nil
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        button
    }
}

#Preview("Popup") {
    @Previewable @State var host = "127.0.0.1"
    @Previewable @State var quant = "q4"

    VStack(alignment: .leading, spacing: 14) {
        Popup(selection: $host, width: 220, options: [
            ("127.0.0.1", "127.0.0.1 (Local only)"),
            ("0.0.0.0", "0.0.0.0 (IPv4 only)"),
            ("::", "0.0.0.0 & :: (All Networks)"),
            ("localhost", "localhost"),
        ])
        Popup(selection: $quant, width: 120, options: [
            ("auto", "Auto"), ("q4", "q4"), ("q5", "q5"), ("q6", "q6"), ("q8", "q8"), ("fp16", "fp16"),
        ])
    }
    .padding(24)
    .omlxThemed()
}
