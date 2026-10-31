// System-wide hotkey through Carbon `RegisterEventHotKey`. It is still the
// only public API that registers a key combo for the whole session, and it
// needs no Accessibility or Input Monitoring permission.

import AppKit
import Carbon.HIToolbox

struct HotKeyCombo: Codable, Equatable, Sendable {
    var keyCode: UInt32
    /// Carbon modifier mask (`cmdKey`, `optionKey`, `controlKey`, `shiftKey`).
    var modifiers: UInt32
    /// Key name from the keyboard layout that was active when recorded.
    var keyLabel: String

    static let optionSpace = HotKeyCombo(
        keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey), keyLabel: "Space"
    )
    static let optionShiftSpace = HotKeyCombo(
        keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey | shiftKey), keyLabel: "Space"
    )

    var displayString: String {
        var out = ""
        if modifiers & UInt32(controlKey) != 0 { out += "⌃" }
        if modifiers & UInt32(optionKey) != 0 { out += "⌥" }
        if modifiers & UInt32(shiftKey) != 0 { out += "⇧" }
        if modifiers & UInt32(cmdKey) != 0 { out += "⌘" }
        return out + keyLabel
    }

    /// Builds a combo from a recorder key press. Returns nil unless the press
    /// holds Command, Option, or Control, or is a function key. Command with
    /// only Shift needs a named key, so editing shortcuts such as Cmd-C,
    /// Cmd-V, and Cmd-Q can never become a system-wide hotkey.
    init?(event: NSEvent) {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        var mods: UInt32 = 0
        if flags.contains(.command) { mods |= UInt32(cmdKey) }
        if flags.contains(.option) { mods |= UInt32(optionKey) }
        if flags.contains(.control) { mods |= UInt32(controlKey) }
        if flags.contains(.shift) { mods |= UInt32(shiftKey) }

        let code = UInt32(event.keyCode)
        let named = Self.namedKeys[Int(code)]
        let isFunctionKey = named?.hasPrefix("F") == true
        guard mods & UInt32(cmdKey | optionKey | controlKey) != 0 || isFunctionKey else { return nil }
        if mods & UInt32(optionKey | controlKey) == 0, named == nil { return nil }

        guard let label = named ?? event.charactersIgnoringModifiers?
            .trimmingCharacters(in: .whitespacesAndNewlines).uppercased(),
            !label.isEmpty
        else { return nil }
        self.init(keyCode: code, modifiers: mods, keyLabel: label)
    }

    init(keyCode: UInt32, modifiers: UInt32, keyLabel: String) {
        self.keyCode = keyCode
        self.modifiers = modifiers
        self.keyLabel = keyLabel
    }

    private static let namedKeys: [Int: String] = [
        kVK_Space: "Space", kVK_Return: "↩", kVK_Tab: "⇥", kVK_Delete: "⌫",
        kVK_ForwardDelete: "⌦", kVK_Escape: "⎋", kVK_LeftArrow: "←",
        kVK_RightArrow: "→", kVK_UpArrow: "↑", kVK_DownArrow: "↓",
        kVK_Home: "↖", kVK_End: "↘", kVK_PageUp: "⇞", kVK_PageDown: "⇟",
        kVK_F1: "F1", kVK_F2: "F2", kVK_F3: "F3", kVK_F4: "F4", kVK_F5: "F5",
        kVK_F6: "F6", kVK_F7: "F7", kVK_F8: "F8", kVK_F9: "F9", kVK_F10: "F10",
        kVK_F11: "F11", kVK_F12: "F12", kVK_F13: "F13", kVK_F14: "F14",
        kVK_F15: "F15", kVK_F16: "F16", kVK_F17: "F17", kVK_F18: "F18",
        kVK_F19: "F19", kVK_F20: "F20",
    ]
}

@MainActor
final class GlobalHotKey {
    private var hotKeyRef: EventHotKeyRef?
    private var handlerRef: EventHandlerRef?
    private var action: (() -> Void)?

    private static let signature: OSType = 0x6F4D4C58  // "oMLX"
    private static var nextID: UInt32 = 1

    /// Each instance's own id; handlers pass other instances' presses on.
    nonisolated let hotKeyID: UInt32

    var isRegistered: Bool { hotKeyRef != nil }

    init() {
        hotKeyID = Self.nextID
        Self.nextID += 1
    }

    /// Replaces any current registration. Returns a Carbon status; non-zero
    /// usually means another app already owns the combo.
    @discardableResult
    func register(_ combo: HotKeyCombo, action: @escaping () -> Void) -> OSStatus {
        unregister()
        installHandlerIfNeeded()
        self.action = action
        var ref: EventHotKeyRef?
        let status = RegisterEventHotKey(
            combo.keyCode,
            combo.modifiers,
            EventHotKeyID(signature: Self.signature, id: hotKeyID),
            GetApplicationEventTarget(),
            OptionBits(kEventHotKeyExclusive),
            &ref
        )
        if status == noErr {
            hotKeyRef = ref
        }
        return status
    }

    func unregister() {
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
        }
        hotKeyRef = nil
    }

    func invalidate() {
        unregister()
        if let handlerRef {
            RemoveEventHandler(handlerRef)
        }
        handlerRef = nil
        action = nil
    }

    private func installHandlerIfNeeded() {
        guard handlerRef == nil else { return }
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                                 eventKind: UInt32(kEventHotKeyPressed))
        let context = Unmanaged.passUnretained(self).toOpaque()
        InstallEventHandler(GetApplicationEventTarget(), { _, event, userData in
            guard let event, let userData else { return OSStatus(eventNotHandledErr) }
            var id = EventHotKeyID()
            let status = GetEventParameter(
                event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                nil, MemoryLayout<EventHotKeyID>.size, nil, &id
            )
            guard status == noErr, id.signature == GlobalHotKey.signature else {
                return OSStatus(eventNotHandledErr)
            }
            let owner = Unmanaged<GlobalHotKey>.fromOpaque(userData).takeUnretainedValue()
            guard id.id == owner.hotKeyID else { return OSStatus(eventNotHandledErr) }
            // Carbon dispatches application-target events on the main thread.
            MainActor.assumeIsolated { owner.action?() }
            return noErr
        }, 1, &spec, context, &handlerRef)
    }
}
