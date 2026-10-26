// Click-to-record shortcut field for the Ask AI hotkey. While recording,
// a local key monitor captures the next combo; Escape cancels, and a click
// elsewhere or leaving the app ends recording.

import AppKit
import Carbon.HIToolbox
import SwiftUI

struct HotKeyRecorder: View {
    @Binding var combo: HotKeyCombo
    let onRecordingChange: (Bool) -> Void

    @State private var session = HotKeyRecordingSession()

    var body: some View {
        Button {
            session.toggle(onRecord: { combo = $0 }, onRecordingChange: onRecordingChange)
        } label: {
            Text(session.isRecording
                 ? String(localized: "askai.hotkey.recording", defaultValue: "Press shortcut…",
                          comment: "Hotkey recorder label while waiting for a key press")
                 : combo.displayString)
                .font(.omlxMono(DesignTokens.FontSize.callout, weight: .medium))
                .frame(minWidth: 110)
        }
        .buttonStyle(.omlx(session.isRecording ? .primary : .normal, size: .small))
        .help(String(localized: "askai.hotkey.help",
                     defaultValue: "Click, then press the new shortcut. Esc cancels.",
                     comment: "Tooltip on the Ask AI hotkey recorder"))
        .onDisappear { session.stop() }
    }
}

@MainActor
@Observable
final class HotKeyRecordingSession {
    private(set) var isRecording = false
    @ObservationIgnored private var monitor: Any?
    @ObservationIgnored private var resignObserver: NSObjectProtocol?
    @ObservationIgnored private var onRecordingChange: ((Bool) -> Void)?
    /// A click on the recorder itself ends recording on mouse down, then the
    /// button fires on mouse up. This keeps that click from restarting it.
    @ObservationIgnored private var stoppedByClickAt = Date.distantPast

    func toggle(onRecord: @escaping (HotKeyCombo) -> Void, onRecordingChange: @escaping (Bool) -> Void) {
        if isRecording {
            stop()
        } else if Date().timeIntervalSince(stoppedByClickAt) > 0.5 {
            start(onRecord: onRecord, onRecordingChange: onRecordingChange)
        }
    }

    private func start(onRecord: @escaping (HotKeyCombo) -> Void, onRecordingChange: @escaping (Bool) -> Void) {
        guard monitor == nil else { return }
        isRecording = true
        self.onRecordingChange = onRecordingChange
        onRecordingChange(true)
        monitor = NSEvent.addLocalMonitorForEvents(
            matching: [.keyDown, .leftMouseDown, .rightMouseDown]
        ) { [weak self] event in
            let consumed = MainActor.assumeIsolated {
                guard let self else { return false }
                guard event.type == .keyDown else {
                    self.stoppedByClickAt = Date()
                    self.stop()
                    return false
                }
                let flags = event.modifierFlags.intersection([.command, .option, .control, .shift])
                if event.keyCode == UInt16(kVK_Escape), flags.isEmpty {
                    self.stop()
                } else if let combo = HotKeyCombo(event: event) {
                    onRecord(combo)
                    self.stop()
                } else {
                    NSSound.beep()
                }
                return true
            }
            return consumed ? nil : event
        }
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.stop() }
        }
    }

    /// Ends recording and tells the owner, so the global hotkey comes back
    /// even when the view disappears mid-recording.
    func stop() {
        guard isRecording else { return }
        if let monitor {
            NSEvent.removeMonitor(monitor)
        }
        if let resignObserver {
            NotificationCenter.default.removeObserver(resignObserver)
        }
        monitor = nil
        resignObserver = nil
        isRecording = false
        let callback = onRecordingChange
        onRecordingChange = nil
        callback?(false)
    }
}
