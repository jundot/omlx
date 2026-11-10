// Microphone permission for Ask AI voice mode. macOS asks once; after a denial
// only System Settings can turn access back on.

import AppKit
import AVFoundation

enum MicrophoneAccess {
    enum State: Equatable {
        case notDetermined
        case authorized
        case denied
    }

    static var state: State {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: .authorized
        case .notDetermined: .notDetermined
        default: .denied
        }
    }

    /// Voice processing cannot start without an input device, such as on a
    /// Mac mini or a MacBook with its lid closed.
    static var hasInputDevice: Bool {
        AVCaptureDevice.default(for: .audio) != nil
    }

    /// Shows the system prompt the first time; later calls return the saved answer.
    static func request() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    static func openSystemSettings() {
        let link = "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_Microphone"
        guard let url = URL(string: link) else { return }
        NSWorkspace.shared.open(url)
    }
}
