// Screen capture for the Ask AI capture_screen tool. macOS needs the
// Screen & System Audio Recording permission, and it often applies only
// after oMLX restarts.

import AppKit
import CoreGraphics
import ImageIO
import ScreenCaptureKit
import UniformTypeIdentifiers

enum ScreenCaptureAccess {
    static var isGranted: Bool { CGPreflightScreenCaptureAccess() }

    /// Asks once and lists oMLX in System Settings; later calls only report.
    @discardableResult
    static func request() -> Bool { CGRequestScreenCaptureAccess() }

    static func openSystemSettings() {
        let link = "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_ScreenCapture"
        guard let url = URL(string: link) else { return }
        NSWorkspace.shared.open(url)
    }
}

enum ScreenCaptureError: LocalizedError {
    case permission
    case noDisplay
    case encoding

    var errorDescription: String? {
        switch self {
        case .permission: "Screen recording permission is off for oMLX."
        case .noDisplay: "No display to capture."
        case .encoding: "The screenshot could not be encoded."
        }
    }
}

enum ScreenCapture {
    /// The display under the pointer without oMLX's own windows, encoded
    /// like a pasted image.
    @MainActor
    static func capture(maxPixelSize: Int = AskAIImage.maxPixelSize) async throws -> AskAIImage {
        guard ScreenCaptureAccess.isGranted else { throw ScreenCaptureError.permission }
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        let mouse = NSEvent.mouseLocation
        let screenID = NSScreen.screens.first { NSMouseInRect(mouse, $0.frame, false) }
            .flatMap { $0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID }
        guard let display = content.displays.first(where: { $0.displayID == screenID }) ?? content.displays.first
        else { throw ScreenCaptureError.noDisplay }
        let own = content.applications.filter { $0.bundleIdentifier == Bundle.main.bundleIdentifier }
        let filter = SCContentFilter(display: display, excludingApplications: own, exceptingWindows: [])

        // Capture near the final size; AskAIImage only downscales.
        let config = SCStreamConfiguration()
        let longSide = Double(max(display.width, display.height))
        let scale = min(2.0, Double(maxPixelSize) / longSide)
        config.width = Int(Double(display.width) * scale)
        config.height = Int(Double(display.height) * scale)
        config.showsCursor = false
        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        guard let png = png(image), let out = await AskAIImage.make(from: .data(png), maxPixelSize: maxPixelSize)
        else { throw ScreenCaptureError.encoding }
        return out
    }

    // Lossless, so the image is JPEG-encoded only once.
    private static func png(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil
        ) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
