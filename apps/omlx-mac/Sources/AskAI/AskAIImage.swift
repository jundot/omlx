// Images pasted into the Ask AI bar for vision models. Each one is
// downscaled and JPEG-encoded once, so a 5K Retina screenshot does not turn
// into a multi-megabyte request on every follow-up turn.

import AppKit
import ImageIO
import UniformTypeIdentifiers

/// Immutable after init, so sharing across tasks is safe.
final class AskAIImage: Identifiable, Equatable, @unchecked Sendable {
    /// Where pasted image bytes come from, read on the main actor.
    enum Source: Sendable {
        case file(URL)
        case data(Data)
    }

    let id = UUID()
    let jpegData: Data
    let thumbnail: CGImage

    static let maxPixelSize = 2048
    static let thumbnailPixelSize = 160
    static let maxCount = 4

    var dataURL: String {
        "data:image/jpeg;base64,\(jpegData.base64EncodedString())"
    }

    static func == (lhs: AskAIImage, rhs: AskAIImage) -> Bool { lhs.id == rhs.id }

    /// Image files copied in Finder, or image data such as a screenshot.
    /// Text wins when present, because rich-text copies from some apps also
    /// carry a picture of the text.
    static func sources(on pasteboard: NSPasteboard) -> [Source] {
        let options: [NSPasteboard.ReadingOptionKey: Any] = [
            .urlReadingFileURLsOnly: true,
            .urlReadingContentsConformToTypes: [UTType.image.identifier],
        ]
        if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: options) as? [URL],
           !urls.isEmpty {
            return urls.map(Source.file)
        }
        guard pasteboard.string(forType: .string) == nil,
              let type = pasteboard.types?.first(where: {
                  UTType($0.rawValue)?.conforms(to: .image) == true
              }),
              let data = pasteboard.data(forType: type)
        else { return [] }
        return [.data(data)]
    }

    /// Pasteboard types to register for so image drops reach the panel.
    static var dropTypes: [NSPasteboard.PasteboardType] {
        [.fileURL] + NSImage.imageTypes.map { NSPasteboard.PasteboardType($0) }
    }

    /// Decoding, downscaling, and encoding a large image takes hundreds of
    /// milliseconds, so it runs off the main actor.
    static func make(from source: Source) async -> AskAIImage? {
        await Task.detached(priority: .userInitiated) { AskAIImage(source: source) }.value
    }

    private init?(source: Source) {
        let imageSource: CGImageSource? = switch source {
        case .file(let url): CGImageSourceCreateWithURL(url as CFURL, nil)
        case .data(let data): CGImageSourceCreateWithData(data as CFData, nil)
        }
        guard let imageSource,
              let full = Self.decode(imageSource, maxPixelSize: Self.maxPixelSize).flatMap(Self.onWhite),
              let small = Self.decode(imageSource, maxPixelSize: Self.thumbnailPixelSize),
              let jpeg = Self.jpeg(full)
        else { return nil }
        self.jpegData = jpeg
        self.thumbnail = small
    }

    // ImageIO decodes straight to the target size and applies EXIF rotation.
    private static func decode(_ source: CGImageSource, maxPixelSize: Int) -> CGImage? {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    // Draws on white so transparent PNGs do not turn black as JPEG.
    private static func onWhite(_ image: CGImage) -> CGImage? {
        let size = CGSize(width: image.width, height: image.height)
        guard let context = CGContext(
            data: nil, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        context.setFillColor(.white)
        context.fill(CGRect(origin: .zero, size: size))
        context.draw(image, in: CGRect(origin: .zero, size: size))
        return context.makeImage()
    }

    private static func jpeg(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil
        ) else { return nil }
        CGImageDestinationAddImage(
            destination, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
