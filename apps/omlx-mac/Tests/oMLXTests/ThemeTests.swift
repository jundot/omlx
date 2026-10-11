import AppKit
import SwiftUI
import XCTest
@testable import oMLX

final class ThemeTests: XCTestCase {

    func testLightWindowBackgroundUsesStandardWindowColor() {
        let actual = resolvedRGBA(OMLXTheme.light.windowBg, appearance: .aqua)
        let expected = resolvedRGBA(Color(nsColor: .windowBackgroundColor),
                                    appearance: .aqua)
        let underPage = resolvedRGBA(Color(nsColor: .underPageBackgroundColor),
                                     appearance: .aqua)

        assertClose(actual, expected)
        XCTAssertGreaterThan(actual.red, 0.95)
        XCTAssertGreaterThan(abs(actual.red - underPage.red), 0.25)
    }

    func testDarkWindowBackgroundKeepsUnderPageColor() {
        let actual = resolvedRGBA(OMLXTheme.dark.windowBg, appearance: .darkAqua)
        let expected = resolvedRGBA(Color(nsColor: .underPageBackgroundColor),
                                    appearance: .darkAqua)

        assertClose(actual, expected)
    }

    func testLightGroupBackgroundIsSubtleGrayWash() {
        let actual = resolvedRGBA(OMLXTheme.light.groupBg, appearance: .aqua)

        XCTAssertLessThan(actual.red, 0.01)
        XCTAssertLessThan(actual.green, 0.01)
        XCTAssertLessThan(actual.blue, 0.01)
        XCTAssertGreaterThan(actual.alpha, 0.025)
        XCTAssertLessThan(actual.alpha, 0.05)
    }

    @MainActor
    func testThemedViewUsesSavedReadabilityPreference() throws {
        for scheme: ColorScheme in [.light, .dark] {
            let base = scheme == .dark ? OMLXTheme.dark : OMLXTheme.light
            for enabled in [false, true] {
                let actual = try renderedTheme(scheme: scheme, readability: enabled)

                XCTAssertEqual(actual.textSecondary, enabled ? base.text : base.textSecondary)
                XCTAssertEqual(actual.textTertiary, enabled ? base.text : base.textTertiary)
                XCTAssertEqual(actual.windowBg, base.windowBg)
                XCTAssertEqual(actual.groupBg, base.groupBg)
                if enabled {
                    let appearance: NSAppearance.Name = scheme == .dark ? .darkAqua : .aqua
                    let expected: RGBA
                    if scheme == .dark {
                        expected = (239 / 255.0, 91 / 255.0, 84 / 255.0, 1)
                    } else {
                        expected = (217 / 255.0, 45 / 255.0, 32 / 255.0, 1)
                    }
                    assertClose(resolvedRGBA(actual.redDot, appearance: appearance), expected)
                    XCTAssertEqual(actual.warningText, actual.redDot)
                } else {
                    XCTAssertEqual(actual.redDot, base.redDot)
                    XCTAssertEqual(actual.warningText, base.warningText)
                }
            }
        }
    }

    @MainActor
    private func renderedTheme(scheme: ColorScheme, readability: Bool) throws -> OMLXTheme {
        let suiteName = "ThemeTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set(readability, forKey: "OMLXEnhancedReadability")

        var actual: OMLXTheme?
        let renderer = ImageRenderer(content:
            ThemeProbe { actual = $0 }
                .omlxThemed()
                .environment(\.colorScheme, scheme)
                .defaultAppStorage(defaults)
        )
        XCTAssertNotNil(renderer.nsImage)
        return try XCTUnwrap(actual)
    }

    private struct ThemeProbe: View {
        @Environment(\.omlxTheme) private var theme
        let record: (OMLXTheme) -> Void

        var body: some View {
            let _ = record(theme)
            Color.clear.frame(width: 1, height: 1)
        }
    }

    private typealias RGBA = (
        red: CGFloat,
        green: CGFloat,
        blue: CGFloat,
        alpha: CGFloat
    )

    private func resolvedRGBA(_ color: Color, appearance: NSAppearance.Name) -> RGBA {
        let nsColor = NSColor(color)
        var components: RGBA?
        NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
            let resolved = nsColor.usingColorSpace(.sRGB)!
            components = (
                red: resolved.redComponent,
                green: resolved.greenComponent,
                blue: resolved.blueComponent,
                alpha: resolved.alphaComponent
            )
        }
        return components!
    }

    private func assertClose(
        _ actual: RGBA,
        _ expected: RGBA,
        accuracy: CGFloat = 0.001,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.red,
            expected.red,
            accuracy: accuracy,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.green,
            expected.green,
            accuracy: accuracy,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.blue,
            expected.blue,
            accuracy: accuracy,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.alpha,
            expected.alpha,
            accuracy: accuracy,
            file: file,
            line: line
        )
    }
}
