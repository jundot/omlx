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
