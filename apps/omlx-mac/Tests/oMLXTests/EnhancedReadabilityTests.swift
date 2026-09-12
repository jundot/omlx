// SPDX-License-Identifier: Apache-2.0

import XCTest
import SwiftUI

#if canImport(AppKit)
import AppKit
#endif

/// Regression coverage for the SwiftUI side of the macOS Enhanced Readability
/// accessibility toggle. Complements `tests/test_admin_enhanced_readability.py`
/// (which covers the web admin side).
///
/// Three contracts pinned here so a future refactor of the `Theme` colour
/// tokens or the `@AppStorage` key cannot silently desynchronise the two
/// surfaces:
///
///   1. The `EnhancedReadability.enabledKey` constant matches the
///      `@AppStorage(...)` declaration on `AppView` (so the toggle
///      state round-trips through UserDefaults across launches).
///   2. `Color(hex:)` round-trips for the two web-admin canonical
///      danger colours: light = `0xd92d20`, dark = `0xef5b54`.
///   3. `OMLXTheme.light` and `OMLXTheme.dark` use distinguishable
///      danger colours (the dark-mode canonical red, not the
///      light-mode one).
///
/// View-snapshot tests for `AppView.resolvedTheme` are deliberately
/// out of scope here — `resolvedTheme` is `private` on `AppView`.
/// Long-term, exposing it (e.g. as `@Testable` style) would let us
/// pin the lift-on / lift-off behaviour from a unit test; the
/// only consumers of this contract today are the SwiftUI render
/// pipeline and integration tests in `ServerProcessIntegrationTests`.
final class EnhancedReadabilityTests: XCTestCase {

    // MARK: constants

    func testPersistedUserDefaultsKey_matchesAppStorage() {
        // The SwiftUI @AppStorage("OMLXEnhancedReadability") source-of-truth
        // lives on AppView. If either side drifts, the toggle stops
        // round-tripping through launch and the feature "forgets" itself.
        XCTAssertEqual(EnhancedReadability.enabledKey, "OMLXEnhancedReadability")
    }

    // MARK: Color(hex:)

    func testColorInitFromHex_lightDanger_d92d20() throws {
        let sut = Color(hex: 0xd92d20)
        let (r, g, b) = try Self.rgbComponents(of: sut)
        XCTAssertEqual(r, 0xD9 / 255.0, accuracy: 0.01)
        XCTAssertEqual(g, 0x2D / 255.0, accuracy: 0.01)
        XCTAssertEqual(b, 0x20 / 255.0, accuracy: 0.01)
    }

    func testColorInitFromHex_darkDanger_ef5b54() throws {
        let sut = Color(hex: 0xef5b54)
        let (r, g, b) = try Self.rgbComponents(of: sut)
        XCTAssertEqual(r, 0xEF / 255.0, accuracy: 0.01)
        XCTAssertEqual(g, 0x5B / 255.0, accuracy: 0.01)
        XCTAssertEqual(b, 0x54 / 255.0, accuracy: 0.01)
    }

    // MARK: theme behaviour

    func testLightTheme_distinctFromDarkTheme() {
        // Light and dark must use different danger colours. Light-mode red
        // leaking into dark rendering is the failure mode we want to catch
        // here — pin the two themes against each other so a refactor
        // cannot make them silently identical.
        XCTAssertNotEqual(OMLXTheme.light.redDot, OMLXTheme.dark.redDot,
                          "light and dark themes must use different danger colours")
        XCTAssertNotEqual(OMLXTheme.light.textSecondary, OMLXTheme.dark.textSecondary,
                          "light and dark themes must use distinguishable secondary text")
        XCTAssertNotEqual(OMLXTheme.light.textTertiary, OMLXTheme.dark.textTertiary,
                          "light and dark themes must use distinguishable tertiary text")
    }

    // MARK: helpers

    /// Resolves a SwiftUI `Color` to its RGB components in `0…1` space, using
    /// `NSColor` round-trips on macOS. Throws `XCTSkip` on non-macOS builds.
    private static func rgbComponents(of color: Color) throws -> (Double, Double, Double) {
        #if canImport(AppKit)
        guard let ns = NSColor(color).usingColorSpace(.deviceRGB) else {
            XCTFail("could not resolve Color to device RGB")
            return (0, 0, 0)
        }
        return (Double(ns.redComponent), Double(ns.greenComponent), Double(ns.blueComponent))
        #else
        throw XCTSkip("AppKit not available in this target; skipping RGB round-trip assertions.")
        #endif
    }
}
