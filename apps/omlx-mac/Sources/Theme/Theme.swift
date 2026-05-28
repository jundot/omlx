// PR 3 — design tokens for the Tahoe (macOS 26) liquid-glass system.
//
// Lifted verbatim from omlx-components.jsx:10-94 in the design bundle.
// Both light and dark variants must stay in sync with the canvas; mismatches
// are caught by manual visual diffs and the snapshot tests in the test
// target.

import SwiftUI

// MARK: - Token table

struct OMLXTheme: Sendable {
    let isDark: Bool

    // Surfaces
    let windowBg: Color
    let sidebarBg: Color
    let sidebarBorder: Color
    let contentBg: Color
    let toolbarBg: Color
    let toolbarBorder: Color
    let groupBg: Color
    let groupBorder: Color
    let rowSep: Color
    let separator: Color

    // Text
    let text: Color
    let textSecondary: Color
    let textTertiary: Color

    // Accent + selection
    let accent: Color
    let accentSoft: Color
    let accentText: Color
    let selBg: Color
    let hoverBg: Color

    // Controls + inputs
    let controlBg: Color
    let controlBgHover: Color
    let glassBg: Color
    let glassBgStrong: Color
    let inputBg: Color
    let inputBorder: Color
    let inputBorderFocus: Color

    // Status
    let greenDot: Color
    let amberDot: Color
    let redDot: Color
    let blueDot: Color

    // Code + status backgrounds
    let codeBg: Color
    let warningBg: Color
    let warningText: Color
    let successBg: Color
    let successText: Color

    // Desktop wash gradient stops (radial background composed in PR 6's shell).
    let desktopWashTopLeft: Color
    let desktopWashBottomRight: Color
    let desktopWashBase: Color

    // Metrics
    let cornerRadius: CGFloat = 14
    let rowRadius: CGFloat = 10
    let groupHighlightTopOpacity: Double
    let groupShadowOpacity: Double
}

extension OMLXTheme {
    static let light = OMLXTheme(
        isDark: false,
        windowBg: .white,
        sidebarBg: .white(0.55),
        sidebarBorder: .black(0.06),
        contentBg: .white(0.75),
        toolbarBg: .white(0.55),
        toolbarBorder: .black(0.06),
        groupBg: .white(0.72),
        groupBorder: .black(0.06),
        rowSep: .black(0.06),
        separator: .black(0.08),
        text: .black(0.88),
        textSecondary: Color(rgb24: 0x3C3C43, opacity: 0.62),
        textTertiary: Color(rgb24: 0x3C3C43, opacity: 0.32),
        accent: Color(rgb24: 0x007AFF),
        accentSoft: Color(rgb24: 0x007AFF, opacity: 0.16),
        accentText: .white,
        selBg: .black(0.06),
        hoverBg: .black(0.04),
        controlBg: .white(0.65),
        controlBgHover: .white(0.85),
        glassBg: .white(0.55),
        glassBgStrong: .white(0.78),
        inputBg: .white(0.85),
        inputBorder: .black(0.10),
        inputBorderFocus: Color(rgb24: 0x007AFF),
        greenDot: Color(rgb24: 0x30D158),
        amberDot: Color(rgb24: 0xFF9500),
        redDot: Color(rgb24: 0xFF3B30),
        blueDot: Color(rgb24: 0x007AFF),
        codeBg: .black(0.05),
        warningBg: Color(rgb24: 0xFFF4E0),
        warningText: Color(rgb24: 0xB35900),
        successBg: Color(rgb24: 0xE3F7EA),
        successText: Color(rgb24: 0x1E7C3A),
        desktopWashTopLeft: Color(rgb24: 0xA88CFF, opacity: 0.30),
        desktopWashBottomRight: Color(rgb24: 0x78C8EB, opacity: 0.30),
        desktopWashBase: Color(rgb24: 0xF3F1EE),
        groupHighlightTopOpacity: 0.90,
        groupShadowOpacity: 0.06
    )

    static let dark = OMLXTheme(
        isDark: true,
        windowBg: Color(rgb24: 0x16161A),
        sidebarBg: Color(rgb24: 0x1C1C20, opacity: 0.55),
        sidebarBorder: .white(0.07),
        contentBg: Color(rgb24: 0x16161A, opacity: 0.72),
        toolbarBg: Color(rgb24: 0x1C1C20, opacity: 0.55),
        toolbarBorder: .white(0.07),
        groupBg: Color(rgb24: 0x303036, opacity: 0.55),
        groupBorder: .white(0.08),
        rowSep: .white(0.06),
        separator: .white(0.10),
        text: .white(0.94),
        textSecondary: Color(rgb24: 0xEBEBF5, opacity: 0.65),
        textTertiary: Color(rgb24: 0xEBEBF5, opacity: 0.35),
        accent: Color(rgb24: 0x0A84FF),
        accentSoft: Color(rgb24: 0x0A84FF, opacity: 0.22),
        accentText: .white,
        selBg: .white(0.10),
        hoverBg: .white(0.05),
        controlBg: .white(0.10),
        controlBgHover: .white(0.16),
        glassBg: .white(0.10),
        glassBgStrong: .white(0.14),
        inputBg: .black(0.30),
        inputBorder: .white(0.10),
        inputBorderFocus: Color(rgb24: 0x0A84FF),
        greenDot: Color(rgb24: 0x30D158),
        amberDot: Color(rgb24: 0xFF9F0A),
        redDot: Color(rgb24: 0xFF453A),
        blueDot: Color(rgb24: 0x0A84FF),
        codeBg: .white(0.07),
        warningBg: Color(rgb24: 0xFF9F0A, opacity: 0.18),
        warningText: Color(rgb24: 0xFFB340),
        successBg: Color(rgb24: 0x30D158, opacity: 0.18),
        successText: Color(rgb24: 0x34C759),
        desktopWashTopLeft: Color(rgb24: 0x6048DC, opacity: 0.22),
        desktopWashBottomRight: Color(rgb24: 0x28A0D2, opacity: 0.18),
        desktopWashBase: Color(rgb24: 0x0E0E12),
        groupHighlightTopOpacity: 0.08,
        groupShadowOpacity: 0.30
    )
}

// MARK: - Environment

private struct OMLXThemeKey: EnvironmentKey {
    static let defaultValue: OMLXTheme = .light
}

extension EnvironmentValues {
    var omlxTheme: OMLXTheme {
        get { self[OMLXThemeKey.self] }
        set { self[OMLXThemeKey.self] = newValue }
    }
}

private struct OMLXThemeBinder: ViewModifier {
    @Environment(\.colorScheme) private var scheme
    func body(content: Content) -> some View {
        content.environment(\.omlxTheme, scheme == .dark ? .dark : .light)
    }
}

extension View {
    /// Resolves `\.omlxTheme` from the current `\.colorScheme`. Apply once at
    /// the AppView shell (PR 6) so every descendant primitive reads the right
    /// palette without explicit prop-drilling.
    func omlxThemed() -> some View { modifier(OMLXThemeBinder()) }
}

// MARK: - Color helpers

extension Color {
    /// Solid black with the given opacity. Mirrors JSX `rgba(0,0,0,X)`.
    static func black(_ alpha: Double) -> Color {
        Color(.sRGB, white: 0, opacity: alpha)
    }

    /// Solid white with the given opacity. Mirrors JSX `rgba(255,255,255,X)`.
    static func white(_ alpha: Double) -> Color {
        Color(.sRGB, white: 1, opacity: alpha)
    }

    /// Construct from packed 24-bit RGB hex, e.g. `Color(rgb24: 0x007AFF)`.
    init(rgb24: UInt32, opacity: Double = 1.0) {
        let r = Double((rgb24 >> 16) & 0xFF) / 255
        let g = Double((rgb24 >> 8) & 0xFF) / 255
        let b = Double(rgb24 & 0xFF) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: opacity)
    }
}

// MARK: - Typography conveniences

extension Font {
    /// SF Pro Text-y body. macOS picks the right SF variant automatically by size.
    static func omlxText(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }
    /// Display weight for headlines (size auto-promotes on macOS at ≥ 20pt).
    static func omlxDisplay(_ size: CGFloat, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight)
    }
    /// Monospaced (SF Mono fallback chain).
    static func omlxMono(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}
