// PR 3 — translucent surface modifier.
//
// macOS 15 baseline: SwiftUI's `.regularMaterial` / `.ultraThinMaterial` ship
// approximation good enough for non-shell surfaces. macOS 26 (Tahoe) brings
// genuine liquid-glass APIs (e.g. `.glassEffect()`); the exact signature is
// confirmed at PR 6 when the SDK header is in front of us. The TODO branch is
// already gated on `if #available(macOS 26.0, *)` so the rewrite lands today
// without that confirmation.

import SwiftUI

enum GlassStrength {
    /// Sidebar / toolbar / list-group surfaces.
    case regular
    /// Hero cards, dashboards (the design's `glassBgStrong`).
    case strong
}

private struct AppGlassModifier: ViewModifier {
    let strength: GlassStrength

    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            // TODO(PR 6): swap in `.glassEffect()` once we confirm the SDK API.
            // The fallback below stays as a safety net.
            content.background(material)
        } else {
            content.background(material)
        }
    }

    private var material: Material {
        switch strength {
        case .regular: return .regularMaterial
        case .strong:  return .thickMaterial
        }
    }
}

extension View {
    /// Background material that approximates Tahoe liquid glass. Use for
    /// sidebar, toolbar, and group surfaces. Hero cards use `.strong`.
    func appGlass(_ strength: GlassStrength = .regular) -> some View {
        modifier(AppGlassModifier(strength: strength))
    }
}

// MARK: - Desktop wash

/// The Tahoe desktop background: two soft radial accents over a flat base.
/// Apply at the AppView shell (PR 6) below the sidebar+content split.
struct DesktopWash: View {
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        ZStack {
            theme.desktopWashBase
            // Top-left accent (radial approximation of JSX 120% × 80% ellipse).
            RadialGradient(
                colors: [theme.desktopWashTopLeft, .clear],
                center: .topLeading,
                startRadius: 0,
                endRadius: 720
            )
            // Bottom-right accent.
            RadialGradient(
                colors: [theme.desktopWashBottomRight, .clear],
                center: .bottomTrailing,
                startRadius: 0,
                endRadius: 680
            )
        }
        .ignoresSafeArea()
    }
}

#Preview("Glass on desktop wash") {
    ZStack {
        DesktopWash()
        VStack(spacing: 16) {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(.clear)
                .frame(width: 320, height: 80)
                .appGlass(.regular)
                .overlay(Text("regular").foregroundStyle(.primary))
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(.clear)
                .frame(width: 320, height: 80)
                .appGlass(.strong)
                .overlay(Text("strong").foregroundStyle(.primary))
        }
    }
    .frame(width: 480, height: 280)
    .omlxThemed()
}
