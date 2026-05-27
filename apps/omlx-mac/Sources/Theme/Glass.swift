// Translucent surface modifier. macOS 26 (Tahoe) ships a real liquid-glass
// API via `View.glassEffect(_:)` in SwiftUICore; earlier macOS versions
// (15.0 deployment-target floor) fall back to the closest material we have.
// The `Glass` type only exposes `.regular`/`.clear`/`.identity` — there's
// no thick variant — so `GlassStrength.strong` maps to `Glass.regular`
// on macOS 26 while the fallback still uses `.thickMaterial` for the same
// approximation we shipped pre-Tahoe.

import SwiftUI

enum GlassStrength {
    /// Sidebar / toolbar / list-group surfaces.
    case regular
    /// Hero cards, dashboards (the design's `glassBgStrong`).
    case strong
}

private struct AppGlassModifier<S: Shape>: ViewModifier {
    let strength: GlassStrength
    let shape: S
    let tint: Color?

    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            // Bake the tint into the glass material so the surface composites
            // in a single pass — separate `.background(tint)` + `.glassEffect`
            // doubles the per-frame work and shows up as stutter at the end of
            // ancestor animations (e.g. NavigationSplitView sidebar slide).
            // `.glassEffectTransition(.identity)` similarly suppresses the
            // glass-materialize animation that would otherwise compound on
            // top of the geometry change.
            let glass: Glass = tint.map { Glass.regular.tint($0) } ?? .regular
            content
                .glassEffect(glass, in: shape)
                .glassEffectTransition(.identity)
        } else {
            // macOS 15 fallback path. No native tinted-material API here, so
            // composite the tint and material separately. This path isn't on
            // the perf-critical track for the macOS 26 sidebar stutter.
            if let tint {
                content
                    .background(material, in: shape)
                    .background(tint, in: shape)
            } else {
                content.background(material, in: shape)
            }
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
    /// Default shape is `Rectangle()`; pass an explicit shape for rounded
    /// surfaces (e.g. `RoundedRectangle(cornerRadius: 12)` for hero cards).
    /// `tint:` bakes a color into the glass material — preferred over
    /// stacking a separate `.background(_:)` for perf reasons.
    func appGlass(_ strength: GlassStrength = .regular, tint: Color? = nil) -> some View {
        modifier(AppGlassModifier(strength: strength, shape: Rectangle(), tint: tint))
    }

    func appGlass<S: Shape>(_ strength: GlassStrength = .regular,
                            in shape: S,
                            tint: Color? = nil) -> some View {
        modifier(AppGlassModifier(strength: strength, shape: shape, tint: tint))
    }
}

// MARK: - Desktop wash

/// The Tahoe desktop background: two soft radial accents over a flat base.
/// Apply at the AppView shell below the TabView content.
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
