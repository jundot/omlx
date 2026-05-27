// PR 3 — gradient rounded-square icon, used for sidebar items and hero cards.
//
// Mirrors the JSX `Squircle` (omlx-icons.jsx:74-88): continuous-corner radius
// at 27% of size, top-down linear gradient fill, inner top highlight + outer
// bottom shadow approximating the iOS app-icon look.

import SwiftUI

struct Squircle<Content: View>: View {
    let gradient: [Color]
    let size: CGFloat
    let content: () -> Content

    init(
        gradient: [Color],
        size: CGFloat = 22,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.gradient = gradient
        self.size = size
        self.content = content
    }

    var body: some View {
        let radius = size * 0.27
        ZStack {
            RoundedRectangle(cornerRadius: radius, style: .continuous)
                .fill(LinearGradient(colors: gradient, startPoint: .top, endPoint: .bottom))
            RoundedRectangle(cornerRadius: radius, style: .continuous)
                .strokeBorder(Color.white.opacity(0.18), lineWidth: 0.5)
            content()
                .foregroundStyle(.white)
        }
        .frame(width: size, height: size)
        .shadow(color: .black.opacity(0.15), radius: 0.5, x: 0, y: 0.5)
    }
}

extension Squircle where Content == AnyView {
    /// Convenience: SF Symbol inside a Squircle. Symbol weight scales with size.
    init(systemSymbol: String, size: CGFloat = 22, gradient: [Color]) {
        self.init(gradient: gradient, size: size) {
            AnyView(
                Image(systemName: systemSymbol)
                    .font(.system(size: size * 0.55, weight: .medium))
            )
        }
    }
}

// MARK: - Sidebar gradient palette

/// Top-to-bottom gradient stops for sidebar / hero squircle icons. Hex values
/// match `omlx-variants.jsx:14-26` (Variant Classic NAV).
enum SquircleGradient {
    static let server: [Color] = [Color(rgb24: 0x5A5A5E), Color(rgb24: 0x2C2C2E)]
    /// Network squircle — cyan → indigo. Distinct from `status` (green) and
    /// `throughputBench` (teal duo) so the Server group icons read at a glance.
    static let network: [Color] = [Color(rgb24: 0x5AC8FA), Color(rgb24: 0x5856D6)]
    /// Performance squircle — yellow → orange, evoking speed/heat. Distinct
    /// from `integrations` (orange→red) and `accuracyBench` (orange→red).
    static let performance: [Color] = [Color(rgb24: 0xFFCC00), Color(rgb24: 0xFF9500)]
    static let status: [Color] = [Color(rgb24: 0x34C759), Color(rgb24: 0x1A8A3A)]
    static let logs: [Color]   = [Color(rgb24: 0x8E8E93), Color(rgb24: 0x5A5A5E)]

    static let models: [Color]       = [Color(rgb24: 0xAF52DE), Color(rgb24: 0x5E5CE6)]
    static let downloads: [Color]    = [Color(rgb24: 0x0A84FF), Color(rgb24: 0x5E5CE6)]
    static let integrations: [Color] = [Color(rgb24: 0xFF9F0A), Color(rgb24: 0xFF453A)]
    static let quantization: [Color] = [Color(rgb24: 0xFF2D55), Color(rgb24: 0xAF52DE)]

    /// Bench group — speedometer (blue → teal) for Throughput, target
    /// (orange → red) for Accuracy. Distinct from the Status gradient so
    /// the two clusters don't visually collide in the sidebar.
    static let throughputBench: [Color] = [Color(rgb24: 0x32ADE6), Color(rgb24: 0x30B0C7)]
    static let accuracyBench: [Color]   = [Color(rgb24: 0xFF9500), Color(rgb24: 0xFF453A)]

    static let security: [Color] = server          // same dark gunmetal as Server
    static let about: [Color]    = logs            // same light gray as Logs

    /// The "update available" Squircle on Status → Updates section (design v2).
    static let update: [Color] = [Color(rgb24: 0x34C759), Color(rgb24: 0x30B350)]
}

#Preview("Squircle gallery") {
    HStack(spacing: 12) {
        Squircle(systemSymbol: "server.rack", gradient: SquircleGradient.server)
        Squircle(systemSymbol: "gauge", gradient: SquircleGradient.status)
        Squircle(systemSymbol: "cube.transparent", size: 32, gradient: SquircleGradient.models)
        Squircle(systemSymbol: "icloud.and.arrow.down",
                 size: 44,
                 gradient: SquircleGradient.downloads)
        Squircle(systemSymbol: "arrow.down.circle.fill",
                 size: 56,
                 gradient: SquircleGradient.update)
    }
    .padding(40)
}
