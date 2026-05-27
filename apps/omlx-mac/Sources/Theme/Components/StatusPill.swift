// PR 3 — colored pill indicating server / model state.
// Mirrors JSX `StatusPill` but accepts a model-state enum so screens
// can render any lifecycle stage uniformly.

import SwiftUI

struct StatusPill: View {
    enum Status: Equatable, Sendable {
        case running
        case starting
        case stopping
        case stopped
        case error
        /// Free-form variant: caller supplies the dot color and label.
        case custom(color: Color, label: String, fillBg: Bool)
    }

    let status: Status
    var compact: Bool = false

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        let cfg = config
        HStack(spacing: 5) {
            Circle()
                .fill(cfg.dot)
                .frame(width: 6, height: 6)
            if !compact {
                Text(cfg.label)
                    .font(.omlxText(11, weight: .medium))
                    .foregroundStyle(cfg.text)
            }
        }
        .padding(.horizontal, compact ? 6 : 8)
        .padding(.vertical, 2.5)
        .background(
            Capsule().fill(cfg.bg)
        )
    }

    private var config: (dot: Color, text: Color, bg: Color, label: String) {
        switch status {
        case .running:
            return (theme.greenDot, theme.successText, theme.successBg, "Running")
        case .starting:
            return (theme.blueDot, theme.blueDot, theme.accentSoft, "Starting")
        case .stopping:
            return (theme.amberDot, theme.warningText, theme.warningBg, "Stopping")
        case .stopped:
            return (theme.textTertiary, theme.textSecondary,
                    theme.controlBg, "Stopped")
        case .error:
            return (theme.redDot, theme.redDot, theme.warningBg, "Error")
        case .custom(let color, let label, let fillBg):
            return (color, color, fillBg ? color.opacity(0.16) : .clear, label)
        }
    }
}

#Preview("StatusPill") {
    VStack(alignment: .leading, spacing: 10) {
        StatusPill(status: .running)
        StatusPill(status: .starting)
        StatusPill(status: .stopping)
        StatusPill(status: .stopped)
        StatusPill(status: .error)
        StatusPill(status: .custom(
            color: Color(rgb24: 0x5E5CE6),
            label: "Prefilling 42%",
            fillBg: true
        ))
    }
    .padding(24)
    .omlxThemed()
}
