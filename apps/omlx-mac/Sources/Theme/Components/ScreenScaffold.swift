// Top-of-screen header + inline status banner shared by the bench /
// quantization screens. Each screen's scaffolding was originally a private
// `HeaderSection` + `BannerSection` pair; centralising them keeps the
// padding/typography in sync.

import SwiftUI

/// Three-line title block rendered at the top of a screen: an uppercase
/// eyebrow, a 20pt title, and an 11.5pt subtitle paragraph.
struct ScreenHeader: View {
    let eyebrow: String
    let title: String
    let subtitle: String

    @Environment(\.omlxTheme) private var theme

    init(eyebrow: String, title: String, subtitle: String) {
        self.eyebrow = eyebrow
        self.title = title
        self.subtitle = subtitle
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(eyebrow)
                .font(.omlxText(11, weight: .semibold))
                .foregroundStyle(theme.textSecondary)
                .textCase(.uppercase)
                .kerning(0.6)
            Text(title)
                .font(.omlxText(20, weight: .semibold))
                .foregroundStyle(theme.text)
            Text(subtitle)
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 14)
        .padding(.top, 18)
        .padding(.bottom, 10)
    }
}

/// Inline error/success message strip rendered just below `ScreenHeader`.
/// Either field can be nil/empty; the view collapses to nothing when both
/// are absent, matching the original per-screen behaviour.
struct MessageBanner: View {
    let error: String?
    let success: String?

    @Environment(\.omlxTheme) private var theme

    init(error: String? = nil, success: String? = nil) {
        self.error = error
        self.success = success
    }

    var body: some View {
        let hasError   = !(error   ?? "").isEmpty
        let hasSuccess = !(success ?? "").isEmpty
        VStack(alignment: .leading, spacing: 6) {
            if hasError, let error {
                row(icon: "exclamationmark.triangle.fill", text: error, color: theme.redDot)
            }
            if hasSuccess, let success {
                row(icon: "checkmark.circle.fill", text: success, color: theme.greenDot)
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, (hasError || hasSuccess) ? 6 : 0)
    }

    private func row(icon: String, text: String, color: Color) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .font(.system(size: 11))
                .padding(.top, 1)
            Text(text)
                .font(.omlxText(11.5))
                .foregroundStyle(theme.text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

/// Single-line tertiary hint with a leading info icon. Used wherever a
/// screen wants to drop an inline tip below a section (Welcome's
/// Storage/API-key reminders, ServerScreen's "restart to apply" note,
/// QuantizationScreen's empty-state pointer to Downloads).
struct HintLine: View {
    let text: String
    var icon: String = "info.circle"

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundStyle(theme.textTertiary)
            Text(text)
                .font(.omlxText(11))
                .foregroundStyle(theme.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Bottom action bar shared by screens with a trailing primary action
/// (Apply and friends): optional hint on the left that wraps under width
/// pressure, action buttons flush right, optional error line underneath.
/// One component so the button/note relationship is identical everywhere.
struct FooterBar<Actions: View>: View {
    var hint: String? = nil
    var error: String? = nil
    @ViewBuilder var actions: () -> Actions

    @Environment(\.omlxTheme) private var theme

    private var hasHint: Bool { !(hint ?? "").isEmpty }
    private var hasError: Bool { !(error ?? "").isEmpty }
    private var hasActions: Bool { Actions.self != EmptyView.self }

    var body: some View {
        if hasHint || hasError || hasActions {
            VStack(alignment: .leading, spacing: 6) {
                if hasHint || hasActions {
                    HStack(alignment: .center, spacing: 12) {
                        if let hint, hasHint {
                            HintLine(text: hint)
                        }
                        Spacer(minLength: 0)
                        actions()
                    }
                }
                if let error, hasError {
                    Text(error)
                        .font(.omlxText(11))
                        .foregroundStyle(theme.redDot)
                }
            }
            .padding(.horizontal, 14)
            .padding(.top, 8)
        }
    }
}

extension FooterBar where Actions == EmptyView {
    /// Error-only (or hint-only) footer — the standard trailing status line
    /// under a screen. Renders nothing while there is nothing to show.
    init(hint: String? = nil, error: String? = nil) {
        self.init(hint: hint, error: error) { EmptyView() }
    }
}
