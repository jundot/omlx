// PR 6 — sidebar for the AppView shell.
//
// Mirrors omlx-components.jsx: SidebarSearch (143-167), SidebarItem (169-209),
// SidebarGroupLabel (211-220), Sidebar (222-245). Sections + ordering match
// VariantClassic (omlx-variants.jsx:12-26).
//
// Search is wired to a binding but does NOT filter the list — that lands in a
// later PR. Keeping it inert here mirrors the design canvas, which captures
// the user's typed string but never reduces the nav set.

import SwiftUI

// MARK: - Section model

enum AppSection: String, Hashable, CaseIterable, Identifiable, Sendable {
    case server, status, logs
    case models, downloads, integrations
    case security, about

    var id: String { rawValue }

    var title: String {
        switch self {
        case .server:       return "Server"
        case .status:       return "Status"
        case .logs:         return "Logs"
        case .models:       return "Models"
        case .downloads:    return "Downloads"
        case .integrations: return "Integrations"
        case .security:     return "Security"
        case .about:        return "About oMLX"
        }
    }

    /// Localized title key resolved against `Localizable.xcstrings`. Falls
    /// back to the source-language `title` when a translation is missing.
    var localizedTitle: LocalizedStringResource {
        LocalizedStringResource(stringLiteral: "sidebar.\(rawValue)")
    }

    var symbol: String {
        switch self {
        case .server:       return "server.rack"
        case .status:       return "gauge.with.dots.needle.50percent"
        case .logs:         return "scroll"
        case .models:       return "cube.transparent"
        case .downloads:    return "icloud.and.arrow.down"
        case .integrations: return "powerplug"
        case .security:     return "lock"
        case .about:        return "info.circle"
        }
    }

    var gradient: [Color] {
        switch self {
        case .server:       return SquircleGradient.server
        case .status:       return SquircleGradient.status
        case .logs:         return SquircleGradient.logs
        case .models:       return SquircleGradient.models
        case .downloads:    return SquircleGradient.downloads
        case .integrations: return SquircleGradient.integrations
        case .security:     return SquircleGradient.security
        case .about:        return SquircleGradient.about
        }
    }

    var group: SidebarGroup {
        switch self {
        case .server, .status, .logs:                return .server
        case .models, .downloads, .integrations:     return .ai
        case .security, .about:                      return .general
        }
    }
}

enum SidebarGroup: String, CaseIterable, Hashable, Sendable {
    case server  = "Server"
    case ai      = "AI"
    case general = "General"

    var sections: [AppSection] {
        AppSection.allCases.filter { $0.group == self }
    }

    var localizedTitle: LocalizedStringResource {
        LocalizedStringResource(stringLiteral: "sidebar.group.\(rawValue.lowercased())")
    }
}

// MARK: - Sidebar

struct Sidebar: View {
    @Binding var selection: AppSection
    @Binding var search: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            SidebarSearchField(text: $search)
                .padding(.top, 6)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(SidebarGroup.allCases, id: \.self) { group in
                        SidebarGroupLabel(title: group.localizedTitle)
                        ForEach(group.sections) { section in
                            SidebarItem(
                                section: section,
                                isSelected: selection == section,
                                onTap: { selection = section }
                            )
                        }
                    }
                }
                .padding(.bottom, 10)
            }
        }
        .background(theme.sidebarBg)
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(theme.sidebarBorder)
                .frame(width: 0.5)
        }
    }
}

// MARK: - Search field

private struct SidebarSearchField: View {
    @Binding var text: String

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11, weight: .regular))
                .foregroundStyle(theme.textTertiary)

            TextField("Search", text: $text)
                .textFieldStyle(.plain)
                .font(.omlxText(12))
                .foregroundStyle(theme.text)
        }
        .padding(.horizontal, 6)
        .frame(height: 22)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(theme.isDark ? Color.black.opacity(0.25) : Color.black.opacity(0.06))
        )
        .padding(.horizontal, 14)
        .padding(.bottom, 6)
    }
}

// MARK: - Group label

private struct SidebarGroupLabel: View {
    let title: LocalizedStringResource

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Text(title)
            .font(.omlxText(11, weight: .medium))
            .foregroundStyle(theme.textSecondary)
            .padding(.leading, 16)
            .padding(.top, 12)
            .padding(.bottom, 4)
    }
}

// MARK: - Row

private struct SidebarItem: View {
    let section: AppSection
    let isSelected: Bool
    let onTap: () -> Void

    @State private var isHovering = false
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 9) {
                Squircle(systemSymbol: section.symbol, size: 20, gradient: section.gradient)
                Text(section.localizedTitle)
                    .font(.omlxText(13, weight: isSelected ? .medium : .regular))
                    .foregroundStyle(theme.text)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background(rowBackground)
            .overlay(rowBorder)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 8)
        .padding(.vertical, 1)
        .onHover { isHovering = $0 }
    }

    private var rowBackground: some View {
        RoundedRectangle(cornerRadius: 9, style: .continuous)
            .fill(
                isSelected
                    ? (theme.isDark ? Color.white.opacity(0.12) : Color.white.opacity(0.85))
                    : (isHovering ? theme.hoverBg : Color.clear)
            )
    }

    @ViewBuilder
    private var rowBorder: some View {
        if isSelected {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(
                    theme.isDark ? Color.white.opacity(0.18) : Color.white.opacity(0.95),
                    lineWidth: 0.5
                )
        }
    }
}

// MARK: - Preview

#Preview("Sidebar — light") {
    SidebarPreview()
        .preferredColorScheme(.light)
}

#Preview("Sidebar — dark") {
    SidebarPreview()
        .preferredColorScheme(.dark)
}

private struct SidebarPreview: View {
    @State private var selection: AppSection = .server
    @State private var search: String = ""

    var body: some View {
        Sidebar(selection: $selection, search: $search)
            .frame(width: 220, height: 600)
            .omlxThemed()
    }
}
