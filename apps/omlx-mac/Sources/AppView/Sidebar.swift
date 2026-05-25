// PR 6 — sidebar for the AppView shell.
//
// Mirrors omlx-components.jsx: SidebarItem (169-209), SidebarGroupLabel
// (211-220), Sidebar (222-245). Sections + ordering match VariantClassic
// (omlx-variants.jsx:12-26).
//
// The design canvas had a search field above the nav list; we shipped it
// inert in PR 6, then ripped it out — without filtering wired up it was
// just visual noise.

import SwiftUI

// MARK: - Section model

enum AppSection: String, Hashable, CaseIterable, Identifiable, Sendable {
    case server, status, network, performance, logs
    case models, downloads, integrations, quantization
    case throughputBench, accuracyBench
    case security, about

    var id: String { rawValue }

    var title: String {
        switch self {
        case .server:
            return String(localized: "sidebar.server",
                          defaultValue: "Server",
                          comment: "Sidebar row label / navigation title for the Server section")
        case .network:
            return String(localized: "sidebar.network",
                          defaultValue: "Network",
                          comment: "Sidebar row label / navigation title for the Network section")
        case .performance:
            return String(localized: "sidebar.performance",
                          defaultValue: "Performance",
                          comment: "Sidebar row label / navigation title for the Performance section")
        case .status:
            return String(localized: "sidebar.status",
                          defaultValue: "Status",
                          comment: "Sidebar row label / navigation title for the Status section")
        case .logs:
            return String(localized: "sidebar.logs",
                          defaultValue: "Logs",
                          comment: "Sidebar row label / navigation title for the Logs section")
        case .models:
            return String(localized: "sidebar.models",
                          defaultValue: "Models",
                          comment: "Sidebar row label / navigation title for the Models section")
        case .downloads:
            return String(localized: "sidebar.downloads",
                          defaultValue: "Downloads",
                          comment: "Sidebar row label / navigation title for the Downloads section")
        case .integrations:
            return String(localized: "sidebar.integrations",
                          defaultValue: "Integrations",
                          comment: "Sidebar row label / navigation title for the Integrations section")
        case .quantization:
            return String(localized: "sidebar.quantization",
                          defaultValue: "Quantization",
                          comment: "Sidebar row label / navigation title for the Quantization section")
        case .throughputBench:
            return String(localized: "sidebar.throughputBench",
                          defaultValue: "Throughput",
                          comment: "Sidebar row label / navigation title for the Throughput benchmark section")
        case .accuracyBench:
            return String(localized: "sidebar.accuracyBench",
                          defaultValue: "Accuracy",
                          comment: "Sidebar row label / navigation title for the Accuracy benchmark section")
        case .security:
            return String(localized: "sidebar.security",
                          defaultValue: "Security",
                          comment: "Sidebar row label / navigation title for the Security section")
        case .about:
            return String(localized: "sidebar.about",
                          defaultValue: "About oMLX",
                          comment: "Sidebar row label / navigation title for the About section")
        }
    }

    /// Localized title key resolved against `Localizable.xcstrings`. Falls
    /// back to the source-language `title` when a translation is missing.
    var localizedTitle: LocalizedStringResource {
        LocalizedStringResource(stringLiteral: "sidebar.\(rawValue)")
    }

    var symbol: String {
        switch self {
        case .server:          return "server.rack"
        case .network:         return "network"
        case .performance:     return "bolt.fill"
        case .status:          return "gauge.with.dots.needle.50percent"
        case .logs:            return "scroll"
        case .models:          return "cube.transparent"
        case .downloads:       return "icloud.and.arrow.down"
        case .integrations:    return "powerplug"
        case .quantization:    return "sparkles"
        case .throughputBench: return "speedometer"
        case .accuracyBench:   return "target"
        case .security:        return "lock"
        case .about:           return "info.circle"
        }
    }

    var gradient: [Color] {
        switch self {
        case .server:          return SquircleGradient.server
        case .network:         return SquircleGradient.network
        case .performance:     return SquircleGradient.performance
        case .status:          return SquircleGradient.status
        case .logs:            return SquircleGradient.logs
        case .models:          return SquircleGradient.models
        case .downloads:       return SquircleGradient.downloads
        case .integrations:    return SquircleGradient.integrations
        case .quantization:    return SquircleGradient.quantization
        case .throughputBench: return SquircleGradient.throughputBench
        case .accuracyBench:   return SquircleGradient.accuracyBench
        case .security:        return SquircleGradient.security
        case .about:           return SquircleGradient.about
        }
    }

    var group: SidebarGroup {
        switch self {
        case .server, .network, .performance, .status, .logs:      return .server
        case .models, .downloads, .integrations, .quantization:    return .models
        case .throughputBench, .accuracyBench:                     return .bench
        case .security, .about:                                    return .general
        }
    }

    /// True when the screen wants to fill the content area vertically rather
    /// than ride inside the default outer scroll view. The Logs pane uses
    /// this so its monospace text block grows with the window.
    var fillsContentArea: Bool {
        switch self {
        case .logs: return true
        default:    return false
        }
    }
}

enum SidebarGroup: String, CaseIterable, Hashable, Sendable {
    case server  = "Server"
    case models  = "Models"
    case bench   = "Benchmark"
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

    @Environment(\.omlxTheme) private var theme

    var body: some View {
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
            .padding(.top, 6)
            .padding(.bottom, 10)
        }
        // Translucent glass surface on macOS 26 (Tahoe liquid glass) with a
        // material approximation fallback on macOS 15. Painting theme.sidebarBg
        // first preserves the existing tint as the base layer behind the glass.
        .background(theme.sidebarBg)
        .appGlass(.regular)
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(theme.sidebarBorder)
                .frame(width: 0.5)
        }
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

    var body: some View {
        Sidebar(selection: $selection)
            .frame(width: 220, height: 600)
            .omlxThemed()
    }
}
