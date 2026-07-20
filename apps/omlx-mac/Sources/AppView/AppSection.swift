// Top-level sections rendered by AppView's settings sidebar. Each case becomes
// one selectable row, with `title` driving the visible label (localized with a
// `defaultValue` fallback so the catalog isn't load-bearing) and `symbol`
// driving the SF Symbol on the row.
//
// Section groupings (Server / Models / Benchmark / General) live inline in
// AppView; the visual grouping is purely a layout decision in the shell.

import SwiftUI

enum AppSection: String, Hashable, CaseIterable, Identifiable, Sendable {
    case server, status, appearance, network, performance, logs
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
        case .appearance:
            return String(localized: "sidebar.appearance",
                          defaultValue: "Appearance",
                          comment: "Sidebar row label / navigation title for the Appearance section")
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

    var symbol: String {
        switch self {
        case .server:          return "server.rack"
        case .network:         return "network"
        case .performance:     return "bolt.fill"
        case .status:          return "gauge.with.dots.needle.50percent"
        case .appearance:      return "paintbrush"
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
