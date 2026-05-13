// Visual components used by the Profiles tab and the Server-defaults block
// on the Server screen. The shapes mirror omlx-screens.jsx:626-1191:
//
//   • ProfileGroup        — chip row labelled by scope
//   • ActiveProfileBanner — top-of-tab summary of the model's attach state
//   • SaveAsPopover       — inline name + scope toggle for "Save as new"
//   • ProfileDetailCard   — visual summary of one profile's settings
//
// The view models live elsewhere — this file is render-only.

import SwiftUI

// MARK: - Scope colors / labels

/// Per-scope visual treatment. Lifted from omlx-screens.jsx:878-884
/// (SCOPE_META) so the chip dots and badges read the same as the canvas.
enum ProfileScopeMeta {
    static func color(_ scope: ProfileScope, theme: OMLXTheme) -> Color {
        switch scope {
        case .preset: return theme.amberDot
        case .global: return Color(rgb24: 0xAF52DE)
        case .model:  return theme.blueDot
        }
    }

    static func label(_ scope: ProfileScope) -> String {
        switch scope {
        case .preset: return "Preset"
        case .global: return "Global"
        case .model:  return "Model"
        }
    }
}

// MARK: - ProfileGroup

/// One chip row, scoped to preset / global / model. The active chip
/// (when its scope matches `activeName`) renders in scope color; the
/// "based-on of the working profile" gets a dashed scope-color border;
/// preview-selected gets a 1px primary-text border.
struct ProfileGroup: View {
    let scope: ProfileScope
    let label: String
    let names: [String]
    /// Name of the currently-active profile in this scope (or nil if the
    /// active profile lives in a different scope).
    let activeName: String?
    /// Name of the profile the working profile forked from (or nil).
    let basedOnName: String?
    /// Name of the profile currently being previewed in the detail card.
    let previewName: String?
    let canSaveCurrent: Bool

    let onSelect: (String) -> Void
    let onSaveCurrent: () -> Void

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            header
            chipRow
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 12)
    }

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(ProfileScopeMeta.color(scope, theme: theme))
                .frame(width: 6, height: 6)
            Text(label)
                .font(.omlxText(10.5, weight: .heavy))
                .kerning(0.7)
                .textCase(.uppercase)
                .foregroundStyle(theme.textSecondary)
            Spacer(minLength: 8)
            if canSaveCurrent {
                Button {
                    onSaveCurrent()
                } label: {
                    Label("Save current as new", systemImage: "plus")
                        .labelStyle(.titleAndIcon)
                        .font(.omlxText(11, weight: .medium))
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .overlay(
                    Capsule().strokeBorder(
                        theme.inputBorder, style: StrokeStyle(lineWidth: 1, dash: [3, 2])
                    )
                )
                .foregroundStyle(theme.textSecondary)
            }
        }
        .padding(.horizontal, 2)
    }

    @ViewBuilder
    private var chipRow: some View {
        let metaColor = ProfileScopeMeta.color(scope, theme: theme)
        Group {
            if names.isEmpty {
                Text("No \(ProfileScopeMeta.label(scope).lowercased()) profiles yet.")
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textTertiary)
                    .padding(.vertical, 10)
                    .padding(.horizontal, 12)
            } else {
                FlowLayout(spacing: 6) {
                    ForEach(names, id: \.self) { name in
                        chip(name: name, metaColor: metaColor)
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 10)
            }
        }
        // The chip row needs to span its container's width so FlowLayout
        // receives a definite width proposal — without this the layout's
        // two passes (sizeThatFits with .unspecified, then placeSubviews
        // with the parent's bounds) disagree on wrap count and chips
        // overflow into the next group's header. `.background` (vs the
        // old ZStack) keeps the rounded surface flush to the same frame.
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(theme.groupBg)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
    }

    @ViewBuilder
    private func chip(name: String, metaColor: Color) -> some View {
        let isActive = (activeName == name)
        let isBase = (basedOnName == name)
        let isPreviewed = (previewName == name)
        HStack(spacing: 5) {
            if isActive {
                Image(systemName: "checkmark")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.white)
            }
            Text(name)
                .font(.omlxText(12, weight: .medium))
                .foregroundStyle(isActive ? .white : theme.text)
        }
        .padding(.horizontal, 11)
        .frame(height: 26)
        .background(
            Capsule().fill(
                isActive
                    ? metaColor
                    : (isPreviewed ? theme.selBg : theme.codeBg)
            )
        )
        .overlay(
            Capsule().strokeBorder(
                isActive ? Color.clear
                    : (isPreviewed ? theme.text
                        : (isBase ? metaColor.opacity(0.6) : theme.inputBorder)),
                style: StrokeStyle(
                    lineWidth: isPreviewed ? 1 : 0.5,
                    dash: (isBase && !isActive && !isPreviewed) ? [3, 2] : []
                )
            )
        )
        .contentShape(Capsule())
        .onTapGesture { onSelect(name) }
    }
}

// MARK: - ActiveProfileBanner

/// Banner that sits above the chip groups (full) or above Basic/Advanced
/// (slim) summarizing what's currently attached + what unsaved work
/// exists. The state-machine logic lives in the VM; this view just
/// renders one of three shapes (working / named / defaults).
struct ActiveProfileBanner: View {
    let state: ActiveProfileState
    let isSlim: Bool

    /// nil when the corresponding action isn't relevant for this state.
    let onUpdateBasedOn: (() -> Void)?
    let onSaveAsNew: (() -> Void)?
    let onRevert: (() -> Void)?

    @Environment(\.omlxTheme) private var theme

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(dotColor)
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: 2) {
                Text(titleText)
                    .font(.omlxText(13.5, weight: .semibold))
                    .foregroundStyle(theme.text)
                subtitleView
            }
            Spacer(minLength: 8)
            actions
        }
        .padding(.horizontal, isSlim ? 12 : 14)
        .padding(.vertical, isSlim ? 10 : 12)
        .background(bannerBackground)
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(bannerBorder, lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .padding(.horizontal, 14)
        .padding(.bottom, 12)
    }

    private var dotColor: Color {
        switch state {
        case .working:
            // Working / unsaved → amber (matches HTML mock and the
            // "you have unsaved edits" affordance pattern).
            return theme.amberDot
        case .named(let scope, _):
            return ProfileScopeMeta.color(scope, theme: theme)
        case .defaults:
            return theme.textTertiary
        }
    }

    private var titleText: String {
        switch state {
        case .working:                   return "Working profile"
        case .named(_, let name):        return name
        case .defaults:                  return "No profile"
        }
    }

    @ViewBuilder
    private var subtitleView: some View {
        switch state {
        case .working(let basedOn):
            if let basedOn {
                Text("Unsaved · based on \(basedOn.name) (\(ProfileScopeMeta.label(basedOn.scope)))")
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            } else {
                Text("Unsaved · based on server defaults")
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            }
        case .named(let scope, _):
            Text("\(ProfileScopeMeta.label(scope)) profile · active on this model")
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
        case .defaults:
            Text("Using server defaults · edit any field to start a working profile")
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
        }
    }

    @ViewBuilder
    private var actions: some View {
        HStack(spacing: 6) {
            switch state {
            case .working(let basedOn):
                if let basedOn, basedOn.scope != .preset, let onUpdateBasedOn {
                    Button("Update \(basedOn.name)") { onUpdateBasedOn() }
                        .buttonStyle(.omlx(.normal, size: .small))
                }
                if let onSaveAsNew {
                    Button("Save as new") { onSaveAsNew() }
                        .buttonStyle(.omlx(.primary, size: .small))
                }
                if let onRevert {
                    Button(basedOn == nil ? "Discard" : "Revert") { onRevert() }
                        .buttonStyle(.omlx(.plain, size: .small))
                }
            case .named, .defaults:
                EmptyView()
            }
        }
    }

    @ViewBuilder
    private var bannerBackground: some View {
        switch state {
        case .working:
            theme.amberDot.opacity(theme.isDark ? 0.08 : 0.07)
        case .named, .defaults:
            theme.groupBg
        }
    }

    private var bannerBorder: Color {
        switch state {
        case .working:           return theme.amberDot.opacity(theme.isDark ? 0.28 : 0.35)
        case .named, .defaults:  return theme.groupBorder
        }
    }
}

// MARK: - SaveAsPopover

struct SaveAsPopover: View {
    @Binding var name: String
    @Binding var scope: ProfileScope
    /// Save as new only supports global / model; preset is read-only.
    let onCommit: () -> Void
    let onCancel: () -> Void

    @Environment(\.omlxTheme) private var theme
    @FocusState private var nameFocused: Bool

    var body: some View {
        HStack(spacing: 10) {
            Text("Save current profile as")
                .font(.omlxText(12, weight: .medium))
                .foregroundStyle(theme.textSecondary)
            Segmented(
                selection: $scope,
                options: [(.global, "Global"), (.model, "Model")]
            )
            .frame(width: 140)
            TextInput(text: $name, placeholder: "profile-name", mono: true)
                .frame(maxWidth: .infinity)
                .focused($nameFocused)
                .onSubmit { onCommit() }
            Button("Cancel") { onCancel() }
                .buttonStyle(.omlx(.normal, size: .small))
            Button("Save") { onCommit() }
                .buttonStyle(.omlx(.primary, size: .small))
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(12)
        .background(theme.groupBg)
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .padding(.horizontal, 14)
        .padding(.bottom, 12)
        .onAppear { nameFocused = true }
    }
}

// MARK: - ProfileDetailCard

/// Compact visual summary of a profile's settings — sampling meters,
/// capacity stats, penalty bars, behavior flag chips, and aliases.
/// Renders below the chip groups (preview-on-click) and at the bottom
/// of the Profiles tab as the read-only Server Defaults card.
struct ProfileDetailCard: View {
    let name: String
    let scope: ProfileScope?  // nil → defaults card (no scope dot)
    let settings: [String: AnyCodable]
    let isActive: Bool
    let isWorking: Bool
    let basedOn: ActiveProfileState.NamedProfileRef?
    /// True when the card is being shown for the chip the working
    /// profile forked from — dashed border treatment in the chip group,
    /// "Base of working" badge here.
    let isWorkingBase: Bool
    /// Compact mode shrinks padding; used for the Server Defaults card.
    let compact: Bool
    let hasWorking: Bool

    // nil means the action isn't relevant for this card.
    var onApply: (() -> Void)? = nil
    var onUpdateFromWorking: (() -> Void)? = nil
    var onDelete: (() -> Void)? = nil
    var onClosePreview: (() -> Void)? = nil

    @Environment(\.omlxTheme) private var theme

    private var scopeColor: Color {
        guard let scope else { return theme.textTertiary }
        return ProfileScopeMeta.color(scope, theme: theme)
    }

    private var showApply: Bool { onApply != nil && !isActive }
    private var showUpdate: Bool { onUpdateFromWorking != nil && hasWorking }
    private var showDelete: Bool {
        onDelete != nil && scope != .preset && scope != nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            sections
        }
        .padding(compact ? 12 : 14)
        .background(theme.groupBg)
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(theme.groupBorder, lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .padding(.horizontal, 14)
        .padding(.bottom, 14)
    }

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 12) {
            iconBox
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(name)
                        .font(.omlxText(16, weight: .semibold))
                        .foregroundStyle(theme.text)
                    if isActive && !isWorking {
                        badge(text: "ACTIVE", fg: .white, bg: theme.greenDot)
                    }
                    if isWorking {
                        badge(text: "UNSAVED", fg: Color(rgb24: 0x1A1407), bg: theme.amberDot)
                    }
                    if isWorkingBase && !isWorking {
                        badge(text: "BASE OF WORKING",
                              fg: theme.textSecondary, bg: .clear,
                              border: theme.inputBorder)
                    }
                }
                HStack(spacing: 6) {
                    Circle()
                        .fill(isWorking ? theme.amberDot : scopeColor)
                        .frame(width: 6, height: 6)
                    Text(subtitleText)
                        .font(.omlxText(11))
                        .foregroundStyle(theme.textSecondary)
                }
            }
            Spacer(minLength: 8)
            HStack(spacing: 6) {
                if let onClosePreview {
                    Button("Done") { onClosePreview() }
                        .buttonStyle(.omlx(.plain, size: .small))
                }
                if showUpdate, let onUpdateFromWorking {
                    Button("Update with working") { onUpdateFromWorking() }
                        .buttonStyle(.omlx(.normal, size: .small))
                }
                if showApply, let onApply {
                    Button("Apply") { onApply() }
                        .buttonStyle(.omlx(.primary, size: .small))
                }
                if showDelete, let onDelete {
                    Button {
                        onDelete()
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 12))
                            .foregroundStyle(theme.redDot)
                    }
                    .buttonStyle(.omlx(.plain, size: .small))
                }
            }
        }
    }

    @ViewBuilder
    private var iconBox: some View {
        let useScopeFill = (isActive && !isWorking && scope != nil) || isWorking
        let fillColor: Color = isWorking ? theme.amberDot : scopeColor
        let symbol: String = isWorking ? "sparkles" : (scope == nil ? "gauge.medium" : "cpu")
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(useScopeFill
                  ? LinearGradient(
                      colors: [fillColor, fillColor.opacity(0.8)],
                      startPoint: .top, endPoint: .bottom
                    )
                  : LinearGradient(
                      colors: [theme.codeBg, theme.codeBg],
                      startPoint: .top, endPoint: .bottom
                    ))
            .frame(width: 36, height: 36)
            .overlay(
                Image(systemName: symbol)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(useScopeFill ? .white : theme.textSecondary)
            )
    }

    private var subtitleText: String {
        if isWorking, let basedOn {
            return "Working profile · based on \(basedOn.name) (\(ProfileScopeMeta.label(basedOn.scope)))"
        }
        if isWorking {
            return "Working profile · based on server defaults"
        }
        if scope == nil {
            return "Falls back here when no profile is set, or when a profile leaves a field empty"
        }
        let count = settings.keys.filter { !isEmptyValue(settings[$0]) }.count
        let scopeLabel = scope.map { ProfileScopeMeta.label($0) } ?? "Profile"
        return "\(scopeLabel) profile · \(count) setting\(count == 1 ? "" : "s")"
    }

    @ViewBuilder
    private var sections: some View {
        let s = settings
        let hasSampling = ["temperature", "top_p", "top_k", "min_p"].contains { s[$0] != nil }
        let hasCapacity = ["model_type_override", "max_context_window", "max_tokens", "ttl_seconds"]
            .contains { s[$0] != nil }
        let hasPenalty = ["repetition_penalty", "presence_penalty"].contains { s[$0] != nil }
        let behaviorKeys = [
            "enable_thinking", "thinking_budget_enabled", "max_tool_result_tokens",
            "force_sampling", "is_pinned",
        ]
        let hasBehavior = behaviorKeys.contains { s[$0] != nil }

        VStack(alignment: .leading, spacing: 14) {
            if hasSampling { samplingSection(s) }
            if hasCapacity { capacitySection(s) }
            if hasPenalty { penaltySection(s) }
            if hasBehavior { behaviorSection(s) }
            if !hasSampling && !hasCapacity && !hasPenalty && !hasBehavior {
                Text("This profile doesn't override any settings.")
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 10)
            }
        }
    }

    @ViewBuilder
    private func samplingSection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle("Sampling")
        let entries: [(String, Double?, Double, Double, String)] = [
            ("Temperature", doubleOf(s["temperature"]), 0, 2, "%.2f"),
            ("Top P",       doubleOf(s["top_p"]),       0, 1, "%.2f"),
            ("Top K",       doubleOf(s["top_k"]),       0, 100, "%.0f"),
            ("Min P",       doubleOf(s["min_p"]),       0, 1, "%.2f"),
        ].filter { $0.1 != nil }
        HStack(alignment: .top, spacing: 18) {
            ForEach(Array(entries.enumerated()), id: \.offset) { _, e in
                meter(label: e.0, value: e.1!, min: e.2, max: e.3, fmt: e.4)
            }
        }
    }

    @ViewBuilder
    private func capacitySection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle("Capacity")
        let entries: [(String, String)] = [
            ("Type", s["model_type_override"].flatMap { ($0.value as? String) }
                .flatMap { capacityType(value: $0) }),
            ("Context", s["max_context_window"].flatMap { intOf($0) }
                .map { "\(fmtCtx(Double($0))) tk" }),
            ("Max Tokens", s["max_tokens"].flatMap { intOf($0) }
                .map { "\($0) tk" }),
            ("TTL", s["ttl_seconds"].flatMap { intOf($0) }
                .map { $0 == 0 ? "Persistent" : "\($0)s" }),
        ].compactMap { (label, value) in value.map { (label, $0) } }
        HStack(alignment: .top, spacing: 18) {
            ForEach(Array(entries.enumerated()), id: \.offset) { _, e in
                stat(label: e.0, value: e.1)
            }
        }
    }

    @ViewBuilder
    private func penaltySection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle("Penalties")
        let entries: [(String, Double?, Double, Double)] = [
            ("Repetition", doubleOf(s["repetition_penalty"]), 0, 2),
            ("Presence", doubleOf(s["presence_penalty"]), -2, 2),
        ].filter { $0.1 != nil }
        HStack(alignment: .top, spacing: 18) {
            ForEach(Array(entries.enumerated()), id: \.offset) { _, e in
                meter(label: e.0, value: e.1!, min: e.2, max: e.3, fmt: "%.2f")
            }
        }
    }

    @ViewBuilder
    private func behaviorSection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle("Behavior")
        FlowLayout(spacing: 6) {
            if let v = s["enable_thinking"].flatMap({ boolOf($0) }) {
                flagChip(label: "Thinking", on: v)
            }
            if let v = s["thinking_budget_enabled"].flatMap({ boolOf($0) }) {
                let n = intOf(s["thinking_budget_tokens"]) ?? 8192
                flagChip(label: v ? "Budget · \(fmtCtx(Double(n))) tk" : "Thinking budget", on: v)
            }
            if let v = s["max_tool_result_tokens"].flatMap({ intOf($0) }) {
                flagChip(label: "Limit tool output", on: v > 0)
            }
            if let v = s["force_sampling"].flatMap({ boolOf($0) }) {
                flagChip(label: "Force sampling", on: v)
            }
            if let v = s["is_pinned"].flatMap({ boolOf($0) }) {
                flagChip(label: "Pinned in memory", on: v)
            }
        }
    }

    @ViewBuilder
    private func sectionTitle(_ s: String) -> some View {
        Text(s)
            .font(.omlxText(10, weight: .heavy))
            .kerning(0.9)
            .textCase(.uppercase)
            .foregroundStyle(theme.textTertiary)
            .padding(.bottom, 2)
    }

    @ViewBuilder
    private func meter(label: String, value: Double, min minV: Double, max maxV: Double, fmt: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .firstTextBaseline) {
                Text(label)
                    .font(.omlxText(10.5, weight: .heavy))
                    .kerning(0.6)
                    .textCase(.uppercase)
                    .foregroundStyle(theme.textSecondary)
                Spacer(minLength: 4)
                Text(String(format: fmt, value))
                    .font(.omlxMono(12.5, weight: .semibold))
                    .foregroundStyle(theme.text)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(theme.text.opacity(theme.isDark ? 0.07 : 0.06))
                    let raw = (value - minV) / Swift.max(0.001, maxV - minV)
                    let pct = Swift.max(0, Swift.min(1, raw))
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(theme.accent)
                        .frame(width: geo.size.width * pct)
                }
            }
            .frame(height: 4)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func stat(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.omlxText(10.5, weight: .heavy))
                .kerning(0.6)
                .textCase(.uppercase)
                .foregroundStyle(theme.textSecondary)
            Text(value)
                .font(.omlxMono(14, weight: .semibold))
                .foregroundStyle(theme.text)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func flagChip(label: String, on: Bool) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(on ? theme.greenDot : theme.text.opacity(0.18))
                .frame(width: 6, height: 6)
            Text(label)
                .font(.omlxText(11.5, weight: .medium))
                .foregroundStyle(on ? theme.greenDot : theme.textSecondary)
        }
        .padding(.horizontal, 9)
        .frame(height: 24)
        .background(
            Capsule().fill(
                on ? theme.greenDot.opacity(theme.isDark ? 0.16 : 0.14)
                   : theme.text.opacity(theme.isDark ? 0.04 : 0.035)
            )
        )
        .overlay(
            Capsule().strokeBorder(
                on ? theme.greenDot.opacity(theme.isDark ? 0.25 : 0.3)
                   : theme.inputBorder,
                lineWidth: 0.5
            )
        )
    }

    @ViewBuilder
    private func badge(text: String, fg: Color, bg: Color, border: Color? = nil) -> some View {
        Text(text)
            .font(.omlxText(9.5, weight: .heavy))
            .kerning(0.7)
            .foregroundStyle(fg)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(bg)
            .overlay(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .strokeBorder(border ?? .clear, lineWidth: border == nil ? 0 : 0.5)
            )
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
    }
}

// MARK: - Helpers (settings dict reads)

private func doubleOf(_ v: AnyCodable?) -> Double? {
    guard let v else { return nil }
    if let d = v.value as? Double { return d }
    if let i = v.value as? Int { return Double(i) }
    return nil
}

private func intOf(_ v: AnyCodable?) -> Int? {
    guard let v else { return nil }
    if let i = v.value as? Int { return i }
    if let d = v.value as? Double { return Int(d) }
    return nil
}

private func boolOf(_ v: AnyCodable?) -> Bool? {
    guard let v else { return nil }
    return v.value as? Bool
}

private func capacityType(value: String) -> String? {
    let v = value.trimmingCharacters(in: .whitespaces)
    return v.isEmpty ? nil : v
}

private func fmtCtx(_ v: Double) -> String {
    if v >= 1_000_000 {
        let q = v / 1_000_000
        return q.truncatingRemainder(dividingBy: 1) == 0
            ? String(format: "%.0fM", q)
            : String(format: "%.1fM", q)
    }
    if v >= 1_000 {
        return String(format: "%.0fK", v / 1_000)
    }
    return String(Int(v))
}

private func isEmptyValue(_ v: AnyCodable?) -> Bool {
    guard let v else { return true }
    if v.value is NSNull { return true }
    if let s = v.value as? String, s.isEmpty { return true }
    if let arr = v.value as? [AnyCodable], arr.isEmpty { return true }
    if let dict = v.value as? [String: AnyCodable], dict.isEmpty { return true }
    return false
}

// MARK: - FlowLayout (chip wrapping)

/// Wrapping chip layout. `arrange` is the single source of truth for
/// both passes (`sizeThatFits` and `placeSubviews`) so the reported size
/// and the actual placement always agree — even when SwiftUI calls the
/// sizing pass with an unspecified proposal before the real bounds are
/// known.
struct FlowLayout: Layout {
    let spacing: CGFloat

    init(spacing: CGFloat = 6) { self.spacing = spacing }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let containerWidth = proposal.width ?? .greatestFiniteMagnitude
        return arrange(containerWidth: containerWidth, subviews: subviews).size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(containerWidth: bounds.width, subviews: subviews)
        for (i, frame) in result.frames.enumerated() {
            subviews[i].place(
                at: CGPoint(x: bounds.minX + frame.origin.x, y: bounds.minY + frame.origin.y),
                proposal: ProposedViewSize(frame.size)
            )
        }
    }

    private struct Arrangement {
        let size: CGSize
        let frames: [CGRect]
    }

    private func arrange(containerWidth: CGFloat, subviews: Subviews) -> Arrangement {
        var frames: [CGRect] = []
        frames.reserveCapacity(subviews.count)
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowH: CGFloat = 0
        var maxRowEnd: CGFloat = 0

        for sub in subviews {
            let size = sub.sizeThatFits(.unspecified)
            // Wrap when this chip doesn't fit *and* there's at least one
            // chip already on the current row. The leading-edge check
            // prevents an infinite-width single chip from being thrown
            // off the layout.
            if x > 0 && x + size.width > containerWidth {
                y += rowH + spacing
                x = 0
                rowH = 0
            }
            frames.append(CGRect(origin: CGPoint(x: x, y: y), size: size))
            x += size.width + spacing
            rowH = max(rowH, size.height)
            maxRowEnd = max(maxRowEnd, x - spacing)
        }
        return Arrangement(
            size: CGSize(width: maxRowEnd, height: y + rowH),
            frames: frames
        )
    }
}
