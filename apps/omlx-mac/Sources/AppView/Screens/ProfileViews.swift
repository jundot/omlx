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
        case .preset: return String(localized: "profile.scope.preset",
                                    defaultValue: "Preset",
                                    comment: "Scope label for shipped preset profiles in Profiles tab")
        case .global: return String(localized: "profile.scope.global",
                                    defaultValue: "Global",
                                    comment: "Scope label for user-defined global profiles")
        case .model:  return String(localized: "profile.scope.model",
                                    defaultValue: "Model",
                                    comment: "Scope label for per-model profiles")
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
    /// Optional refresh affordance — when non-nil, a small refresh icon
    /// renders in the header instead of (or alongside) the save button.
    /// Used by the preset chip strip to pull the latest bundle from
    /// omlx.ai via `POST /api/presets/refresh`.
    var onRefresh: (() -> Void)? = nil
    /// Spinner state for the refresh icon. Disables the button while a
    /// refresh is in flight so a fast user can't queue duplicates.
    var isRefreshing: Bool = false
    /// Optional inline-rename callback. When provided, double-clicking a
    /// chip swaps its label for a `TextField`; Enter commits via
    /// `(originalName, newName)`. Pass nil for read-only groups (preset
    /// chips backed by the shipped JSON bundle).
    var onRename: ((String, String) -> Void)? = nil

    @Environment(\.omlxTheme) private var theme
    @State private var renamingName: String? = nil
    @State private var renameText: String = ""
    @FocusState private var renameFieldFocused: Bool

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
            if let onRefresh {
                Button {
                    onRefresh()
                } label: {
                    Image(systemName: isRefreshing
                          ? "arrow.triangle.2.circlepath"
                          : "arrow.clockwise")
                        .font(.system(size: 11, weight: .semibold))
                        .symbolEffect(.rotate, isActive: isRefreshing)
                }
                .buttonStyle(.omlx(.plain, size: .small))
                .help(String(localized: "profile.group.refresh.help",
                             defaultValue: "Refresh presets from omlx.ai",
                             comment: "Tooltip on the refresh button in the preset chip group header"))
                .disabled(isRefreshing)
            }
            if canSaveCurrent {
                Button {
                    onSaveCurrent()
                } label: {
                    Label(String(localized: "profile.group.save_current_as_new",
                                 defaultValue: "Save current as new",
                                 comment: "Button in profile chip-group header that saves the working profile as a new named profile"),
                          systemImage: "plus")
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
                Text(String(localized: "profile.group.empty",
                            defaultValue: "No \(ProfileScopeMeta.label(scope).lowercased()) profiles yet.",
                            comment: "Placeholder text in an empty profile chip group; placeholder is the lowercased scope name"))
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
        let isEditing = (renamingName == name)
        HStack(spacing: 5) {
            if isActive && !isEditing {
                Image(systemName: "checkmark")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.white)
            }
            if isEditing {
                TextField("", text: $renameText)
                    .textFieldStyle(.plain)
                    .font(.omlxText(12, weight: .medium))
                    .foregroundStyle(isActive ? .white : theme.text)
                    .fixedSize(horizontal: true, vertical: false)
                    .focused($renameFieldFocused)
                    .onSubmit { commitRename(original: name) }
                    .onExitCommand { renamingName = nil }
                    .onChange(of: renameFieldFocused) { _, focused in
                        // Commit on focus loss (click outside / tab away)
                        // — same validation path as Enter. The TextField
                        // only loses focus *after* `onSubmit` runs, and
                        // `commitRename` clears `renamingName` before
                        // returning, so this guard prevents a double-fire
                        // on a clean Enter. Escape still cancels via
                        // `.onExitCommand` (which nils `renamingName`
                        // before focus loss propagates here).
                        if !focused && renamingName == name {
                            commitRename(original: name)
                        }
                    }
            } else {
                Text(name)
                    .font(.omlxText(12, weight: .medium))
                    .foregroundStyle(isActive ? .white : theme.text)
            }
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
        .gesture(
            // ExclusiveGesture(first, second): the double-tap is tried
            // first; if it succeeds the single-tap doesn't fire. Without
            // this, double-clicking a chip would briefly flash the
            // preview before entering rename mode.
            ExclusiveGesture(
                TapGesture(count: 2).onEnded {
                    guard onRename != nil, !isEditing else { return }
                    startRename(name)
                },
                TapGesture(count: 1).onEnded {
                    guard !isEditing else { return }
                    onSelect(name)
                }
            )
        )
    }

    private func startRename(_ name: String) {
        renamingName = name
        renameText = name
        // Focus on the next runloop tick so the @FocusState observer sees
        // the TextField after it's been mounted into the hierarchy.
        DispatchQueue.main.async { renameFieldFocused = true }
    }

    private func commitRename(original: String) {
        let trimmed = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        // Always exit rename mode — validation failures silently revert
        // to the original name without an error banner.
        defer { renamingName = nil }
        guard !trimmed.isEmpty,
              trimmed != original,
              !names.contains(trimmed),
              Self.isValidSlug(trimmed)
        else { return }
        onRename?(original, trimmed)
    }

    /// Mirror of the server's profile-name slug rule
    /// (`omlx/model_profiles.py:validate_profile_name`). Pre-checking
    /// client-side avoids a doomed PUT round-trip for invalid names.
    private static func isValidSlug(_ s: String) -> Bool {
        guard let re = try? NSRegularExpression(
            pattern: #"^[a-z0-9][a-z0-9_-]{0,31}$"#
        ) else { return false }
        let range = NSRange(s.startIndex..., in: s)
        return re.firstMatch(in: s, range: range) != nil
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
        case .working:                   return String(localized: "profile.banner.working.title",
                                                       defaultValue: "Working profile",
                                                       comment: "Profile banner title when the user has unsaved edits")
        case .named(_, let name):        return name
        case .defaults:                  return String(localized: "profile.banner.defaults.title",
                                                       defaultValue: "No profile",
                                                       comment: "Profile banner title when the model uses server defaults")
        }
    }

    @ViewBuilder
    private var subtitleView: some View {
        switch state {
        case .working(let basedOn):
            if let basedOn {
                Text(String(localized: "profile.banner.working.subtitle.based_on",
                            defaultValue: "Unsaved · based on \(basedOn.name) (\(ProfileScopeMeta.label(basedOn.scope)))",
                            comment: "Profile banner subtitle for working state with a base profile; placeholders are profile name and scope label"))
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            } else {
                Text(String(localized: "profile.banner.working.subtitle.defaults",
                            defaultValue: "Unsaved · based on server defaults",
                            comment: "Profile banner subtitle when working profile has no named base"))
                    .font(.omlxText(11.5))
                    .foregroundStyle(theme.textSecondary)
            }
        case .named(let scope, _):
            Text(String(localized: "profile.banner.named.subtitle",
                        defaultValue: "\(ProfileScopeMeta.label(scope)) profile · active on this model",
                        comment: "Profile banner subtitle when a named profile is active; placeholder is scope label"))
                .font(.omlxText(11.5))
                .foregroundStyle(theme.textSecondary)
        case .defaults:
            Text(String(localized: "profile.banner.defaults.subtitle",
                        defaultValue: "Using server defaults · edit any field to start a working profile",
                        comment: "Profile banner subtitle when no profile is assigned"))
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
                    Button(String(localized: "profile.banner.action.update_based_on",
                                  defaultValue: "Update \(basedOn.name)",
                                  comment: "Profile banner action to overwrite the base profile with working edits; placeholder is the base profile's name")) { onUpdateBasedOn() }
                        .buttonStyle(.omlx(.normal, size: .small))
                }
                if let onSaveAsNew {
                    Button(String(localized: "profile.banner.action.save_as_new",
                                  defaultValue: "Save as new",
                                  comment: "Profile banner action that saves working edits as a new profile")) { onSaveAsNew() }
                        .buttonStyle(.omlx(.primary, size: .small))
                }
                if let onRevert {
                    Button(basedOn == nil
                           ? String(localized: "profile.banner.action.discard",
                                    defaultValue: "Discard",
                                    comment: "Profile banner action that discards working edits when there's no base profile")
                           : String(localized: "profile.banner.action.revert",
                                    defaultValue: "Revert",
                                    comment: "Profile banner action that reverts working edits back to the base profile")) { onRevert() }
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
            Text(String(localized: "profile.save_as.title",
                        defaultValue: "Save current profile as",
                        comment: "Lead label inside the Save-as popover for naming a new profile"))
                .font(.omlxText(12, weight: .medium))
                .foregroundStyle(theme.textSecondary)
            Segmented(
                selection: $scope,
                options: [
                    (.global, String(localized: "profile.scope.global",
                                     defaultValue: "Global",
                                     comment: "Scope label for user-defined global profiles")),
                    (.model,  String(localized: "profile.scope.model",
                                     defaultValue: "Model",
                                     comment: "Scope label for per-model profiles")),
                ]
            )
            TextInput(text: $name,
                      placeholder: String(localized: "profile.save_as.name.placeholder",
                                          defaultValue: "profile-name",
                                          comment: "Placeholder text inside the new-profile name field"),
                      mono: true)
                .frame(maxWidth: .infinity)
                .focused($nameFocused)
                .onSubmit { onCommit() }
            Button(String(localized: "common.cancel",
                          defaultValue: "Cancel",
                          comment: "Generic Cancel button label")) { onCancel() }
                .buttonStyle(.omlx(.normal, size: .small))
            Button(String(localized: "common.save",
                          defaultValue: "Save",
                          comment: "Generic Save button label")) { onCommit() }
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
    /// Server `expose_as_model` state for this profile. Only meaningful
    /// when `onToggleExpose` is wired (model-scope profiles).
    var exposeAsModel: Bool = false
    /// Derived API model ID (`<base-model>:<profile-name>`) shown next to
    /// the toggle while exposure is on.
    var exposedModelId: String? = nil
    /// Server-derived `has_engine_fields` — true when the profile carries
    /// engine-construction overrides, which the exposed-model overlay
    /// ignores. Shows a warning under the toggle while exposure is on.
    var hasEngineFields: Bool = false
    /// Non-nil renders the "Expose as model" toggle; the callback receives
    /// the requested state. Pass nil for templates, presets, and the
    /// defaults card.
    var onToggleExpose: ((Bool) -> Void)? = nil

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
            exposeRow
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
                        badge(text: String(localized: "profile.detail.badge.active",
                                           defaultValue: "ACTIVE",
                                           comment: "Badge shown on the profile detail card when this profile is active on the model"),
                              fg: .white, bg: theme.greenDot)
                    }
                    if isWorking {
                        badge(text: String(localized: "profile.detail.badge.unsaved",
                                           defaultValue: "UNSAVED",
                                           comment: "Badge on the profile detail card indicating unsaved working-state edits"),
                              fg: Color(rgb24: 0x1A1407), bg: theme.amberDot)
                    }
                    if isWorkingBase && !isWorking {
                        badge(text: String(localized: "profile.detail.badge.base_of_working",
                                           defaultValue: "BASE OF WORKING",
                                           comment: "Badge on the profile detail card marking the profile the working profile forked from"),
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
                    Button(String(localized: "profile.detail.action.done",
                                  defaultValue: "Done",
                                  comment: "Profile detail card button that closes the preview overlay")) { onClosePreview() }
                        .buttonStyle(.omlx(.plain, size: .small))
                }
                if showUpdate, let onUpdateFromWorking {
                    Button(String(localized: "profile.detail.action.update_with_working",
                                  defaultValue: "Update with working",
                                  comment: "Profile detail card button that writes the current working edits into this named profile")) { onUpdateFromWorking() }
                        .buttonStyle(.omlx(.normal, size: .small))
                }
                if showApply, let onApply {
                    Button(String(localized: "profile.detail.action.apply",
                                  defaultValue: "Apply",
                                  comment: "Profile detail card button that activates this profile on the model")) { onApply() }
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
            return String(localized: "profile.detail.subtitle.working_based_on",
                          defaultValue: "Working profile · based on \(basedOn.name) (\(ProfileScopeMeta.label(basedOn.scope)))",
                          comment: "Profile detail card subtitle in working state with a base profile; placeholders are name and scope label")
        }
        if isWorking {
            return String(localized: "profile.detail.subtitle.working_defaults",
                          defaultValue: "Working profile · based on server defaults",
                          comment: "Profile detail card subtitle in working state without a base profile")
        }
        if scope == nil {
            return String(localized: "profile.detail.subtitle.defaults",
                          defaultValue: "Falls back here when no profile is set, or when a profile leaves a field empty",
                          comment: "Profile detail card subtitle for the Server Defaults card")
        }
        let count = settings.keys.filter { !isEmptyValue(settings[$0]) }.count
        let scopeLabel = scope.map { ProfileScopeMeta.label($0) } ?? String(localized: "profile.detail.subtitle.scope.profile",
                                                                            defaultValue: "Profile",
                                                                            comment: "Fallback scope label in the profile detail subtitle when scope is unknown")
        return String(localized: "profile.detail.subtitle.named",
                      defaultValue: "\(scopeLabel) profile · \(count) setting\(count == 1 ? "" : "s")",
                      comment: "Profile detail card subtitle for a named profile; placeholders are scope label and a setting count with pluralization")
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
        // Per-model "Advanced" surface — fields from
        // `MODEL_SPECIFIC_PROFILE_FIELDS` in `omlx/model_profiles.py`.
        // None of these are universal, so they're absent from preset and
        // global templates; they only appear on per-model profiles.
        let accelKeys = [
            "turboquant_kv_enabled", "dflash_enabled", "mtp_enabled",
            "specprefill_enabled", "index_cache_freq", "vlm_mtp_enabled",
        ]
        let hasAcceleration = accelKeys.contains { s[$0] != nil }
        let templateKeys = ["reasoning_parser", "chat_template_kwargs", "forced_ct_kwargs"]
        let hasTemplates = templateKeys.contains { k in
            guard let v = s[k] else { return false }
            return !isEmptyValue(v)
        }

        VStack(alignment: .leading, spacing: 14) {
            if hasSampling { samplingSection(s) }
            if hasCapacity { capacitySection(s) }
            if hasPenalty { penaltySection(s) }
            if hasBehavior { behaviorSection(s) }
            if hasAcceleration { accelerationSection(s) }
            if hasTemplates { templatesSection(s) }
            if !hasSampling && !hasCapacity && !hasPenalty && !hasBehavior
                && !hasAcceleration && !hasTemplates {
                Text(String(localized: "profile.detail.empty",
                            defaultValue: "This profile doesn't override any settings.",
                            comment: "Placeholder text in the profile detail card when no settings are set"))
                    .font(.omlxText(12))
                    .foregroundStyle(theme.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 10)
            }
        }
    }

    @ViewBuilder
    private func samplingSection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle(String(localized: "profile.detail.section.sampling",
                            defaultValue: "Sampling",
                            comment: "Profile detail card section header for sampler-related fields"))
        let entries: [(String, Double?, Double, Double, String)] = [
            (String(localized: "profile.detail.sampling.temperature",
                    defaultValue: "Temperature",
                    comment: "Sampling meter label: temperature"),
             doubleOf(s["temperature"]), 0, 2, "%.2f"),
            (String(localized: "profile.detail.sampling.top_p",
                    defaultValue: "Top P",
                    comment: "Sampling meter label: top-p"),
             doubleOf(s["top_p"]),       0, 1, "%.2f"),
            (String(localized: "profile.detail.sampling.top_k",
                    defaultValue: "Top K",
                    comment: "Sampling meter label: top-k"),
             doubleOf(s["top_k"]),       0, 100, "%.0f"),
            (String(localized: "profile.detail.sampling.min_p",
                    defaultValue: "Min P",
                    comment: "Sampling meter label: min-p"),
             doubleOf(s["min_p"]),       0, 1, "%.2f"),
        ].filter { $0.1 != nil }
        HStack(alignment: .top, spacing: 18) {
            ForEach(Array(entries.enumerated()), id: \.offset) { _, e in
                meter(label: e.0, value: e.1!, min: e.2, max: e.3, fmt: e.4)
            }
        }
    }

    @ViewBuilder
    private func capacitySection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle(String(localized: "profile.detail.section.capacity",
                            defaultValue: "Capacity",
                            comment: "Profile detail card section header for capacity-related fields"))
        let entries: [(String, String)] = [
            (String(localized: "profile.detail.capacity.type",
                    defaultValue: "Type",
                    comment: "Capacity stat label: model type override"),
             s["model_type_override"].flatMap { ($0.value as? String) }
                .flatMap { capacityType(value: $0) }),
            (String(localized: "profile.detail.capacity.context",
                    defaultValue: "Context",
                    comment: "Capacity stat label: max context window"),
             s["max_context_window"].flatMap { intOf($0) }
                .map { String(localized: "profile.detail.capacity.tokens",
                              defaultValue: "\(fmtCtx(Double($0))) tk",
                              comment: "Token count rendered next to a Capacity stat; placeholder is the formatted count") }),
            (String(localized: "profile.detail.capacity.max_tokens",
                    defaultValue: "Max Tokens",
                    comment: "Capacity stat label: max output tokens"),
             s["max_tokens"].flatMap { intOf($0) }
                .map { String(localized: "profile.detail.capacity.tokens.raw",
                              defaultValue: "\($0) tk",
                              comment: "Raw token count for max tokens; placeholder is the integer count") }),
            (String(localized: "profile.detail.capacity.ttl",
                    defaultValue: "TTL",
                    comment: "Capacity stat label: time-to-live for the model in memory"),
             s["ttl_seconds"].flatMap { intOf($0) }
                .map { $0 == 0
                       ? String(localized: "profile.detail.capacity.ttl.persistent",
                                defaultValue: "Persistent",
                                comment: "TTL value rendered when the model is pinned indefinitely")
                       : String(localized: "profile.detail.capacity.ttl.seconds",
                                defaultValue: "\($0)s",
                                comment: "TTL value in seconds; placeholder is the integer seconds") }),
        ].compactMap { (label, value) in value.map { (label, $0) } }
        HStack(alignment: .top, spacing: 18) {
            ForEach(Array(entries.enumerated()), id: \.offset) { _, e in
                stat(label: e.0, value: e.1)
            }
        }
    }

    @ViewBuilder
    private func penaltySection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle(String(localized: "profile.detail.section.penalties",
                            defaultValue: "Penalties",
                            comment: "Profile detail card section header for repetition/presence penalty fields"))
        let entries: [(String, Double?, Double, Double)] = [
            (String(localized: "profile.detail.penalty.repetition",
                    defaultValue: "Repetition",
                    comment: "Penalty meter label: repetition"),
             doubleOf(s["repetition_penalty"]), 0, 2),
            (String(localized: "profile.detail.penalty.presence",
                    defaultValue: "Presence",
                    comment: "Penalty meter label: presence"),
             doubleOf(s["presence_penalty"]), -2, 2),
        ].filter { $0.1 != nil }
        HStack(alignment: .top, spacing: 18) {
            ForEach(Array(entries.enumerated()), id: \.offset) { _, e in
                meter(label: e.0, value: e.1!, min: e.2, max: e.3, fmt: "%.2f")
            }
        }
    }

    @ViewBuilder
    private func behaviorSection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle(String(localized: "profile.detail.section.behavior",
                            defaultValue: "Behavior",
                            comment: "Profile detail card section header for behavior-flag chips"))
        FlowLayout(spacing: 6) {
            if let v = s["enable_thinking"].flatMap({ boolOf($0) }) {
                flagChip(label: String(localized: "profile.detail.behavior.thinking",
                                       defaultValue: "Thinking",
                                       comment: "Behavior chip: extended-thinking flag"),
                         on: v)
            }
            if let v = s["thinking_budget_enabled"].flatMap({ boolOf($0) }) {
                let n = intOf(s["thinking_budget_tokens"]) ?? 8192
                flagChip(label: v
                         ? String(localized: "profile.detail.behavior.thinking_budget.on",
                                  defaultValue: "Budget · \(fmtCtx(Double(n))) tk",
                                  comment: "Behavior chip when thinking budget is enabled; placeholder is the formatted token count")
                         : String(localized: "profile.detail.behavior.thinking_budget.off",
                                  defaultValue: "Thinking budget",
                                  comment: "Behavior chip label when thinking budget is disabled"),
                         on: v)
            }
            if let v = s["max_tool_result_tokens"].flatMap({ intOf($0) }) {
                flagChip(label: String(localized: "profile.detail.behavior.limit_tool_output",
                                       defaultValue: "Limit tool output",
                                       comment: "Behavior chip: cap on tool result tokens"),
                         on: v > 0)
            }
            if let v = s["force_sampling"].flatMap({ boolOf($0) }) {
                flagChip(label: String(localized: "profile.detail.behavior.force_sampling",
                                       defaultValue: "Force sampling",
                                       comment: "Behavior chip: force sampling flag"),
                         on: v)
            }
            if let v = s["is_pinned"].flatMap({ boolOf($0) }) {
                flagChip(label: String(localized: "profile.detail.behavior.pinned",
                                       defaultValue: "Pinned in memory",
                                       comment: "Behavior chip: pin-in-memory flag"),
                         on: v)
            }
        }
    }

    /// Acceleration / experimental knobs that only land on per-model
    /// profiles (TurboQuant KV, DFlash + drafter quant, native MTP,
    /// SpecPrefill, IndexCache). Renders one chip per feature so the
    /// preview surfaces whether a profile flips them — and at what
    /// settings — without forcing the user back into the Advanced tab.
    @ViewBuilder
    private func accelerationSection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle(String(localized: "profile.detail.section.acceleration",
                            defaultValue: "Acceleration",
                            comment: "Profile detail card section header for experimental acceleration knobs"))
        FlowLayout(spacing: 6) {
            if let on = s["turboquant_kv_enabled"].flatMap({ boolOf($0) }) {
                flagChip(label: turboquantLabel(s, on: on), on: on)
            }
            if let on = s["dflash_enabled"].flatMap({ boolOf($0) }) {
                flagChip(label: dflashLabel(s, on: on), on: on)
                if on, let q = s["dflash_draft_quant_enabled"].flatMap({ boolOf($0) }), q {
                    flagChip(label: dflashQuantLabel(s), on: true)
                }
            }
            if let on = s["mtp_enabled"].flatMap({ boolOf($0) }) {
                flagChip(label: String(localized: "profile.detail.acceleration.mtp",
                                       defaultValue: "Native MTP",
                                       comment: "Acceleration chip: native multi-token prediction"),
                         on: on)
            }
            if let on = s["vlm_mtp_enabled"].flatMap({ boolOf($0) }) {
                flagChip(label: String(localized: "profile.detail.acceleration.vlm_mtp",
                                       defaultValue: "VLM MTP",
                                       comment: "Acceleration chip: VLM multi-token prediction"),
                         on: on)
            }
            if let on = s["specprefill_enabled"].flatMap({ boolOf($0) }) {
                flagChip(label: specprefillLabel(s, on: on), on: on)
            }
            if let freq = intOf(s["index_cache_freq"]), freq > 0 {
                flagChip(label: String(localized: "profile.detail.acceleration.index_cache",
                                       defaultValue: "IndexCache · every \(freq)",
                                       comment: "Acceleration chip describing IndexCache frequency; placeholder is the integer frequency"),
                         on: true)
            }
        }
    }

    /// Tokenizer / chat-template overrides — universal fields that the
    /// existing Sampling / Capacity / Penalty / Behavior sections don't
    /// cover. Kept separate so the visual weight matches its semantic
    /// distance from the sampler knobs above.
    @ViewBuilder
    private func templatesSection(_ s: [String: AnyCodable]) -> some View {
        sectionTitle(String(localized: "profile.detail.section.templates",
                            defaultValue: "Templates",
                            comment: "Profile detail card section header for tokenizer/chat-template fields"))
        FlowLayout(spacing: 6) {
            if let parser = (s["reasoning_parser"]?.value as? String)?
                .trimmingCharacters(in: .whitespaces), !parser.isEmpty {
                flagChip(label: String(localized: "profile.detail.templates.reasoning_parser",
                                       defaultValue: "Reasoning · \(parser)",
                                       comment: "Templates chip naming the active reasoning parser; placeholder is the parser identifier"),
                         on: true)
            }
            if let count = nonEmptyKwargCount(s["chat_template_kwargs"]) {
                flagChip(
                    label: String(localized: "profile.detail.templates.chat_template",
                                  defaultValue: "Chat template · \(count) override\(count == 1 ? "" : "s")",
                                  comment: "Templates chip describing chat-template kwarg override count; placeholder is the count with pluralization"),
                    on: true
                )
            }
            if let count = nonEmptyKwargCount(s["forced_ct_kwargs"]) {
                flagChip(
                    label: String(localized: "profile.detail.templates.forced_ct",
                                  defaultValue: "Forced CT · \(count) key\(count == 1 ? "" : "s")",
                                  comment: "Templates chip describing forced chat-template key count; placeholder is the count with pluralization"),
                    on: true
                )
            }
        }
    }

    private func turboquantLabel(_ s: [String: AnyCodable], on: Bool) -> String {
        let baseName = String(localized: "profile.detail.acceleration.turboquant",
                              defaultValue: "TurboQuant KV",
                              comment: "Acceleration chip base name for TurboQuant KV-cache compression")
        guard on else { return baseName }
        let bits = doubleOf(s["turboquant_kv_bits"])
        let skip = intOf(s["turboquant_skip_last"]) ?? 0
        let bitsText = bits.map { v in
            v.truncatingRemainder(dividingBy: 1) == 0
                ? String(localized: "profile.detail.acceleration.turboquant.bits_int",
                         defaultValue: "\(Int(v))bit",
                         comment: "TurboQuant bit-width suffix when the value is an integer; placeholder is the int")
                : String(format: String(localized: "profile.detail.acceleration.turboquant.bits_decimal",
                                        defaultValue: "%.1fbit",
                                        comment: "TurboQuant bit-width suffix when fractional; %.1f gets the decimal width"), v)
        }
        let skipText = skip > 0
            ? String(localized: "profile.detail.acceleration.turboquant.skip",
                     defaultValue: "skip \(skip)",
                     comment: "TurboQuant skip-last-N suffix; placeholder is the layer count")
            : nil
        let parts = [bitsText, skipText].compactMap { $0 }
        return parts.isEmpty
            ? baseName
            : String(localized: "profile.detail.acceleration.turboquant.with_parts",
                     defaultValue: "TurboQuant KV · \(parts.joined(separator: " / "))",
                     comment: "TurboQuant chip with detail suffix; placeholder is the joined detail parts")
    }

    private func dflashLabel(_ s: [String: AnyCodable], on: Bool) -> String {
        let baseName = String(localized: "profile.detail.acceleration.dflash",
                              defaultValue: "DFlash",
                              comment: "Acceleration chip base name for DFlash drafter")
        guard on else { return baseName }
        let drafter = (s["dflash_draft_model"]?.value as? String)?
            .trimmingCharacters(in: .whitespaces)
        if let d = drafter, !d.isEmpty {
            return String(localized: "profile.detail.acceleration.dflash.with_drafter",
                          defaultValue: "DFlash · \(d)",
                          comment: "DFlash chip with drafter model id; placeholder is the model identifier")
        }
        return baseName
    }

    private func dflashQuantLabel(_ s: [String: AnyCodable]) -> String {
        let w = intOf(s["dflash_draft_quant_weight_bits"]) ?? 4
        let a = intOf(s["dflash_draft_quant_activation_bits"]) ?? 8
        let g = intOf(s["dflash_draft_quant_group_size"]) ?? 64
        return String(localized: "profile.detail.acceleration.dflash_quant",
                      defaultValue: "Drafter quant · W\(w)/A\(a) G\(g)",
                      comment: "DFlash drafter quantization chip; placeholders are weight bits, activation bits, group size")
    }

    private func specprefillLabel(_ s: [String: AnyCodable], on: Bool) -> String {
        let baseName = String(localized: "profile.detail.acceleration.specprefill",
                              defaultValue: "SpecPrefill",
                              comment: "Acceleration chip base name for SpecPrefill")
        guard on else { return baseName }
        // `specprefill_keep_pct` is a fraction (0.1–0.5 per
        // `omlx/model_settings.py:62`); render as a percent so the chip
        // reads naturally.
        let keep = doubleOf(s["specprefill_keep_pct"])
            .map { String(format: "%.0f%%", $0 * 100) }
        let thresh = intOf(s["specprefill_threshold"])
            .map { String(localized: "profile.detail.acceleration.specprefill.threshold",
                          defaultValue: "≥\(fmtCtx(Double($0))) tk",
                          comment: "SpecPrefill threshold suffix; placeholder is the formatted token count") }
        let parts = [keep, thresh].compactMap { $0 }
        return parts.isEmpty
            ? baseName
            : String(localized: "profile.detail.acceleration.specprefill.with_parts",
                     defaultValue: "SpecPrefill · \(parts.joined(separator: " / "))",
                     comment: "SpecPrefill chip with detail suffix; placeholder is the joined detail parts")
    }

    /// Count of non-empty key-value pairs in a `[String: AnyCodable]`
    /// dictionary blob. Returns nil if the field is absent, empty, or
    /// not a dict — those cases should hide the chip entirely.
    private func nonEmptyKwargCount(_ v: AnyCodable?) -> Int? {
        guard let v else { return nil }
        if let dict = v.value as? [String: AnyCodable] {
            return dict.isEmpty ? nil : dict.count
        }
        if let dict = v.value as? [String: Any] {
            return dict.isEmpty ? nil : dict.count
        }
        return nil
    }

    /// "Expose as model" toggle — mirrors the web dashboard's checkbox on
    /// per-model profiles. While on, the profile serves as its own model
    /// ID on /v1/models, overlaying its settings on the base model.
    @ViewBuilder
    private var exposeRow: some View {
        if let onToggleExpose {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Toggle(isOn: Binding(
                        get: { exposeAsModel },
                        set: { onToggleExpose($0) }
                    )) {
                        Text(String(localized: "profile.detail.expose_as_model",
                                    defaultValue: "Expose as model",
                                    comment: "Toggle on the profile detail card that publishes the profile as its own model ID"))
                            .font(.omlxText(11.5, weight: .medium))
                            .foregroundStyle(theme.textSecondary)
                    }
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    if exposeAsModel, let exposedModelId {
                        Text(exposedModelId)
                            .font(.omlxMono(11))
                            .foregroundStyle(theme.textTertiary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Spacer(minLength: 0)
                }
                .help(String(localized: "profile.detail.expose_as_model.help",
                             defaultValue: "Serve this profile as its own model ID on /v1/models, sharing the base model's engine",
                             comment: "Tooltip on the expose-as-model toggle"))
                if exposeAsModel && hasEngineFields {
                    Text(String(localized: "profile.detail.expose_as_model.engine_fields_hint",
                                defaultValue: "Engine-level settings in this profile only take effect when it is applied to the base model — they don't change the exposed model.",
                                comment: "Warning under the expose-as-model toggle when the profile carries engine-construction settings"))
                        .font(.omlxText(10.5))
                        .foregroundStyle(theme.amberDot)
                        .fixedSize(horizontal: false, vertical: true)
                }
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
