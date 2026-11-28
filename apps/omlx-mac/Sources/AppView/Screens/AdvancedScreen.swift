// Settings → Advanced. App-level maintenance that is not a server knob.
//
// Today it hosts the D1 danger zone: "Uninstall oMLX…" opens one sheet with a
// keep switch per scope, a byte summary before anything is removed, and the
// model directories it keeps named verbatim. Models and cluster/identity.json
// are kept by default. The helper is the only admin prompt; the app quits last.
// The model layer lives in Utils/Uninstall.swift and is driven by XCTests over
// injected paths — nothing real is deleted in a test.

import SwiftUI
import AppKit

struct AdvancedScreen: View {
    @Environment(AppServices.self) private var services
    @Environment(\.omlxTheme) private var theme
    @State private var showUninstall = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(
                String(localized: "advanced.section.title",
                       defaultValue: "Advanced",
                       comment: "Section header for the app-level Advanced settings screen"),
                subtitle: String(localized: "advanced.section.sub",
                                 defaultValue: """
                                 Maintenance for the app itself. These actions affect your whole installation.
                                 """,
                                 comment: "Subtitle for the Advanced settings screen")
            )

            ListGroup {
                Row(
                    label: String(localized: "advanced.uninstall.label",
                                  defaultValue: "Uninstall oMLX",
                                  comment: "Row label for the uninstall danger zone"),
                    sublabel: String(localized: "advanced.uninstall.sub",
                                     defaultValue: """
                                     Remove the app, helper, runtime and settings. \
                                     Models and the node identity are kept unless you choose otherwise.
                                     """,
                                     comment: "Sublabel explaining what the uninstall removes and keeps"),
                    isLast: true
                ) {
                    Button(String(localized: "advanced.uninstall.button",
                                  defaultValue: "Uninstall…",
                                  comment: "Button that opens the uninstall sheet")) {
                        showUninstall = true
                    }
                    .buttonStyle(.omlx(.normal))
                }
            }
        }
        .sheet(isPresented: $showUninstall) {
            UninstallSheet(vm: UninstallSheetVM(services: services))
        }
    }
}

// MARK: - Sheet view model

@MainActor
@Observable
final class UninstallSheetVM {
    let locator: UninstallLocator
    var options: UninstallOptions = .default
    private(set) var plan: UninstallPlan
    private(set) var report: UninstallReport?
    private(set) var isRunning = false

    private let actions: UninstallActions
    private let stopServer: () async -> Void
    private let quitApp: () -> Void

    /// Designated init — everything injectable so XCTests can drive the sheet
    /// against temporary directories with a recording seam and a fake quit.
    init(
        locator: UninstallLocator,
        actions: UninstallActions,
        stopServer: @escaping () async -> Void = {},
        quitApp: @escaping () -> Void = {}
    ) {
        self.locator = locator
        self.actions = actions
        self.stopServer = stopServer
        self.quitApp = quitApp
        self.plan = UninstallPlanner.build(locator: locator, options: .default)
    }

    convenience init(services: AppServices) {
        let live = UninstallActions.live()
        self.init(
            locator: .live(config: services.config),
            actions: live,
            stopServer: { await services.stopServer() },
            quitApp: { live.quitApp() }
        )
    }

    func recomputePlan() {
        plan = UninstallPlanner.build(locator: locator, options: options)
    }

    func setModelDir(_ dir: String, remove: Bool) {
        if remove { options.modelDirsToRemove.insert(dir) }
        else { options.modelDirsToRemove.remove(dir) }
        recomputePlan()
    }

    /// Stop the server, then remove every chosen scope. The runner records a
    /// cancelled helper dialog and still finishes the rest; quitting is the
    /// explicit final step (`finish`) so the report can be shown first.
    func run() async {
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }
        await stopServer()
        recomputePlan()
        report = UninstallRunner.run(plan: plan, actions: actions, quitAfter: false)
    }

    /// The app quits last. Called from the report's final button.
    func finish() {
        quitApp()
    }

    // Summary helpers for the sheet.

    var keptLine: String {
        var parts: [String] = []
        if plan.keepsIdentity {
            parts.append(String(localized: "advanced.uninstall.kept_identity",
                                defaultValue: "node identity (cluster/identity.json)",
                                comment: "Kept-item label for the cluster node identity"))
        }
        for dir in plan.keptModelDirs { parts.append(dir) }
        if parts.isEmpty {
            return String(localized: "advanced.uninstall.kept_none",
                          defaultValue: "Nothing is kept.",
                          comment: "Kept-items line when the user chose to remove everything")
        }
        return String(localized: "advanced.uninstall.kept_prefix",
                      defaultValue: "Kept: ",
                      comment: "Prefix for the list of items the uninstall keeps")
            + parts.joined(separator: ", ")
    }
}

// MARK: - Sheet

private struct UninstallSheet: View {
    @Bindable var vm: UninstallSheetVM
    @Environment(\.dismiss) private var dismiss
    @Environment(\.omlxTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let report = vm.report {
                        reportBody(report)
                    } else {
                        optionsBody
                        summaryBody
                    }
                }
                .padding(20)
            }
            footer
        }
        .frame(width: 560, height: 620)
        .onChange(of: vm.options) { vm.recomputePlan() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(String(localized: "advanced.uninstall.sheet_title",
                        defaultValue: "Uninstall oMLX",
                        comment: "Title of the uninstall sheet"))
                .font(.omlxDisplay(20, weight: .semibold))
                .foregroundStyle(theme.text)
            Text(String(localized: "advanced.uninstall.sheet_sub",
                        defaultValue: """
                        Choose what to remove. Models and the node identity are kept by default. \
                        The helper needs one admin dialog; the app quits last.
                        """,
                        comment: "Subtitle of the uninstall sheet"))
                .font(.omlxText(DesignTokens.FontSize.subheadline))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var optionsBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            ListGroup {
                scopeRow(String(localized: "advanced.uninstall.scope_app", defaultValue: "App",
                                comment: "Uninstall scope: the app bundle"),
                         sub: "Move oMLX.app to the Trash.", binding: $vm.options.removeApp)
                scopeRow(String(localized: "advanced.uninstall.scope_helper", defaultValue: "System helper",
                                comment: "Uninstall scope: the privileged helper"),
                         sub: "Remove the privileged helper (one admin dialog).", binding: $vm.options.removeHelper)
                scopeRow(String(localized: "advanced.uninstall.scope_runtime", defaultValue: "Runtime",
                                comment: "Uninstall scope: the Python runtime"),
                         sub: "Python runtime, logs and updater state.", binding: $vm.options.removeRuntime)
                scopeRow(String(localized: "advanced.uninstall.scope_settings", defaultValue: "Settings",
                                comment: "Uninstall scope: app settings"),
                         sub: "settings.json, model settings and preferences.", binding: $vm.options.removeSettings)
                scopeRow(String(localized: "advanced.uninstall.scope_cluster", defaultValue: "Cluster state",
                                comment: "Uninstall scope: cluster state"),
                         sub: "Paired devices, deployments, measurements and operations.",
                         binding: $vm.options.removeClusterState)
                scopeRow(String(localized: "advanced.uninstall.scope_identity", defaultValue: "Node identity",
                                comment: "Uninstall scope: the cluster node identity"),
                         sub: "cluster/identity.json — removing it re-rolls this Mac's node id.",
                         binding: $vm.options.removeIdentity)
                scopeRow(String(localized: "advanced.uninstall.scope_diagnostics", defaultValue: "Diagnostics",
                                comment: "Uninstall scope: logs and caches"),
                         sub: "Logs and caches under ~/Library.", binding: $vm.options.removeDiagnostics, isLast: true)
            }

            if !vm.locator.modelDirs.isEmpty {
                Text(String(localized: "advanced.uninstall.models_header",
                            defaultValue: "Model directories (kept unless you opt in)",
                            comment: "Header for the per-directory model deletion opt-in"))
                    .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                    .foregroundStyle(theme.text)
                ListGroup {
                    ForEach(Array(vm.locator.modelDirs.enumerated()), id: \.element) { index, dir in
                        Row(
                            label: dir,
                            sublabel: String(localized: "advanced.uninstall.models_remove_sub",
                                             defaultValue: "Remove this directory and every model in it.",
                                             comment: "Sublabel for opting into removing a model directory"),
                            isLast: index == vm.locator.modelDirs.count - 1
                        ) {
                            RowSwitch(isOn: Binding(
                                get: { vm.options.modelDirsToRemove.contains(dir) },
                                set: { vm.setModelDir(dir, remove: $0) }
                            ))
                        }
                    }
                }
            }
        }
    }

    private func scopeRow(
        _ label: String, sub: String,
        binding: Binding<Bool>, isLast: Bool = false
    ) -> some View {
        Row(label: label, sublabel: sub, isLast: isLast) {
            RowSwitch(isOn: binding)
        }
    }

    private var summaryBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(String(localized: "advanced.uninstall.summary_title",
                        defaultValue: "Will remove",
                        comment: "Header for the byte summary of what the uninstall removes"))
                .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                .foregroundStyle(theme.text)
            if vm.plan.items.isEmpty {
                Text(String(localized: "advanced.uninstall.summary_empty",
                            defaultValue: "Nothing selected.",
                            comment: "Byte summary empty state"))
                    .font(.omlxText(DesignTokens.FontSize.subheadline))
                    .foregroundStyle(theme.textSecondary)
            } else {
                ForEach(Array(vm.plan.items.enumerated()), id: \.offset) { _, item in
                    HStack {
                        Text(item.url.path)
                            .font(.omlxMono(11))
                            .foregroundStyle(theme.textSecondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: 12)
                        Text(UninstallFormat.bytes(item.bytes))
                            .font(.omlxMono(11))
                            .foregroundStyle(theme.text)
                    }
                }
                HStack {
                    Text(String(localized: "advanced.uninstall.summary_total",
                                defaultValue: "Total",
                                comment: "Total byte count label"))
                        .font(.omlxText(DesignTokens.FontSize.body, weight: .medium))
                        .foregroundStyle(theme.text)
                    Spacer()
                    Text(UninstallFormat.bytes(vm.plan.totalBytes))
                        .font(.omlxMono(12))
                        .foregroundStyle(theme.text)
                }
            }
            Text(vm.keptLine)
                .font(.omlxText(DesignTokens.FontSize.subheadline))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(theme.groupBg)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private func reportBody(_ report: UninstallReport) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(String(localized: "advanced.uninstall.report_title",
                        defaultValue: "Uninstall finished",
                        comment: "Title shown after the uninstall ran"))
                .font(.omlxText(DesignTokens.FontSize.body, weight: .semibold))
                .foregroundStyle(theme.text)

            if report.helperLeftInPlace {
                Text(String(localized: "advanced.uninstall.report_helper_left",
                            defaultValue: """
                            The system helper was left in place (the admin dialog was cancelled or failed). \
                            Everything else was removed.
                            """,
                            comment: "Report line when the helper removal was cancelled or failed"))
                    .font(.omlxText(DesignTokens.FontSize.subheadline))
                    .foregroundStyle(theme.warningText)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(Array(report.errors.enumerated()), id: \.offset) { _, error in
                Text(error)
                    .font(.omlxText(DesignTokens.FontSize.caption))
                    .foregroundStyle(theme.warningText)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(vm.keptLine)
                .font(.omlxText(DesignTokens.FontSize.subheadline))
                .foregroundStyle(theme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if report.appRemoved {
                Text(String(localized: "advanced.uninstall.report_quit_hint",
                            defaultValue: """
                            oMLX was moved to the Trash. Quit now to finish; relaunch from the DMG to restore.
                            """,
                            comment: "Hint that the app was trashed and quitting is the final step"))
                    .font(.omlxText(DesignTokens.FontSize.subheadline))
                    .foregroundStyle(theme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var footer: some View {
        HStack {
            if vm.report == nil {
                Button(String(localized: "common.cancel", defaultValue: "Cancel")) { dismiss() }
                    .buttonStyle(.omlx(.normal))
                    .disabled(vm.isRunning)
                Spacer()
                if vm.isRunning { ProgressView().controlSize(.small) }
                Button(String(localized: "advanced.uninstall.confirm",
                              defaultValue: "Uninstall…",
                              comment: "Confirm button that runs the uninstall")) {
                    Task { await vm.run() }
                }
                .buttonStyle(.omlx(.primary))
                .disabled(vm.isRunning || vm.plan.items.isEmpty)
            } else {
                Spacer()
                if vm.report?.appRemoved == true {
                    Button(String(localized: "advanced.uninstall.quit",
                                  defaultValue: "Quit oMLX",
                                  comment: "Final button that quits the app after uninstall")) {
                        vm.finish()
                    }
                    .buttonStyle(.omlx(.primary))
                } else {
                    Button(String(localized: "common.done", defaultValue: "Done")) { dismiss() }
                        .buttonStyle(.omlx(.primary))
                }
            }
        }
        .padding(20)
    }
}