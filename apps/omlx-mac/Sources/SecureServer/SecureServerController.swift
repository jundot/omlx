// SPDX-License-Identifier: Apache-2.0

//
//  SecureServerController.swift
//  oMLX
//
//  App-side half of the Secure Server toggle: registers/observes the
//  privileged helper, asks for administrator authorization with the standard
//  macOS dialog, and relays the request over XPC.
//
//  The XPC plumbing lives in `SecureServerXPCClient`, which is deliberately
//  NOT @MainActor: NSXPCConnection invokes reply and error blocks on its own
//  queue, and a closure formed inside a MainActor context is asserted to run
//  on the main queue — doing that traps with
//  "Block was expected to execute on queue [com.apple.main-thread]".
//

import Foundation
import Observation
import Security
import ServiceManagement

enum SecureServerHelperStatus: Equatable {
    /// The daemon is installed, approved and answering.
    case ready
    /// Registered, but the user has to approve it in System Settings.
    case requiresApproval
    /// Not registered yet (first launch, or the user removed it).
    case notRegistered
    /// Registered but rejected by the system (an unsigned/ad-hoc build cannot
    /// register a LaunchDaemon) or not answering.
    case unavailable
}

enum SecureServerError: LocalizedError {
    case authorizationCancelled
    case authorizationFailed(OSStatus)
    case helperUnavailable
    case rejected(String)

    var errorDescription: String? {
        switch self {
        case .authorizationCancelled:
            return String(localized: "security.secure_server.error.cancelled",
                          defaultValue: "Administrator authorization was cancelled.",
                          comment: "Shown when the admin password dialog is dismissed")
        case let .authorizationFailed(status):
            return String(localized: "security.secure_server.error.authorization_failed",
                          defaultValue: "Administrator authorization failed. The Secure Server setting was not changed.",
                          comment: "Shown when authorization fails before reaching the helper")
                + " (OSStatus \(status))"
        case .helperUnavailable:
            return String(localized: "security.secure_server.error.helper_unavailable",
                          defaultValue: "The Secure Server helper is not running. Approve it in System Settings, then try again.",
                          comment: "Shown when the privileged helper cannot be reached")
        case let .rejected(message):
            return message
        }
    }
}

/// XPC may deliver an error and a reply for the same call; only the first
/// resumption of the continuation is legal.
private final class ReplyGate: @unchecked Sendable {
    private let lock = NSLock()
    private var used = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !used else { return false }
        used = true
        return true
    }
}

/// Owns the privileged Mach service connection. Non-isolated on purpose: see
/// the note at the top of this file.
final class SecureServerXPCClient: @unchecked Sendable {
    private let lock = NSLock()
    private var connection: NSXPCConnection?
    private var connectionID: UUID?

    private func proxy(
        onError: @escaping @Sendable (Error) -> Void
    ) -> SecureServerHelperProtocol? {
        lock.lock()
        defer { lock.unlock() }

        if connection == nil {
            let id = UUID()
            let newConnection = NSXPCConnection(
                machServiceName: SecureServer.machServiceName,
                options: .privileged
            )
            newConnection.remoteObjectInterface = NSXPCInterface(
                with: SecureServerHelperProtocol.self
            )
            // The daemon going away must not poison a later, healthy
            // connection: only clear the one this handler belongs to.
            newConnection.invalidationHandler = { [weak self] in
                self?.drop(id)
            }
            newConnection.resume()
            connection = newConnection
            connectionID = id
        }
        return connection?.remoteObjectProxyWithErrorHandler(onError)
            as? SecureServerHelperProtocol
    }

    private func drop(_ id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        guard connectionID == id else { return }
        connection = nil
        connectionID = nil
    }

    /// Current state from the helper, or nil when it cannot be reached.
    func getState() async -> SecureServerState? {
        await withCheckedContinuation { continuation in
            let gate = ReplyGate()
            let reply: @Sendable (Data?) -> Void = { data in
                guard gate.claim() else { return }
                let state = data.flatMap {
                    try? JSONDecoder().decode(SecureServerState.self, from: $0)
                } ?? .off
                continuation.resume(returning: state)
            }
            let onError: @Sendable (Error) -> Void = { _ in
                guard gate.claim() else { return }
                continuation.resume(returning: nil)
            }
            guard let proxy = proxy(onError: onError) else {
                if gate.claim() { continuation.resume(returning: nil) }
                return
            }
            proxy.getState(with: reply)
        }
    }

    /// Apply a change. Throws rather than latching, so the UI can never sit
    /// busy on a dead Mach service.
    func setEnabled(_ value: Bool, form: Data) async throws -> (Bool, String?) {
        try await withCheckedThrowingContinuation { continuation in
            let gate = ReplyGate()
            let reply: @Sendable (Bool, String?) -> Void = { success, message in
                guard gate.claim() else { return }
                continuation.resume(returning: (success, message))
            }
            let onError: @Sendable (Error) -> Void = { error in
                guard gate.claim() else { return }
                continuation.resume(
                    throwing: SecureServerError.rejected(error.localizedDescription)
                )
            }
            guard let proxy = proxy(onError: onError) else {
                if gate.claim() {
                    continuation.resume(throwing: SecureServerError.helperUnavailable)
                }
                return
            }
            proxy.setEnabled(value, authorization: form, with: reply)
        }
    }

    /// Measured enforcement, or nil when the helper cannot be reached. This
    /// replaces reading `state.json`: the file records what was asked for, and a
    /// hand-run `pfctl -d` leaves it asserting a lockdown that is not happening.
    func status() async -> SecureServerStatus? {
        await withCheckedContinuation { continuation in
            let gate = ReplyGate()
            let reply: @Sendable (Data) -> Void = { data in
                guard gate.claim() else { return }
                continuation.resume(
                    returning: try? JSONDecoder().decode(SecureServerStatus.self, from: data)
                )
            }
            let onError: @Sendable (Error) -> Void = { _ in
                guard gate.claim() else { return }
                continuation.resume(returning: nil)
            }
            guard let proxy = proxy(onError: onError) else {
                if gate.claim() { continuation.resume(returning: nil) }
                return
            }
            proxy.status(with: reply)
        }
    }

    /// Run the server under the sandbox account. No authorization form: this is
    /// strictly less privilege than the app already holds.
    func startServer(_ launch: SecureServerLaunch) async throws -> SecureServerLaunchResult {
        let request: Data
        do {
            request = try JSONEncoder().encode(launch)
        } catch {
            throw SecureServerError.rejected(error.localizedDescription)
        }
        return try await withCheckedThrowingContinuation { continuation in
            let gate = ReplyGate()
            let reply: @Sendable (Data) -> Void = { data in
                guard gate.claim() else { return }
                if let result = try? JSONDecoder().decode(SecureServerLaunchResult.self, from: data) {
                    continuation.resume(returning: result)
                } else {
                    continuation.resume(throwing: SecureServerError.helperUnavailable)
                }
            }
            let onError: @Sendable (Error) -> Void = { error in
                guard gate.claim() else { return }
                continuation.resume(
                    throwing: SecureServerError.rejected(error.localizedDescription)
                )
            }
            guard let proxy = proxy(onError: onError) else {
                if gate.claim() {
                    continuation.resume(throwing: SecureServerError.helperUnavailable)
                }
                return
            }
            proxy.startServer(request, with: reply)
        }
    }

    func stopServer() async -> Bool {
        await withCheckedContinuation { continuation in
            let gate = ReplyGate()
            let reply: @Sendable (Data) -> Void = { data in
                guard gate.claim() else { return }
                continuation.resume(
                    returning: (try? JSONDecoder().decode(SecureServerStopResult.self, from: data))?
                        .stopped ?? false
                )
            }
            let onError: @Sendable (Error) -> Void = { _ in
                guard gate.claim() else { return }
                continuation.resume(returning: false)
            }
            guard let proxy = proxy(onError: onError) else {
                if gate.claim() { continuation.resume(returning: false) }
                return
            }
            proxy.stopServer(with: reply)
        }
    }
}

@MainActor
@Observable
final class SecureServerController {
    /// One privileged connection per app process.
    static let shared = SecureServerController()

    /// Posted after a change really took effect. Whoever runs the server has to
    /// listen: whether that server may reach the network is decided by the uid
    /// owning its sockets, which is fixed at spawn, so a mode change means the
    /// running server is now the wrong kind of process.
    static let didChangeNotification = Notification.Name("app.omlx.secure-server.didChange")

    private(set) var isEnabled = false
    private(set) var isBusy = false
    var errorMessage: String?
    private(set) var helperStatus: SecureServerHelperStatus = .notRegistered
    /// Last measured status from the helper; nil until the first reply. The
    /// toggle shows what was *requested*, this shows what is *true*.
    private(set) var status: SecureServerStatus?

    private let service = SMAppService.daemon(plistName: SecureServer.helperPlistName)
    private let client = SecureServerXPCClient()
    /// Registering is what raises the approval prompt, so ask at most once per
    /// launch instead of on every refresh.
    private var didAttemptRegistration = false
    /// Set when the system rejects the daemon outright (unsigned build), so the
    /// UI can explain that instead of looping on "not registered".
    private var registrationRejected = false

    init() {
        // Seed from the record, not from the XPC reply. Whoever starts the server
        // reads `isEnabled` before the first reply can land, and a server can only be
        // sandboxed if the answer is already known when it is spawned.
        isEnabled = SecureServer.enabledOnRecord
        refresh()
    }

    // MARK: - Helper registration

    /// Registering a LaunchDaemon is what prompts for administrator approval;
    /// macOS then bootstraps it on every subsequent boot, which is what makes
    /// a reboot unable to silently drop the lockdown.
    /// One measured status, asked for now rather than read from the snapshot that
    /// `refresh` keeps: adopting a server needs its pid at launch, when the cached
    /// value may still be nil.
    func measuredStatus() async -> SecureServerStatus? {
        await client.status()
    }

    func refreshHelperRegistration() {
        if service.status == .notRegistered, !didAttemptRegistration {
            didAttemptRegistration = true
            do {
                try service.register()
            } catch {
                // An ad-hoc/unsigned build cannot register a system daemon.
                // Record it so the failure is visible instead of silent.
                registrationRejected = true
                NSLog("Secure Server: helper registration failed: %@",
                      error.localizedDescription)
            }
        }
        helperStatus = Self.map(service.status, rejected: registrationRejected)
    }

    private static func map(
        _ status: SMAppService.Status,
        rejected: Bool
    ) -> SecureServerHelperStatus {
        if rejected { return .unavailable }
        switch status {
        case .enabled: return .ready
        case .requiresApproval: return .requiresApproval
        case .notRegistered: return .notRegistered
        case .notFound: return .unavailable
        @unknown default: return .unavailable
        }
    }

    func openSystemSettingsLoginItems() {
        SMAppService.openSystemSettingsLoginItems()
    }

    // MARK: - State

    /// Re-read what the helper reports, measured rather than remembered: the
    /// state file only records the last requested change, and pf can be switched
    /// off from a terminal without the file ever learning about it. Also the
    /// app's recovery hook, because the first connection is made before the
    /// daemon is approved.
    func refresh() {
        refreshHelperRegistration()
        Task {
            if let reported = await client.status() {
                status = reported
                helperStatus = .ready
                // A measured change has to reach whoever runs the server exactly as a
                // requested one does: the record seeded `isEnabled` at launch, and if
                // the helper now disagrees, the running server is the wrong kind of
                // process and has to be replaced.
                if reported.enabled != isEnabled {
                    isEnabled = reported.enabled
                    NotificationCenter.default.post(
                        name: Self.didChangeNotification,
                        object: self,
                        userInfo: ["enabled": reported.enabled]
                    )
                }
            } else if helperStatus == .ready {
                helperStatus = .unavailable
            }
        }
    }

    /// True only when pf is enabled *and* holding our uid-scoped rules. Nothing
    /// else may claim the server is protected.
    var isEnforced: Bool { status?.enforced == true }

    /// Non-nil when the operator asked for the sandbox and it is not in force.
    var enforcementWarning: String? {
        guard let status, status.enabled, !status.enforced else { return nil }
        return status.detail ?? String(
            localized: "security.secure_server.error.not_enforced",
            defaultValue: "Secure Server is on but nothing is being blocked.",
            comment: "Shown when the lockdown was requested but pf is not enforcing it"
        )
    }

    // MARK: - Sandboxed server

    /// Run the server under the sandbox account. Used instead of spawning it in
    /// process: a server started as the console user owns sockets that match none
    /// of the rules, so starting it that way while the mode is on would leave the
    /// UI claiming protection that does not exist.
    func startServer(_ launch: SecureServerLaunch) async throws -> SecureServerLaunchResult {
        try await client.startServer(launch)
    }

    func stopServer() async -> Bool {
        await client.stopServer()
    }

    // MARK: - Toggle

    /// The switch's full bootstrap. One idempotent sequence spanning app and
    /// helper: register the privileged helper (one-time System Settings approval),
    /// ask for a fresh administrator authorization, let the helper lay out the
    /// config, create the account and home, load and verify the uid-scoped pf rules
    /// and enable pf (its single `bootstrap()`), then restart the server under
    /// `_omlx` via `didChangeNotification`. A failure leaves the mode off.
    func setEnabled(_ value: Bool) {
        guard !isBusy, value != isEnabled else { return }
        isBusy = true
        errorMessage = nil
        Task {
            defer { isBusy = false }
            do {
                refreshHelperRegistration()
                // Off the main actor: AuthorizationCopyRights below BLOCKS until the
                // administrator dialog is answered, and this Task inherits @MainActor
                // isolation, so calling it inline froze the whole app — every window and
                // the menubar — for as long as the dialog was up.
                let (form, auth) = try await Task.detached(priority: .userInitiated) {
                    try Self.authorization()
                }.value
                // The reference must outlive the helper's verification.
                defer { AuthorizationFree(auth.ref, []) }
                let (success, message) = try await client.setEnabled(value, form: form)
                if success {
                    isEnabled = value
                    helperStatus = .ready
                    NotificationCenter.default.post(
                        name: Self.didChangeNotification,
                        object: self,
                        userInfo: ["enabled": value]
                    )
                    // Measure it. `status` is what the pill reads, and it still holds
                    // the pre-flip answer (pf disabled, no rules) until something asks
                    // the helper again — which is how a working lockdown showed up as
                    // NOT ENFORCED until the screen was reopened or "Re-check" pressed.
                    refresh()
                } else {
                    throw SecureServerError.rejected(
                        message ?? String(localized: "security.secure_server.error.rejected",
                                          defaultValue: "The Secure Server helper refused the change.",
                                          comment: "Fallback error when the helper rejects a change")
                    )
                }
            } catch {
                // Keep the old value: a refused change must never look applied.
                errorMessage = error.localizedDescription
                refresh()
            }
        }
    }

    /// Run the standard authorization dialog for the manage right and hand
    /// back the 32-byte external form the helper can verify.
    ///
    /// The `AuthorizationRef` is returned, not freed here: an external form
    /// only names a live reference, and freeing it first makes the daemon's
    /// `AuthorizationCreateFromExternalForm` fail with errAuthorizationDenied
    /// (-60005). The caller frees it once the helper has answered.
    nonisolated private static func authorization() throws -> (form: Data, ref: AuthRefBox) {
        var auth: AuthorizationRef?
        var status = AuthorizationCreate(nil, nil, [], &auth)
        guard status == errAuthorizationSuccess, let auth else {
            throw SecureServerError.authorizationFailed(status)
        }
        // One owner for the reference: every early return below has to release it, and
        // a cancelled dialog is the common path.
        var handedOver = false
        defer { if !handedOver { AuthorizationFree(auth, []) } }

        status = SecureServer.authRightName.withCString { name in
            var item = AuthorizationItem(name: name, valueLength: 0, value: nil, flags: 0)
            return withUnsafeMutablePointer(to: &item) { itemPointer in
                var rights = AuthorizationRights(count: 1, items: itemPointer)
                return AuthorizationCopyRights(
                    auth,
                    &rights,
                    nil,
                    // No .preAuthorize: that obtains the credential without
                    // adding the right to this reference, so the daemon's
                    // later CopyRights would have nothing to extend and fails
                    // with errAuthorizationInteractionNotAllowed (-60007).
                    [.interactionAllowed, .extendRights],
                    nil
                )
            }
        }

        if status == errAuthorizationCanceled {
            throw SecureServerError.authorizationCancelled
        }
        guard status == errAuthorizationSuccess else {
            throw SecureServerError.authorizationFailed(status)
        }

        var form = AuthorizationExternalForm()
        status = AuthorizationMakeExternalForm(auth, &form)
        guard status == errAuthorizationSuccess else {
            // The defer above owns the release now.
            throw SecureServerError.authorizationFailed(status)
        }
        handedOver = true
        return (withUnsafeBytes(of: &form) { Data($0) }, AuthRefBox(auth))
    }
}

/// Carries an `AuthorizationRef` back from the detached authorization task.
///
/// `AuthorizationRef` is a C handle with no Swift concurrency annotation, so it cannot
/// cross an isolation boundary on its own. The box is `@unchecked` because the invariant
/// is enforced by construction rather than by the type: the reference is created and last
/// used inside `authorization()` on the detached task, then only read and freed by the
/// caller on the main actor after that task has completed. It is never used from two
/// places at once.
private struct AuthRefBox: @unchecked Sendable {
    let ref: AuthorizationRef
    init(_ ref: AuthorizationRef) { self.ref = ref }
}
