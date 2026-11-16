// SPDX-License-Identifier: Apache-2.0

//
//  SecureServerShared.swift
//  oMLX
//
//  Layout, protocol and payload for the Secure Server lockdown. Compiled into
//  both the app and the privileged helper, so it must stay Foundation-only.
//

import Foundation

/// Everything the app and the helper must agree on.
///
/// The config directory is root-owned: only the privileged helper writes there,
/// which is what makes "turning it off needs an administrator" true. The files
/// are world-readable so both the app and the Python server can report status
/// without privileges.
enum SecureServer {
    /// `MachServices` key and the name the app connects to.
    static let machServiceName = "com.omlx.secure-server-helper"
    /// LaunchDaemon plist name inside Contents/Library/LaunchDaemons.
    static let helperPlistName = "com.omlx.secure-server-helper.plist"
    /// Custom authorization right every state change is gated on.
    static let authRightName = "app.omlx.secure-server.manage"

    static let configDirectory = "/Library/Application Support/oMLX/secure-server"
    static let statePath = configDirectory + "/state.json"
    static let pfAnchorConfigPath = configDirectory + "/pf-anchor.conf"

    /// Dedicated account the server runs as while Secure Server mode is on.
    /// pf can only match a packet to a process through the effective uid that
    /// owned the socket when it was created, so "jail just this process" means
    /// "run that process under its own uid" and give every rule a `user`
    /// qualifier. Nothing else on the Mac belongs to it, so nothing else is
    /// affected. It stays in group `staff` (20) because a home directory is
    /// `drwxr-x---`: group membership keeps existing world-readable models
    /// readable, while writes are redirected to `sandboxHome`.
    static let sandboxUser = "_omlx"
    static let sandboxGroup = "staff"
    /// Writable HOME/TMPDIR for the sandboxed server: the user's real home is
    /// not writable by it.
    static let sandboxHome = configDirectory + "/sandbox"
    /// stdout/stderr of the sandboxed server. The helper opens this as root and
    /// hands the descriptor to the child, so the sandbox needs no write access.
    static let serverLogPath = configDirectory + "/server.log"
    /// Where the sandboxed server puts what it *creates*: logs and the disk cache
    /// default into the user's base directory, which the sandbox cannot write, so
    /// both are redirected here. `omlx/settings.py` reads exactly these variables.
    static let sandboxLogDir = sandboxHome + "/logs"
    static let sandboxCacheDir = sandboxHome + "/cache"
    static let logDirEnvironmentKey = "OMLX_LOG_DIR"
    static let cacheDirEnvironmentKey = "OMLX_SSD_CACHE_DIR"

    /// Where the sandboxed server keeps *its* settings, and the variable that sends
    /// it there. The file is a copy seeded at every start: the server saves by
    /// renaming a temp file over the target, so the saved file belongs to the sandbox
    /// account, while the app reads the real `settings.json` as the logged-in user —
    /// measured as the app coming up on defaults whenever the mode was on. A copy
    /// keeps both true, and `omlx/settings.py` reads this variable.
    static let sandboxSettingsPath = sandboxHome + "/.omlx/settings.json"
    static let settingsFileEnvironmentKey = "OMLX_SETTINGS_FILE"
    /// What the copy looked like when it was seeded, so teardown can tell a save made
    /// by the sandbox from an edit the user made while the server was running. It sits
    /// in the root-owned config directory, *not* beside the copy: teardown writes the
    /// file it names as root and takes its owner from it, so a record the sandbox
    /// could rewrite would turn the account this feature contains into an arbitrary
    /// root write.
    static let settingsSeedPath = configDirectory + "/settings.seed.json"

    /// Where the daemon records the sandboxed server's pid once the spawn is
    /// confirmed. As soon as anything has run under a uid, launchd also owns
    /// per-user agents (`distnoted`, `lsd`, `cfprefsd`) under it, so "the oldest
    /// process with this uid" is not the server; this pid is validated — alive, and
    /// owned by the sandbox account — before being treated as one.
    static let sandboxPIDPath = sandboxHome + "/server.pid"

    /// The port the sandboxed server was launched on, so teardown can wait for the
    /// *port* to be free rather than only for the recorded pid to disappear: a
    /// descendant or an orphan can still hold the listening socket after that.
    static let sandboxPortPath = sandboxHome + "/server.port"

    /// The only external programs the helper still runs (see docs/secure-server.md):
    /// pfctl owns the kernel firewall, dscl creates the account once, chmod appends
    /// ACL entries. Everything else is done with in-process syscalls.
    static let dsclPath = "/usr/bin/dscl"
    static let chmodPath = "/bin/chmod"

    /// pf sub-anchor under the `anchor "com.apple/*"` point in /etc/pf.conf,
    /// so the main ruleset is never edited or replaced.
    static let pfAnchor = "com.apple/900.omlx.secure"
    static let pfctlPath = "/sbin/pfctl"

    /// The rules file: a deny-by-default block for the sandbox account, scoped by
    /// `user`, with loopback allowed first.
    ///
    /// Every rule names `user \(SecureServer.sandboxUser)`. That is the whole
    /// point: an unqualified `block drop out all` locks the entire Mac, which
    /// takes the updater, the browser and the terminal down with the server.
    ///
    /// `user` holds in both directions, which is what lets the mode mean what it
    /// says. For packets arriving at a local listener it matches the uid listening
    /// on the destination port (pf.conf(5)), so the block rules keep *other
    /// machines* off the sandboxed server while the loopback passes keep this Mac
    /// able to reach it. Measured on macOS 26 against a `_omlx` listener: with only
    /// the inbound block loaded the console user's connection is dropped (that
    /// rule's `Packets:` counter climbs, curl times out); with the loopback pass
    /// loaded the same connection answers 200 and the pass rule takes the hits.
    ///
    /// Only TCP and UDP can be associated with a user, so no protocol-less rule
    /// appears here — one would match every other process's ICMP and IPIP traffic
    /// too. `quick` stops at the first match, so the loopback passes must stay
    /// above the blocks.
    static let pfAnchorRules = """
    # oMLX Secure Server - lockdown of the sandboxed server account only.
    # Loaded via: pfctl -a com.apple/900.omlx.secure -f <this file>
    # Every rule is scoped with `user`: no other process on this Mac is matched.
    # Loopback first: local clients keep working, `quick` ends the search here.
    # Both protocols in both directions, because a UDP listener under this account
    # would otherwise be reachable from the network while the docs say it cannot be.
    pass in  quick on lo0 proto tcp from any to any user \(SecureServer.sandboxUser) keep state
    pass in  quick on lo0 proto udp from any to any user \(SecureServer.sandboxUser) keep state
    pass out quick on lo0 proto tcp from any user \(SecureServer.sandboxUser) keep state
    pass out quick on lo0 proto udp from any user \(SecureServer.sandboxUser) keep state
    # Everything off-box is refused, in both directions.
    block drop in  quick proto tcp from any to any user \(SecureServer.sandboxUser)
    block drop in  quick proto udp from any to any user \(SecureServer.sandboxUser)
    block drop out quick proto tcp from any user \(SecureServer.sandboxUser)
    block drop out quick proto udp from any user \(SecureServer.sandboxUser)

    """

    /// How the kernel reports a rule scoped to `uid`. pf resolves names as the rules
    /// are loaded, so `user _omlx` never appears in `pfctl -sr` output: a guard that
    /// looked for the name would match nothing and approve everything.
    static func scopeMarker(uid: Int) -> String { "user = \(uid)" }

    /// The first rule not scoped to `uid`, or nil when every one of them is. The
    /// daemon runs this against what the kernel has *actually* loaded and refuses to
    /// enable pf until it returns nil — the check that makes "an older, unqualified
    /// ruleset was still sitting in the anchor, so enabling pf blacked out the Mac"
    /// impossible rather than merely unlikely.
    ///
    /// The uid has to match as a whole number: a plain substring test would accept
    /// `user = 5020` as proof that a rule was scoped to `502`, and an anchor holding
    /// somebody else's uid would be enabled on the strength of it.
    static func firstUnscopedRule(_ rules: [String], sandboxUID uid: Int?) -> String? {
        guard !rules.isEmpty else { return "the anchor is empty" }
        guard let uid else { return "the \(sandboxUser) account does not exist" }
        let bound = try? NSRegularExpression(pattern: "user = \(uid)(?![0-9])")
        return rules.first { rule in
            guard let bound else { return true }
            return bound.firstMatch(in: rule, range: NSRange(rule.startIndex..., in: rule)) == nil
        }
    }

    /// Overrides the state file location for tests and a development install. The
    /// Python server follows the same variable (`omlx/secure_mode.py`), so one
    /// setting moves both sides.
    ///
    /// **Debug builds only.** This decides whether the server may be spawned by the
    /// helper at all, so an environment variable that can answer "off" would be a
    /// way to make the app start an ordinary, unrestricted server while the real
    /// record says the mode is on — exactly the outcome the guard exists to prevent.
    static let statePathEnvironmentKey = "OMLX_SECURE_MODE_STATE_PATH"

    /// What was last *requested*, read straight off the state file.
    ///
    /// The measured status arrives over XPC, and the launch path cannot wait for a
    /// round trip: whether a server may reach the network is decided by the uid
    /// owning its listening socket, which is fixed the moment it is spawned. So the
    /// question "must this server go through the helper?" is answered here,
    /// synchronously, from a file the world can read and only root can write.
    /// Missing, unreadable or malformed means off — the rule the server follows too.
    static func recordedState() -> SecureServerState {
        #if DEBUG
        let path = ProcessInfo.processInfo.environment[statePathEnvironmentKey] ?? statePath
        #else
        let path = statePath
        #endif
        guard let data = FileManager.default.contents(atPath: path),
              let state = try? JSONDecoder().decode(SecureServerState.self, from: data)
        else { return .off }
        return state
    }

    static var enabledOnRecord: Bool { recordedState().enabled }

    /// The reference token from `pfctl -E`, out of the text it printed.
    ///
    /// macOS writes `pf enabled` and `Token : <n>` to **stderr**, not stdout, while
    /// every other `pfctl` call puts its useful output on stdout and its ALTQ banner
    /// on stderr — which is why the runner separates the streams everywhere else.
    /// Reading only stdout here is what made a real flip fail with "did not report a
    /// token" *after* pf had already been enabled: the reference was taken and the
    /// daemon could not name it, so it could not release it either. Hence: the caller
    /// hands in both streams, and only this one line is looked at.
    static func enableToken(in output: String) -> String? {
        for line in output.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2,
                  parts[0].trimmingCharacters(in: .whitespaces) == "Token"
            else { continue }
            let value = parts[1].trimmingCharacters(in: .whitespaces)
            return value.isEmpty ? nil : value
        }
        return nil
    }
}

/// `state.json` payload. `pfToken` is the reference token from `pfctl -E`,
/// released again on the way out so pf stops when nothing else needs it.
struct SecureServerState: Codable, Equatable {
    var enabled: Bool
    var pfToken: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case pfToken = "pf_token"
    }

    static let off = SecureServerState(enabled: false, pfToken: nil)
}

/// What the app asks the helper to run. The helper is the only process that can
/// change uid, and pf tags a socket with the uid that created it, so the
/// sandboxed server has to be *spawned* as `SecureServer.sandboxUser` — a root
/// server that drops privileges later keeps root's credentials and would slip
/// past every rule.
struct SecureServerLaunch: Codable, Equatable {
    var executable: String
    var arguments: [String]
    var environment: [String: String]
    /// Port the server binds, so a conflict can be named instead of guessed.
    var port: Int
    /// Everything the sandbox must be able to reach: the interpreter, the
    /// package it imports, the models it loads. Changing uid means losing the
    /// access the logged-in user had, and a `drwx------` ancestor anywhere on the
    /// path makes the server fail to start rather than fail loudly.
    var accessPaths: [String]
    /// Paths the sandbox may also modify — the settings files the server owns and
    /// the directory their atomic replace needs. Split out from `accessPaths`
    /// because write is a different ask: reading the user's models must not imply
    /// the right to delete them. Still capped to paths the requesting user owns.
    var writablePaths: [String]?
    /// The settings file the *user* reads, which the sandbox only ever reads: the
    /// helper seeds `sandboxSettingsPath` from it at every start and hands the server
    /// `OMLX_SETTINGS_FILE`, so the real file never becomes sandbox-owned.
    var settingsPath: String?
}

/// Reply to a start request: exactly one field is set.
struct SecureServerLaunchResult: Codable, Equatable {
    var pid: Int?
    var error: String?

    static func failure(_ reason: String) -> SecureServerLaunchResult {
        SecureServerLaunchResult(pid: nil, error: reason)
    }
}

/// Measured state, as opposed to `SecureServerState`, which only records what
/// was asked for. The app may show the mode as ON only when `enforced` is true:
/// `pfctl -d` from a terminal, or a reboot into a pf-disabled state, leaves the
/// state file asserting a lockdown that is not happening, and a security
/// control that lies about itself is worse than none.
struct SecureServerStatus: Codable, Equatable {
    var enabled: Bool
    var pfEnabled: Bool
    var rulesLoaded: Bool
    var sandboxUID: Int?
    var serverPID: Int?
    var detail: String?

    var enforced: Bool { enabled && pfEnabled && rulesLoaded }

    /// What a refused caller is told: nothing is claimed, nothing measured.
    static let unavailable = SecureServerStatus(
        enabled: false, pfEnabled: false, rulesLoaded: false,
        sandboxUID: nil, serverPID: nil,
        detail: "the calling process is not the oMLX application"
    )
}

/// Reply to `stopServer`.
struct SecureServerStopResult: Codable, Equatable {
    var stopped: Bool
    var error: String?
}

/// The helper's XPC surface. Changing the flag needs proof of administrator
/// authorization; running the server under the sandbox account does not,
/// because it is strictly *less* privilege than the app already has.
@objc(SecureServerHelperProtocol)
protocol SecureServerHelperProtocol {
    /// Raw bytes of `state.json`, or nil when the helper has never written it.
    func getState(with reply: @escaping (Data?) -> Void)

    /// `authorization` is a 32-byte `AuthorizationExternalForm` for the
    /// `SecureServer.authRightName` right. `reply` carries (success, error).
    func setEnabled(_ enabled: Bool, authorization: Data, with reply: @escaping (Bool, String?) -> Void)

    /// `launch` is JSON `SecureServerLaunch`; the reply is JSON
    /// `SecureServerLaunchResult`. Replaces any server the helper already runs.
    func startServer(_ launch: Data, with reply: @escaping (Data) -> Void)

    /// Kills the sandboxed server, then sweeps the account so nothing it
    /// spawned survives. Reply is JSON `SecureServerStopResult`.
    func stopServer(with reply: @escaping (Data) -> Void)

    /// JSON `SecureServerStatus`, measured from pf and the child process.
    func status(with reply: @escaping (Data) -> Void)
}
