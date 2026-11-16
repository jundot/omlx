// SPDX-License-Identifier: Apache-2.0

//
//  main.swift
//  oMLXSecureServerHelper
//
//  Privileged daemon behind the Secure Server toggle. Runs as root from
//  launchd. Two jobs, both of which need root and neither of which the app can
//  do for itself:
//
//    1. own the pf anchor, so the sandbox account has no route off the box;
//    2. spawn the server as that account, since changing uid at exec time is
//       what makes pf see the sandbox uid on the listening socket.
//
//  Every state change is gated on a fresh administrator authorization; running
//  the server needs only proof that the caller is the oMLX app itself.
//

import CryptoKit
import Foundation
import Security
import os

private let log = Logger(subsystem: "app.omlx.secure-server", category: "helper")

// MARK: - Errors

private enum HelperError: Error, CustomStringConvertible {
    case run(String, Int32, String)
    case noToken(String)
    case emptyAnchor
    case unscopedRules(String)
    case noAccount
    case authorization(String)
    case notEnabled
    case serverStillRunning
    case startFailed(String)

    var description: String {
        switch self {
        case let .run(command, status, output):
            let detail = output.isEmpty ? "no output" : output
            return "\(command) failed (exit \(status)): \(detail)"
        case let .noToken(output):
            // Seen in the wild as "did not report a token:" with nothing after it, when
            // only stdout was read and the token was on stderr. Say which it was.
            let detail = output.isEmpty ? "no output on either stream" : output
            return "pfctl -E did not report a token (\(detail))"
        case .emptyAnchor:
            return "pf anchor \(SecureServer.pfAnchor) is empty after loading the rules"
        case let .unscopedRules(line):
            return "pf anchor \(SecureServer.pfAnchor) holds a rule that is not scoped to \(SecureServer.sandboxUser): \(line)"
        case .noAccount:
            return "could not create or find the \(SecureServer.sandboxUser) account"
        case let .authorization(reason):
            return reason
        case .notEnabled:
            return "Secure Server mode is off, so the app runs the server itself"
        case .serverStillRunning:
            return "the sandboxed server could not be stopped, so Secure Server stays on"
        case let .startFailed(reason):
            return reason
        }
    }
}

// MARK: - Running programs

private enum Runner {
    /// Run a program and return its stdout.
    ///
    /// stderr is deliberately kept out of the result: pfctl prints
    /// `No ALTQ support in kernel` there on *every* invocation, and merging the
    /// streams injects that banner into parsed output — it turned a rule list
    /// into seven "rules" and made the scope check unreadable. stderr is used
    /// only for failure text. (Output is assumed small enough to fit a pipe;
    /// pfctl, dscl and chmod never come close.)
    /// - Parameter includingStandardError: return both streams, in that order. Only
    ///   `pfctl -E` needs this: it reports its reference token on stderr, while every
    ///   other call here puts its data on stdout and only the ALTQ banner on stderr.
    @discardableResult
    static func run(
        _ path: String, _ arguments: [String], includingStandardError: Bool = false
    ) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        process.standardInput = FileHandle.nullDevice

        try process.run()
        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        var output = String(decoding: outData, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if includingStandardError {
            let text = String(decoding: errData, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            output = [output, text].filter { !$0.isEmpty }.joined(separator: "\n")
        }
        guard process.terminationStatus == 0 else {
            let detail = String(decoding: errData, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw HelperError.run(
                (path as NSString).lastPathComponent + " " + arguments.joined(separator: " "),
                process.terminationStatus,
                detail.isEmpty ? output : detail
            )
        }
        return output
    }
}

// MARK: - pf and the config directory

private enum Store {
    static func ensureLayout() throws {
        if !FileManager.default.fileExists(atPath: SecureServer.configDirectory) {
            try FileManager.default.createDirectory(
                atPath: SecureServer.configDirectory,
                withIntermediateDirectories: true
            )
        }
        try own(SecureServer.configDirectory, mode: 0o755)

        let rules = SecureServer.pfAnchorRules
        let existing = try? String(contentsOfFile: SecureServer.pfAnchorConfigPath, encoding: .utf8)
        if existing != rules {
            try rules.write(toFile: SecureServer.pfAnchorConfigPath, atomically: true, encoding: .utf8)
        }
        try own(SecureServer.pfAnchorConfigPath, mode: 0o644)

        if !FileManager.default.fileExists(atPath: SecureServer.statePath) {
            try write(.off)
        }
    }

    /// Re-assert root:wheel and the exact mode even when the path pre-existed:
    /// the whole point is that no user-writable copy can shadow the real one.
    private static func own(_ path: String, mode: Int) throws {
        try FileManager.default.setAttributes(
            [
                .ownerAccountID: 0,
                .groupOwnerAccountID: 0,
                .posixPermissions: mode
            ],
            ofItemAtPath: path
        )
    }

    static func read() -> SecureServerState {
        guard
            let data = FileManager.default.contents(atPath: SecureServer.statePath),
            let state = try? JSONDecoder().decode(SecureServerState.self, from: data)
        else {
            return .off
        }
        return state
    }

    static func write(_ state: SecureServerState) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try data.write(to: URL(fileURLWithPath: SecureServer.statePath), options: .atomic)
        try own(SecureServer.statePath, mode: 0o644)
    }

    @discardableResult
    static func pfctl(_ arguments: [String]) throws -> String {
        try Runner.run(SecureServer.pfctlPath, arguments)
    }

    /// The reference token of an `-E`/`-X` pair. Parsed by the shared implementation
    /// so the app's tests cover exactly the code this daemon runs.
    static func token(in output: String) -> String? { SecureServer.enableToken(in: output) }

    // MARK: Measurement

    static func pfIsEnabled() -> Bool {
        guard let info = try? pfctl(["-s", "info"]) else { return false }
        return info.contains("Status: Enabled")
    }

    /// The rules the kernel currently holds for our anchor.
    static func loadedRules() -> [String] {
        guard let output = try? pfctl(["-a", SecureServer.pfAnchor, "-sr"]) else { return [] }
        return output
            .split(whereSeparator: \.isNewline)
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    }

    /// The offending rule, or nil when every loaded rule is scoped to the sandbox
    /// account. The implementation lives in the shared target so the app's tests
    /// exercise the same code that guards the transition here, instead of a copy
    /// that could drift while the real check stayed broken.
    static func rulesAreScoped(_ rules: [String]) -> String? {
        SecureServer.firstUnscopedRule(rules, sandboxUID: Sandbox.uid.map(Int.init))
    }

    // MARK: Transitions

    /// Flush, load, verify, enable — in that order, and never any other.
    ///
    /// Enabling pf is the last step because pf may be sitting on an anchor that
    /// still holds an older ruleset. An earlier build of this daemon loaded the
    /// rules *after* `pfctl -E`, which meant that a stale host-wide ruleset
    /// (`block drop out all`) became active the moment pf was enabled: the whole
    /// Mac lost its network, not just the server. Flushing first and refusing to
    /// enable unless the anchor reads back as uid-scoped removes that class of
    /// failure entirely.
    static func applyLockdown() throws -> String {
        try ensureLayout()
        _ = try? pfctl(["-a", SecureServer.pfAnchor, "-F", "all"])
        try pfctl(["-a", SecureServer.pfAnchor, "-f", SecureServer.pfAnchorConfigPath])

        let rules = loadedRules()
        guard !rules.isEmpty else { throw HelperError.emptyAnchor }
        if let offender = rulesAreScoped(rules) {
            // Do not leave unverified rules in the kernel.
            _ = try? pfctl(["-a", SecureServer.pfAnchor, "-F", "all"])
            throw HelperError.unscopedRules(offender)
        }

        // `pfctl -E` writes `pf enabled` and `Token : <n>` to stderr, so both streams
        // are requested for this one call. A reference we cannot name is a reference
        // we cannot release, which is how this transition used to leave pf enabled
        // with nobody owning it — so the failure path restores the pf state it found,
        // and only does so when pf was off before we touched it.
        let wasEnabled = pfIsEnabled()
        let output = try Runner.run(SecureServer.pfctlPath, ["-E"], includingStandardError: true)
        guard let token = token(in: output) else {
            _ = try? pfctl(["-a", SecureServer.pfAnchor, "-F", "all"])
            if !wasEnabled {
                _ = try? pfctl(["-d"])
                log.error("pfctl -E produced no token; pf was off before, so it was disabled again")
            } else {
                log.error("pfctl -E produced no token; pf was already enabled, so it was left alone")
            }
            throw HelperError.noToken(output)
        }
        return token
    }

    /// Rules out, then release our enable reference. Flushing first is what
    /// guarantees connectivity even if the reference cannot be released: with an
    /// empty anchor, pf being enabled is harmless to everyone.
    static func dropLockdown() {
        var failure: String?
        for _ in 0..<2 {
            _ = try? pfctl(["-a", SecureServer.pfAnchor, "-F", "all"])
            let remaining = loadedRules()
            guard !remaining.isEmpty else { failure = nil; break }
            failure = "rules still loaded after flushing: \(remaining.joined(separator: "; "))"
        }
        let state = read()
        if let token = state.pfToken, !token.isEmpty {
            // A token released by hand (`pfctl -d`) is already gone; that is not
            // a reason to report a failure, since nothing is enforced any more.
            _ = try? pfctl(["-X", token])
        }
        if let failure {
            log.error("dropLockdown incomplete: \(failure, privacy: .public)")
        }
    }
}

// MARK: - Sandbox account

/// The account the server runs as. pf associates a packet with the effective uid
/// that owned the socket when it was created, so "restrict one process" has to
/// mean "run that process under its own uid" and then name that uid in every
/// rule. Nothing else on the Mac belongs to it, so nothing else is affected.
private enum Sandbox {
    static var uid: uid_t? { getpwnam(SecureServer.sandboxUser)?.pointee.pw_uid }
    /// The account's primary group, which `ensureAccount` sets to `staff`. There is
    /// no group named `_omlx`, so this has to come from the user record.
    static var gid: gid_t? { getpwnam(SecureServer.sandboxUser)?.pointee.pw_gid }

    /// Idempotent. No password and `/usr/bin/false` as the shell: the account
    /// exists so pf has something to match and so the server's files can be told
    /// apart from the user's, not so anyone can log in as it.
    static func ensureAccount() throws {
        guard uid == nil else { return }
        guard let group = getgrnam(SecureServer.sandboxGroup)?.pointee.gr_gid else {
            throw HelperError.noAccount
        }
        guard let identifier = freeUID() else { throw HelperError.noAccount }

        // Group `staff` matters: a home directory is drwxr-x---, so group
        // membership keeps existing world-readable models readable while writes
        // still go to `SecureServer.sandboxHome`.
        let base = "/Users/\(SecureServer.sandboxUser)"
        try Runner.run(SecureServer.dsclPath, ["-create", base])
        try dscl(base, "RecordName", SecureServer.sandboxUser)
        try dscl(base, "RealName", "oMLX Secure Server Sandbox")
        try dscl(base, "UniqueID", String(identifier))
        try dscl(base, "PrimaryGroupID", String(group))
        try dscl(base, "UserShell", "/usr/bin/false")
        try dscl(base, "NFSHomeDirectory", SecureServer.sandboxHome)
        try? dscl(base, "IsHidden", "1")
        guard uid != nil else { throw HelperError.noAccount }
        log.notice("created sandbox account \(SecureServer.sandboxUser, privacy: .public) uid=\(identifier)")
    }

    private static func dscl(_ base: String, _ key: String, _ value: String) throws {
        try Runner.run(SecureServer.dsclPath, [".", "-create", base, key, value])
    }

    /// First free id in the range macOS leaves to local system accounts.
    private static func freeUID() -> uid_t? {
        for candidate in uid_t(460)...uid_t(499) where getpwuid(candidate) == nil {
            return candidate
        }
        return nil
    }

    /// Writable HOME/TMPDIR for the server. The user's real home is group-read
    /// only, so the sandbox needs somewhere of its own for caches and temp files.
    static func ensureHome() throws {
        guard let uid, let gid else { throw HelperError.noAccount }
        let path = SecureServer.sandboxHome
        if !FileManager.default.fileExists(atPath: path) {
            try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        }
        try FileManager.default.setAttributes(
            [
                .ownerAccountID: Int(uid),
                .groupOwnerAccountID: Int(gid),
                .posixPermissions: 0o700
            ],
            ofItemAtPath: path
        )
    }
}

// MARK: - File access for the sandbox

/// What the sandbox keeps of the requesting user's files.
///
/// A macOS home directory is not traversable by other accounts, and toolchains
/// hide their interpreters behind directories that are worse: on this machine
/// `~/.local` is `drwx------`, so a uv-managed Python underneath it is unreachable
/// however open the leaf is. Rather than loosening modes, each directory that
/// denies the world search permission gets one narrow ACL entry for the sandbox
/// account and every path touched is recorded, so turning the mode off takes them
/// back. Traversal is the ceiling here; the only write access is the separate,
/// explicitly named set in `writableDirectoryEntry`.
///
/// The gate is *ownership*, and it is not cosmetic. The app names these paths and
/// this daemon runs as root, so an ungated grant is a local privilege escalation:
/// "root, make this root-owned file readable by `_omlx`", followed by "now run the
/// program I chose as `_omlx`", reads any file on the machine. So a path is opened
/// only when the user at the other end of the connection owns it — nothing that was
/// private to its owner becomes readable — and anything the world can already reach
/// is left completely alone.
private enum Access {
    private static let directoryEntry = "\(SecureServer.sandboxUser) allow list,search"
    private static let fileEntry = "\(SecureServer.sandboxUser) allow read,execute"
    /// Directory rights an atomic settings save needs and nothing more. Measured: a
    /// temp file plus `rename` over the target succeeds with `add_file,delete_child`,
    /// while `mkdir` in the same directory is still refused. Anything the server
    /// *creates* at scale — logs, the disk cache — is redirected into the sandbox
    /// home instead of asking for a writable tree here.
    private static let writableDirectoryEntry =
        "\(SecureServer.sandboxUser) allow list,search,add_file,delete_child"
    private static let writableFileEntry = "\(SecureServer.sandboxUser) allow read,write,execute"
    private static let recordPath = SecureServer.configDirectory + "/access.json"
    /// Bounds on following a private directory down to wherever the bits open up.
    /// The case worth handling is a closed `~/models` tree, not repainting a home
    /// directory, and the transition has to finish inside a second or two.
    private static let maxDepth = 6
    private static let maxEntriesPerDirectory = 512
    private static let maxGrants = 2000

    /// A directory whose own bits already let the sandbox in still gets walked,
    /// because its children may not: `~/.omlx` is open while `~/.omlx/models/x` is
    /// not. Walking is therefore bounded on its own, independently of how many
    /// entries end up granted.
    private static let maxDirectories = 4000

    /// One ACL entry this daemon added, plus who owned the path when it did. The
    /// owner is recorded because a replace leaves the *writer* as owner: see
    /// `restoreOwnership`. `directory` records what the entry was decided against so
    /// that revoking does not have to ask the filesystem again — by then the path may
    /// be a different file. Records written before either field existed decode with
    /// `nil` and are handled by `entry(for:)`.
    struct Grant: Codable, Hashable {
        var path: String
        var writable: Bool
        var directory: Bool?
        var owner: Int
        var group: Int
    }

    private struct Budget {
        var granted: [Grant] = []
        var walked = 0
        var stopped = false
        var exhausted: Bool { granted.count >= maxGrants || walked >= maxDirectories }
    }

    /// Open `paths` for reading and `writable` for modification, on behalf of
    /// `clientUID`.
    ///
    /// A refusal is logged, not thrown: one path the caller does not own should not
    /// fail the transition, and a server that cannot read what it needs says so in
    /// its own log.
    static func ensure(paths: [String], writable: [String], clientUID: uid_t) throws {
        var budget = Budget()
        // Writable first: it is a handful of paths and the server cannot save its
        // settings without them, so the read-only walk below must not be able to
        // starve it of budget.
        for path in writable where FileManager.default.fileExists(atPath: path) {
            grantAncestors(of: path, clientUID: clientUID, into: &budget)
            grantWritable(path, clientUID: clientUID, into: &budget)
        }
        for path in paths where FileManager.default.fileExists(atPath: path) {
            grantAncestors(of: path, clientUID: clientUID, into: &budget)
            grant(path, clientUID: clientUID, depth: 0, into: &budget)
            if budget.stopped {
                log.warning("access grants stopped at \(Self.maxGrants); later paths were not opened")
                break
            }
        }
        do {
            try record(budget.granted)
        } catch {
            // Nothing can take these entries back without a record, so undo them here
            // rather than leave access that root no longer knows it granted.
            for grant in budget.granted {
                _ = try? Runner.run(SecureServer.chmodPath, ["-a", entry(for: grant), grant.path])
                restoreOwnership(of: grant)
            }
            log.error("could not record access grants; removed the \(budget.granted.count) entries just added")
            throw error
        }
    }

    /// Remove exactly what `ensure` added, then hand back whatever the sandbox ended
    /// up owning. Complete because every path touched is in the record, including the
    /// ones found by descending.
    static func revoke() {
        var failed: [Grant] = []
        for grant in recorded() {
            // `chmod -a` exits non-zero when there is nothing to remove, and that is
            // the normal case for a file the sandbox replaced: an atomic save writes a
            // new inode, which carries no ACL, so the recorded entry refers to an inode
            // that no longer exists. Absence has to count as success, or the record
            // never empties and every later revoke retries an entry that is already
            // gone.
            let removed = (try? Runner.run(
                SecureServer.chmodPath, ["-a", entry(for: grant), grant.path]
            )) != nil || !hasSandboxEntry(grant.path)
            if !removed {
                // Keep it on record: dropping the entry would leave an ACE on disk that
                // nothing knows about and no later revoke could find.
                failed.append(grant)
                log.error("could not remove the access entry on \(grant.path, privacy: .public)")
            }
            restoreOwnership(of: grant)
        }
        if failed.isEmpty {
            try? FileManager.default.removeItem(atPath: recordPath)
        } else {
            try? write(failed)
        }
    }

    /// Whether the sandbox account still has an entry on `path`, read in-process
    /// from the ACL itself. `acl_to_text` renders the same account name the
    /// `chmod -a` revoke uses, so this replaces `ls -led | grep _omlx`.
    private static func hasSandboxEntry(_ path: String) -> Bool {
        guard let acl = acl_get_file(path, ACL_TYPE_EXTENDED) else { return false }
        defer { acl_free(UnsafeMutableRawPointer(acl)) }
        guard let text = acl_to_text(acl, nil) else { return false }
        defer { acl_free(UnsafeMutableRawPointer(text)) }
        return String(cString: text).contains(SecureServer.sandboxUser)
    }

    private static func entry(for grant: Grant) -> String {
        let directory = grant.directory ?? isDirectory(grant.path)
        if grant.writable { return directory ? writableDirectoryEntry : writableFileEntry }
        return directory ? directoryEntry : fileEntry
    }

    /// Anything in a path we opened that now belongs to the sandbox goes back to the
    /// user who owned it. A settings save replaces its file rather than editing it in
    /// place, and a replacement takes its writer as owner — so after one save through
    /// the sandbox, the user's own settings file belongs to `_omlx` and the app can
    /// no longer write its own configuration. `lchown`, so that a symlink planted in
    /// one of those directories cannot move the change onto somebody else's file.
    private static func restoreOwnership(of grant: Grant) {
        guard let sandboxUID = Sandbox.uid, grant.owner >= 0 else { return }
        var info = stat()
        guard lstat(grant.path, &info) == 0 else { return }
        if info.st_uid == sandboxUID {
            lchown(grant.path, uid_t(grant.owner), gid_t(grant.group))
        }
        guard grant.writable, (mode_t(S_IFMT) & info.st_mode) == mode_t(S_IFDIR) else { return }
        guard let children = try? FileManager.default.contentsOfDirectory(atPath: grant.path) else { return }
        for child in children.prefix(maxEntriesPerDirectory) {
            let childPath = (grant.path as NSString).appendingPathComponent(child)
            var childInfo = stat()
            guard lstat(childPath, &childInfo) == 0, childInfo.st_uid == sandboxUID else { continue }
            lchown(childPath, uid_t(grant.owner), gid_t(grant.group))
        }
    }

    /// The directories above `path`: one `drwx------` ancestor hides everything below
    /// it no matter how open the leaf is. The chain is resolved first, because `exec`
    /// follows a symlink — a `.venv/bin/python` that points into a uv installation
    /// needs *that* chain open, and those are the directories that are private in
    /// practice (`~/.local` is `drwx------`).
    private static func grantAncestors(of path: String, clientUID: uid_t, into budget: inout Budget) {
        var directory = ((path as NSString).resolvingSymlinksInPath as NSString).deletingLastPathComponent
        var visited = Set<String>()
        while visited.insert(directory).inserted {
            budget.walked += 1
            guard decide(directory, writable: false, mask: 0o001, clientUID: clientUID, into: &budget)
                    != .refused else { return }
            if budget.exhausted { budget.stopped = true; return }
            let parent = (directory as NSString).deletingLastPathComponent
            if parent == directory { break }
            directory = parent
        }
    }

    /// The named path itself and, when it is a directory, the closed subdirectories
    /// underneath it. Traversal is useless unless the whole chain is open, but the
    /// walk never grants more than traversal: recursing `read` into a tree is exposure
    /// this feature has no need of, and model files are group-readable in practice.
    ///
    /// Descending from the *named* path does not depend on it having needed a grant —
    /// an open parent over `0700` children is exactly the shape a model tree has.
    /// Below that, only directories we opened are walked: an already-open tree (a
    /// site-packages, a checkout) cannot need an entry and would cost the whole
    /// per-launch budget to discover that.
    private static func grant(_ rawPath: String, clientUID: uid_t, depth: Int, into budget: inout Budget) {
        guard !budget.exhausted else { budget.stopped = true; return }
        let path = (rawPath as NSString).resolvingSymlinksInPath
        let directory = isDirectory(path)
        let decision = decide(path, writable: false, mask: directory ? 0o001 : 0o004,
                              clientUID: clientUID, into: &budget)
        guard decision != .refused else { return }
        guard directory, depth < maxDepth else { return }
        if depth > 0, decision == .alreadyOpen { return }
        budget.walked += 1
        guard let children = try? FileManager.default.contentsOfDirectory(atPath: path) else { return }
        for child in children.prefix(maxEntriesPerDirectory) {
            let childPath = (path as NSString).appendingPathComponent(child)
            guard isDirectory(childPath) else { continue }
            grant(childPath, clientUID: clientUID, depth: depth + 1, into: &budget)
            if budget.exhausted { budget.stopped = true; return }
        }
    }

    /// Whether the sandbox may have `path`, and whether an entry had to be written.
    ///
    /// Two conditions, both required: the world bits do not already allow it — so
    /// system binaries and group-readable models are never touched at all — and
    /// `clientUID` owns the file. Both are judged on the inode reached by *resolving*
    /// the path, because a symlink is the caller's to create while whatever it points
    /// at may not be theirs, and `chmod +a` acts on the file the path refers to at the
    /// moment it runs. Deciding on the link and granting on the target is how "open my
    /// models folder" turns into "open `/etc/master.passwd` for `_omlx`".
    ///
    /// The inode is confirmed again after the entry is written: a caller that owns a
    /// directory can replace a component at any time, so if the path no longer refers
    /// to what was approved the entry is taken back and nothing is recorded. What that
    /// leaves is a race measured in microseconds — no check performed on a *path* can
    /// close it — against a caller who is already inside the account that owns the
    /// file, and it is a grant on the sandbox's own account, never a mode change.
    private static func decide(
        _ path: String, writable: Bool, mask: mode_t, clientUID: uid_t, into budget: inout Budget
    ) -> Decision {
        guard !budget.exhausted else { return .refused }
        let resolved = (path as NSString).resolvingSymlinksInPath
        guard let before = inode(of: resolved) else { return .refused }
        let directory = (before.mode & S_IFMT) == S_IFDIR
        // What the sandbox account already has, asked as "are all the bits it needs
        // present": a writable directory needs write *and* search, a writable file
        // write, a read-only directory search, a read-only file read. Testing a
        // directory for search alone made a 0755 data folder look open and silently
        // dropped the `add_file,delete_child` entry its settings save depends on.
        let needed: mode_t = directory ? (writable ? 0o003 : 0o001) : (writable ? 0o002 : 0o004)
        guard before.mode & needed != needed else { return .alreadyOpen }
        // A path the sandbox itself already owns is eligible as well: the server's
        // atomic settings save replaces the file, so from that moment the inode
        // belongs to the sandbox account. Refusing it would make the next grant fail
        // forever — measured on this machine, where every start after the first one
        // refused `settings.json` — even though the principal being granted access is
        // the one that already has it, so nothing widens. The client is recorded as
        // the owner either way, so teardown hands the file back where it came from.
        let sandboxUID = Sandbox.uid
        guard before.uid == clientUID || before.uid == sandboxUID else {
            log.warning("refused \(SecureServer.sandboxUser, privacy: .public) access to \(resolved, privacy: .public): owned by \(before.uid), not by the requesting user \(clientUID)")
            return .refused
        }
        let grant = Grant(
            path: resolved, writable: writable, directory: directory,
            owner: Int(clientUID), group: Int(before.gid)
        )
        guard (try? Runner.run(SecureServer.chmodPath, ["+a", entry(for: grant), resolved])) != nil else {
            log.warning("chmod +a failed for \(resolved, privacy: .public)")
            return .refused
        }
        guard let after = inode(of: resolved), after.dev == before.dev, after.ino == before.ino,
              after.uid == clientUID || after.uid == sandboxUID else {
            _ = try? Runner.run(SecureServer.chmodPath, ["-a", entry(for: grant), resolved])
            log.error("refused \(SecureServer.sandboxUser, privacy: .public) access to \(resolved, privacy: .public): the path changed while the entry was being written")
            return .refused
        }
        budget.granted.append(grant)
        return .granted
    }

    enum Decision: Equatable {
        /// An entry was written.
        case granted
        /// Nothing to do: the mode bits already let the sandbox account in.
        case alreadyOpen
        /// Not the caller's file, unreadable, or the path moved underneath us.
        case refused
    }

    /// Identity and permissions of the inode a path reaches, without following its
    /// final symlink. `O_NONBLOCK` so a fifo named by the caller cannot wedge the
    /// daemon; a directory opens read-only either way.
    private static func inode(of path: String) -> (dev: dev_t, ino: ino_t, uid: uid_t, gid: gid_t, mode: mode_t)? {
        let descriptor = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        guard descriptor >= 0 else { return nil }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { return nil }
        return (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_mode)
    }

    /// A path the server has to modify. Same gate, different entry, and deliberately
    /// no descent: the mutable set is decided in the app — the settings files and the
    /// directory they are replaced in — and walking a tree looking for things that
    /// might want writing is how a narrow grant becomes a wide one.
    private static func grantWritable(_ path: String, clientUID: uid_t, into budget: inout Budget) {
        _ = decide(path, writable: true, mask: 0o002, clientUID: clientUID, into: &budget)
    }

    private static func isDirectory(_ path: String) -> Bool {
        var flag: ObjCBool = false
        FileManager.default.fileExists(atPath: path, isDirectory: &flag)
        return flag.boolValue
    }

    private static func recorded() -> [Grant] {
        guard let data = FileManager.default.contents(atPath: recordPath) else { return [] }
        if let grants = try? JSONDecoder().decode([Grant].self, from: data) { return grants }
        // Records written before writable grants existed were bare paths. They still
        // have to be revocable; all of them were read-only, and with no owner on
        // record nothing is chowned back for them (-1 suppresses the restore).
        if let paths = try? JSONDecoder().decode([String].self, from: data) {
            return paths.map { Grant(path: $0, writable: false, directory: nil, owner: -1, group: -1) }
        }
        return []
    }

    private static func record(_ grants: [Grant]) throws {
        // Union with what is already recorded. `ensure` runs on every start, and a
        // second pass finds the first pass's ACLs in place and grants nothing new;
        // overwriting the record then would strand those entries on the filesystem
        // with nothing left to revoke, and leave the sandbox owning what it replaced.
        let unique = Set(recorded() + grants).sorted { $0.path < $1.path }
        try write(unique)
    }

    /// Replace the record with exactly `grants` — used to keep the entries a revoke
    /// could not remove, so a later revoke can still find them.
    private static func write(_ grants: [Grant]) throws {
        let data = try JSONEncoder().encode(grants)
        try data.write(to: URL(fileURLWithPath: recordPath), options: .atomic)
        try FileManager.default.setAttributes(
            [.ownerAccountID: 0, .groupOwnerAccountID: 0, .posixPermissions: 0o600],
            ofItemAtPath: recordPath
        )
    }
}

// MARK: - Process and socket introspection (libproc)

/// Direct syscalls on the kernel's process table, replacing the `pgrep`, `lsof`,
/// `netstat` and `ls` probes this daemon used to spawn. Everything here runs
/// in-process; the port and process-tree checks sit in tight loops around every
/// start and stop, so not launching a helper program for each one matters.
private enum Proc {
    static func allPIDs() -> [Int32] {
        let needed = proc_listpids(UInt32(PROC_ALL_PIDS), 0, nil, 0)
        guard needed > 0 else { return [] }
        var buffer = [Int32](repeating: 0, count: Int(needed) / 4 + 32)
        let written = buffer.withUnsafeMutableBytes { raw -> Int32 in
            proc_listpids(UInt32(PROC_ALL_PIDS), 0, raw.baseAddress, Int32(raw.count))
        }
        guard written > 0 else { return [] }
        let n = Int(written) / MemoryLayout<Int32>.size
        return (0..<min(n, buffer.count)).map { buffer[$0] }.filter { $0 > 0 }
    }

    /// (uid, ppid) of a running process, from the kernel's own record of it.
    static func bsdInfo(of pid: Int32) -> (uid: uid_t, ppid: Int32)? {
        var info = proc_bsdinfo()
        let written = proc_pidinfo(
            pid, PROC_PIDTBSDINFO, 0, &info,
            Int32(MemoryLayout<proc_bsdinfo>.size)
        )
        guard written > 0 else { return nil }
        return (info.pbi_uid, Int32(info.pbi_ppid))
    }

    /// The executable image actually running as `pid`.
    static func path(of pid: Int32) -> String {
        var buffer = [CChar](repeating: 0, count: 4096)
        guard proc_pidpath(pid, &buffer, 4096) > 0 else { return "" }
        return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Every descendant of `root`, deepest first, bounded to eight levels. Built
    /// from one snapshot of ppids rather than `proc_listchildpids`, whose null-
    /// buffer sizing call is unusable on this OS, and rather than a recursive
    /// `pgrep -P` per level.
    static func descendants(of root: Int32) -> [Int32] {
        var parent: [Int32: Int32] = [:]
        for pid in allPIDs() {
            if let info = bsdInfo(of: pid) { parent[pid] = info.ppid }
        }
        var result: [Int32] = []
        var frontier = [root]
        var depth = 0
        while !frontier.isEmpty, depth < 8 {
            let children = parent.compactMap { frontier.contains($0.value) ? $0.key : nil }
            result = children + result
            frontier = children
            depth += 1
        }
        return result
    }

    /// Pids owned by `uid` whose image is the Python/oMLX server. The uid gate
    /// excludes everything else, and the image check excludes the per-user
    /// launchd agents (`distnoted`, `lsd`, `cfprefsd`) that also run under it.
    static func servers(ownedBy uid: uid_t) -> [Int32] {
        allPIDs().filter { pid in
            guard bsdInfo(of: pid)?.uid == uid else { return false }
            let image = path(of: pid).lowercased()
            return image.contains("python") || image.contains("omlx")
        }
    }

    /// Any pid holding a TCP LISTEN socket on `port` — the in-process equivalent
    /// of `lsof -ti :port -sTCP:LISTEN`.
    static func listeners(on port: Int) -> [Int32] {
        allPIDs().filter { listens($0, on: port) }
    }

    /// Whether one specific pid holds the listening socket. Used for the spawn
    /// handshake, where scanning the whole table on every poll would be wasteful.
    static func listens(_ pid: Int32, on port: Int) -> Bool {
        let target = in_port_t(port).bigEndian
        for fd in socketFDs(of: pid) {
            var info = socket_fdinfo()
            let written = proc_pidfdinfo(
                pid, fd, PROC_PIDFDSOCKETINFO, &info,
                Int32(MemoryLayout<socket_fdinfo>.size)
            )
            guard written == MemoryLayout<socket_fdinfo>.size,
                  info.psi.soi_kind == SOCKINFO_TCP else { continue }
            let tcp = info.psi.soi_proto.pri_tcp
            if tcp.tcpsi_state == TCPS_LISTEN, tcp.tcpsi_ini.insi_lport == target {
                return true
            }
        }
        return false
    }

    /// Socket descriptors of one process.
    private static func socketFDs(of pid: Int32) -> [Int32] {
        let needed = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, nil, 0)
        guard needed > 0 else { return [] }
        let capacity = Int(needed) / MemoryLayout<proc_fdinfo>.size + 16
        var buffer = [proc_fdinfo](repeating: proc_fdinfo(), count: capacity)
        let written = buffer.withUnsafeMutableBytes { raw -> Int32 in
            proc_pidinfo(pid, PROC_PIDLISTFDS, 0, raw.baseAddress, Int32(raw.count))
        }
        guard written > 0 else { return [] }
        let n = Int(written) / MemoryLayout<proc_fdinfo>.size
        return (0..<min(n, buffer.count)).compactMap { index in
            buffer[index].proc_fdtype == PROX_FDTYPE_SOCKET ? buffer[index].proc_fd : nil
        }
    }
}

// MARK: - Sandboxed server

/// One sandboxed server at a time, supervised by the daemon.
///
/// macOS has no `POSIX_SPAWN_SETEUID`, and the identity has to be in place at
/// exec time for pf to tag the listening socket. The daemon therefore re-executes
/// its own binary (see `DropAndExec`): that child drops to `_omlx` and execs the
/// server. A root listener that called setuid afterwards would keep root's
/// credentials and match no rule at all.
private enum Server {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var child: Process?

    /// The running sandboxed server, if any.
    ///
    /// Identity comes from the pid the daemon records once the spawn is confirmed,
    /// not from "the oldest process owned by the sandbox account". That account
    /// also owns per-user launchd agents — `distnoted`, `lsd` and `cfprefsd` appear
    /// as soon as anything has run under a uid — so a scan of the process table
    /// answers with a system daemon instead of the server: status would name
    /// something that is not serving requests, teardown would signal daemons that
    /// were never ours, and "is anything left under the uid" would never go false.
    /// The tracked child is only the fallback for the window before the record exists.
    static var pid: Int32? {
        if let recorded = recordedPID() { return recorded }
        if let child = lock.withLock({ child }), child.isRunning { return child.processIdentifier }
        return discoveredServers(excluding: nil).first
    }

    /// Servers that have no pidfile — a failed start, or a server spawned by an
    /// older helper. The uid gate alone is not enough: launchd runs per-user agents
    /// under the same account, so the executable image is checked too. libproc
    /// answers both in-process, without scanning the uid or spawning `pgrep`.
    private static func discoveredServers(excluding known: Int32?) -> [Int32] {
        guard let uid = Sandbox.uid else { return [] }
        return Proc.servers(ownedBy: uid).filter { $0 != known }
    }

    /// The recorded pid, validated. Pids are recycled: a file left behind by an
    /// earlier run must not be reported as a running server, and certainly not
    /// signalled on the way out.
    private static func recordedPort() -> Int? {
        guard let body = try? String(
            contentsOfFile: SecureServer.sandboxPortPath, encoding: .utf8
        ) else { return nil }
        return Int(body.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func recordedPID() -> Int32? {
        guard let body = try? String(
            contentsOfFile: SecureServer.sandboxPIDPath, encoding: .utf8
        ), let value = Int32(body.trimmingCharacters(in: .whitespacesAndNewlines)) else { return nil }
        guard isAlive(value), Caller.euid(of: value) == Sandbox.uid else {
            clearRecord()
            return nil
        }
        return value
    }

    /// Existence, not permission: the daemon is root and the server is not, and a
    /// process that exists but cannot be signalled still counts as running.
    private static func isAlive(_ pid: Int32) -> Bool {
        kill(pid, 0) == 0 || errno == EPERM
    }

    /// Every process below `pid`, deepest first, bounded to eight levels. The
    /// tree is built in-process from one libproc snapshot of ppids, so stopping
    /// the server no longer spawns `pgrep` once per level.
    private static func descendants(of pid: Int32) -> [Int32] {
        Proc.descendants(of: pid)
    }

    static func start(_ launch: SecureServerLaunch, clientUID: uid_t) throws -> Int32 {
        guard Store.read().enabled else { throw HelperError.notEnabled }
        stop()
        // Whoever still owns the listening socket — including a uid-501 child of
        // the server that was just stopped — has to go before a replacement can bind.
        drainPort(launch.port, seconds: 20)
        // Idempotent re-assertions: bootstrap() owns the setup, but a start can be
        // requested on its own (a restart, a relaunch), so both must hold here too.
        try Sandbox.ensureAccount()
        try Sandbox.ensureHome()
        // Running as somebody else means the sandbox can no longer assume it can
        // reach the interpreter or the model files, and the paths to open are named
        // by the app, so they are opened only where the app's own user owns them.
        try Access.ensure(
            paths: launch.accessPaths,
            writable: launch.writablePaths ?? [],
            clientUID: clientUID
        )

        // The sandbox gets its own HOME/TMPDIR; everything else is whatever the
        // app normally passes (model directories, API keys, ports). It travels in
        // the spawn's environment, never on a command line — the old sudo wrapper
        // kept every KEY=VALUE in an argv visible to all local users.
        var environment = launch.environment
        environment["HOME"] = SecureServer.sandboxHome
        environment["TMPDIR"] = SecureServer.sandboxHome
        environment["USER"] = SecureServer.sandboxUser
        environment["LOGNAME"] = SecureServer.sandboxUser
        environment["PYTHONUNBUFFERED"] = "1"
        // The server writes settings, and a save is a rename over the target, which
        // would leave the file owned by the sandbox account where the app — running as
        // the user — can no longer read it. It gets a copy instead; teardown hands
        // back what it saved.
        if let settings = try seedSettings(launch.settingsPath, clientUID: clientUID) {
            environment[SecureServer.settingsFileEnvironmentKey] = settings
        }

        // Root opens the log and hands the descriptor down, so the sandbox needs
        // no write access to the root-owned config directory.
        let handle = try logHandle()
        try? "\(launch.port)\n".write(toFile: SecureServer.sandboxPortPath, atomically: true, encoding: .utf8)

        // Do not report a pid until the child actually owns the port. Uvicorn exits
        // with STARTUP_FAILURE (3) when its bind fails; handing that dead pid to the
        // app made it burn three auto-restarts after every flip. Verify here and retry
        // the spawn in the supervisor — the only layer that can drain and try again.
        for attempt in 1...3 {
            // The port may look clear to the socket scan while a just-killed
            // server's socket is still bound-but-not-listening. Wait for the exact
            // bind the child will do.
            _ = waitUntilBindable(launch.port, host: "127.0.0.1", seconds: 30)
            clearRecord()
            let process = Process()
            // Re-execute our own binary in trampoline mode: the child drops to
            // _omlx and execs the interpreter. No sudo, no shell, no env file —
            // the daemon is root, so the child's setuid succeeds there, and the
            // full environment is passed straight through the spawn.
            process.executableURL = URL(fileURLWithPath: ownExecutable)
            process.arguments = [DropAndExec.flag, launch.executable] + launch.arguments
            process.environment = environment
            process.standardOutput = handle
            process.standardError = handle
            // Why the child went away, in one line: a signal says something killed it,
            // an exit status says it decided to leave.
            process.terminationHandler = { finished in
                log.notice(
                    "sandboxed server pid=\(finished.processIdentifier, privacy: .public) ended reason=\(finished.terminationReason.rawValue, privacy: .public) status=\(finished.terminationStatus, privacy: .public)"
                )
            }
            try process.run()
            lock.withLock { child = process }

            if waitUntilPortOwnedBySandbox(process, port: launch.port, seconds: 25) {
                // Record the pid only now. Until the trampoline has exec'd the
                // server the child is still root, and recordedPID() would see uid 0
                // and discard a record written any earlier as invalid.
                let pid = process.processIdentifier
                try? "\(pid)\n".write(
                    toFile: SecureServer.sandboxPIDPath, atomically: true, encoding: .utf8
                )
                log.notice(
                    "started sandboxed server pid=\(pid) on port \(launch.port) (attempt \(attempt))"
                )
                return pid
            }
            log.error(
                "sandboxed server attempt \(attempt, privacy: .public) never bound port \(launch.port, privacy: .public); draining and retrying: \(socketSnapshot(launch.port), privacy: .public)"
            )
            lock.withLock { child = nil }
            drainPort(launch.port, seconds: 15)
        }
        throw HelperError.startFailed(
            "the sandboxed server never bound port \(launch.port) after 3 attempts"
        )
    }

    /// Absolute path to this helper binary, for the self-exec trampoline.
    private static var ownExecutable: String {
        URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().path
    }

    /// Any pid holding a TCP LISTEN socket on `port`, whatever uid owns it. The
    /// in-process libproc scan sees a uid-501 child that inherited the socket just
    /// as `lsof -sTCP:LISTEN` did, without spawning a program.
    private static func portListener(_ port: Int) -> Int32? {
        Proc.listeners(on: port).first
    }

    /// What the kernel says is listening on the port, for the failure log.
    private static func socketSnapshot(_ port: Int) -> String {
        "listeners=\(Proc.listeners(on: port))"
    }

    /// Waits until the process we spawned is the sandbox-owned listener on `port`.
    /// True is the only proof the spawn worked; uvicorn's STARTUP_FAILURE (3) would
    /// otherwise leave a dead pid to hand to the app. Checking only our own pid's
    /// descriptors (not the whole process table) keeps this 25-second poll cheap.
    private static func waitUntilPortOwnedBySandbox(
        _ process: Process, port: Int, seconds: Int
    ) -> Bool {
        let pid = process.processIdentifier
        let deadline = Date().addingTimeInterval(TimeInterval(seconds))
        while Date() < deadline {
            if Proc.listens(pid, on: port), Caller.euid(of: pid) == Sandbox.uid {
                return true
            }
            if !process.isRunning { return false }
            Thread.sleep(forTimeInterval: 0.1)
        }
        return false
    }

    /// Clears the port of any listener, whatever uid owns it, before a spawn.
    ///
    /// The libproc socket scan names the socket's owner directly: uvicorn binds
    /// the listening socket at cli.py:238, *before* `init_server` forks the engine
    /// pool at 356, so a child of the user-owned server inherits that fd and
    /// survives its parent's death. There is deliberately no bind probe here:
    /// binding the port, even briefly, competes with the child's own bind and was
    /// itself a source of `EADDRINUSE`.
    private static func drainPort(_ port: Int, seconds: Int) {
        let deadline = Date().addingTimeInterval(TimeInterval(seconds))
        while Date() < deadline, let owner = portListener(port) {
            // SIGTERM lets an oMLX server begin a graceful uvicorn shutdown, which
            // closes the listening socket while its children still hold bound
            // sockets on that local port. These are strays being replaced, so kill
            // them outright.
            kill(owner, SIGKILL)
            for stray in discoveredServers(excluding: owner) where isAlive(stray) {
                kill(stray, SIGKILL)
            }
            Thread.sleep(forTimeInterval: 0.2)
        }
        if let owner = portListener(port) {
            log.error("port \(port, privacy: .public) still has listener \(owner, privacy: .public) after \(seconds, privacy: .public)s")
        }
    }

    /// Whether a socket with the exact options uvicorn uses can bind right now.
    /// Run only *before* a spawn (never concurrently with one), so it cannot compete
    /// with the child's own bind. A LISTEN-state socket scan cannot see a bound-
    /// but-not-listening socket or a lingering FIN_WAIT, but this asks the kernel
    /// the same question the child asks.
    private static func isPortBindable(_ port: Int, host: String) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var reuse: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = inet_addr(host)
        let result = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return result == 0
    }

    private static func waitUntilBindable(_ port: Int, host: String, seconds: Int) -> Bool {
        let deadline = Date().addingTimeInterval(TimeInterval(seconds))
        while Date() < deadline {
            if isPortBindable(port, host: host) { return true }
            Thread.sleep(forTimeInterval: 0.25)
        }
        log.error("port \(port, privacy: .public) never became bindable in \(seconds, privacy: .public)s")
        return false
    }

    /// Gives the sandbox its own copy of the user's settings and records what it
    /// started from, so a save it makes can be told apart from an edit the user made
    /// while the server was running.
    private static func seedSettings(_ path: String?, clientUID: uid_t) throws -> String? {
        guard let path, let uid = Sandbox.uid, let gid = Sandbox.gid else { return nil }
        // A record left behind by a crashed run still holds the only copy of a save the
        // sandbox made, so it is settled before it is replaced.
        reconcileSettings()
        let source = (path as NSString).resolvingSymlinksInPath
        guard let data = FileManager.default.contents(atPath: source) else { return nil }
        let attributes = try? FileManager.default.attributesOfItem(atPath: source)
        let ownerGID = (attributes?[.groupOwnerAccountID] as? NSNumber)?.int32Value ?? Int32(gid)
        let directory = (SecureServer.sandboxSettingsPath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        // The directory belongs to the sandbox as well: a save creates its temp file
        // beside the target, so a directory left root-owned here would let the server
        // boot cleanly and then fail every settings save with EACCES.
        try setOwnership(directory, uid: uid, gid: gid, mode: 0o700)
        try data.write(to: URL(fileURLWithPath: SecureServer.sandboxSettingsPath), options: .atomic)
        try setOwnership(SecureServer.sandboxSettingsPath, uid: uid, gid: gid)
        let seed: [String: String] = [
            "clientUID": String(clientUID),
            "clientGID": String(ownerGID),
            "path": source,
            "sha256": digest(data),
        ]
        let encoded = try JSONSerialization.data(withJSONObject: seed)
        try encoded.write(to: URL(fileURLWithPath: SecureServer.settingsSeedPath), options: .atomic)
        // Root-owned: this record decides where root writes at teardown.
        try setOwnership(SecureServer.settingsSeedPath, uid: 0, gid: 0)
        return SecureServer.sandboxSettingsPath
    }

    /// Hands back what the sandbox saved — but only when the user did not edit the
    /// file in the meantime: the app keeps writing settings while the mode is on, and
    /// its newer version must not be replaced by the sandbox's older one. A failure
    /// keeps the seed record on disk instead of dropping the only copy of a change.
    private static func reconcileSettings() {
        guard let raw = FileManager.default.contents(atPath: SecureServer.settingsSeedPath),
              let seed = try? JSONSerialization.jsonObject(with: raw) as? [String: String],
              let path = seed["path"],
              let uid = uid_t(seed["clientUID"] ?? ""),
              let gid = gid_t(seed["clientGID"] ?? ""),
              let saved = FileManager.default.contents(atPath: SecureServer.sandboxSettingsPath),
              let current = FileManager.default.contents(atPath: path),
              digest(saved) != seed["sha256"],
              digest(current) == seed["sha256"],
              // Defence in depth over the root-owned record: root only ever writes a
              // settings file that still belongs to the user who asked for the mode.
              (try? FileManager.default.attributesOfItem(atPath: path))?[.ownerAccountID] as? NSNumber
                  == NSNumber(value: uid)
        else {
            try? FileManager.default.removeItem(atPath: SecureServer.settingsSeedPath)
            try? FileManager.default.removeItem(atPath: SecureServer.sandboxSettingsPath)
            return
        }
        do {
            try saved.write(to: URL(fileURLWithPath: path), options: .atomic)
            try setOwnership(path, uid: uid, gid: gid)
            try? FileManager.default.removeItem(atPath: SecureServer.settingsSeedPath)
            try? FileManager.default.removeItem(atPath: SecureServer.sandboxSettingsPath)
            log.notice("handed the sandboxed server's settings back to the user")
        } catch {
            log.error("could not write back the sandboxed settings: \(error.localizedDescription, privacy: .public)")
        }
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func setOwnership(_ path: String, uid: uid_t, gid: gid_t, mode: mode_t = 0o600) throws {
        try FileManager.default.setAttributes(
            [.ownerAccountID: Int(uid), .groupOwnerAccountID: Int(gid), .posixPermissions: mode],
            ofItemAtPath: path
        )
    }

    private static func logHandle() throws -> FileHandle {
        let path = SecureServer.serverLogPath
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        // The server's output names models, paths and request metadata, so this is
        // root plus the admin group rather than world-readable, which is what a file
        // under /Library/Application Support would otherwise become.
        try FileManager.default.setAttributes(
            [.ownerAccountID: 0, .groupOwnerAccountID: Int(adminGID() ?? 0), .posixPermissions: 0o640],
            ofItemAtPath: path
        )
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        // `forWritingTo` starts at offset 0: without this every sandboxed child
        // overwrote the log from the top and the previous run's tail survived, which is
        // exactly how "the child wrote nothing" became the wrong conclusion.
        try handle.seekToEnd()
        return handle
    }

    /// The admin group, so the log stays readable by whoever is allowed to manage
    /// the mode instead of by every local account.
    private static func adminGID() -> gid_t? {
        getgrnam("admin").map { $0.pointee.gr_gid }
    }

    /// Stop the server and everything it spawned, then clean up after it.
    ///
    /// Not `pkill -u <sandbox>`: the account owns per-user launchd agents that
    /// launchd keeps alive, so signalling the account would take down things that
    /// were never ours and would never report success. The sweep is the tree below
    /// the server — captured *before* the parent is signalled, because once the
    /// server exits its workers are reparented to launchd and nothing links them
    /// any more. A survivor would otherwise keep a listening socket open under rules
    /// that no longer have a purpose.
    ///
    /// Returns false when the server itself could not be confirmed dead. The caller
    /// must not drop pf then: a live server with the rules gone has unrestricted
    /// egress, which is the one state this mode exists to make impossible. A server
    /// that ignores SIGTERM is therefore killed outright and the result checked,
    /// not assumed from the attempt.
    /// The server this daemon knows about: its own tracked child if it spawned one,
    /// otherwise the validated pid record. A helper is started on demand and exits
    /// after each call, so "the child this process spawned" is regularly empty while a
    /// server is running — reporting only that made the app unable to adopt a server
    /// that existed.
    static var runningPID: Int32? {
        if let child, child.isRunning, let pid = recordedPID() { return pid }
        return recordedPID()
    }

    @discardableResult
    static func stop() -> Bool {
        let dying = lock.withLock { () -> Process? in
            let previous = child
            child = nil
            return previous
        }
        let server = recordedPID()
        if let server {
            // Named so a start that stops the previous server is distinguishable from a
            // server that crashed: nothing else in the logs says a kill happened.
            log.notice("stopping sandboxed server pid=\(server, privacy: .public)")
        }
        let tree = (server ?? dying?.processIdentifier).map { descendants(of: $0) } ?? []
        if let dying, dying.isRunning {
            dying.terminate()      // direct child: signals the trampoline/server itself
            waitUntilExit(dying, seconds: 8)
        }
        // The record is the authority when the daemon was restarted since the spawn,
        // which is exactly when there is no tracked child to terminate.
        if let server, isAlive(server) {
            kill(server, SIGTERM)
            waitForDeath(server, seconds: 8)
            if isAlive(server) {
                log.warning("sandboxed server \(server) ignored SIGTERM; killing it")
                kill(server, SIGKILL)
                waitForDeath(server, seconds: 4)
            }
        }
        for stray in tree where isAlive(stray) { kill(stray, SIGKILL) }

        // A server with no pidfile is still a server — a failed start, or one spawned
        // by an older helper — and it would otherwise keep :8000 and its sockets while
        // this daemon reports that everything stopped.
        let orphans = discoveredServers(excluding: server)
        for orphan in orphans where isAlive(orphan) {
            kill(orphan, SIGTERM)
            waitForDeath(orphan, seconds: 4)
            if isAlive(orphan) { kill(orphan, SIGKILL); waitForDeath(orphan, seconds: 2) }
        }

        let gone = server.map { !isAlive($0) } ?? !(dying?.isRunning ?? false)
        guard gone, orphans.allSatisfy({ !isAlive($0) }) else {
            log.error("a sandboxed server survived SIGKILL; keeping the lockdown")
            return false
        }
        // The recorded pid being gone is not the same as the port being free: a
        // descendant, or a server spawned by an earlier helper, can still own the
        // listening socket, and the replacement then exits with `Address already in
        // use`. End on the port, sweeping anything that still answers for it.
        if let port = recordedPort() {
            drainPort(port, seconds: 20)
            if portListener(port) == nil {
                log.notice("port \(port, privacy: .public) released")
            }
        }
        try? FileManager.default.removeItem(atPath: SecureServer.sandboxPortPath)

        reconcileSettings()
        clearRecord()
        return true
    }

    private static func waitForDeath(_ pid: Int32, seconds: Int) {
        var patience = seconds * 4
        while isAlive(pid), patience > 0 {
            patience -= 1
            Thread.sleep(forTimeInterval: 0.25)
        }
    }

    private static func clearRecord() {
        try? FileManager.default.removeItem(atPath: SecureServer.sandboxPIDPath)
    }

    private static func waitUntilExit(_ process: Process, seconds: Int) {
        var patience = seconds * 2
        while process.isRunning, patience > 0 {
            patience -= 1
            Thread.sleep(forTimeInterval: 0.5)
        }
    }
}

// MARK: - Authorization

private enum Authorization {
    /// Register the right in the system database. `timeout = 0` and
    /// `shared = false` mean a cached credential never satisfies a later call:
    /// every change asks for the password again.
    static func registerRight() {
        var auth: AuthorizationRef?
        guard AuthorizationCreate(nil, nil, [], &auth) == errAuthorizationSuccess,
              let auth
        else {
            log.error("could not create an authorization reference for right registration")
            return
        }
        defer { AuthorizationFree(auth, []) }

        // timeout 0 / not shared: a cached credential never satisfies a later
        // call, so every state change prompts for the administrator again.
        let definition: [String: Any] = [
            "class": "user",
            "group": "admin",
            "timeout": 0,
            "shared": false
        ]
        let comment = "oMLX needs administrator approval to change Secure Server mode, which runs the oMLX server as its own sandboxed account."
        let status = AuthorizationRightSet(
            auth,
            SecureServer.authRightName,
            definition as CFDictionary,
            comment as CFString,
            nil,
            nil
        )
        if status != errAuthorizationSuccess {
            log.error("AuthorizationRightSet failed with \(status)")
        }
    }

    /// Verify a caller that also has to prove it authenticated.
    ///
    /// Two independent checks, both required: the form has to carry the manage right
    /// (re-imported and asked non-interactively, so 32 arbitrary bytes fail), and the
    /// process on the other end has to be the oMLX binary. The daemon itself never
    /// prompts — the human gate is the app's own dialog, one fresh credential per
    /// change because the right is registered with `timeout = 0` and `shared = false`.
    static func failure(for clientPID: Int32, form: Data) -> String? {
        guard form.count == MemoryLayout<AuthorizationExternalForm>.size else {
            return "Malformed authorization form (\(form.count) bytes)."
        }
        if let detail = rightMissing(from: form) { return detail }
        return Caller.failure(for: clientPID)
    }

    /// Re-import the external form and ask the authorization server, without
    /// allowing a prompt, whether it grants the manage right.
    private static func rightMissing(from form: Data) -> String? {
        var external = AuthorizationExternalForm()
        form.withUnsafeBytes { source in
            withUnsafeMutableBytes(of: &external) { destination in
                destination.copyBytes(from: source)
            }
        }
        var authorization: AuthorizationRef?
        let created = AuthorizationCreateFromExternalForm(&external, &authorization)
        guard created == errAuthorizationSuccess, let authorization else {
            return "The authorization form could not be read (\(created))."
        }
        defer { AuthorizationFree(authorization, []) }

        var status = errAuthorizationSuccess
        SecureServer.authRightName.withCString { name in
            var item = AuthorizationItem(name: name, valueLength: 0, value: nil, flags: 0)
            // The item has to stay put for the duration of the call: `AuthorizationRights`
            // only holds a pointer to it.
            withUnsafeMutablePointer(to: &item) { itemPtr in
                var rights = AuthorizationRights(count: 1, items: itemPtr)
                status = AuthorizationCopyRights(
                    authorization, &rights, nil, [.extendRights, .preAuthorize], nil
                )
            }
        }
        guard status == errAuthorizationSuccess else {
            return "The authorization form does not grant Secure Server management (\(status))."
        }
        return nil
    }
}

/// Code identity of the process on the other end of the XPC connection.
private enum Caller {
    /// Verify the caller is the oMLX application this helper ships inside.
    /// Without this, any local process that could reach the mach service could
    /// start a server as the sandbox user or drop the lockdown.
    static func failure(for clientPID: Int32) -> String? {
        guard let expected = expectedCDHash() else {
            return "Cannot determine the oMLX application's code identity; refusing."
        }

        let attributes: [String: Any] = [kSecGuestAttributePid as String: clientPID]
        var code: SecCode?
        guard SecCodeCopyGuestWithAttributes(nil, attributes as CFDictionary, [], &code) == errSecSuccess,
              let code
        else {
            return "Cannot inspect the calling process (\(clientPID))."
        }
        // Signing information is read from the static code behind the running process.
        var staticCode: SecStaticCode?
        guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode else {
            return "Cannot inspect the calling process (\(clientPID))."
        }
        var information: CFDictionary?
        guard SecCodeCopySigningInformation(
            staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &information
        ) == errSecSuccess,
            let dict = information as? [String: Any],
            let cdhash = dict[kSecCodeInfoUnique as String] as? Data,
            !cdhash.isEmpty
        else {
            return "Cannot read the code signature of the calling process (\(clientPID))."
        }
        // The code-directory hash, compared directly. A `cdhash H"…"` requirement
        // answers the same question but only after the whole signature evaluates as
        // valid, which a linker-signed build can fail for reasons that have nothing to
        // do with identity — measured as `status` refused for the real app while the
        // authorization-gated `setEnabled` was accepted. Both hashes are named on a
        // mismatch so this is never opaque again.
        guard cdhash == expected else {
            return "The calling process is not the oMLX application (\(describe(clientPID)) cdhash \(hex(cdhash)) != \(hex(expected)))."
        }
        return nil
    }

    private static func decode(_ buffer: [CChar]) -> String {
        String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Who the caller is, for the refusal message: a pid alone does not say whether
    /// the peer is the app, a stale build, or something else entirely.
    private static func describe(_ pid: Int32) -> String {
        var name = [CChar](repeating: 0, count: 256)
        proc_name(pid, &name, 256)
        var path = [CChar](repeating: 0, count: 4096)
        proc_pidpath(pid, &path, 4096)
        return "\(decode(name)) at \(decode(path))"
    }

    /// The effective uid of the process at the other end of the connection, taken
    /// from the kernel's record of it rather than from anything the caller sent.
    static func euid(of pid: Int32) -> uid_t? {
        var info = proc_bsdinfo()
        let written = proc_pidinfo(
            pid, PROC_PIDTBSDINFO, 0, &info,
            Int32(MemoryLayout<proc_bsdinfo>.size)
        )
        guard written > 0 else { return nil }
        return info.pbi_uid
    }

    /// The code-directory hash of the app executable this helper ships inside.
    ///
    /// Content-addressed, so it also identifies an ad-hoc signed build, where there is
    /// no Team ID to compare against.
    private static func expectedCDHash() -> Data? {
        let helper = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let contents = helper.deletingLastPathComponent().deletingLastPathComponent()
        let appRoot = contents.deletingLastPathComponent()
        guard
            let info = NSDictionary(
                contentsOf: appRoot.appendingPathComponent("Contents/Info.plist")
            ),
            let executable = info["CFBundleExecutable"] as? String
        else {
            return nil
        }
        let appExecutable = contents.appendingPathComponent("MacOS/\(executable)")

        var staticCode: SecStaticCode?
        guard SecStaticCodeCreateWithPath(appExecutable as CFURL, [], &staticCode) == errSecSuccess,
              let staticCode
        else {
            return nil
        }
        var information: CFDictionary?
        guard SecCodeCopySigningInformation(
            staticCode,
            SecCSFlags(rawValue: kSecCSSigningInformation),
            &information
        ) == errSecSuccess,
            let dict = information as? [String: Any],
            let cdhash = dict[kSecCodeInfoUnique as String] as? Data,
            !cdhash.isEmpty
        else {
            return nil
        }
        return cdhash
    }

    private static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Lockdown

private enum Lockdown {
    /// The one-shot, idempotent bootstrap for turning the switch on: everything
    /// that has to be true, in one place. Each step is safe to repeat, so the
    /// switch, a startup recovery and a retry after a partial failure all take
    /// this same path. `applyLockdown()` flushes, loads, reads the rules back
    /// from the kernel and refuses to enable pf unless every rule is uid-scoped.
    static func bootstrap() throws {
        try Store.ensureLayout()
        try Sandbox.ensureAccount()
        try Sandbox.ensureHome()
        let token = try Store.applyLockdown()
        do {
            try Store.write(SecureServerState(enabled: true, pfToken: token))
        } catch {
            // A reference that was taken but never recorded would leave pf
            // enabled with nobody owning it: release it and flush the anchor.
            _ = try? Store.pfctl(["-X", token])
            _ = try? Store.pfctl(["-a", SecureServer.pfAnchor, "-F", "all"])
            throw error
        }
    }

    static func set(enabled: Bool) throws {
        try Store.ensureLayout()
        let previous = Store.read()

        if enabled {
            if previous.enabled, let token = previous.pfToken, !token.isEmpty, Store.pfIsEnabled() {
                // Already on: refresh the rules and keep the existing reference
                // instead of taking a second one.
                try Store.pfctl(["-a", SecureServer.pfAnchor, "-f", SecureServer.pfAnchorConfigPath])
                try Store.write(previous)
                return
            }
            do {
                try bootstrap()
            } catch {
                // Never leave the record claiming a lockdown that failed.
                try? Store.write(previous)
                throw error
            }
        } else {
            // The server goes first: it is the only thing the rules constrain,
            // and leaving it up after the rules are gone would be the one state
            // where it is neither sandboxed nor stopped. If it cannot be stopped,
            // the lockdown stays.
            guard Server.stop() else {
                throw HelperError.serverStillRunning
            }
            Store.dropLockdown()
            Access.revoke()
            try Store.write(.off)
        }
    }

    /// Re-assert the lockdown on helper start (login/boot). An enabled state
    /// file only wins if the rules can be verified as scoped; if they cannot,
    /// the state is reset to off rather than risking a host-wide block.
    static func recover() {
        let state = Store.read()
        guard state.enabled else { return }
        do {
            // This daemon is starting again — an update, a kickstart, a reboot —
            // and a fresh -E below would take a *second* reference while the one
            // on record is never released. Release the recorded reference first;
            // if it was the only one, pf stops for a moment and bootstrap starts
            // it again, which is the state we want either way.
            if let previous = state.pfToken, !previous.isEmpty, Store.pfIsEnabled() {
                _ = try? Store.pfctl(["-X", previous])
            }
            try bootstrap()
            log.notice("re-applied Secure Server lockdown on startup")
        } catch {
            log.error("could not re-apply lockdown on startup: \(String(describing: error), privacy: .public)")
            try? Store.write(.off)
        }
    }

    /// What is actually in force, measured rather than remembered.
    static func status() -> SecureServerStatus {
        let state = Store.read()
        let rules = Store.loadedRules()
        let offender = Store.rulesAreScoped(rules)
        let pfEnabled = Store.pfIsEnabled()
        var detail: String?
        if state.enabled, !pfEnabled {
            detail = "pf is disabled, so nothing is being blocked"
        } else if state.enabled, let offender {
            detail = "refusing to enforce a rule that is not scoped to \(SecureServer.sandboxUser): \(offender)"
        }
        return SecureServerStatus(
            enabled: state.enabled,
            pfEnabled: pfEnabled,
            // An empty anchor is not "loaded", and an unscoped one protects
            // nothing, so neither counts.
            rulesLoaded: !rules.isEmpty && offender == nil,
            sandboxUID: Sandbox.uid.map { Int($0) },
            serverPID: Server.runningPID.map(Int.init),
            detail: detail
        )
    }
}

// MARK: - Service

private final class SecureServerService: NSObject, SecureServerHelperProtocol {
    private let clientPID: Int32
    /// The user running the app. Grants are limited to what this uid owns, so the
    /// identity is read from the kernel rather than taken from the request.
    private let clientUID: uid_t?

    init(clientPID: Int32) {
        self.clientPID = clientPID
        self.clientUID = Caller.euid(of: clientPID)
    }

    func getState(with reply: @escaping (Data?) -> Void) {
        // Pinned like every other method: these only read, but they read as root and
        // an unauthenticated listener is still an unauthenticated listener.
        if let failure = Caller.failure(for: clientPID) {
            log.error("refused getState: \(failure, privacy: .public)")
            reply(nil)
            return
        }
        reply(FileManager.default.contents(atPath: SecureServer.statePath))
    }

    func status(with reply: @escaping (Data) -> Void) {
        if let failure = Caller.failure(for: clientPID) {
            log.error("refused status: \(failure, privacy: .public)")
            reply(encode(SecureServerStatus.unavailable))
            return
        }
        reply(encode(Lockdown.status()))
    }

    func setEnabled(
        _ enabled: Bool,
        authorization: Data,
        with reply: @escaping (Bool, String?) -> Void
    ) {
        // Both gates: the form proves an administrator asked, the pin proves the ask came
        // from this app. Either one alone leaves a path open — a cached form was the old
        // credential-caching bug, and a form alone can be minted by any admin process.
        if let failure = Caller.failure(for: clientPID) {
            log.error("refused setEnabled(\(enabled)): \(failure, privacy: .public)")
            reply(false, failure)
            return
        }
        if let failure = Authorization.failure(for: clientPID, form: authorization) {
            log.error("refused setEnabled(\(enabled)): \(failure, privacy: .public)")
            reply(false, failure)
            return
        }
        do {
            try Lockdown.set(enabled: enabled)
            log.notice("Secure Server \(enabled ? "enabled" : "disabled", privacy: .public)")
            reply(true, nil)
        } catch {
            let detail = String(describing: error)
            log.error("setEnabled(\(enabled)) failed: \(detail, privacy: .public)")
            reply(false, detail)
        }
    }

    /// No authorization form, because the grants below are bounded to files the
    /// caller already owns: with that bound in place this asks for nothing the app
    /// could not do to itself. The caller still has to be the oMLX binary.
    func startServer(_ launch: Data, with reply: @escaping (Data) -> Void) {
        if let failure = Caller.failure(for: clientPID) {
            log.error("refused startServer: \(failure, privacy: .public)")
            reply(encode(SecureServerLaunchResult.failure(failure)))
            return
        }
        guard let request = decode(SecureServerLaunch.self, from: launch) else {
            reply(encode(SecureServerLaunchResult.failure("Malformed launch request.")))
            return
        }
        guard let clientUID else {
            log.error("refused startServer: cannot identify the calling user (pid \(self.clientPID))")
            reply(encode(SecureServerLaunchResult.failure("Cannot identify the calling user.")))
            return
        }
        do {
            let pid = try Server.start(request, clientUID: clientUID)
            reply(encode(SecureServerLaunchResult(pid: Int(pid), error: nil)))
        } catch {
            let detail = String(describing: error)
            log.error("startServer failed: \(detail, privacy: .public)")
            reply(encode(SecureServerLaunchResult.failure(detail)))
        }
    }

    func stopServer(with reply: @escaping (Data) -> Void) {
        if let failure = Caller.failure(for: clientPID) {
            log.error("refused stopServer: \(failure, privacy: .public)")
            reply(encode(SecureServerStopResult(stopped: false, error: failure)))
            return
        }
        Server.stop()
        // Truthful: the app must not start a second server on a port the first one
        // still holds because the stop silently did not happen.
        let stopped = Server.pid == nil
        reply(encode(SecureServerStopResult(
            stopped: stopped,
            error: stopped ? nil : "the sandboxed server did not stop"
        )))
    }

    private func encode<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return (try? encoder.encode(value)) ?? Data("{}".utf8)
    }

    private func decode<T: Decodable>(_ type: T.Type, from data: Data) -> T? {
        try? JSONDecoder().decode(type, from: data)
    }
}

/// The client that could have started the server is gone: stop it. The app owns
/// the server's lifecycle and cannot signal a process owned by the sandbox account,
/// so this is the only place that can.
///
/// A dropped connection on its own is not the signal — a helper update or
/// `launchctl kickstart` drops it while the app keeps running, and killing the
/// server then would take inference down with it. The check is whether the client
/// process still exists, with a moment's patience because a quit is not
/// instantaneous and a relaunch (`open -n`, an update) reconnects right away.
private func clientGone(_ clientPID: pid_t) {
    DispatchQueue.global(qos: .utility).async {
        for _ in 0..<4 {
            if kill(clientPID, 0) == 0 || errno != ESRCH { return }
            Thread.sleep(forTimeInterval: 0.5)
        }
        guard Server.pid != nil else { return }
        log.notice("oMLX (pid \(clientPID)) exited; stopping the sandboxed server it owned")
        Server.stop()
    }
}

private final class ListenerDelegate: NSObject, NSXPCListenerDelegate {
    func listener(
        _ listener: NSXPCListener,
        shouldAcceptNewConnection connection: NSXPCConnection
    ) -> Bool {
        let clientPID = connection.processIdentifier
        connection.exportedInterface = NSXPCInterface(with: SecureServerHelperProtocol.self)
        connection.exportedObject = SecureServerService(clientPID: clientPID)
        connection.interruptionHandler = { clientGone(clientPID) }
        connection.invalidationHandler = { clientGone(clientPID) }
        connection.resume()
        return true
    }
}

// MARK: - Drop-and-exec trampoline

/// The root daemon re-executes its own binary in this mode to become the server.
/// macOS has no `POSIX_SPAWN_SETEUID` and Swift 6 strict concurrency forbids
/// `fork()`, so the uid change has to happen inside a root process that then
/// `exec`s the interpreter. This short-lived child is that process; it is handled
/// before anything starts an XPC listener, and `run()` never returns.
private enum DropAndExec {
    static let flag = "--drop-and-exec"

    static var isTrampoline: Bool {
        CommandLine.arguments.dropFirst().first == flag
    }

    /// Drop to `_omlx` and exec the command in argv. Called only in a child of
    /// the root daemon. uid/gid come from the account record, not argv, so a
    /// local caller cannot name an arbitrary identity; from a non-root parent
    /// `setgroups`/`setuid` simply fail with EPERM and the child exits.
    static func run() -> Never {
        guard let record = getpwnam(SecureServer.sandboxUser)?.pointee else {
            fail("the \(SecureServer.sandboxUser) account is missing")
        }
        // Supplementary groups must be dropped while still root: setuid() below
        // removes the privilege setgroups() needs.
        guard setgroups(0, nil) == 0 else { fail("setgroups failed: \(errno)") }
        guard setgid(record.pw_gid) == 0 else { fail("setgid failed: \(errno)") }
        guard setuid(record.pw_uid) == 0 else { fail("setuid failed: \(errno)") }
        // Working directory is inherited from launchd (`/`), the same cwd the
        // previous sudo-based spawn gave the server. Everything the sandbox
        // writes is named by absolute environment variables (HOME/TMPDIR,
        // OMLX_LOG_DIR, OMLX_SETTINGS_FILE, …) or by the ACL grants, so cwd is
        // never written to. Do not chdir here: it can fail and abort the spawn,
        // and moving cwd would change relative-path resolution compared with
        // the behaviour the old spawn was proven to have.
        let command = Array(CommandLine.arguments.dropFirst(2))
        guard let executable = command.first, !executable.isEmpty else {
            fail("no server executable in argv")
        }
        // argv/envp for execve. The daemon passes the complete environment via
        // the spawn, so no token or key ever appears on a command line. The
        // strdup'd strings are deliberately not freed: execve replaces this image.
        var argv: [UnsafeMutablePointer<CChar>?] = command.map { strdup($0) }
        argv.append(nil)
        var envp: [UnsafeMutablePointer<CChar>?] = ProcessInfo.processInfo.environment
            .map { strdup("\($0.key)=\($0.value)") }
        envp.append(nil)
        execve(executable, &argv, &envp)
        fail("execve(\(executable)) failed: \(errno)")
    }

    private static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data(("oMLX drop-and-exec: " + message + "\n").utf8))
        _exit(127)
    }
}

// MARK: - Entry point

// A --drop-and-exec launch is the daemon's own binary re-executed as the child
// that becomes the server. Handle it before the XPC listener starts as root.
if DropAndExec.isTrampoline {
    DropAndExec.run()          // noreturn
}

Authorization.registerRight()
Lockdown.recover()

private let delegate = ListenerDelegate()
let listener = NSXPCListener(machServiceName: SecureServer.machServiceName)
listener.delegate = delegate
listener.resume()
RunLoop.current.run()
