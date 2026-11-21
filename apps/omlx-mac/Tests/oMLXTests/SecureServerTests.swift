// SPDX-License-Identifier: Apache-2.0

// Guards the Secure Server values that cross a process and a language boundary,
// and the one property the whole feature exists for: what gets loaded restricts
// the sandbox account and nothing else.
//
// The Python server reads the state file at a fixed path, so a typo here means
// the dashboard reports "off" forever. The ruleset literal *is* the pf
// configuration, so drift in it is drift in the security property — which is why
// the rules are checked for the invariants that decide whether this Mac keeps its
// network (every rule names the sandbox uid, the loopback passes outrank the
// blocks, every rule names a protocol) instead of only being compared against
// themselves.

import XCTest
@testable import oMLX

final class SecureServerStateTests: XCTestCase {
    private func decode(_ json: String) throws -> SecureServerState {
        try JSONDecoder().decode(SecureServerState.self, from: Data(json.utf8))
    }

    func testDecodesEnabledFlag() throws {
        let state = try decode(#"{"enabled":true}"#)
        XCTAssertTrue(state.enabled)
        XCTAssertNil(state.pfToken)
    }

    func testDecodesDisabledFlag() throws {
        XCTAssertFalse(try decode(#"{"enabled":false}"#).enabled)
    }

    func testDecodesSnakeCaseToken() throws {
        let state = try decode(#"{"enabled":true,"pf_token":"2410633657"}"#)
        XCTAssertTrue(state.enabled)
        XCTAssertEqual(state.pfToken, "2410633657")
    }

    func testEncodesSnakeCaseToken() throws {
        let encoded = try JSONEncoder().encode(SecureServerState(enabled: true, pfToken: "42"))
        let text = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        XCTAssertTrue(text.contains(#""pf_token":"42""#), text)
    }

    func testOffIsDisabledWithNoToken() {
        XCTAssertEqual(SecureServerState.off, SecureServerState(enabled: false, pfToken: nil))
    }
}

final class SecureServerLayoutTests: XCTestCase {
    /// These literals are the contract with omlx/secure_mode.py.
    func testPathsMatchThePythonReader() {
        XCTAssertEqual(SecureServer.statePath, "/Library/Application Support/oMLX/secure-server/state.json")
        XCTAssertEqual(SecureServer.pfAnchorConfigPath, "/Library/Application Support/oMLX/secure-server/pf-anchor.conf")
        XCTAssertTrue(SecureServer.configDirectory.hasPrefix("/Library/Application Support/oMLX/"))
    }

    /// The sandbox's HOME/TMPDIR, its log and its cache have to sit under the
    /// root-owned directory: the sandbox must be able to use them without being able
    /// to change the rules stored next to them.
    func testSandboxFilesLiveInsideTheConfigDirectory() {
        for path in [SecureServer.sandboxHome, SecureServer.serverLogPath,
                     SecureServer.sandboxLogDir, SecureServer.sandboxCacheDir] {
            XCTAssertTrue(path.hasPrefix(SecureServer.configDirectory + "/"), path)
        }
    }

    /// The variables `omlx/settings.py` reads to move the server's own writes out of
    /// the base directory. Getting a name wrong fails silently: the sandboxed server
    /// simply cannot create its log or cache directory.
    func testSandboxEnvironmentKeysMatchThePythonReader() {
        XCTAssertEqual(SecureServer.logDirEnvironmentKey, "OMLX_LOG_DIR")
        XCTAssertEqual(SecureServer.cacheDirEnvironmentKey, "OMLX_SSD_CACHE_DIR")
        XCTAssertEqual(SecureServer.settingsFileEnvironmentKey, "OMLX_SETTINGS_FILE")
    }

    /// The record teardown trusts names a file, an owner and a digest, and root acts on
    /// all three — so it must live in the root-owned configuration directory. Beside
    /// the sandbox's own copy, the account this feature contains could rewrite it and
    /// turn teardown into an arbitrary root write and chown.
    func testSettingsSeedRecordLivesOutsideTheSandboxHome() {
        XCTAssertTrue(SecureServer.settingsSeedPath.hasPrefix(SecureServer.configDirectory + "/"))
        XCTAssertFalse(SecureServer.settingsSeedPath.hasPrefix(SecureServer.sandboxHome + "/"))
        XCTAssertTrue(SecureServer.sandboxSettingsPath.hasPrefix(SecureServer.sandboxHome + "/"))
    }

    func testPFAnchorAndServiceNames() {
        XCTAssertEqual(SecureServer.pfAnchor, "com.apple/900.omlx.secure")
        XCTAssertEqual(SecureServer.machServiceName, "com.omlx.secure-server-helper")
        XCTAssertEqual(SecureServer.helperPlistName, "com.omlx.secure-server-helper.plist")
    }
}

final class SecureServerRulesTests: XCTestCase {
    /// The rules as pf will hold them: comments and blanks stripped.
    private var rules: [String] {
        SecureServer.pfAnchorRules
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    /// Loopback in and out, then drop every off-box packet, in both directions,
    /// for the sandbox account only. Pinned verbatim so that loosening it is a
    /// deliberate edit and not a surprise six months from now.
    func testRulesetLiteral() {
        let expected = """
        # oMLX Secure Server - lockdown of the sandboxed server account only.
        # Loaded via: pfctl -a com.apple/900.omlx.secure -f <this file>
        # Every rule is scoped with `user`: no other process on this Mac is matched.
        # Loopback first: local clients keep working, `quick` ends the search here.
        # Both protocols in both directions, because a UDP listener under this account
        # would otherwise be reachable from the network while the docs say it cannot be.
        pass in  quick on lo0 proto tcp from any to any user _omlx keep state
        pass in  quick on lo0 proto udp from any to any user _omlx keep state
        pass out quick on lo0 proto tcp from any user _omlx keep state
        pass out quick on lo0 proto udp from any user _omlx keep state
        # Everything off-box is refused, in both directions.
        block drop in  quick proto tcp from any to any user _omlx
        block drop in  quick proto udp from any to any user _omlx
        block drop out quick proto tcp from any user _omlx
        block drop out quick proto udp from any user _omlx

        """
        XCTAssertEqual(SecureServer.pfAnchorRules, expected)
    }

    /// One unqualified rule and the lockdown is host-wide instead of server-wide —
    /// the failure mode that once took this Mac's whole network down.
    func testEveryRuleIsScopedToTheSandboxAccount() {
        XCTAssertEqual(rules.count, 8)
        for rule in rules {
            XCTAssertTrue(rule.contains("user \(SecureServer.sandboxUser)"), rule)
        }
    }

    /// `quick` stops at the first match, so a block ranked above the loopback
    /// passes would cut this Mac off from its own server.
    func testLoopbackPassesPrecedeEveryBlock() throws {
        let firstBlock = try XCTUnwrap(rules.firstIndex { $0.hasPrefix("block") })
        let lastPass = try XCTUnwrap(rules.lastIndex { $0.hasPrefix("pass") })
        XCTAssertLessThan(lastPass, firstBlock)
        for rule in rules where rule.hasPrefix("pass") {
            XCTAssertTrue(rule.contains(" on lo0 "), rule)
        }
    }

    /// Only TCP and UDP can be associated with a uid. A rule with no protocol
    /// would also match other processes' ICMP — again, host-wide damage.
    func testEveryRuleNamesAProtocol() {
        for rule in rules {
            XCTAssertTrue(rule.contains("proto tcp") || rule.contains("proto udp"), rule)
        }
    }
}

/// The guard that runs before pf is enabled, against the kernel's own rendering of
/// the loaded rules. Measured rendering: pf resolves account names as the rules are
/// loaded, so `user _omlx` comes back as `user = 502`.
final class SecureServerScopeCheckTests: XCTestCase {
    private let loopbackPass = "pass in quick on lo0 proto tcp all user = 502 flags S/SA keep state"
    private let egressBlock = "block drop out quick proto tcp all user = 502"
    private let hostWideBlock = "block drop out quick proto tcp all"

    func testAcceptsTheKernelRenderingOfTheShippedRuleset() {
        XCTAssertNil(SecureServer.firstUnscopedRule([loopbackPass, egressBlock], sandboxUID: 502))
    }

    /// A stale or hand-edited rule is reported, not activated: it is the first
    /// thing in the list that is not scoped, so the daemon can name it.
    func testReportsTheFirstUnqualifiedRule() {
        XCTAssertEqual(
            SecureServer.firstUnscopedRule([loopbackPass, hostWideBlock, egressBlock], sandboxUID: 502),
            hostWideBlock
        )
    }

    func testRejectsAnEmptyAnchor() {
        XCTAssertEqual(SecureServer.firstUnscopedRule([], sandboxUID: 502), "the anchor is empty")
    }

    /// `user = 5020` is not `user = 502`. A substring test would accept an anchor
    /// scoped to somebody else's uid and then report the server as jailed.
    func testRejectsALongerUidThatStartsWithOurs() {
        let otherAccount = "block drop out quick proto tcp all user = 5020"
        XCTAssertEqual(
            SecureServer.firstUnscopedRule([otherAccount], sandboxUID: 502),
            otherAccount
        )
    }

    /// Without the account there is nothing to scope to; refusing is the only
    /// safe answer, because the alternative is loading rules that match nobody and
    /// reporting the server as jailed.
    func testRejectsMissingSandboxAccount() {
        XCTAssertEqual(
            SecureServer.firstUnscopedRule([egressBlock], sandboxUID: nil),
            "the \(SecureServer.sandboxUser) account does not exist"
        )
    }
}

/// The UI reports `enforced`, never the flag the operator touched: a state file
/// asserting a lockdown that pf is not applying must not render as success.
final class SecureServerStatusTests: XCTestCase {
    private func status(enabled: Bool, pf: Bool, rules: Bool) -> SecureServerStatus {
        SecureServerStatus(
            enabled: enabled, pfEnabled: pf, rulesLoaded: rules,
            sandboxUID: 502, serverPID: nil, detail: nil
        )
    }

    func testEnforcedNeedsEveryCondition() {
        XCTAssertTrue(status(enabled: true, pf: true, rules: true).enforced)
        // Someone ran `pfctl -d` in a terminal; the flag still says on.
        XCTAssertFalse(status(enabled: true, pf: false, rules: true).enforced)
        // pf is up but our anchor was flushed.
        XCTAssertFalse(status(enabled: true, pf: true, rules: false).enforced)
        XCTAssertFalse(status(enabled: false, pf: true, rules: true).enforced)
    }
}

/// The launch path asks the record, not the XPC round trip, whether the mode is on:
/// a server's uid is fixed when its listening socket is created, so the answer has to
/// be available before the first reply could land. These stand a temp file in for the
/// root-owned one.
final class SecureServerRecordTests: XCTestCase {
    private func withStatePath(_ path: String, _ body: () throws -> Void) rethrows {
        setenv(SecureServer.statePathEnvironmentKey, path, 1)
        addTeardownBlock { unsetenv(SecureServer.statePathEnvironmentKey) }
        try body()
    }

    private func stateFile(_ contents: String) throws -> String {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-record-\(UUID().uuidString).json").path
        try contents.write(toFile: path, atomically: true, encoding: .utf8)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: path) }
        return path
    }

    func testEnabledRecordIsVisibleSynchronously() throws {
        let path = try stateFile(#"{"enabled":true,"pf_token":"0123456789ABCDEF"}"#)
        try withStatePath(path) {
            XCTAssertTrue(SecureServer.enabledOnRecord)
            XCTAssertEqual(SecureServer.recordedState().pfToken, "0123456789ABCDEF")
        }
    }

    func testAbsentRecordReadsAsOff() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("omlx-record-\(UUID().uuidString).json").path
        try withStatePath(missing) { XCTAssertFalse(SecureServer.enabledOnRecord) }
    }

    /// Truncated or hand-edited content must never be read as a request it does not
    /// contain — the same rule the Python side follows, so both halves agree.
    func testMalformedRecordReadsAsOff() throws {
        for body in ["", "{", "not json", #"{"pf_token":"0123456789ABCDEF"}"#] {
            let path = try stateFile(body)
            try withStatePath(path) { XCTAssertFalse(SecureServer.enabledOnRecord, "read as on: \(body)") }
        }
    }
}

/// `pfctl -E` is the one call whose useful output lands on stderr, and reading only
/// stdout is what made a real flip fail — after pf had already been enabled, with a
/// reference the daemon could then not release. The daemon parses through this, so
/// the behaviour is pinned here rather than in a comment.
final class SecureServerTokenTests: XCTestCase {
    func testFindsTheTokenAmongTheAltqBanner() {
        let output = """
        No ALTQ support in kernel
        ALTQ related functions disabled
        pf enabled
        Token : 1978904160
        """
        XCTAssertEqual(SecureServer.enableToken(in: output), "1978904160")
    }

    func testRequiresTheTokenLabel() {
        XCTAssertNil(SecureServer.enableToken(in: "pf enabled"))
        XCTAssertNil(SecureServer.enableToken(in: ""))
        XCTAssertNil(SecureServer.enableToken(in: "Token :"))
        XCTAssertNil(SecureServer.enableToken(in: "Token :   "))
    }

    func testTakesTheFirstTokenOnly() {
        let output = "Token : 1\nToken : 2"
        XCTAssertEqual(SecureServer.enableToken(in: output), "1")
    }

    /// A rule list must not be mistaken for a token: the marker is a *label*, and the
    /// only line that has both the word and a value after a colon is the reference.
    func testIgnoresRuleText() {
        let rules = """
        block drop out quick proto tcp from any user = 502
        pass in quick on lo0 proto tcp from any to any user = 502 keep state
        """
        XCTAssertNil(SecureServer.enableToken(in: rules))
    }
}
