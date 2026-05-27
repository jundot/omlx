// The kwargs codec is the only place the editor's flat entry list meets
// the server's (chat_template_kwargs, forced_ct_kwargs) pair. These
// tests pin the round-trip so a future refactor of either side fails
// at compile/test time rather than silently dropping the user's
// settings on save.

import XCTest
@testable import oMLX

final class ChatTemplateKwargsCodecTests: XCTestCase {

    // MARK: - Encode

    func testEmptyEntriesEncodeToNilPair() {
        let (kwargs, forced) = ChatTemplateKwargsCodec.encode([])
        XCTAssertNil(kwargs)
        XCTAssertNil(forced)
    }

    func testEnableThinkingEncodesAsBool() {
        let entries = [
            ChatTemplateKwargEntry(kind: .enableThinking, value: "true", force: false),
        ]
        let (kwargs, forced) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?["enable_thinking"]?.value as? Bool, true)
        XCTAssertNil(forced)
    }

    func testEnableThinkingFalseEncodesAsBool() {
        let entries = [
            ChatTemplateKwargEntry(kind: .enableThinking, value: "false", force: false),
        ]
        let (kwargs, _) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?["enable_thinking"]?.value as? Bool, false)
    }

    func testReasoningEffortEncodesAsString() {
        let entries = [
            ChatTemplateKwargEntry(kind: .reasoningEffort, value: "medium", force: false),
        ]
        let (kwargs, _) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?["reasoning_effort"]?.value as? String, "medium")
    }

    func testCustomBoolCoercion() {
        let entries = [
            ChatTemplateKwargEntry(kind: .custom, customKey: "do_thing", value: "true"),
            ChatTemplateKwargEntry(kind: .custom, customKey: "skip_thing", value: "false"),
        ]
        let (kwargs, _) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?["do_thing"]?.value as? Bool, true)
        XCTAssertEqual(kwargs?["skip_thing"]?.value as? Bool, false)
    }

    func testCustomIntCoercion() {
        let entries = [
            ChatTemplateKwargEntry(kind: .custom, customKey: "n_layers", value: "42"),
        ]
        let (kwargs, _) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?["n_layers"]?.value as? Int, 42)
    }

    func testCustomDoubleCoercion() {
        let entries = [
            ChatTemplateKwargEntry(kind: .custom, customKey: "alpha", value: "0.25"),
        ]
        let (kwargs, _) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?["alpha"]?.value as? Double, 0.25)
    }

    func testCustomStringFallback() {
        // Anything that isn't bool-literal or numeric stays a string.
        let entries = [
            ChatTemplateKwargEntry(kind: .custom, customKey: "role", value: "expert"),
        ]
        let (kwargs, _) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?["role"]?.value as? String, "expert")
    }

    func testCustomBlankKeyIsDropped() {
        // Mirrors dashboard.js's `e.key && e.key.trim()` guard.
        let entries = [
            ChatTemplateKwargEntry(kind: .custom, customKey: "   ", value: "v"),
            ChatTemplateKwargEntry(kind: .custom, customKey: "real", value: "v"),
        ]
        let (kwargs, _) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?.count, 1)
        XCTAssertEqual(kwargs?["real"]?.value as? String, "v")
    }

    func testForceRoutesIntoForcedList() {
        let entries = [
            ChatTemplateKwargEntry(kind: .enableThinking, value: "true", force: true),
            ChatTemplateKwargEntry(kind: .reasoningEffort, value: "high", force: false),
            ChatTemplateKwargEntry(kind: .custom, customKey: "k", value: "v", force: true),
        ]
        let (kwargs, forced) = ChatTemplateKwargsCodec.encode(entries)
        XCTAssertEqual(kwargs?.count, 3)
        XCTAssertEqual(Set(forced ?? []), Set(["enable_thinking", "k"]))
    }

    // MARK: - Decode

    func testDecodeEmptyDictReturnsEmpty() {
        XCTAssertTrue(ChatTemplateKwargsCodec.decode(kwargs: nil, forced: nil).isEmpty)
        XCTAssertTrue(ChatTemplateKwargsCodec.decode(kwargs: [:], forced: nil).isEmpty)
    }

    func testDecodeKnownKindsAppearFirst() {
        // The HTML editor's UX intuition is that the canonical kinds
        // ride at the top; custom entries below in alphabetical order.
        let kwargs: [String: AnyCodable] = [
            "z_custom": AnyCodable("v"),
            "enable_thinking": AnyCodable(true),
            "a_custom": AnyCodable("v"),
            "reasoning_effort": AnyCodable("low"),
        ]
        let entries = ChatTemplateKwargsCodec.decode(kwargs: kwargs, forced: nil)
        XCTAssertEqual(entries.count, 4)
        XCTAssertEqual(entries[0].kind, .enableThinking)
        XCTAssertEqual(entries[1].kind, .reasoningEffort)
        XCTAssertEqual(entries[2].kind, .custom)
        XCTAssertEqual(entries[2].customKey, "a_custom")
        XCTAssertEqual(entries[3].customKey, "z_custom")
    }

    func testDecodeAppliesForcedFlag() {
        let kwargs: [String: AnyCodable] = [
            "enable_thinking": AnyCodable(true),
            "custom_a": AnyCodable("v"),
        ]
        let entries = ChatTemplateKwargsCodec.decode(
            kwargs: kwargs,
            forced: ["enable_thinking"]
        )
        XCTAssertTrue(entries.first(where: { $0.kind == .enableThinking })!.force)
        XCTAssertFalse(entries.first(where: { $0.kind == .custom })!.force)
    }

    func testDecodeStringifiesValues() {
        let kwargs: [String: AnyCodable] = [
            "enable_thinking": AnyCodable(true),
            "n":               AnyCodable(7),
            "alpha":           AnyCodable(0.25),
            "role":            AnyCodable("expert"),
        ]
        let entries = ChatTemplateKwargsCodec.decode(kwargs: kwargs, forced: nil)
        let byKey: [String: String] = entries.reduce(into: [:]) {
            $0[$1.resolvedKey ?? ""] = $1.value
        }
        XCTAssertEqual(byKey["enable_thinking"], "true")
        XCTAssertEqual(byKey["n"], "7")
        // 0.25 is non-integral → keeps decimal form.
        XCTAssertEqual(byKey["alpha"], "0.25")
        XCTAssertEqual(byKey["role"], "expert")
    }

    // MARK: - Round-trip

    func testRoundTripPreservesKnownAndCustomEntries() {
        let original = [
            ChatTemplateKwargEntry(kind: .enableThinking, value: "false", force: true),
            ChatTemplateKwargEntry(kind: .reasoningEffort, value: "high", force: false),
            ChatTemplateKwargEntry(kind: .custom, customKey: "max_branches", value: "5", force: true),
        ]
        let (kwargs, forced) = ChatTemplateKwargsCodec.encode(original)
        let roundtripped = ChatTemplateKwargsCodec.decode(kwargs: kwargs, forced: forced)
        XCTAssertEqual(roundtripped.count, 3)

        let byKey: [String: ChatTemplateKwargEntry] = roundtripped.reduce(into: [:]) {
            $0[$1.resolvedKey ?? ""] = $1
        }
        XCTAssertEqual(byKey["enable_thinking"]?.value, "false")
        XCTAssertEqual(byKey["enable_thinking"]?.force, true)
        XCTAssertEqual(byKey["reasoning_effort"]?.value, "high")
        XCTAssertEqual(byKey["reasoning_effort"]?.force, false)
        XCTAssertEqual(byKey["max_branches"]?.value, "5")
        XCTAssertEqual(byKey["max_branches"]?.force, true)
    }
}
