// Ask AI pure logic: SSE line parsing, slash-command parsing, request body
// shape, and tolerant preference decoding.

import Carbon.HIToolbox
import XCTest
@testable import oMLX

final class AskAITests: XCTestCase {

    func testStreamLineParsing() throws {
        XCTAssertEqual(
            try ChatStreamLine.parse(#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#),
            .delta(content: "Hi", reasoning: "")
        )
        XCTAssertEqual(
            try ChatStreamLine.parse("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\r"),
            .delta(content: "", reasoning: "hmm")
        )
        XCTAssertEqual(try ChatStreamLine.parse("data: [DONE]"), .done)
        // Role preamble, server keepalive chunk, and comment keepalive carry no text.
        XCTAssertEqual(try ChatStreamLine.parse(#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#), .ignore)
        XCTAssertEqual(try ChatStreamLine.parse(#"data: {"choices":[{"delta":{"content":""}}]}"#), .ignore)
        XCTAssertEqual(try ChatStreamLine.parse(": keep-alive"), .ignore)
        XCTAssertThrowsError(try ChatStreamLine.parse(#"data: {"error":{"message":"boom","type":"x"}}"#)) {
            XCTAssertEqual($0 as? ChatStreamError, .server("boom"))
        }
        XCTAssertEqual(
            ChatStreamLine.errorMessage(fromBody: Data(#"{"error":{"message":"Model not found"}}"#.utf8)),
            "Model not found"
        )
    }

    func testSlashCommandParsing() {
        let translate = AskAIPreset(command: "gemma-translate", model: "gemma-3-12b")
        let tldr = AskAIPreset(command: "tldr", model: "qwen")
        let presets = [translate, tldr]

        XCTAssertEqual(AskAIInput.parse("/gemma-translate 오늘 주간업무 보고 있습니다.", presets: presets),
                       .preset(translate, body: "오늘 주간업무 보고 있습니다."))
        XCTAssertEqual(AskAIInput.parse("/Gemma-Translate\nline two", presets: presets),
                       .preset(translate, body: "line two"))
        XCTAssertEqual(AskAIInput.parse("/tldr", presets: presets), .preset(tldr, body: ""))
        let korean = AskAIPreset(command: "번역", model: "gemma-3-12b")
        XCTAssertEqual(AskAIInput.parse("/번역 안녕하세요?", presets: [korean]), .preset(korean, body: "안녕하세요?"))
        XCTAssertEqual(AskAIInput.suggestions(for: "/번", presets: [korean]), [korean])
        XCTAssertEqual(AskAIInput.parse("/nope hi", presets: presets), .unknownCommand("nope"))
        // A path is a question, not a command.
        XCTAssertEqual(AskAIInput.parse("/usr/bin is what?", presets: presets), .plain("/usr/bin is what?"))
        XCTAssertEqual(AskAIInput.parse("  hello ", presets: presets), .plain("hello"))

        XCTAssertEqual(AskAIInput.suggestions(for: "/", presets: presets), [translate, tldr])
        XCTAssertEqual(AskAIInput.suggestions(for: "/gem", presets: presets), [translate])
        XCTAssertEqual(AskAIInput.suggestions(for: "/gemma-translate ", presets: presets), [])
        XCTAssertEqual(AskAIInput.suggestions(for: "hi", presets: presets), [])
    }

    @MainActor
    func testRequestBodyOrdersSystemHistoryAndQuestion() throws {
        var answered = AskAIConversation.Turn(question: "q1")
        answered.answer = "a1"
        var failed = AskAIConversation.Turn(question: "q2")
        failed.error = "boom"
        let profile = AskAIProfile(model: "m", systemPrompt: " Be brief. ", allowThinking: false, autoCopy: false)
        let messages = AskAIConversation.messages(profile: profile, history: [answered, failed], question: "q3")
        XCTAssertEqual(messages.map(\.role), [.system, .user, .assistant, .user])
        XCTAssertEqual(messages.map(\.content), ["Be brief.", "q1", "a1", "q3"])

        let off = try JSONSerialization.jsonObject(
            with: ChatCompletionBody(model: "m", messages: messages, allowThinking: false).encoded()
        ) as! [String: Any]
        XCTAssertEqual(off["stream"] as? Bool, true)
        XCTAssertEqual((off["chat_template_kwargs"] as? [String: Bool])?["enable_thinking"], false)

        let on = try JSONSerialization.jsonObject(
            with: ChatCompletionBody(model: "m", messages: messages, allowThinking: true).encoded()
        ) as! [String: Any]
        XCTAssertNil(on["chat_template_kwargs"])
        XCTAssertNil(on["gpu_power_limit_w"])

        let capped = try JSONSerialization.jsonObject(
            with: ChatCompletionBody(model: "m", messages: messages, allowThinking: true,
                                     gpuPowerLimitW: 10).encoded()
        ) as! [String: Any]
        XCTAssertEqual(capped["gpu_power_limit_w"] as? Int, 10)

        let withImage = ChatMessage(role: .user, content: "What is this?", imageURLs: ["data:image/jpeg;base64,AAAA"])
        let body = try JSONSerialization.jsonObject(
            with: ChatCompletionBody(model: "vlm", messages: [withImage], allowThinking: false).encoded()
        ) as! [String: Any]
        let parts = (body["messages"] as! [[String: Any]])[0]["content"] as! [[String: Any]]
        XCTAssertEqual(parts.map { $0["type"] as? String }, ["image_url", "text"])
        XCTAssertEqual((parts[0]["image_url"] as? [String: String])?["url"], "data:image/jpeg;base64,AAAA")
        XCTAssertEqual(parts[1]["text"] as? String, "What is this?")
    }

    func testPrefsDecodeFillsMissingKeys() throws {
        let json = #"{"enabled":true,"presets":[{"command":"tldr","model":"qwen"}]}"#
        let prefs = try JSONDecoder().decode(AskAIPrefs.self, from: Data(json.utf8))
        XCTAssertTrue(prefs.enabled)
        XCTAssertEqual(prefs.hotKey, .optionSpace)
        XCTAssertEqual(prefs.newConversationAfter, 300)
        XCTAssertNil(prefs.panelPosition)
        XCTAssertEqual(prefs.panelWidth, AskAIPrefs.defaultPanelWidth)
        XCTAssertEqual(prefs.presets.first?.command, "tldr")
        XCTAssertEqual(prefs.presets.first?.allowThinking, false)
        XCTAssertFalse(prefs.gpuPowerLimitEnabled)
        XCTAssertNil(prefs.requestGPUPowerLimit)
    }

    func testGPUPowerLimitIsClampedToTheServerRange() throws {
        let json = #"{"gpuPowerLimitEnabled":true,"gpuPowerLimitWatts":1}"#
        var prefs = try JSONDecoder().decode(AskAIPrefs.self, from: Data(json.utf8))
        XCTAssertEqual(prefs.requestGPUPowerLimit, 2)
        prefs.gpuPowerLimitWatts = 12
        XCTAssertEqual(prefs.requestGPUPowerLimit, 12)
        prefs.gpuPowerLimitEnabled = false
        XCTAssertNil(prefs.requestGPUPowerLimit)
    }

    func testPrefsDecodeKeepsGoodFieldsWhenOthersAreMalformed() throws {
        let json = #"""
        {"enabled":true,"hotKey":{"keyCode":49},"panelPosition":"left","gpuPowerLimitWatts":"ten",
         "presets":[{"id":"not-a-uuid","command":"tldr","model":"qwen"},{"command":7},
                    {"command":"번역","model":"gemma"}]}
        """#
        let prefs = try JSONDecoder().decode(AskAIPrefs.self, from: Data(json.utf8))
        XCTAssertTrue(prefs.enabled)
        XCTAssertEqual(prefs.hotKey, .optionSpace)
        XCTAssertNil(prefs.panelPosition)
        XCTAssertEqual(prefs.gpuPowerLimitWatts, 10)
        XCTAssertEqual(prefs.presets.map(\.command), ["tldr", "", "번역"])
    }

    @MainActor
    func testStoreBacksUpUnreadableFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AskAITests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("ask-ai.json")
        try Data("{broken".utf8).write(to: url)

        let store = AskAIStore(url: url)
        XCTAssertFalse(store.prefs.enabled)
        XCTAssertEqual(try Data(contentsOf: url.appendingPathExtension("bak")), Data("{broken".utf8))
    }

    func testHotKeyRejectsEditingShortcuts() {
        func combo(_ flags: NSEvent.ModifierFlags, _ key: Int, _ chars: String) -> HotKeyCombo? {
            let event = NSEvent.keyEvent(
                with: .keyDown, location: .zero, modifierFlags: flags, timestamp: 0,
                windowNumber: 0, context: nil, characters: chars,
                charactersIgnoringModifiers: chars, isARepeat: false, keyCode: UInt16(key)
            )
            return event.flatMap(HotKeyCombo.init(event:))
        }
        XCTAssertNil(combo(.command, kVK_ANSI_V, "v"))
        XCTAssertNil(combo([.command, .shift], kVK_ANSI_Z, "z"))
        XCTAssertNil(combo([], kVK_ANSI_A, "a"))
        XCTAssertEqual(combo(.command, kVK_Space, " ")?.displayString, "⌘Space")
        XCTAssertEqual(combo([.option, .command], kVK_ANSI_K, "k")?.displayString, "⌥⌘K")
        XCTAssertEqual(combo([], kVK_F5, "")?.displayString, "F5")
    }

    func testPromptPreviewSkipsBlankLinesAndCaps() {
        let long = String(repeating: "x", count: 500)
        let preset = AskAIPreset(systemPrompt: "\n\n  # Role\n" + long)
        XCTAssertEqual(preset.promptPreview, "# Role")
        XCTAssertEqual(AskAIPreset(systemPrompt: long + "\nmore").promptPreview.count, 120)
    }

    func testPresetValidation() {
        let existing = AskAIPreset(command: "tldr", model: "m")
        func check(_ command: String, model: String = "m") -> AskAIPresetValidation? {
            AskAIPresetValidation.check(AskAIPreset(command: command, model: model), among: [existing])
        }
        XCTAssertNil(check("gemma-translate_2"))
        XCTAssertNil(check("번역"))
        XCTAssertNil(check("翻訳"))
        XCTAssertEqual(check(""), .emptyCommand)
        XCTAssertEqual(check("Bad"), .invalidCommand)
        XCTAssertEqual(check("-x"), .invalidCommand)
        XCTAssertEqual(check("a b"), .invalidCommand)
        XCTAssertEqual(check("tldr"), .duplicateCommand)
        XCTAssertEqual(check("new", model: ""), .missingModel)
    }
}
