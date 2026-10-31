// Ask AI pure logic: SSE line parsing, slash-command parsing, request body
// shape, tolerant preference decoding, and Ask AI voice turn detection and
// realtime event decoding.

@preconcurrency import AVFoundation
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
        XCTAssertEqual(
            try ChatStreamLine.parse(#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c1","type":"function","function":{"name":"web_search","arguments":"{\"query\":"}}]}}]}"#),
            .toolCalls([ChatToolCallDelta(index: 1, id: "c1", name: "web_search", arguments: #"{"query":"#)])
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
        XCTAssertNil(body["tools"])

        let call = ChatToolCall(id: "c1", name: "web_search", arguments: #"{"query":"mlx"}"#)
        let toolBody = try JSONSerialization.jsonObject(with: ChatCompletionBody(
            model: "m",
            messages: [ChatMessage(role: .assistant, content: "", toolCalls: [call]),
                       ChatMessage(role: .tool, content: #"{"ok":true}"#, toolCallID: "c1")],
            allowThinking: false, tools: [AskAITool.webSearch]
        ).encoded()) as! [String: Any]
        let sent = toolBody["messages"] as! [[String: Any]]
        let sentCall = (sent[0]["tool_calls"] as! [[String: Any]])[0]
        XCTAssertEqual(sentCall["id"] as? String, "c1")
        XCTAssertEqual(sentCall["type"] as? String, "function")
        XCTAssertEqual((sentCall["function"] as? [String: String])?["arguments"], #"{"query":"mlx"}"#)
        XCTAssertEqual(sent[1]["tool_call_id"] as? String, "c1")
        let function = (toolBody["tools"] as! [[String: Any]])[0]["function"] as! [String: Any]
        let parameters = function["parameters"] as! [String: Any]
        XCTAssertEqual(function["name"] as? String, "web_search")
        XCTAssertEqual(parameters["required"] as? [String], ["query"])
    }

    @MainActor
    func testConversationRunsToolCallsAndKeepsThemForFollowUps() async throws {
        let chat = ScriptedChat([
            [.content("Let me check."),
             .toolCall(ChatToolCallDelta(index: 0, id: "c1", name: "web_search", arguments: #"{"query":"#)),
             .toolCall(ChatToolCallDelta(index: 0, arguments: #""mlx"}"#))],
            [.content("Answer.")],
            [.content("Follow-up answer.")],
        ])
        let profile = AskAIProfile(model: "m", systemPrompt: "", allowThinking: false, autoCopy: false)
        let conversation = AskAIConversation()
        conversation.begin(profile: profile, command: nil)
        XCTAssertTrue(conversation.send("q1", tools: [AskAITool.webSearch], using: chat, runner: StubToolRunner()))
        try await waitUntilDone(conversation)

        let turn = try XCTUnwrap(conversation.turns.first)
        XCTAssertNil(turn.error)
        XCTAssertEqual(turn.activities, [AskAIToolActivity(kind: .search, detail: "mlx", isRunning: false)])
        XCTAssertEqual(turn.answer, "Let me check.\n\nAnswer.")
        XCTAssertEqual(turn.reply, "Answer.")
        let second = chat.bodies[1]["messages"] as! [[String: Any]]
        XCTAssertEqual(second.map { $0["role"] as? String }, ["user", "assistant", "tool"])
        XCTAssertEqual(second[1]["content"] as? String, "Let me check.")
        XCTAssertEqual(second[2]["tool_call_id"] as? String, "c1")

        // The follow-up resends the calls and results, then the final reply.
        XCTAssertTrue(conversation.send("q2", tools: [AskAITool.webSearch], using: chat, runner: StubToolRunner()))
        try await waitUntilDone(conversation)
        let third = chat.bodies[2]["messages"] as! [[String: Any]]
        XCTAssertEqual(third.map { $0["role"] as? String }, ["user", "assistant", "tool", "assistant", "user"])
        XCTAssertEqual(third[3]["content"] as? String, "Answer.")
    }

    @MainActor
    func testToolRoundsEndWithARoundWithoutTools() async throws {
        let chat = ScriptedChat([
            [.toolCall(ChatToolCallDelta(index: 0, id: "c", name: "web_search", arguments: "{}"))],
        ])
        let conversation = AskAIConversation()
        conversation.begin(profile: AskAIProfile(model: "m", systemPrompt: "", allowThinking: false,
                                                 autoCopy: false), command: nil)
        conversation.send("q", tools: [AskAITool.webSearch], using: chat, runner: StubToolRunner())
        try await waitUntilDone(conversation)
        XCTAssertEqual(chat.bodies.count, AskAIConversation.maxToolRounds + 1)
        XCTAssertNotNil(chat.bodies[AskAIConversation.maxToolRounds - 1]["tools"])
        XCTAssertNil(chat.bodies[AskAIConversation.maxToolRounds]["tools"])
    }

    func testToolPrefsDecodeAndScreenCaptureNeedsVision() throws {
        let json = #"{"defaultTools":{"enabled":true,"screenCapture":true,"readLinks":"no"}}"#
        let prefs = try JSONDecoder().decode(AskAIPrefs.self, from: Data(json.utf8))
        XCTAssertEqual(prefs.defaultProfile.tools,
                       { var t = AskAIToolPrefs(); t.enabled = true; t.screenCapture = true; return t }())
        XCTAssertEqual(AskAITool.tools(for: prefs.defaultTools, vision: false).map(\.name),
                       ["web_search", "fetch_url"])
        XCTAssertEqual(AskAITool.tools(for: prefs.defaultTools, vision: true).map(\.name),
                       ["web_search", "fetch_url", "capture_screen"])
        XCTAssertEqual(AskAITool.tools(for: AskAIToolPrefs(), vision: true), [])
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
        XCTAssertFalse(prefs.sendOnEnter)
        XCTAssertNil(prefs.requestGPUPowerLimit)
        XCTAssertEqual(prefs.talk, TalkPrefs())
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
         "sendOnEnter":true,
         "presets":[{"id":"not-a-uuid","command":"tldr","model":"qwen"},{"command":7},
                    {"command":"번역","model":"gemma"}],
         "talk":{"enabled":true,"hotKey":7,"model":"lfm3","voice":["ethan"],"speechSpeed":9}}
        """#
        let prefs = try JSONDecoder().decode(AskAIPrefs.self, from: Data(json.utf8))
        XCTAssertTrue(prefs.enabled)
        XCTAssertEqual(prefs.hotKey, .optionSpace)
        XCTAssertNil(prefs.panelPosition)
        XCTAssertEqual(prefs.gpuPowerLimitWatts, 10)
        XCTAssertTrue(prefs.sendOnEnter)
        XCTAssertEqual(prefs.presets.map(\.command), ["tldr", "", "번역"])
        XCTAssertTrue(prefs.talk.enabled)
        XCTAssertEqual(prefs.talk.hotKey, .optionShiftSpace)
        XCTAssertEqual(prefs.talk.model, "lfm3")
        XCTAssertEqual(prefs.talk.voice, "")
        XCTAssertEqual(prefs.talk.speechSpeed, 2.0)
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

    func testVoiceActivityDetectorTurns() {
        func feed(_ vad: inout VoiceActivityDetector, _ level: Double, frames: Int,
                  replying: Bool = false) -> [VoiceActivityDetector.Event] {
            (0..<frames).compactMap { _ in vad.process(level: level, assistantSpeaking: replying) }
        }
        var vad = VoiceActivityDetector()
        XCTAssertEqual(feed(&vad, -70, frames: 200), [])
        // A 100 ms click starts speech but is too short to be a turn.
        XCTAssertEqual(feed(&vad, -30, frames: 5), [.speechStarted])
        XCTAssertEqual(feed(&vad, -70, frames: 40), [.discarded])
        // 400 ms of speech ends the turn after the end silence.
        let silenceFrames = VoiceActivityDetector.endSilenceMs / VoiceActivityDetector.frameMs
        XCTAssertEqual(feed(&vad, -30, frames: 20), [.speechStarted])
        XCTAssertEqual(feed(&vad, -70, frames: silenceFrames - 1), [])
        XCTAssertEqual(feed(&vad, -70, frames: 1), [.speechEnded])

        // Over the reply, a level that starts a turn in silence does not interrupt.
        var noisy = VoiceActivityDetector()
        _ = feed(&noisy, -50, frames: 200)
        var silent = noisy
        XCTAssertEqual(feed(&silent, -36, frames: 3), [.speechStarted])
        XCTAssertEqual(feed(&noisy, -36, frames: 15, replying: true), [])
        XCTAssertEqual(feed(&noisy, -20, frames: 9, replying: true), [])
        XCTAssertEqual(feed(&noisy, -20, frames: 1, replying: true), [.interrupted])
    }

    func testRealtimeEventDecoding() {
        func decode(_ json: String) -> RealtimeVoiceEvent? { RealtimeVoiceEvent.decode(json) }
        XCTAssertEqual(decode(#"{"type":"session.created","session":{}}"#),
                       .sessionCreated(instructions: "", capabilities: RealtimeVoiceCapabilities()))
        XCTAssertEqual(
            decode(#"{"type":"session.created","session":{"instructions":"Speak briefly.","capabilities":{"input_image":true,"tools":true}}}"#),
            .sessionCreated(instructions: "Speak briefly.",
                            capabilities: RealtimeVoiceCapabilities(inputImage: true, tools: true))
        )
        XCTAssertEqual(
            decode(#"{"type":"response.function_call_arguments.done","response_id":"resp_1","call_id":"call_1","name":"web_search","arguments":"{\"query\":\"mlx\"}"}"#),
            .functionCall(callID: "call_1", name: "web_search", arguments: #"{"query":"mlx"}"#, responseID: "resp_1")
        )
        // The function call item itself is not a reply to play.
        XCTAssertNil(decode(#"{"type":"response.output_item.added","item":{"id":"item_2","type":"function_call"}}"#))
        XCTAssertEqual(decode(#"{"type":"response.output_item.added","response_id":"resp_1","item":{"id":"item_1"}}"#),
                       .responseStarted(itemID: "item_1", responseID: "resp_1"))
        XCTAssertEqual(decode(#"{"type":"response.output_audio.delta","item_id":"item_1","delta":"AAEC"}"#),
                       .audio(itemID: "item_1", pcm: Data([0, 1, 2])))
        XCTAssertEqual(decode(#"{"type":"response.output_audio_transcript.delta","item_id":"item_1","delta":"Hi"}"#),
                       .transcript(itemID: "item_1", text: "Hi"))
        XCTAssertEqual(decode(#"{"type":"response.done","response":{"status":"cancelled","output":[{"id":"item_1"}]}}"#),
                       .responseDone(itemID: "item_1", status: "cancelled"))
        XCTAssertEqual(decode(#"{"type":"error","error":{"code":"session_limit","message":"busy"}}"#),
                       .error(code: "session_limit", message: "busy"))
        XCTAssertNil(decode(#"{"type":"response.created","response":{}}"#))
        XCTAssertNil(decode("not json"))
    }

    func testMicrophoneTapBlockRunsOnAnAudioThread() throws {
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1))
        let frames = expectation(description: "20 ms frames at 24 kHz")
        frames.expectedFulfillmentCount = 5
        frames.assertForOverFulfill = false
        let tap = try XCTUnwrap(MicrophoneTap(input: format) { data, _ in
            if data.count == MicrophoneTap.frameSamples * 2 { frames.fulfill() }
        })
        // 200 ms in; resampler latency holds back part of the first call.
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 9600))
        buffer.frameLength = 9600
        nonisolated(unsafe) let block = VoiceAudioEngine.tapBlock(for: tap)
        DispatchQueue.global().async { block(buffer, AVAudioTime(sampleTime: 0, atRate: 48000)) }
        wait(for: [frames], timeout: 5)
    }

    func testReplyPacingSizesTheNextStartBuffer() {
        XCTAssertNil(ReplyPacing().startBuffer(sampleRate: 24000, speed: 1))
        // Faster than real time keeps the shortest start.
        var fast = ReplyPacing()
        for step in 0..<10 { fast.add(seconds: 0.08, at: Double(step) * 0.04, speed: 1) }
        XCTAssertEqual(fast.startBuffer(sampleRate: 24000, speed: 1), ReplyPacing.minimumStartBuffer)
        // Each 0.5 s chunk lands 0.25 s after the audio before it ran out.
        var slow = ReplyPacing()
        slow.add(seconds: 0.25, at: 0, speed: 1)
        for step in 1...3 { slow.add(seconds: 0.5, at: Double(step) * 0.5, speed: 1) }
        XCTAssertEqual(slow.startBuffer(sampleRate: 24000, speed: 1), 7500)
        // Faster playback needs more lead, bounded by the cap.
        XCTAssertEqual(slow.startBuffer(sampleRate: 24000, speed: 4), ReplyPacing.maximumStartBuffer)
    }

    func testCaptionFollowsPlayback() {
        func caption(_ text: String, played: Double, received: Double, done: Bool = true) -> String {
            TalkAIController.caption(text, played: played, received: received, done: done,
                                     secondsPerChar: TalkAIController.defaultSecondsPerChar(text))
        }
        let text = "The capital of France is Paris."
        XCTAssertEqual(caption(text, played: 0, received: 3), "")
        // A third of the audio shows a third of the text, finishing the word.
        XCTAssertEqual(caption(text, played: 1, received: 3), "The capital")
        XCTAssertEqual(caption(text, played: 3, received: 3), text)
        // Before the reply is done, text that arrived early waits for its estimated speech time.
        XCTAssertEqual(caption(text, played: 0.3, received: 0.5, done: false), "The capital")
        XCTAssertEqual(caption("프랑스의 수도는 파리입니다", played: 1.5, received: 4), "프랑스의 수도는")
        XCTAssertEqual(caption("フランスの首都はパリです", played: 1, received: 4), "フラン")
    }

    func testCaptionLinesBreakAtSentenceEnds() {
        func lines(_ text: String) -> [String] { TalkAIController.captionLines(text) }
        XCTAssertEqual(lines("Hi there.\nHow are you? I'm fine"), ["Hi there.", "How are you?", "I'm fine"])
        XCTAssertEqual(lines("Well... it costs 3.5 dollars."), ["Well... it costs 3.5 dollars."])
        XCTAssertEqual(lines("One. Two. Three. Four"), ["Two.", "Three.", "Four"])
        XCTAssertEqual(lines("He said \"Wow!\" Then left."), ["He said \"Wow!\"", "Then left."])
        XCTAssertEqual(lines("안녕하세요.반가워요"), ["안녕하세요.반가워요"])
        XCTAssertEqual(lines("こんにちは。元気？うん"), ["こんにちは。", "元気？", "うん"])
    }

    func testForcedTurnEndTakesTheSoundAsBackground() {
        var vad = VoiceActivityDetector()
        for _ in 0..<200 { _ = vad.process(level: -70, assistantSpeaking: false) }
        // A fan starts while the user talks, so the turn never ends on its own.
        var events: [VoiceActivityDetector.Event] = []
        for _ in 0..<100 { vad.process(level: -35, assistantSpeaking: false).map { events.append($0) } }
        XCTAssertEqual(events, [.speechStarted])
        vad.forceEnd()
        XCTAssertFalse(vad.isSpeaking)
        // The same fan over the reply does not interrupt it.
        for _ in 0..<100 { XCTAssertNil(vad.process(level: -35, assistantSpeaking: true)) }

        // Space pressed by mistake mid-sentence: the pauses between words keep the floor low.
        var talking = VoiceActivityDetector()
        for _ in 0..<200 { _ = talking.process(level: -70, assistantSpeaking: false) }
        for frame in 0..<100 { _ = talking.process(level: frame % 25 < 15 ? -25 : -70, assistantSpeaking: false) }
        XCTAssertTrue(talking.isSpeaking)
        talking.forceEnd()
        XCTAssertLessThan(talking.noiseFloor, -60)
        let next = (0..<3).compactMap { _ in talking.process(level: -30, assistantSpeaking: false) }
        XCTAssertEqual(next, [.speechStarted])
    }

    func testReplyEchoDoesNotRaiseTheNoiseFloor() {
        var vad = VoiceActivityDetector()
        for _ in 0..<200 { _ = vad.process(level: -70, assistantSpeaking: false) }
        // Five seconds of leftover echo under the interrupt threshold.
        for _ in 0..<250 { XCTAssertNil(vad.process(level: -50, assistantSpeaking: true)) }
        XCTAssertLessThan(vad.noiseFloor, -65)
        // Quiet speech right after the reply still opens a turn.
        let next = (0..<3).compactMap { _ in vad.process(level: -42, assistantSpeaking: false) }
        XCTAssertEqual(next, [.speechStarted])
    }

    func testPromptPlaceholdersFill() throws {
        var reads = 0
        var context = PromptContext()
        context.now = try XCTUnwrap(ISO8601DateFormatter().date(from: "2026-10-30T05:05:09Z"))
        context.timeZone = try XCTUnwrap(TimeZone(identifier: "Asia/Seoul"))
        context.language = "ko-KR"
        context.app = "Safari"
        context.model = "qwen"
        context.readClipboard = { reads += 1; return String(repeating: "x", count: 5000) }

        let filled = PromptPlaceholders.fill(
            #"{weekday}, {datetime} ({timezone}) {date} {time} {language} {app} {model} {"a": 1} {unknown}"#,
            context: context)
        XCTAssertEqual(filled, #"Friday, 2026-10-30 14:05:09 (Asia/Seoul (UTC+09:00)) 2026-10-30 14:05:09 Korean (ko-KR) Safari qwen {"a": 1} {unknown}"#)
        XCTAssertEqual(reads, 0)
        XCTAssertEqual(PromptPlaceholders.fill("{clipboard}", context: context).count, PromptPlaceholders.clipboardLimit)
        XCTAssertEqual(reads, 1)
    }

    @MainActor
    func testConversationFillsThePromptOnce() throws {
        var first = PromptContext()
        first.now = Date(timeIntervalSince1970: 0)
        first.timeZone = try XCTUnwrap(TimeZone(identifier: "UTC"))
        var later = first
        later.now = Date(timeIntervalSince1970: 3600)
        let profile = AskAIProfile(model: "m", systemPrompt: "Now {time}", allowThinking: false, autoCopy: false)

        let conversation = AskAIConversation()
        conversation.begin(profile: profile, command: nil, context: first)
        XCTAssertEqual(conversation.profile?.systemPrompt, "Now 00:00:00")
        // A follow-up keeps the prompt text, so the server's prefix cache matches.
        conversation.updateProfile(profile, context: later)
        XCTAssertEqual(conversation.profile?.systemPrompt, "Now 00:00:00")
        var edited = profile
        edited.systemPrompt = "At {time}"
        conversation.updateProfile(edited, context: later)
        XCTAssertEqual(conversation.profile?.systemPrompt, "At 01:00:00")
    }

    func testVoiceInstructionsPutTheModelPromptFirst() throws {
        let json = #"""
        {"model":"omni","systemPrompt":"Today is {weekday}.","modelPrompts":{"lfm":"Be calm."},
         "gpuPowerLimitEnabled":true,"gpuPowerLimitWatts":1,"tools":{"enabled":true}}
        """#
        var talk = try JSONDecoder().decode(TalkPrefs.self, from: Data(json.utf8))
        var context = PromptContext()
        context.now = Date(timeIntervalSince1970: 0)
        context.timeZone = try XCTUnwrap(TimeZone(identifier: "UTC"))
        XCTAssertEqual(talk.instructions(modelDefault: "Speak briefly.", context: context),
                       "Speak briefly.\n\nToday is Thursday.")
        // An edited model prompt replaces the server's default for that model only.
        talk.model = "lfm"
        XCTAssertEqual(talk.instructions(modelDefault: "", context: context), "Be calm.\n\nToday is Thursday.")
        talk.modelPrompts["lfm"] = ""
        talk.systemPrompt = ""
        XCTAssertEqual(talk.instructions(modelDefault: "Speak briefly.", context: context), "")
        XCTAssertEqual(talk.requestGPUPowerLimit, 2)
        XCTAssertTrue(talk.tools.enabled)

        let tools = AskAITool.realtimeTools([AskAITool.webSearch, AskAITool.captureScreen])
        XCTAssertEqual(tools.map { $0["name"] as? String }, ["web_search", "capture_screen"])
        let search = try XCTUnwrap(tools[0]["parameters"] as? [String: Any])
        XCTAssertEqual(search["required"] as? [String], ["query"])
        XCTAssertEqual((tools[1]["parameters"] as? [String: Any])?["required"] as? [String], [])
    }

    func testEmptyPromptsGetTheDefaultOnce() throws {
        let old = #"{"defaultSystemPrompt":"","talk":{"systemPrompt":""}}"#
        let migrated = try JSONDecoder().decode(AskAIPrefs.self, from: Data(old.utf8))
        XCTAssertEqual(migrated.defaultSystemPrompt, AskAIPrefs.defaultPrompt)
        XCTAssertEqual(migrated.talk.systemPrompt, TalkPrefs.defaultPrompt)
        // Cleared after the default arrived: stays empty.
        let cleared = #"{"defaultSystemPrompt":"","talk":{"systemPrompt":""},"promptDefaultsApplied":true}"#
        let kept = try JSONDecoder().decode(AskAIPrefs.self, from: Data(cleared.utf8))
        XCTAssertEqual(kept.defaultSystemPrompt, "")
        XCTAssertEqual(kept.talk.systemPrompt, "")
    }

    @MainActor
    private func waitUntilDone(_ conversation: AskAIConversation) async throws {
        let deadline = ContinuousClock.now + .seconds(5)
        while conversation.isStreaming, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertFalse(conversation.isStreaming)
    }
}

/// Replies with the next scripted events per request; the last script repeats.
private final class ScriptedChat: ChatStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var scripts: [[ChatStreamEvent]]
    private var sent: [[String: Any]] = []

    init(_ scripts: [[ChatStreamEvent]]) {
        self.scripts = scripts
    }

    var bodies: [[String: Any]] { lock.withLock { sent } }

    func stream(_ body: ChatCompletionBody) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        let json = (try? body.encoded()).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let events = lock.withLock {
            sent.append(json ?? [:])
            return scripts.count > 1 ? scripts.removeFirst() : scripts[0]
        }
        return AsyncThrowingStream { continuation in
            events.forEach { continuation.yield($0) }
            continuation.finish()
        }
    }
}

private struct StubToolRunner: AskAIToolRunning {
    func run(_ call: ChatToolCall) async -> AskAIToolResult {
        AskAIToolResult(text: #"{"ok":true,"results":[]}"#)
    }
}
