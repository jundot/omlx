// Owns Ask AI voice mode at runtime: its hotkey, the voice panel, and one spoken
// conversation over /v1/realtime. The app decides when a turn ends: speech
// followed by silence commits the turn, and speech over the reply stops the
// reply and trims it to what was heard.

import AppKit
import AVFoundation
import Foundation
import Observation

@MainActor
@Observable
final class TalkAIController {
    enum Phase: Equatable {
        case idle
        case connecting
        case listening
        case userSpeaking
        case thinking
        case speaking
        case failed(String)
    }

    /// What the panel offers next to a failure.
    enum FailureAction: Equatable {
        case none
        case microphoneSettings
        case appSettings
    }

    /// Microphone frames kept before speech starts, so its first syllable is sent.
    static let preRollFrames = 15  // 300 ms
    /// A turn this long ends on its own; the server holds at most 120 s per turn.
    static let maxTurnFrames = 3000  // 60 s
    /// Retries while the server still closes the previous conversation's session.
    static let maxConnectAttempts = 10

    let store: AskAIStore
    private(set) var phase: Phase = .idle
    private(set) var failureAction: FailureAction = .none
    private(set) var isMuted = false
    /// The part of the current reply's transcript that has been spoken.
    private(set) var transcript = ""
    /// 0...1 for the orb: the microphone while listening, the reply while speaking.
    private(set) var level = 0.0
    private(set) var hotKeyError: String?
    /// True once a reply of this conversation can be copied.
    private(set) var canCopyReplies = false

    /// Hides the Ask AI bar before the card takes its place.
    @ObservationIgnored var willOpen: (() -> Void)?
    /// Shows the Ask AI bar in place of the card.
    @ObservationIgnored var openText: (() -> Void)?
    @ObservationIgnored private let hotKey = GlobalHotKey()
    @ObservationIgnored private var panel: TalkAIPanelController?
    @ObservationIgnored private weak var services: AppServices?
    @ObservationIgnored private var openSettingsAction: (() -> Void)?
    @ObservationIgnored private var hotKeySuspended = false
    @ObservationIgnored private var client: RealtimeVoiceClient?
    @ObservationIgnored private var audio: VoiceAudioEngine?
    @ObservationIgnored private var detector = VoiceActivityDetector()
    @ObservationIgnored private var preRoll: [Data] = []
    @ObservationIgnored private var pendingAudio = Data()
    @ObservationIgnored private var micLevel = -120.0
    /// Set from commit until the reply's response.done arrives.
    @ObservationIgnored private var awaitingReply = false
    /// The reply being played; events of other items belong to a stopped reply.
    @ObservationIgnored private var replyItemID: String?
    /// Set when the user talks over a reply that the server has not announced yet.
    @ObservationIgnored private var dropNextReply = false
    /// Full text of the current reply; `transcript` shows the spoken part.
    @ObservationIgnored private var replyText = ""
    @ObservationIgnored private var replyDone = false
    /// Speech time per character, measured on finished replies.
    @ObservationIgnored private var secondsPerChar: Double?
    /// Replies of this conversation as heard; the user's speech has no text.
    @ObservationIgnored private var replies: [String] = []
    @ObservationIgnored private var turnFrames = 0
    @ObservationIgnored private var connectAttempts = 0
    @ObservationIgnored private var meterTimer: Timer?
    /// Start buffer each model settled on, so the next conversation begins with it.
    @ObservationIgnored private var startBuffers: [String: AVAudioFrameCount] = [:]
    @ObservationIgnored private var sessionModel = ""

    init(store: AskAIStore) {
        self.store = store
    }

    func start(services: AppServices, openSettings: @escaping () -> Void) {
        self.services = services
        openSettingsAction = openSettings
        applyHotKey()
    }

    func shutdown() {
        endConversation()
        hotKey.invalidate()
    }

    func prefsChanged(old: TalkPrefs, new: TalkPrefs) {
        if old.enabled != new.enabled || old.hotKey != new.hotKey {
            applyHotKey()
        }
        if !new.enabled {
            endConversation()
        }
        audio?.speechSpeed = new.speechSpeed
    }

    func setHotKeySuspended(_ suspended: Bool) {
        hotKeySuspended = suspended
        applyHotKey()
    }

    private func applyHotKey() {
        let prefs = store.prefs.talk
        guard services != nil, prefs.enabled, !hotKeySuspended else {
            hotKey.unregister()
            if !prefs.enabled { hotKeyError = nil }
            return
        }
        let combo = prefs.hotKey
        let status = hotKey.register(combo) { [weak self] in
            self?.toggle()
        }
        hotKeyError = status == noErr ? nil : String(
            localized: "talkai.hotkey.error.in_use",
            defaultValue: "\(combo.displayString) is already used by macOS, Ask AI, or another app. Choose a different shortcut.",
            comment: "Shown when the Talk with AI hotkey cannot be registered; the argument is the shortcut"
        )
    }

    // MARK: - Conversation

    var isActive: Bool { phase != .idle }

    func toggle() {
        if isActive {
            endConversation()
        } else {
            open()
        }
    }

    func open() {
        guard store.prefs.talk.enabled, !isActive else { return }
        willOpen?()
        beginConversation()
    }

    func switchToText() {
        endConversation(animated: true)
        openText?()
    }

    private func beginConversation() {
        let panel = self.panel ?? TalkAIPanelController(controller: self)
        self.panel = panel
        resetCaption()
        replies = []
        canCopyReplies = false
        connectAttempts = 0
        level = 0
        panel.show()
        let prefs = store.prefs.talk
        guard !prefs.model.isEmpty else {
            fail(TalkAIStrings.noModel, action: .appSettings)
            return
        }
        guard MicrophoneAccess.hasInputDevice else {
            fail(TalkAIStrings.noMicrophone, action: .none)
            return
        }
        switch MicrophoneAccess.state {
        case .authorized:
            connect(prefs)
        case .notDetermined:
            phase = .connecting
            Task {
                let granted = await MicrophoneAccess.request()
                // The user may have ended or restarted the conversation during the prompt.
                guard self.phase == .connecting, self.client == nil else { return }
                if granted {
                    self.connect(prefs)
                } else {
                    self.fail(TalkAIStrings.microphoneDenied, action: .microphoneSettings)
                }
            }
        case .denied:
            fail(TalkAIStrings.microphoneDenied, action: .microphoneSettings)
        }
    }

    func endConversation(animated: Bool = false) {
        teardown()
        phase = .idle
        panel?.hide(animated: animated)
    }

    func toggleMute() {
        isMuted.toggle()
        audio?.isMuted = isMuted
        if isMuted, detector.isSpeaking {
            detector = VoiceActivityDetector()
            pendingAudio = Data()
            client?.send(["type": "input_audio_buffer.clear"])
            if phase == .userSpeaking { phase = .listening }
        }
    }

    /// Ends the user's turn now, for when ongoing sound keeps it open.
    func finishTurn() {
        guard phase == .userSpeaking else { return }
        detector.forceEnd()
        commitTurn()
    }

    func copyReplies() {
        var text = replies
        let current = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty, current != replies.last { text.append(current) }
        guard !text.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text.joined(separator: "\n\n"), forType: .string)
    }

    private func keepReply(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        replies.append(trimmed)
        canCopyReplies = true
    }

    /// Opens the settings that fix the failure and closes the panel, which would cover them.
    func performFailureAction() {
        let action = failureAction
        endConversation()
        switch action {
        case .microphoneSettings:
            MicrophoneAccess.openSystemSettings()
        case .appSettings:
            openSettingsAction?()
        case .none:
            break
        }
    }

    private func connect(_ prefs: TalkPrefs) {
        guard let config = services?.config,
              let url = AppConfig.webSocketURL(
                  host: config.host, port: config.port, path: "/v1/realtime",
                  query: [URLQueryItem(name: "model", value: prefs.model)]
              )
        else {
            fail(TalkAIStrings.disconnected, action: .none)
            return
        }
        phase = .connecting
        failureAction = .none
        let client = RealtimeVoiceClient(url: url, apiKey: config.apiKey)
        client.onEvent = { [weak self, weak client] event in
            guard let self, let client, self.client === client else { return }
            self.handle(event)
        }
        self.client = client
        client.connect()
    }

    private func handle(_ event: RealtimeVoiceEvent?) {
        guard let event else {
            if case .failed = phase { return }
            if isActive { fail(TalkAIStrings.disconnected, action: .none) }
            return
        }
        switch event {
        case .sessionCreated:
            connectAttempts = 0
            startListening()
        case .responseStarted(let itemID):
            guard dropNextReply else {
                replyItemID = itemID
                return
            }
            // Nothing of it was heard, so keep none of it in the conversation.
            dropNextReply = false
            truncate(itemID, heardMs: 0)
        case .audio(let itemID, let pcm):
            guard itemID == replyItemID else { return }
            audio?.enqueue(pcm)
            if phase == .thinking { phase = .speaking }
        case .transcript(let itemID, let delta):
            guard itemID == replyItemID else { return }
            replyText += delta
            updateCaption()
        case .responseDone(let itemID, _):
            guard itemID == replyItemID else { return }
            if let audio, audio.receivedSeconds > 0, !replyText.isEmpty {
                let measured = audio.receivedSeconds / Double(replyText.count)
                secondsPerChar = secondsPerChar.map { ($0 + measured) / 2 } ?? measured
            }
            replyDone = true
            replyFinished()
            updateCaption()
        case .error(let code, _) where code == "session_limit" && phase == .connecting
            && connectAttempts < Self.maxConnectAttempts:
            // Reopened right after closing: the server may still be closing the old session.
            connectAttempts += 1
            client?.close()
            client = nil
            let prefs = store.prefs.talk
            Task {
                try? await Task.sleep(for: .milliseconds(300))
                guard self.phase == .connecting, self.client == nil else { return }
                self.connect(prefs)
            }
        case .error(let code, let message):
            // Turn-level refusals leave the conversation usable.
            if Self.turnErrors.contains(code) {
                dropNextReply = false
                replyFinished()
                return
            }
            fail(message.isEmpty ? code : message, action: .none)
        }
    }

    private func replyFinished() {
        awaitingReply = false
        audio?.finishReply()
        if phase == .thinking { phase = .listening }
    }

    private static let turnErrors: Set<String> = [
        "input_audio_buffer_commit_empty",
        "input_audio_buffer_empty",
        "conversation_already_has_active_response",
        "input_audio_buffer_too_large",
    ]

    private func startListening() {
        let prefs = store.prefs.talk
        var output: [String: Any] = [:]
        if !prefs.voice.isEmpty { output["voice"] = prefs.voice }
        var session: [String: Any] = [
            "type": "realtime",
            "audio": ["input": ["turn_detection": NSNull()], "output": output],
        ]
        let prompt = prefs.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !prompt.isEmpty { session["instructions"] = prompt }
        client?.send(["type": "session.update", "session": session])

        let audio = VoiceAudioEngine()
        audio.onFrame = { [weak self] frame, level in self?.handleFrame(frame, level: level) }
        do {
            try audio.start()
        } catch {
            NSLog("oMLX: Ask AI voice audio start failed - \(error)")
            audio.stop()
            fail("\(TalkAIStrings.audioFailed) (\((error as NSError).code))", action: .microphoneSettings)
            return
        }
        audio.isMuted = isMuted
        audio.speechSpeed = prefs.speechSpeed
        sessionModel = prefs.model
        if let learned = startBuffers[prefs.model] { audio.startBufferSamples = learned }
        self.audio = audio
        detector = VoiceActivityDetector()
        preRoll = []
        phase = .listening
        let timer = Timer(timeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    private func handleFrame(_ frame: Data, level: Double) {
        guard client != nil, audio != nil, !isMuted else { return }
        micLevel = level
        let replying = awaitingReply || audio?.isPlayingReply == true
        switch detector.process(level: level, assistantSpeaking: replying) {
        case .speechStarted:
            beginTurn(interrupting: false)
        case .interrupted:
            beginTurn(interrupting: true)
        case .speechEnded:
            commitTurn()
        case .discarded:
            pendingAudio = Data()
            client?.send(["type": "input_audio_buffer.clear"])
            if phase == .userSpeaking { phase = .listening }
        case nil:
            break
        }
        if detector.isSpeaking {
            append(frame)
            if turnFrames >= Self.maxTurnFrames { finishTurn() }
        } else {
            preRoll.append(frame)
            if preRoll.count > Self.preRollFrames { preRoll.removeFirst() }
        }
    }

    private func beginTurn(interrupting: Bool) {
        // Speech can start between the end of playback and the next tick.
        if !interrupting, phase == .speaking { finishPlayedReply() }
        turnFrames = 0
        if interrupting {
            // Keep the caption at what was heard.
            updateCaption()
            replyText = transcript
            if awaitingReply || audio?.isPlayingReply == true { keepReply(transcript) }
            let heardMs = audio?.stopPlayback() ?? 0
            if awaitingReply {
                client?.send(["type": "response.cancel"])
                awaitingReply = false
                dropNextReply = replyItemID == nil
            }
            if let replyItemID { truncate(replyItemID, heardMs: heardMs) }
        }
        replyItemID = nil
        for frame in preRoll { append(frame) }
        preRoll = []
        phase = .userSpeaking
    }

    private func truncate(_ itemID: String, heardMs: Int) {
        client?.send([
            "type": "conversation.item.truncate",
            "item_id": itemID,
            "content_index": 0,
            "audio_end_ms": heardMs,
        ])
    }

    private func append(_ frame: Data) {
        turnFrames += 1
        pendingAudio.append(frame)
        // Two 20 ms frames per event.
        if pendingAudio.count >= MicrophoneTap.frameSamples * 4 { flushAudio() }
    }

    private func flushAudio() {
        guard !pendingAudio.isEmpty else { return }
        client?.send(["type": "input_audio_buffer.append", "audio": pendingAudio.base64EncodedString()])
        pendingAudio = Data()
    }

    private func commitTurn() {
        flushAudio()
        client?.send(["type": "input_audio_buffer.commit"])
        client?.send(["type": "response.create"])
        awaitingReply = true
        resetCaption()
        phase = .thinking
    }

    private func tick() {
        switch phase {
        case .speaking:
            level = Self.normalized(audio?.outputLevel ?? -120)
            updateCaption()
            if !awaitingReply, audio?.isPlayingReply != true {
                finishPlayedReply()
                phase = .listening
            }
        case .listening, .userSpeaking:
            level = Self.normalized(micLevel)
        default:
            level = 0
        }
    }

    /// Ends a reply whose audio has played: keep its text, reset playback.
    private func finishPlayedReply() {
        audio?.stopPlayback()
        transcript = replyText
        keepReply(replyText)
    }

    private func resetCaption() {
        transcript = ""
        replyText = ""
        replyDone = false
    }

    private func updateCaption() {
        guard let audio, audio.isPlayingReply || audio.playedSeconds > 0 else {
            transcript = ""
            return
        }
        transcript = Self.caption(
            replyText, played: audio.playedSeconds, received: audio.receivedSeconds, done: replyDone,
            secondsPerChar: secondsPerChar ?? Self.defaultSecondsPerChar(replyText)
        )
    }

    /// The part of a reply's text that matches how much of its audio has
    /// played. Text and audio have no exact alignment, so the text is spread
    /// over the reply's length: the measured length once it is done, an
    /// estimate from the character count before that.
    nonisolated static func caption(_ text: String, played: Double, received: Double, done: Bool,
                        secondsPerChar: Double) -> String {
        let total = done ? received : max(received, Double(text.count) * secondsPerChar)
        guard !text.isEmpty, total > 0 else { return "" }
        let shown = Int((Double(text.count) * min(played / total, 1)).rounded(.up))
        guard shown > 0 else { return "" }
        var end = text.index(text.startIndex, offsetBy: shown)
        // Finish the word being spoken; CJK and kana have no word breaks.
        while end < text.endIndex, text[end].isLetter || text[end].isNumber, !isUnspaced(text[end]) {
            end = text.index(after: end)
        }
        return String(text[..<end])
    }

    /// The last `limit` sentences of a caption, one per line. Model line
    /// breaks become spaces; a line ends at a sentence end that is not part
    /// of an ellipsis or a decimal number.
    nonisolated static func captionLines(_ text: String, limit: Int = 3) -> [String] {
        let chars = Array(text.map { $0.isNewline ? " " : $0 })
        var lines: [String] = []
        var line = ""
        var i = 0
        while i < chars.count {
            let c = chars[i]
            line.append(c)
            i += 1
            guard "。？！.?!".contains(c), c != "." || line.dropLast().last != "." else { continue }
            // Runs like "?!" and closing quotes stay on the sentence.
            while i < chars.count, "。？！?!\"'”’」』)]".contains(chars[i]) {
                line.append(chars[i])
                i += 1
            }
            let ascii = c == "." || c == "?" || c == "!"
            if i < chars.count, ascii, !chars[i].isWhitespace { continue }
            if i < chars.count, chars[i] == "." { continue }
            lines.append(line)
            line = ""
        }
        lines.append(line)
        return Array(
            lines.map { $0.split(whereSeparator: \.isWhitespace).joined(separator: " ") }
                .filter { !$0.isEmpty }
                .suffix(limit)
        )
    }

    /// Hangul, CJK, and kana take about a syllable per character.
    nonisolated static func defaultSecondsPerChar(_ text: String) -> Double {
        text.contains { isUnspaced($0) || isHangul($0) } ? 0.17 : 0.06
    }

    nonisolated private static func isHangul(_ c: Character) -> Bool {
        c.unicodeScalars.contains { (0xAC00...0xD7AF).contains($0.value) || (0x1100...0x11FF).contains($0.value) }
    }

    nonisolated private static func isUnspaced(_ c: Character) -> Bool {
        c.unicodeScalars.contains { (0x3040...0x30FF).contains($0.value) || (0x3400...0x9FFF).contains($0.value) }
    }

    /// dBFS to 0...1 over the speech range.
    static func normalized(_ db: Double) -> Double {
        min(max((db + 60) / 50, 0), 1)
    }

    private func fail(_ message: String, action: FailureAction) {
        teardown()
        failureAction = action
        phase = .failed(message)
    }

    private func teardown() {
        meterTimer?.invalidate()
        meterTimer = nil
        client?.close()
        client = nil
        if let audio { startBuffers[sessionModel] = audio.startBufferSamples }
        audio?.stop()
        audio = nil
        awaitingReply = false
        replyItemID = nil
        dropNextReply = false
        pendingAudio = Data()
        preRoll = []
        level = 0
    }
}

enum TalkAIStrings {
    static var noModel: String {
        String(localized: "talkai.error.no_model",
               defaultValue: "Choose a voice model in Ask AI settings.",
               comment: "Shown in the Talk with AI panel when no model is selected")
    }

    static var noMicrophone: String {
        String(localized: "talkai.error.no_microphone",
               defaultValue: "No microphone found. Connect a microphone or headset and try again.",
               comment: "Shown in the Talk with AI panel when the Mac has no audio input device")
    }

    static var microphoneDenied: String {
        String(localized: "talkai.error.microphone",
               defaultValue: "oMLX can't use the microphone. Turn it on in System Settings.",
               comment: "Shown in the Talk with AI panel when microphone access is denied")
    }

    static var audioFailed: String {
        String(localized: "talkai.error.audio",
               defaultValue: "Couldn't start the microphone or speakers.",
               comment: "Shown in the Talk with AI panel when the audio engine fails to start")
    }

    static var disconnected: String {
        String(localized: "talkai.error.disconnected",
               defaultValue: "Lost the connection to the server. Make sure it is running.",
               comment: "Shown in the Talk with AI panel when the realtime connection closes")
    }
}
