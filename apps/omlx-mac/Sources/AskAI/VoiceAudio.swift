// Microphone capture and reply playback for Ask AI voice mode. One AVAudioEngine
// runs both with voice processing on, so the model's own voice is removed
// from the microphone signal and the user can talk over it.

@preconcurrency import AVFoundation
import Foundation

/// Finds where the user starts and stops talking from 20 ms frame levels.
struct VoiceActivityDetector: Equatable {
    enum Event: Equatable {
        /// Speech began while the assistant was quiet.
        case speechStarted
        /// Speech began over the assistant's reply.
        case interrupted
        /// The turn ended after enough speech.
        case speechEnded
        /// The sound was too short to be a turn.
        case discarded
    }

    static let frameMs = 20
    static let startMs = 60
    static let interruptMs = 200
    static let endSilenceMs = 500
    static let minSpeechMs = 300
    static let startMargin = 12.0
    /// A stricter margin over the reply, so leftover echo does not interrupt it.
    static let interruptMargin = 18.0
    static let minimumThreshold = -45.0
    /// A forced end looks for the background in this much recent sound.
    static let backgroundWindowFrames = 75  // 1.5 s
    /// A forced end never raises the floor past this, so speech still opens turns.
    static let maxForcedFloor = -40.0

    private(set) var noiseFloor = -60.0
    private(set) var isSpeaking = false
    /// Briefly smoothed levels of the last frames, speech or not.
    private var smoothedLevel = -60.0
    private var recentLevels: [Double] = []
    private var aboveMs = 0
    private var silenceMs = 0
    private var speechMs = 0

    mutating func process(level: Double, assistantSpeaking: Bool) -> Event? {
        smoothedLevel = smoothedLevel * 0.7 + level * 0.3
        recentLevels.append(smoothedLevel)
        if recentLevels.count > Self.backgroundWindowFrames { recentLevels.removeFirst() }
        if isSpeaking {
            let threshold = max(noiseFloor + Self.startMargin, Self.minimumThreshold)
            speechMs += Self.frameMs
            silenceMs = level > threshold ? 0 : silenceMs + Self.frameMs
            guard silenceMs >= Self.endSilenceMs else { return nil }
            let spoken = speechMs - silenceMs
            isSpeaking = false
            speechMs = 0
            silenceMs = 0
            return spoken >= Self.minSpeechMs ? .speechEnded : .discarded
        }
        let margin = assistantSpeaking ? Self.interruptMargin : Self.startMargin
        guard level > max(noiseFloor + margin, Self.minimumThreshold) else {
            aboveMs = 0
            // Quiet frames follow the room's noise level. The reply's echo
            // is not room noise: following it would raise the floor and cut
            // the next turn into fragments.
            if !assistantSpeaking {
                noiseFloor = min(max(noiseFloor * 0.95 + level * 0.05, -90), -30)
            }
            return nil
        }
        aboveMs += Self.frameMs
        guard aboveMs >= (assistantSpeaking ? Self.interruptMs : Self.startMs) else { return nil }
        isSpeaking = true
        speechMs = aboveMs
        aboveMs = 0
        return assistantSpeaking ? .interrupted : .speechStarted
    }

    /// Ends the turn on request. The floor only follows quiet frames, so
    /// sound that rose during the turn kept it open. The quietest moment of
    /// the last 1.5 s becomes the floor: steady noise lifts it, while speech
    /// has pauses between words and leaves it near the room level.
    mutating func forceEnd() {
        let background = recentLevels.min() ?? noiseFloor
        noiseFloor = max(noiseFloor, min(background, Self.maxForcedFloor))
        isSpeaking = false
        aboveMs = 0
        speechMs = 0
        silenceMs = 0
    }

    /// Level of PCM16 samples in dBFS.
    static func level(of samples: UnsafeBufferPointer<Int16>) -> Double {
        guard !samples.isEmpty else { return -120 }
        var sum = 0.0
        for sample in samples {
            let value = Double(sample) / 32768
            sum += value * value
        }
        return 20 * log10(max((sum / Double(samples.count)).squareRoot(), 1e-6))
    }
}

/// Converts microphone buffers to 24 kHz mono PCM16 and cuts them into 20 ms
/// frames. It runs on the audio thread and hands each frame to `deliver`.
final class MicrophoneTap: @unchecked Sendable {
    static let frameSamples = 480

    private let converter: AVAudioConverter
    private let format: AVAudioFormat
    private let deliver: @Sendable (Data, Double) -> Void
    private var carry: [Int16] = []

    init?(input: AVAudioFormat, deliver: @escaping @Sendable (Data, Double) -> Void) {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24000,
                                         channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: input, to: format)
        else { return nil }
        // Voice processing can report several input channels; the first holds the voice.
        if input.channelCount > 1 { converter.channelMap = [0] }
        self.converter = converter
        self.format = format
        self.deliver = deliver
    }

    func process(_ buffer: AVAudioPCMBuffer) {
        let ratio = format.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        // The input block runs synchronously inside convert.
        nonisolated(unsafe) var supplied = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, let channel = out.int16ChannelData else { return }
        carry.append(contentsOf: UnsafeBufferPointer(start: channel[0], count: Int(out.frameLength)))
        while carry.count >= Self.frameSamples {
            let frame = Array(carry[..<Self.frameSamples])
            carry.removeFirst(Self.frameSamples)
            let level = frame.withUnsafeBufferPointer { VoiceActivityDetector.level(of: $0) }
            deliver(frame.withUnsafeBytes { Data($0) }, level)
        }
    }
}

/// How far one reply's audio fell behind real-time playback.
struct ReplyPacing: Equatable {
    static let minimumStartBuffer: AVAudioFrameCount = 3840  // 160 ms
    static let maximumStartBuffer: AVAudioFrameCount = 24000  // 1 s

    private var start: TimeInterval?
    private var seconds = 0.0
    /// Delay after the first chunk that plays the reply without a gap.
    private var need = 0.0

    mutating func add(seconds chunk: Double, at now: TimeInterval, speed: Double) {
        if let start {
            need = max(need, now - start - seconds / speed)
        } else {
            start = now
        }
        seconds += chunk
    }

    /// Start buffer for the next reply with a 25% margin; nil before any audio.
    func startBuffer(sampleRate: Double, speed: Double) -> AVAudioFrameCount? {
        guard start != nil else { return nil }
        let samples = AVAudioFrameCount(need * 1.25 * speed * sampleRate)
        return min(max(samples, Self.minimumStartBuffer), Self.maximumStartBuffer)
    }
}

@MainActor
final class VoiceAudioEngine {

    /// Each 20 ms microphone frame as 24 kHz PCM16, with its level in dBFS.
    var onFrame: ((Data, Double) -> Void)?

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let timePitch = AVAudioUnitTimePitch()
    /// Reply audio buffered before playback starts. Each finished reply sets
    /// it from how far its audio fell behind real time, so a slow model and
    /// Mac buffer more and a fast one starts sooner. An underrun doubles it.
    var startBufferSamples = ReplyPacing.minimumStartBuffer
    private var pacing = ReplyPacing()
    private let playbackFormat = AVAudioFormat(standardFormatWithSampleRate: 24000, channels: 1)!
    private var tap: MicrophoneTap?
    private var waiting: [AVAudioPCMBuffer] = []
    private var waitingSamples: AVAudioFrameCount = 0
    /// Reply samples played in earlier runs. The player clock keeps running when
    /// its queue runs dry, so each refill starts a new run from zero.
    private var finishedSamples: AVAudioFramePosition = 0
    private var runSamples: AVAudioFramePosition = 0
    /// End sample in the reply and level of each scheduled chunk, for the speaking animation.
    private var chunkLevels: [(end: AVAudioFramePosition, level: Double)] = []
    private var playing = false
    private var configObserver: NSObjectProtocol?

    func start() throws {
        try engine.inputNode.setVoiceProcessingEnabled(true)
        engine.attach(player)
        engine.attach(timePitch)
        guard connectGraph() else { throw CocoaError(.featureUnsupported) }
        try installTap()
        engine.prepare()
        try engine.start()
        // Device switches (AirPods, displays) reset the graph.
        configObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.restart() }
        }
    }

    func stop() {
        if let configObserver {
            NotificationCenter.default.removeObserver(configObserver)
        }
        configObserver = nil
        stopPlayback()
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        tap = nil
    }

    /// Playback rate of the reply. The player still counts source samples,
    /// so the heard length stays in reply time.
    var speechSpeed: Double {
        get { Double(timePitch.rate) }
        set {
            timePitch.rate = Float(newValue)
            timePitch.bypass = newValue == 1
        }
    }

    var isMuted: Bool {
        get { engine.inputNode.isVoiceProcessingInputMuted }
        set { engine.inputNode.isVoiceProcessingInputMuted = newValue }
    }

    /// Voice processing needs the same client format on its input and output,
    /// so the mixer feeds the output in the microphone's format. Without this
    /// a mono microphone with stereo speakers fails with -10875.
    private func connectGraph() -> Bool {
        let voiceFormat = engine.inputNode.outputFormat(forBus: 0)
        guard voiceFormat.sampleRate > 0, voiceFormat.channelCount > 0 else { return false }
        engine.connect(player, to: timePitch, format: playbackFormat)
        engine.connect(timePitch, to: engine.mainMixerNode, format: playbackFormat)
        engine.connect(engine.mainMixerNode, to: engine.outputNode, format: voiceFormat)
        return true
    }

    private func installTap() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        let deliver: @Sendable (Data, Double) -> Void = { [weak self] data, level in
            Task { @MainActor in self?.onFrame?(data, level) }
        }
        guard let tap = MicrophoneTap(input: format, deliver: deliver) else {
            throw CocoaError(.featureUnsupported)
        }
        self.tap = tap
        input.installTap(onBus: 0, bufferSize: 1024, format: format, block: Self.tapBlock(for: tap))
    }

    /// The tap block runs on an audio thread. A closure made in a main actor
    /// method would inherit that isolation and trap on its first buffer.
    nonisolated static func tapBlock(for tap: MicrophoneTap) -> AVAudioNodeTapBlock {
        { buffer, _ in tap.process(buffer) }
    }

    private func restart() {
        stopPlayback()
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        guard connectGraph() else {
            NSLog("oMLX: Ask AI voice audio restart failed - no usable input format")
            return
        }
        do {
            try installTap()
            engine.prepare()
            try engine.start()
        } catch {
            NSLog("oMLX: Ask AI voice audio restart failed - \(error)")
        }
    }

    // MARK: Playback

    /// Queues 24 kHz PCM16 reply audio.
    func enqueue(_ pcm16: Data) {
        guard let buffer = Self.floatBuffer(pcm16, format: playbackFormat) else { return }
        let seconds = Double(buffer.frameLength) / playbackFormat.sampleRate
        receivedSeconds += seconds
        pacing.add(seconds: seconds, at: ProcessInfo.processInfo.systemUptime, speed: speechSpeed)
        if playing {
            if runPlayed < runSamples {
                schedule(buffer)
                return
            }
            // The queue ran dry: buffer more before resuming.
            startBufferSamples = min(startBufferSamples * 2, ReplyPacing.maximumStartBuffer)
            finishedSamples += runSamples
            runSamples = 0
            player.stop()
            playing = false
        }
        waiting.append(buffer)
        waitingSamples += buffer.frameLength
        if waitingSamples >= startBufferSamples { startPlayback() }
    }

    /// The reply is complete, so play what is still buffered and size the
    /// next reply's start buffer from this one.
    func finishReply() {
        if !playing, !waiting.isEmpty { startPlayback() }
        if let next = pacing.startBuffer(sampleRate: playbackFormat.sampleRate, speed: speechSpeed) {
            startBufferSamples = next
        }
        pacing = ReplyPacing()
    }

    /// Reply audio played so far, in seconds of reply time.
    var playedSeconds: Double {
        Double(finishedSamples + runPlayed) / playbackFormat.sampleRate
    }

    /// Reply audio received so far, in seconds.
    private(set) var receivedSeconds = 0.0

    var isPlayingReply: Bool {
        !waiting.isEmpty || (playing && runPlayed < runSamples)
    }

    var outputLevel: Double {
        guard playing else { return -120 }
        let played = finishedSamples + runPlayed
        return chunkLevels.first { $0.end > played }?.level ?? -120
    }

    /// Stops playback and returns how much of the reply was heard, in ms.
    @discardableResult
    func stopPlayback() -> Int {
        let played = finishedSamples + runPlayed
        player.stop()
        waiting = []
        waitingSamples = 0
        finishedSamples = 0
        runSamples = 0
        chunkLevels = []
        playing = false
        pacing = ReplyPacing()
        receivedSeconds = 0
        return Int(played * 1000 / 24000)
    }

    private var runPlayed: AVAudioFramePosition {
        guard playing, let nodeTime = player.lastRenderTime,
              let time = player.playerTime(forNodeTime: nodeTime)
        else { return 0 }
        return min(max(time.sampleTime, 0), runSamples)
    }

    private func startPlayback() {
        playing = true
        for buffer in waiting { schedule(buffer) }
        waiting = []
        waitingSamples = 0
        player.play()
    }

    private func schedule(_ buffer: AVAudioPCMBuffer) {
        runSamples += AVAudioFramePosition(buffer.frameLength)
        chunkLevels.append((finishedSamples + runSamples, Self.level(of: buffer)))
        player.scheduleBuffer(buffer)
    }

    static func floatBuffer(_ pcm16: Data, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let count = pcm16.count / 2
        guard count > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)),
              let channel = buffer.floatChannelData
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(count)
        pcm16.withUnsafeBytes { raw in
            let samples = raw.bindMemory(to: Int16.self)
            for i in 0..<count {
                channel[0][i] = Float(Int16(littleEndian: samples[i])) / 32768
            }
        }
        return buffer
    }

    private static func level(of buffer: AVAudioPCMBuffer) -> Double {
        guard let channel = buffer.floatChannelData, buffer.frameLength > 0 else { return -120 }
        var sum: Float = 0
        for i in 0..<Int(buffer.frameLength) { sum += channel[0][i] * channel[0][i] }
        return 20 * log10(max(Double((sum / Float(buffer.frameLength)).squareRoot()), 1e-6))
    }
}
