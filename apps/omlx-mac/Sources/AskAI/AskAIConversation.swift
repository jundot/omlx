// One Ask AI conversation: the profile it runs with, its turns, and the
// streaming task for the newest answer. The conversation lives on the
// controller, so an answer keeps streaming while the panel is hidden.

import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class AskAIConversation {
    struct Turn: Identifiable, Equatable {
        let id = UUID()
        let question: String
        var images: [AskAIImage] = []
        /// Everything the model wrote, across tool rounds.
        var answer = ""
        var reasoning = ""
        var error: String?
        var isStreaming = true
        var wasStopped = false
        /// Tools used for this answer, in call order.
        var activities: [AskAIToolActivity] = []
        /// Calls and results of finished tool rounds. They are resent with
        /// the turn, so follow-up questions keep what the tools returned.
        var exchanges: [ChatMessage] = []
        /// Where the text after the last tool round starts in `answer`.
        var replyStart = 0

        /// The final answer, without text written before tool calls.
        var reply: String { String(answer.dropFirst(replyStart)) }
    }

    /// Tool rounds per question. One more round without tools makes the
    /// model answer with what it has.
    static let maxToolRounds = 4

    private(set) var turns: [Turn] = []
    private(set) var profile: AskAIProfile?
    /// Preset command that started the conversation, nil for the default profile.
    private(set) var command: String?
    private(set) var lastActivity = Date.distantPast
    /// Set when an automatic copy lands; the panel shows a short notice.
    private(set) var autoCopiedTurnID: UUID?

    @ObservationIgnored private var streamTask: Task<Void, Never>?
    @ObservationIgnored private var generation = 0
    /// The profile's prompt before its placeholders were filled, to tell an
    /// edited prompt from the same prompt at a later time.
    @ObservationIgnored private var promptTemplate = ""

    var isStreaming: Bool { turns.last?.isStreaming == true }
    var isEmpty: Bool { turns.isEmpty }

    var lastAnswer: String? {
        turns.last(where: { !$0.reply.isEmpty })?.reply
    }

    func begin(profile: AskAIProfile, command: String?, context: PromptContext = PromptContext()) {
        reset()
        promptTemplate = profile.systemPrompt
        self.profile = Self.filled(profile, context: context)
        self.command = command
    }

    /// Applies settings changed during the conversation to the next turn. An
    /// unchanged prompt keeps its filled text, so the prefix cache still matches.
    func updateProfile(_ profile: AskAIProfile, context: PromptContext = PromptContext()) {
        guard let current = self.profile else { return }
        var next = profile
        if profile.systemPrompt == promptTemplate {
            next.systemPrompt = current.systemPrompt
        } else {
            promptTemplate = profile.systemPrompt
            next = Self.filled(profile, context: context)
        }
        self.profile = next
    }

    private static func filled(_ profile: AskAIProfile, context: PromptContext) -> AskAIProfile {
        var context = context
        context.model = profile.model
        var out = profile
        out.systemPrompt = PromptPlaceholders.fill(profile.systemPrompt, context: context)
        return out
    }

    func reset() {
        cancelStream()
        turns = []
        profile = nil
        command = nil
        autoCopiedTurnID = nil
    }

    /// Prior answered turns plus the new question, after the system prompt.
    /// Earlier images are resent so follow-up questions can refer to them.
    static func messages(
        profile: AskAIProfile, history: [Turn], question: String, images: [AskAIImage] = []
    ) -> [ChatMessage] {
        var out: [ChatMessage] = []
        let system = profile.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty {
            out.append(ChatMessage(role: .system, content: system))
        }
        for turn in history where turn.error == nil && !turn.reply.isEmpty {
            out.append(ChatMessage(role: .user, content: turn.question,
                                   imageURLs: turn.images.map(\.dataURL)))
            out += turn.exchanges
            out.append(ChatMessage(role: .assistant, content: turn.reply))
        }
        out.append(ChatMessage(role: .user, content: question, imageURLs: images.map(\.dataURL)))
        return out
    }

    /// Returns false when there is no profile or an answer is still streaming.
    /// Tools are offered only with a runner to run their calls.
    @discardableResult
    func send(_ question: String, images: [AskAIImage] = [], gpuPowerLimitW: Int? = nil,
              tools: [ChatTool] = [], using client: some ChatStreaming,
              runner: (any AskAIToolRunning)? = nil) -> Bool {
        guard let profile, !isStreaming else { return false }
        let base = Self.messages(profile: profile, history: turns, question: question, images: images)
        let turn = Turn(question: question, images: images)
        turns.append(turn)
        autoCopiedTurnID = nil
        lastActivity = Date()

        generation += 1
        let gen = generation
        let tools = runner == nil ? [] : tools
        streamTask = Task { [weak self] in
            var exchanges: [ChatMessage] = []
            do {
                for round in 0...Self.maxToolRounds {
                    let body = ChatCompletionBody(
                        model: profile.model,
                        messages: base + exchanges,
                        allowThinking: profile.allowThinking,
                        gpuPowerLimitW: gpuPowerLimitW,
                        tools: round < Self.maxToolRounds ? tools : []
                    )
                    guard let result = try await self?.streamRound(client.stream(body), turn: turn.id,
                                                                   generation: gen),
                          !result.calls.isEmpty, let runner else { break }
                    guard let messages = await self?.runTools(result.calls, text: result.text,
                                                              runner: runner, turn: turn.id, generation: gen)
                    else { return }
                    exchanges += messages
                    self?.finishRound(exchanges, turn: turn.id, generation: gen)
                }
                self?.finish(turn.id, generation: gen, error: nil)
            } catch {
                self?.finish(turn.id, generation: gen, error: error)
            }
        }
        return true
    }

    /// Streams one completion into the turn. Returns the text it wrote and
    /// the tool calls it asked for.
    private func streamRound(
        _ stream: AsyncThrowingStream<ChatStreamEvent, Error>, turn id: UUID, generation gen: Int
    ) async throws -> (text: String, calls: [ChatToolCall]) {
        var text = ""
        var content = ""
        var reasoning = ""
        var calls: [Int: ChatToolCall] = [:]
        var lastFlush = ContinuousClock.now
        do {
            for try await event in stream {
                switch event {
                case .content(let delta):
                    content += delta
                    text += delta
                case .reasoning(let delta):
                    reasoning += delta
                case .toolCall(let delta):
                    var call = calls[delta.index] ?? ChatToolCall(id: "", name: "", arguments: "")
                    if let callID = delta.id { call.id = callID }
                    if let name = delta.name { call.name = name }
                    call.arguments += delta.arguments ?? ""
                    calls[delta.index] = call
                }
                // Batch deltas: Markdown re-parses the whole answer per update.
                if ContinuousClock.now - lastFlush >= .milliseconds(50) {
                    append(content, reasoning, to: id, generation: gen)
                    content = ""
                    reasoning = ""
                    lastFlush = .now
                }
            }
        } catch {
            append(content, reasoning, to: id, generation: gen)
            throw error
        }
        append(content, reasoning, to: id, generation: gen)
        let ordered = calls.sorted { $0.key < $1.key }.map { _, call in
            var call = call
            if call.id.isEmpty { call.id = "call_\(UUID().uuidString.prefix(8).lowercased())" }
            return call
        }
        return (text, ordered.filter { !$0.name.isEmpty })
    }

    /// Runs one round of calls and returns the messages that record it, or
    /// nil when the answer was stopped meanwhile.
    private func runTools(_ calls: [ChatToolCall], text: String, runner: any AskAIToolRunning,
                          turn id: UUID, generation gen: Int) async -> [ChatMessage]? {
        var messages = [ChatMessage(role: .assistant, content: text, toolCalls: calls)]
        var images: [AskAIImage] = []
        for call in calls {
            guard gen == generation, let index = turns.firstIndex(where: { $0.id == id }) else { return nil }
            let slot = AskAITool.activity(for: call).map { activity in
                turns[index].activities.append(activity)
                return turns[index].activities.count - 1
            }
            let result = await runner.run(call)
            guard gen == generation, let index = turns.firstIndex(where: { $0.id == id }) else { return nil }
            if let slot {
                turns[index].activities[slot].isRunning = false
                turns[index].activities[slot].failed = result.failed
            }
            messages.append(ChatMessage(role: .tool, content: result.text, toolCallID: call.id))
            if let image = result.image { images.append(image) }
        }
        if !images.isEmpty {
            messages.append(ChatMessage(role: .user, content: "Screenshot from capture_screen.",
                                        imageURLs: images.map(\.dataURL)))
        }
        return messages
    }

    /// Keeps the finished rounds and starts the reply after the text
    /// written so far.
    private func finishRound(_ exchanges: [ChatMessage], turn id: UUID, generation gen: Int) {
        guard gen == generation, let index = turns.firstIndex(where: { $0.id == id }) else { return }
        turns[index].exchanges = exchanges
        if !turns[index].answer.isEmpty, !turns[index].answer.hasSuffix("\n\n") {
            turns[index].answer += "\n\n"
        }
        turns[index].replyStart = turns[index].answer.count
        lastActivity = Date()
    }

    func stop() {
        guard isStreaming, let index = turns.indices.last else { return }
        cancelStream()
        turns[index].isStreaming = false
        turns[index].wasStopped = true
        Self.endActivities(&turns[index])
        lastActivity = Date()
    }

    private func cancelStream() {
        generation += 1
        streamTask?.cancel()
        streamTask = nil
    }

    private func append(_ content: String, _ reasoning: String, to id: UUID, generation gen: Int) {
        guard gen == generation, !(content.isEmpty && reasoning.isEmpty),
              let index = turns.firstIndex(where: { $0.id == id }) else { return }
        turns[index].answer += content
        turns[index].reasoning += reasoning
    }

    private func finish(_ id: UUID, generation gen: Int, error: Error?) {
        guard gen == generation, let index = turns.firstIndex(where: { $0.id == id }) else { return }
        streamTask = nil
        turns[index].isStreaming = false
        Self.endActivities(&turns[index])
        lastActivity = Date()
        if let error {
            turns[index].error = Self.describe(error)
            return
        }
        if profile?.autoCopy == true {
            let answer = turns[index].reply.trimmingCharacters(in: .whitespacesAndNewlines)
            if !answer.isEmpty {
                AskAIClipboard.copy(answer)
                autoCopiedTurnID = id
            }
        }
    }

    private static func endActivities(_ turn: inout Turn) {
        for i in turn.activities.indices {
            turn.activities[i].isRunning = false
        }
    }

    private static func describe(_ error: Error) -> String {
        if let urlError = error as? URLError {
            switch urlError.code {
            case .cannotConnectToHost, .cannotFindHost, .networkConnectionLost, .notConnectedToInternet:
                return String(localized: "askai.error.server_unreachable",
                              defaultValue: "Can't reach the oMLX server. Make sure it is running.",
                              comment: "Error when the chat request cannot connect to the local server")
            case .timedOut:
                return String(localized: "askai.error.timed_out",
                              defaultValue: "The server took too long to respond.",
                              comment: "Error when the chat request times out")
            default:
                break
            }
        }
        if let streamError = error as? ChatStreamError, case .http(401, _) = streamError {
            return String(localized: "askai.error.unauthorized",
                          defaultValue: "The server rejected the API key. Check Security settings.",
                          comment: "Error when the chat request is rejected with HTTP 401")
        }
        return error.localizedDescription
    }
}

enum AskAIClipboard {
    static func copy(_ text: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
    }
}
