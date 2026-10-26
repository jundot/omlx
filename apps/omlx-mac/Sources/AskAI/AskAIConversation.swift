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
        var answer = ""
        var reasoning = ""
        var error: String?
        var isStreaming = true
        var wasStopped = false
    }

    private(set) var turns: [Turn] = []
    private(set) var profile: AskAIProfile?
    /// Preset command that started the conversation, nil for the default profile.
    private(set) var command: String?
    private(set) var lastActivity = Date.distantPast
    /// Set when an automatic copy lands; the panel shows a short notice.
    private(set) var autoCopiedTurnID: UUID?

    @ObservationIgnored private var streamTask: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    var isStreaming: Bool { turns.last?.isStreaming == true }
    var isEmpty: Bool { turns.isEmpty }

    var lastAnswer: String? {
        turns.last(where: { !$0.answer.isEmpty })?.answer
    }

    func begin(profile: AskAIProfile, command: String?) {
        reset()
        self.profile = profile
        self.command = command
    }

    /// Applies settings changed during the conversation to the next turn.
    func updateProfile(_ profile: AskAIProfile) {
        guard self.profile != nil else { return }
        self.profile = profile
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
        for turn in history where turn.error == nil && !turn.answer.isEmpty {
            out.append(ChatMessage(role: .user, content: turn.question,
                                   imageURLs: turn.images.map(\.dataURL)))
            out.append(ChatMessage(role: .assistant, content: turn.answer))
        }
        out.append(ChatMessage(role: .user, content: question, imageURLs: images.map(\.dataURL)))
        return out
    }

    /// Returns false when there is no profile or an answer is still streaming.
    @discardableResult
    func send(_ question: String, images: [AskAIImage] = [], gpuPowerLimitW: Int? = nil,
              using client: ChatCompletionClient) -> Bool {
        guard let profile, !isStreaming else { return false }
        let body = ChatCompletionBody(
            model: profile.model,
            messages: Self.messages(profile: profile, history: turns, question: question, images: images),
            allowThinking: profile.allowThinking,
            gpuPowerLimitW: gpuPowerLimitW
        )
        let turn = Turn(question: question, images: images)
        turns.append(turn)
        autoCopiedTurnID = nil
        lastActivity = Date()

        generation += 1
        let gen = generation
        let stream = client.stream(body)
        streamTask = Task { [weak self] in
            var content = ""
            var reasoning = ""
            var lastFlush = ContinuousClock.now
            do {
                for try await event in stream {
                    switch event {
                    case .content(let text): content += text
                    case .reasoning(let text): reasoning += text
                    }
                    // Batch deltas: Markdown re-parses the whole answer per update.
                    if ContinuousClock.now - lastFlush >= .milliseconds(50) {
                        self?.append(content, reasoning, to: turn.id, generation: gen)
                        content = ""
                        reasoning = ""
                        lastFlush = .now
                    }
                }
                self?.append(content, reasoning, to: turn.id, generation: gen)
                self?.finish(turn.id, generation: gen, error: nil)
            } catch {
                self?.append(content, reasoning, to: turn.id, generation: gen)
                self?.finish(turn.id, generation: gen, error: error)
            }
        }
        return true
    }

    func stop() {
        guard isStreaming, let index = turns.indices.last else { return }
        cancelStream()
        turns[index].isStreaming = false
        turns[index].wasStopped = true
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
        lastActivity = Date()
        if let error {
            turns[index].error = Self.describe(error)
            return
        }
        if profile?.autoCopy == true {
            let answer = turns[index].answer.trimmingCharacters(in: .whitespacesAndNewlines)
            if !answer.isEmpty {
                AskAIClipboard.copy(answer)
                autoCopiedTurnID = id
            }
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
