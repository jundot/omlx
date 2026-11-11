// Streaming client for the server's OpenAI-compatible
// `/v1/chat/completions`. `/v1` does not accept the admin session cookie,
// so every request carries the main API key as a Bearer token.

import Foundation

struct ChatMessage: Equatable, Sendable {
    enum Role: String, Sendable { case system, user, assistant, tool }
    let role: Role
    let content: String
    /// `data:` image URLs, sent as image parts before the text.
    var imageURLs: [String] = []
    /// Calls made by an assistant message, whose content may be empty.
    var toolCalls: [ChatToolCall] = []
    /// The call a tool message answers.
    var toolCallID: String?
}

/// A function call the model asked for, with its arguments as JSON text.
struct ChatToolCall: Equatable, Sendable {
    var id: String
    var name: String
    var arguments: String
}

/// A function the model may call; every argument is a required string.
struct ChatTool: Equatable, Sendable {
    struct Argument: Equatable, Sendable {
        let name: String
        let description: String
    }

    let name: String
    let description: String
    var arguments: [Argument] = []
}

/// One streamed piece of a tool call. The first piece of a call carries its
/// id and name; later pieces with the same index add to its arguments.
struct ChatToolCallDelta: Equatable, Sendable {
    let index: Int
    var id: String?
    var name: String?
    var arguments: String?
}

enum ChatStreamEvent: Equatable, Sendable {
    case content(String)
    case reasoning(String)
    case toolCall(ChatToolCallDelta)
}

/// Streams one chat completion; tests script the replies through it.
protocol ChatStreaming: Sendable {
    func stream(_ body: ChatCompletionBody) -> AsyncThrowingStream<ChatStreamEvent, Error>
}

enum ChatStreamError: LocalizedError, Equatable {
    case server(String)
    case http(status: Int, message: String?)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .server(let message):
            return message
        case .http(let status, let message):
            return message ?? HTTPURLResponse.localizedString(forStatusCode: status)
        case .invalidResponse:
            return String(localized: "askai.error.invalid_response",
                          defaultValue: "The server sent a response Ask AI could not read.",
                          comment: "Error when the chat stream response is not HTTP")
        }
    }
}

struct ChatCompletionBody: Encodable {
    /// Plain string content, or OpenAI content parts when images are attached.
    struct Message: Encodable {
        let role: String
        let content: String
        let imageURLs: [String]
        var toolCalls: [ChatToolCall] = []
        var toolCallID: String?

        private enum CodingKeys: String, CodingKey { case role, content, toolCalls, toolCallId }

        private struct Call: Encodable {
            struct Function: Encodable {
                let name: String
                let arguments: String
            }
            let id: String
            let type = "function"
            let function: Function
        }

        private struct Part: Encodable {
            struct ImageURL: Encodable { let url: String }
            let type: String
            var text: String?
            var imageUrl: ImageURL?
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(role, forKey: .role)
            if !toolCalls.isEmpty {
                try container.encode(toolCalls.map {
                    Call(id: $0.id, function: .init(name: $0.name, arguments: $0.arguments))
                }, forKey: .toolCalls)
            }
            try container.encodeIfPresent(toolCallID, forKey: .toolCallId)
            guard !imageURLs.isEmpty else {
                try container.encode(content, forKey: .content)
                return
            }
            var parts = imageURLs.map { Part(type: "image_url", imageUrl: .init(url: $0)) }
            if !content.isEmpty {
                parts.append(Part(type: "text", text: content))
            }
            try container.encode(parts, forKey: .content)
        }
    }

    struct TemplateKwargs: Encodable {
        let enableThinking: Bool
    }

    /// OpenAI function tool with string arguments.
    struct Tool: Encodable {
        struct Function: Encodable {
            struct Parameters: Encodable {
                struct Property: Encodable {
                    let type = "string"
                    let description: String
                }
                let type = "object"
                let properties: [String: Property]
                let required: [String]
            }
            let name: String
            let description: String
            let parameters: Parameters
        }
        let type = "function"
        let function: Function

        init(_ tool: ChatTool) {
            let properties = Dictionary(uniqueKeysWithValues: tool.arguments.map {
                ($0.name, Function.Parameters.Property(description: $0.description))
            })
            function = Function(name: tool.name, description: tool.description,
                                parameters: .init(properties: properties,
                                                  required: tool.arguments.map(\.name)))
        }
    }

    let model: String
    let messages: [Message]
    let stream = true
    let chatTemplateKwargs: TemplateKwargs?
    let gpuPowerLimitW: Int?
    let tools: [Tool]?

    /// Thinking off sends `enable_thinking: false`. Thinking on sends nothing,
    /// so the model's own setting applies. The server ignores the value for
    /// models whose template forces it.
    init(model: String, messages: [ChatMessage], allowThinking: Bool, gpuPowerLimitW: Int? = nil,
         tools: [ChatTool] = []) {
        self.model = model
        self.messages = messages.map {
            Message(role: $0.role.rawValue, content: $0.content, imageURLs: $0.imageURLs,
                    toolCalls: $0.toolCalls, toolCallID: $0.toolCallID)
        }
        self.chatTemplateKwargs = allowThinking ? nil : TemplateKwargs(enableThinking: false)
        self.gpuPowerLimitW = gpuPowerLimitW
        self.tools = tools.isEmpty ? nil : tools.map(Tool.init)
    }

    func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return try encoder.encode(self)
    }
}

/// One SSE line of the chat stream.
enum ChatStreamLine: Equatable {
    case ignore
    case delta(content: String, reasoning: String)
    case toolCalls([ChatToolCallDelta])
    case done

    private struct Chunk: Decodable {
        struct Choice: Decodable {
            struct Delta: Decodable {
                struct ToolCall: Decodable {
                    struct Function: Decodable {
                        let name: String?
                        let arguments: String?
                    }
                    let index: Int?
                    let id: String?
                    let function: Function?
                }
                let content: String?
                let reasoningContent: String?
                let toolCalls: [ToolCall]?
            }
            let delta: Delta?
        }
        struct ErrorBody: Decodable {
            let message: String?
        }
        let choices: [Choice]?
        let error: ErrorBody?
    }

    static func parse(_ rawLine: String) throws -> ChatStreamLine {
        let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : rawLine
        guard line.hasPrefix("data:") else { return .ignore }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" { return .done }
        guard !payload.isEmpty else { return .ignore }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let chunk = try? decoder.decode(Chunk.self, from: Data(payload.utf8)) else {
            return .ignore
        }
        if let error = chunk.error {
            throw ChatStreamError.server(error.message ?? "Unknown server error")
        }
        let delta = chunk.choices?.first?.delta
        if let calls = delta?.toolCalls, !calls.isEmpty {
            return .toolCalls(calls.enumerated().map { position, call in
                ChatToolCallDelta(index: call.index ?? position, id: call.id,
                                  name: call.function?.name, arguments: call.function?.arguments)
            })
        }
        let content = delta?.content ?? ""
        let reasoning = delta?.reasoningContent ?? ""
        if content.isEmpty && reasoning.isEmpty { return .ignore }
        return .delta(content: content, reasoning: reasoning)
    }

    /// Pulls `error.message` out of an OpenAI-style error body.
    static func errorMessage(fromBody data: Data) -> String? {
        struct Body: Decodable {
            struct Inner: Decodable { let message: String? }
            let error: Inner?
            let detail: String?
        }
        guard let body = try? JSONDecoder().decode(Body.self, from: data) else { return nil }
        return body.error?.message ?? body.detail
    }
}

struct ChatCompletionClient: ChatStreaming {
    let url: URL
    let apiKey: String?

    // The response headers arrive only after the server has loaded the
    // model, which can take minutes, so the default 60 s timeout is too short.
    private static let session: URLSession = {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 600
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.urlCache = nil
        return URLSession(configuration: cfg)
    }()

    func stream(_ body: ChatCompletionBody) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        let request: URLRequest
        do {
            request = try makeRequest(body)
        } catch {
            return AsyncThrowingStream { $0.finish(throwing: error) }
        }
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await Self.session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        throw ChatStreamError.invalidResponse
                    }
                    guard http.statusCode == 200 else {
                        var data = Data()
                        for try await byte in bytes {
                            data.append(byte)
                            if data.count >= 8192 { break }
                        }
                        throw ChatStreamError.http(
                            status: http.statusCode,
                            message: ChatStreamLine.errorMessage(fromBody: data)
                        )
                    }
                    for try await line in bytes.lines {
                        switch try ChatStreamLine.parse(line) {
                        case .ignore:
                            continue
                        case .done:
                            continuation.finish()
                            return
                        case let .delta(content, reasoning):
                            if !reasoning.isEmpty { continuation.yield(.reasoning(reasoning)) }
                            if !content.isEmpty { continuation.yield(.content(content)) }
                        case .toolCalls(let calls):
                            for call in calls { continuation.yield(.toolCall(call)) }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func makeRequest(_ body: ChatCompletionBody) throws -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        if let apiKey, !apiKey.isEmpty {
            req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = try body.encoded()
        return req
    }
}
