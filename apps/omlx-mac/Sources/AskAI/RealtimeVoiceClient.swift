// WebSocket client for the server's /v1/realtime endpoint, which follows a
// subset of the OpenAI Realtime API. The app decides when a turn ends, so
// the session turns server turn detection off.

import Foundation

/// What a voice model's realtime backend accepts besides speech.
struct RealtimeVoiceCapabilities: Equatable, Sendable {
    var inputImage = false
    var tools = false
}

enum RealtimeVoiceEvent: Equatable {
    /// Carries the model's default instructions, which the session update replaces.
    case sessionCreated(instructions: String, capabilities: RealtimeVoiceCapabilities)
    case responseStarted(itemID: String, responseID: String)
    /// A tool call that ends a reply; `arguments` is JSON text.
    case functionCall(callID: String, name: String, arguments: String, responseID: String)
    case audio(itemID: String, pcm: Data)
    case transcript(itemID: String, text: String)
    case responseDone(itemID: String, status: String)
    case error(code: String, message: String)
}

extension RealtimeVoiceEvent {
    /// Decodes one server event; nil for events the app does not use.
    static func decode(_ text: String) -> RealtimeVoiceEvent? {
        guard let data = text.data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = event["type"] as? String
        else { return nil }
        switch type {
        case "session.created":
            let session = event["session"] as? [String: Any]
            let capabilities = session?["capabilities"] as? [String: Any]
            return .sessionCreated(
                instructions: session?["instructions"] as? String ?? "",
                capabilities: RealtimeVoiceCapabilities(
                    inputImage: capabilities?["input_image"] as? Bool ?? false,
                    tools: capabilities?["tools"] as? Bool ?? false
                )
            )
        case "response.function_call_arguments.done":
            guard let callID = event["call_id"] as? String, let name = event["name"] as? String
            else { return nil }
            return .functionCall(callID: callID, name: name, arguments: event["arguments"] as? String ?? "",
                                 responseID: event["response_id"] as? String ?? "")
        case "response.output_item.added":
            // Function call items arrive as their own events at the end of a reply.
            let item = event["item"] as? [String: Any]
            guard item?["type"] as? String ?? "message" == "message" else { return nil }
            let responseID = event["response_id"] as? String ?? ""
            return (item?["id"] as? String).map { .responseStarted(itemID: $0, responseID: responseID) }
        case "response.output_audio.delta":
            let itemID = event["item_id"] as? String ?? ""
            return (event["delta"] as? String).flatMap { Data(base64Encoded: $0) }
                .map { .audio(itemID: itemID, pcm: $0) }
        case "response.output_audio_transcript.delta":
            let itemID = event["item_id"] as? String ?? ""
            return (event["delta"] as? String).map { .transcript(itemID: itemID, text: $0) }
        case "response.done":
            let response = event["response"] as? [String: Any]
            let item = (response?["output"] as? [[String: Any]])?.first
            return .responseDone(itemID: item?["id"] as? String ?? "",
                                 status: response?["status"] as? String ?? "completed")
        case "error":
            let error = event["error"] as? [String: Any]
            return .error(code: error?["code"] as? String ?? "",
                          message: error?["message"] as? String ?? "")
        default:
            return nil
        }
    }
}

@MainActor
final class RealtimeVoiceClient {
    /// Server events, then nil once the connection closes.
    var onEvent: ((RealtimeVoiceEvent?) -> Void)?

    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 30
        return URLSession(configuration: config)
    }()

    private let task: URLSessionWebSocketTask
    private var pingTask: Task<Void, Never>?
    private var closed = false

    init(url: URL, apiKey: String?) {
        var request = URLRequest(url: url)
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        task = Self.session.webSocketTask(with: request)
        task.maximumMessageSize = 16 << 20
    }

    func connect() {
        task.resume()
        receive()
        // Pings keep the socket open through proxies. The server's idle
        // timeout counts events, not pings, and nothing is sent while the user
        // is silent, so an empty session.update goes out every two minutes.
        pingTask = Task { [weak self] in
            var ticks = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(20))
                self?.task.sendPing { _ in }
                ticks += 1
                if ticks % 6 == 0 {
                    self?.send(["type": "session.update", "session": [String: Any]()])
                }
            }
        }
    }

    func send(_ event: [String: Any]) {
        guard !closed,
              let data = try? JSONSerialization.data(withJSONObject: event),
              let text = String(data: data, encoding: .utf8)
        else { return }
        task.send(.string(text)) { _ in }
    }

    func close() {
        guard !closed else { return }
        closed = true
        pingTask?.cancel()
        task.cancel(with: .normalClosure, reason: nil)
    }

    private func receive() {
        task.receive { [weak self] result in
            Task { @MainActor in self?.handle(result) }
        }
    }

    private func handle(_ result: Result<URLSessionWebSocketTask.Message, Error>) {
        guard !closed else { return }
        switch result {
        case .success(.string(let text)):
            if let event = RealtimeVoiceEvent.decode(text) { onEvent?(event) }
            receive()
        case .success:
            receive()
        case .failure:
            closed = true
            pingTask?.cancel()
            onEvent?(nil)
        }
    }
}
