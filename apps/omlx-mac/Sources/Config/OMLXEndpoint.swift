// OMLXEndpoint — one address for "the oMLX server this app talks to".
//
// It is either the server this app manages on this Mac (`isRemote == false`,
// always plain http on the configured bind port) or another oMLX server on
// the network (`isRemote == true`, parsed from the text the user typed:
// `host`, `host:port`, or `http(s)://host[:port]`). The CLI parses the same
// forms in `omlx/settings.py` (`parse_endpoint_address`); keep the accepted
// and rejected inputs in step with the shared test vectors there.

import Foundation

struct OMLXEndpoint: Equatable, Sendable {
    /// Port a bare address uses: oMLX's default, or 443 for `https`.
    static let defaultPort = 8000
    static let defaultSecurePort = 443

    var scheme: String
    var host: String
    var port: Int
    var apiKey: String?
    var isRemote: Bool

    init(scheme: String, host: String, port: Int, apiKey: String?, isRemote: Bool) {
        self.scheme = scheme
        self.host = host
        self.port = port
        self.apiKey = apiKey
        self.isRemote = isRemote
    }

    /// The server this app runs on this Mac.
    init(localServer config: AppConfig) {
        scheme = "http"
        host = config.host
        port = config.port
        apiKey = config.apiKey
        isRemote = false
    }

    /// Parses `host`, `host:port`, or `http(s)://host[:port]`. Nil when the
    /// text is empty or not an address this app can reach: another scheme,
    /// a path, a query, credentials, or a port outside 1...65535. Mirrors
    /// `parse_endpoint_address` in `omlx/settings.py` step for step so the
    /// CLI accepts exactly what this app saves.
    init?(remoteAddress: String, apiKey: String?) {
        guard let parsed = Self.parse(remoteAddress) else { return nil }
        self.scheme = parsed.scheme
        self.host = parsed.host
        self.port = parsed.port
        let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.apiKey = (key?.isEmpty ?? true) ? nil : key
        self.isRemote = true
    }

    private static let hostnameCharacters = CharacterSet(charactersIn:
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")

    private static func parse(_ raw: String) -> (scheme: String, host: String, port: Int)? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty,
              text.unicodeScalars.allSatisfy({ !"?#@%".unicodeScalars.contains($0) && !CharacterSet.whitespacesAndNewlines.contains($0) })
        else { return nil }

        var scheme = "http"
        if let range = text.range(of: "://") {
            scheme = text[..<range.lowerBound].lowercased()
            guard scheme == "http" || scheme == "https" else { return nil }
            text = String(text[range.upperBound...])
        }
        if text.hasSuffix("/") { text.removeLast() }
        guard !text.isEmpty, !text.contains("/") else { return nil }

        let host: String
        let portText: Substring?
        if text.hasPrefix("[") {
            guard let close = text.firstIndex(of: "]") else { return nil }
            host = String(text[text.index(after: text.startIndex)..<close])
            var addr = in6_addr()
            guard inet_pton(AF_INET6, host, &addr) == 1 else { return nil }
            let rest = text[text.index(after: close)...]
            if rest.isEmpty {
                portText = nil
            } else {
                guard rest.hasPrefix(":") else { return nil }
                portText = rest.dropFirst()
            }
        } else {
            guard text.filter({ $0 == ":" }).count <= 1 else { return nil }  // bare IPv6 is ambiguous
            let pieces = text.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
            host = String(pieces[0])
            portText = pieces.count == 2 ? pieces[1] : nil
            guard !host.isEmpty, host.unicodeScalars.allSatisfy({ hostnameCharacters.contains($0) }) else { return nil }
        }

        let port: Int
        if let portText {
            guard !portText.isEmpty, portText.allSatisfy({ $0.isASCII && $0.isNumber }),
                  let value = Int(portText), (1...65535).contains(value)
            else { return nil }
            port = value
        } else {
            port = scheme == "https" ? defaultSecurePort : defaultPort
        }
        return (scheme, host, port)
    }

    var hasAPIKey: Bool {
        !(apiKey?.isEmpty ?? true)
    }

    /// `http://host:port` with an IPv6 host bracketed.
    var baseURL: URL? {
        AppConfig.httpURL(host: host, port: port, scheme: scheme)
    }

    /// What the UI shows for this server: `host:port`, or the full origin
    /// when the scheme is not the plain-http default.
    var displayAddress: String {
        let shownHost = host.contains(":") ? "[\(host)]" : host
        return scheme == "http" ? "\(shownHost):\(port)" : "\(scheme)://\(shownHost):\(port)"
    }

    /// The key travels in the clear: plain http to a host other than this Mac.
    var sendsKeyInTheClear: Bool {
        scheme == "http" && !["localhost", "127.0.0.1", "::1"].contains(host.lowercased())
    }

    func httpURL(path: String) -> URL? {
        AppConfig.httpURL(host: host, port: port, path: path, scheme: scheme)
    }

    func webSocketURL(path: String, query: [URLQueryItem] = []) -> URL? {
        AppConfig.webSocketURL(host: host, port: port, path: path, query: query, scheme: scheme)
    }

    private func bearerRequest(path: String, timeout: TimeInterval = 15) throws -> URLRequest {
        guard let url = httpURL(path: path) else { throw OMLXClientError.invalidURL }
        var request = URLRequest(url: url, timeoutInterval: timeout)
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// The server's model list. `/v1/models/status` also accepts sub keys;
    /// the admin model list accepts only the main key.
    func models() async throws -> [ModelDTO] {
        let (data, response) = try await URLSession.shared.data(for: bearerRequest(path: "/v1/models/status"))
        guard let http = response as? HTTPURLResponse else { throw OMLXClientError.invalidResponse }
        if http.statusCode == 401, !hasAPIKey { throw OMLXClientError.unauthenticated }
        guard 200..<300 ~= http.statusCode else {
            throw OMLXClientError.http(status: http.statusCode, body: String(data: data, encoding: .utf8))
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ListModelsResponse.self, from: data).models
    }

    /// Whether the server answers `/health`. Throws with the transport
    /// error when it does not; returns the round trip in milliseconds.
    @discardableResult
    func checkHealth(timeout: TimeInterval = 5) async throws -> Int {
        let started = Date()
        let (_, response) = try await URLSession.shared.data(for: bearerRequest(path: "/health", timeout: timeout))
        guard let http = response as? HTTPURLResponse else { throw OMLXClientError.invalidResponse }
        guard 200..<300 ~= http.statusCode else {
            throw OMLXClientError.http(status: http.statusCode, body: nil)
        }
        return Int(Date().timeIntervalSince(started) * 1000)
    }

    /// Whether the key opens the admin API, which the app's screens need.
    /// `/v1/models/status` accepts sub keys, so a passing model list does
    /// not prove this. Uses a throwaway session so no cookie lingers.
    func checkAdminLogin() async throws {
        guard let url = httpURL(path: "/admin/api/login") else { throw OMLXClientError.invalidURL }
        var request = URLRequest(url: url, timeoutInterval: 15)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["api_key": apiKey ?? "", "remember": false]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let session = URLSession(configuration: .ephemeral)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw OMLXClientError.invalidResponse }
        if http.statusCode == 401 { throw OMLXClientError.unauthenticated }
        guard 200..<300 ~= http.statusCode else {
            throw OMLXClientError.http(status: http.statusCode, body: String(data: data, encoding: .utf8))
        }
    }
}
