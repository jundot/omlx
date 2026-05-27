// Shared random-key generator for the Security screen and the Welcome
// wizard. Kept tiny — anything more elaborate (configurable prefix,
// extra alphabet) earns its own type.
//
// Crypto note: `Array.randomElement()` is backed by Swift's
// `SystemRandomNumberGenerator`, which on Apple platforms uses the
// system's cryptographic random source (getentropy/SecRandomCopyBytes
// under the hood). So this generator is crypto-grade without us
// reaching into the Security framework directly.

import Foundation

enum APIKeyGenerator {
    static let prefix = "sk-omlx-"
    static let bodyAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    static let bodyLength = 24

    /// 24-char alphanumeric body — ~143 bits of entropy, comfortably above
    /// the server's "≥ 4 printable, no whitespace" floor and short enough
    /// to fit the editor row's field without truncation.
    static func random() -> String {
        let body = String((0..<bodyLength).map { _ in bodyAlphabet.randomElement()! })
        return "\(prefix)\(body)"
    }
}
