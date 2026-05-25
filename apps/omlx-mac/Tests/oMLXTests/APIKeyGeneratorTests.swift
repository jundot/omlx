// The random-key generator backs the regenerate button in the Security
// screen and the Generate button in the Welcome wizard. The server-side
// validator (`validate_api_key` in `omlx/admin/routes.py`) requires ≥ 4
// printable characters with no whitespace. These tests pin the shape so
// a future tweak to the alphabet, prefix, or length can't silently
// produce keys the server will reject.

import XCTest
@testable import oMLX

final class APIKeyGeneratorTests: XCTestCase {

    func testHasExpectedPrefix() {
        let key = APIKeyGenerator.random()
        XCTAssertTrue(key.hasPrefix(APIKeyGenerator.prefix),
                      "key should start with prefix; got: \(key)")
    }

    func testTotalLengthMatchesPrefixPlusBody() {
        let key = APIKeyGenerator.random()
        XCTAssertEqual(key.count,
                       APIKeyGenerator.prefix.count + APIKeyGenerator.bodyLength)
    }

    func testBodyOnlyUsesDeclaredAlphabet() {
        let key = APIKeyGenerator.random()
        let body = String(key.dropFirst(APIKeyGenerator.prefix.count))
        let allowed = Set(APIKeyGenerator.bodyAlphabet)
        XCTAssertTrue(body.allSatisfy { allowed.contains($0) },
                      "body contained non-alphabet chars: \(body)")
    }

    func testNoWhitespace() {
        // Server rejects whitespace — the alphabet excludes it but pin
        // the invariant explicitly in case someone extends the alphabet
        // and forgets the constraint.
        for _ in 0..<32 {
            let key = APIKeyGenerator.random()
            XCTAssertFalse(key.contains(where: { $0.isWhitespace }),
                           "key contained whitespace: \(key)")
        }
    }

    func testSucceedsServerSideFloor() {
        // ≥ 4 printable characters: trivially satisfied by an 8-char
        // prefix + 24-char body, but checked here so a future shorter
        // body still passes the server floor.
        let key = APIKeyGenerator.random()
        let trimmed = key.trimmingCharacters(in: .whitespaces)
        XCTAssertGreaterThanOrEqual(trimmed.count, 4)
    }

    func testDistinctOutputsBetweenCalls() {
        // 24 chars of ~62-symbol alphabet → collision probability is
        // vanishing. A literal-equality collision across 16 calls would
        // mean the RNG is broken.
        let samples = (0..<16).map { _ in APIKeyGenerator.random() }
        XCTAssertEqual(Set(samples).count, samples.count,
                       "two random keys collided: \(samples)")
    }
}
