// Validation for sampling-parameter text inputs. The text-field UX accepts
// any string, so the validator is the only thing standing between a slipped
// keystroke and an out-of-band patch hitting the server. These tests pin
// the documented ranges so a future widening/narrowing has to update them.

import XCTest
@testable import oMLX

final class SamplingValidatorTests: XCTestCase {

    // MARK: - Temperature (≥ 0)

    func testTemperatureEmptyIsNil() throws {
        XCTAssertNil(try SamplingValidator.temperature("").get())
        XCTAssertNil(try SamplingValidator.temperature("   ").get())
    }

    func testTemperatureAcceptsBoundary() throws {
        XCTAssertEqual(try SamplingValidator.temperature("0").get(), 0)
        XCTAssertEqual(try SamplingValidator.temperature("0.0").get(), 0)
        XCTAssertEqual(try SamplingValidator.temperature("2.5").get(), 2.5)
    }

    func testTemperatureRejectsNegative() {
        XCTAssertThrowsError(try SamplingValidator.temperature("-0.01").get())
    }

    func testTemperatureRejectsNonNumeric() {
        XCTAssertThrowsError(try SamplingValidator.temperature("hot").get())
    }

    // MARK: - Top P (0 < p ≤ 1)

    func testTopPEmptyIsNil() throws {
        XCTAssertNil(try SamplingValidator.topP("").get())
    }

    func testTopPAcceptsUpperBoundary() throws {
        XCTAssertEqual(try SamplingValidator.topP("1").get(), 1)
        XCTAssertEqual(try SamplingValidator.topP("0.95").get(), 0.95)
    }

    func testTopPRejectsZero() {
        // 0 would mean "no candidates pass the nucleus filter" — not valid.
        XCTAssertThrowsError(try SamplingValidator.topP("0").get())
    }

    func testTopPRejectsAboveOne() {
        XCTAssertThrowsError(try SamplingValidator.topP("1.01").get())
    }

    func testTopPRejectsNegative() {
        XCTAssertThrowsError(try SamplingValidator.topP("-0.1").get())
    }

    // MARK: - Min P (0 ≤ p ≤ 1)

    func testMinPAcceptsZero() throws {
        // 0 is the documented "disabled" value for min-p.
        XCTAssertEqual(try SamplingValidator.minP("0").get(), 0)
        XCTAssertEqual(try SamplingValidator.minP("0.05").get(), 0.05)
        XCTAssertEqual(try SamplingValidator.minP("1").get(), 1)
    }

    func testMinPRejectsAboveOne() {
        XCTAssertThrowsError(try SamplingValidator.minP("1.5").get())
    }

    func testMinPRejectsNegative() {
        XCTAssertThrowsError(try SamplingValidator.minP("-0.01").get())
    }

    // MARK: - Top K (Z+)

    func testTopKEmptyIsNil() throws {
        XCTAssertNil(try SamplingValidator.topK("").get())
    }

    func testTopKAcceptsPositiveInteger() throws {
        XCTAssertEqual(try SamplingValidator.topK("1").get(), 1)
        XCTAssertEqual(try SamplingValidator.topK("20").get(), 20)
    }

    func testTopKRejectsZero() {
        XCTAssertThrowsError(try SamplingValidator.topK("0").get())
    }

    func testTopKRejectsNegative() {
        XCTAssertThrowsError(try SamplingValidator.topK("-3").get())
    }

    func testTopKRejectsFloat() {
        // 1.5 is not an integer — must reject so the int-typed patch stays honest.
        XCTAssertThrowsError(try SamplingValidator.topK("1.5").get())
    }

    func testTopKRejectsNonNumeric() {
        XCTAssertThrowsError(try SamplingValidator.topK("twenty").get())
    }

    // MARK: - Penalty (-2 to 2)

    func testPenaltyEmptyIsNil() throws {
        XCTAssertNil(try SamplingValidator.penalty("", name: "Repetition Penalty").get())
    }

    func testPenaltyAcceptsBoundaries() throws {
        XCTAssertEqual(try SamplingValidator.penalty("-2", name: "Repetition Penalty").get(), -2)
        XCTAssertEqual(try SamplingValidator.penalty("0", name: "Repetition Penalty").get(), 0)
        XCTAssertEqual(try SamplingValidator.penalty("2", name: "Repetition Penalty").get(), 2)
        XCTAssertEqual(try SamplingValidator.penalty("1.0", name: "Presence Penalty").get(), 1.0)
    }

    func testPenaltyRejectsOutOfRange() {
        XCTAssertThrowsError(try SamplingValidator.penalty("-2.01", name: "Repetition Penalty").get())
        XCTAssertThrowsError(try SamplingValidator.penalty("2.01", name: "Presence Penalty").get())
    }

    func testPenaltyMessageMentionsFieldName() {
        // The UI surfaces this string verbatim — keep the field name in the message
        // so the user knows which row to fix.
        if case .failure(let err) = SamplingValidator.penalty("9", name: "Repetition Penalty") {
            XCTAssertTrue(err.message.contains("Repetition Penalty"), "message was: \(err.message)")
        } else {
            XCTFail("expected failure")
        }
    }
}
