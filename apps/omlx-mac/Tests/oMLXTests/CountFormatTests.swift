import XCTest
@testable import oMLX

final class CountFormatTests: XCTestCase {

    private let zh = Locale(identifier: "zh-Hans")
    private let zhTW = Locale(identifier: "zh-Hant")

    func testChineseLadder() {
        XCTAssertEqual(CountFormat.compact(9_999, locale: zh), "9,999")
        XCTAssertEqual(CountFormat.compact(10_000, locale: zh), "1万")
        XCTAssertEqual(CountFormat.compact(12_345_678, locale: zh), "1,234.6万")
        XCTAssertEqual(CountFormat.compact(123_456_789, locale: zh), "1.2亿")
        XCTAssertEqual(CountFormat.compact(1_200_000_000_000, locale: zh), "1.2万亿")
        XCTAssertEqual(CountFormat.compact(1_200_000_000_000, locale: zhTW), "1.2兆")
        XCTAssertEqual(CountFormat.compact(12_345, locale: zhTW), "1.2萬")
    }

    func testChineseUnitFollowsRoundedMantissa() {
        XCTAssertEqual(CountFormat.compact(99_999_999, locale: zh), "1亿")
        XCTAssertEqual(CountFormat.compact(999_999_999_999, locale: zhTW), "1兆")
        XCTAssertEqual(CountFormat.compact(99_990_000, locale: zh), "9,999万")
    }

    func testOtherLanguagesUseEnglishLadder() {
        for id in ["en-US", "ko-KR", "ja-JP", "ru-RU"] {
            let locale = Locale(identifier: id)
            XCTAssertEqual(CountFormat.compact(999, locale: locale), "999", id)
            XCTAssertEqual(CountFormat.compact(12_345, locale: locale), "12.3K", id)
            XCTAssertEqual(CountFormat.compact(12_345_000, locale: locale), "12.3M", id)
        }
        XCTAssertEqual(CountFormat.compact(nil), "—")
    }
}
