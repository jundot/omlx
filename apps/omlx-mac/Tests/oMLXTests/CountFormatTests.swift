import XCTest
@testable import oMLX

final class CountFormatTests: XCTestCase {

    private let zh = Locale(identifier: "zh-Hans")
    private let zhTW = Locale(identifier: "zh-Hant")

    func testChineseLadder() {
        XCTAssertEqual(CountFormat.compact(9_999, locale: zh), "9,999")
        XCTAssertEqual(CountFormat.compact(10_000, locale: zh), "1.0万")
        XCTAssertEqual(CountFormat.compact(12_345_678, locale: zh), "1,234.6万")
        XCTAssertEqual(CountFormat.compact(123_456_789, locale: zh), "1.2亿")
        XCTAssertEqual(CountFormat.compact(1_200_000_000_000, locale: zh), "1.2万亿")
        XCTAssertEqual(CountFormat.compact(1_200_000_000_000, locale: zhTW), "1.2兆")
        XCTAssertEqual(CountFormat.compact(12_345, locale: zhTW), "1.2萬")
    }

    func testChineseUnitFollowsRoundedMantissa() {
        XCTAssertEqual(CountFormat.compact(99_999_999, locale: zh), "1.0亿")
        XCTAssertEqual(CountFormat.compact(999_999_999_999, locale: zhTW), "1.0兆")
        XCTAssertEqual(CountFormat.compact(99_990_000, locale: zh), "9,999.0万")
    }

    func testOtherLanguagesUseEnglishLadder() {
        for id in ["en-US", "ru-RU", "fr-FR"] {
            let locale = Locale(identifier: id)
            XCTAssertEqual(CountFormat.compact(999, locale: locale), "999", id)
            XCTAssertEqual(CountFormat.compact(12_345, locale: locale), "12.3K", id)
            XCTAssertEqual(CountFormat.compact(12_345_000, locale: locale), "12.3M", id)
        }
        XCTAssertEqual(CountFormat.compact(nil), "—")
    }

    func testJapaneseAndKoreanUseTenThousandUnits() {
        let ladders: [(id: String, units: (String, String, String))] = [
            ("ja-JP", ("万", "億", "兆")),
            ("ko-KR", ("만", "억", "조")),
        ]
        for (id, units) in ladders {
            let locale = Locale(identifier: id)
            XCTAssertEqual(CountFormat.compact(9_999, locale: locale), "9,999", id)
            XCTAssertEqual(CountFormat.compact(12_345, locale: locale), "1.2\(units.0)", id)
            XCTAssertEqual(CountFormat.compact(123_456_789, locale: locale), "1.2\(units.1)", id)
            XCTAssertEqual(
                CountFormat.compact(1_200_000_000_000, locale: locale), "1.2\(units.2)", id)
            XCTAssertEqual(CountFormat.compact(99_999_999, locale: locale), "1.0\(units.1)", id)
        }
    }

    func testNumberScaleSwitchesBetweenExactDigitsAndCompactUnits() {
        let key = NumberScale.storageKey
        let previous = UserDefaults.standard.object(forKey: key)
        defer {
            if let previous {
                UserDefaults.standard.set(previous, forKey: key)
            } else {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }

        UserDefaults.standard.removeObject(forKey: key)
        XCTAssertEqual(CountFormat.count(104_166_321, locale: zh), "1.0亿")

        UserDefaults.standard.set(NumberScale.exact.rawValue, forKey: key)
        XCTAssertEqual(CountFormat.count(104_166_321, locale: zh), "104,166,321")
        XCTAssertEqual(CountFormat.count(12_345, locale: Locale(identifier: "en_US")), "12,345")
        XCTAssertEqual(CountFormat.count(nil, locale: zh), "—")
    }
}
