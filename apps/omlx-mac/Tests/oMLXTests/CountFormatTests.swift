// The app has one count formatter (Sources/Theme/CountFormat.swift) and these
// tests pin its output per language: Chinese reads its own ten-thousand ladder,
// its mantissa grouped, and every other language the English SI ladder.
//
// The Chinese vectors are the ones the web console pins, so the two surfaces are
// held to one table on the Chinese ladder.

import XCTest
@testable import oMLX

final class CountFormatTests: XCTestCase {

    private let zh = Locale(identifier: "zh-Hans")
    private let zhTW = Locale(identifier: "zh-Hant")
    private let en = Locale(identifier: "en-US")

    // MARK: - The Chinese ladder

    func testChineseCountsUseWanAndYiAndNeverKM() {
        XCTAssertEqual(CountFormat.compact(0, locale: zh), "0")
        XCTAssertEqual(CountFormat.compact(999, locale: zh), "999")
        // Ten-thousands are the first rung: below it the exact figure stands.
        XCTAssertEqual(CountFormat.compact(9_999, locale: zh), "9,999")
        XCTAssertEqual(CountFormat.compact(10_000, locale: zh), "1万")
        XCTAssertEqual(CountFormat.compact(12_345, locale: zh), "1.2万")
        XCTAssertEqual(CountFormat.compact(19_290, locale: zh), "1.9万")
        XCTAssertEqual(CountFormat.compact(1_070_000, locale: zh), "107万")
        // The mantissa is grouped like any other count.
        XCTAssertEqual(CountFormat.compact(12_345_000, locale: zh), "1,234.5万")
        XCTAssertEqual(CountFormat.compact(12_345_678, locale: zh), "1,234.6万")
        XCTAssertEqual(CountFormat.compact(123_456_789, locale: zh), "1.2亿")
        // The top of the ladder: 万亿, the rung Intl engines disagree about, so
        // it is spelled out here.
        XCTAssertEqual(CountFormat.compact(560_000_000_000, locale: zh), "5,600亿")
        XCTAssertEqual(CountFormat.compact(1_200_000_000_000, locale: zh), "1.2万亿")
        XCTAssertEqual(CountFormat.compact(34_000_000_000_000, locale: zh), "34万亿")
        XCTAssertEqual(CountFormat.compact(-4_200_000_000_000, locale: zh), "-4.2万亿")
    }

    /// The unit follows what the mantissa prints, not the raw magnitude: one
    /// that rounds up to 10,000 has left its unit behind. These are the console
    /// formatter's own vectors, so both surfaces stay on one table.
    func testAMantissaThatRoundsUpStepsToTheNextUnit() {
        XCTAssertEqual(CountFormat.compact(99_999_999, locale: zh), "1亿")
        XCTAssertEqual(CountFormat.compact(99_999_999, locale: zhTW), "1億")
        XCTAssertEqual(CountFormat.compact(999_999_999_999, locale: zh), "1万亿")
        XCTAssertEqual(CountFormat.compact(999_999_999_999, locale: zhTW), "1兆")
        // Just below the roll-over the mantissa still has room to round down.
        XCTAssertEqual(CountFormat.compact(99_990_000, locale: zh), "9,999万")
        XCTAssertEqual(CountFormat.compact(9_999_999, locale: zh), "1,000万")
    }

    func testTraditionalChineseKeepsItsOwnCharacters() {
        XCTAssertEqual(CountFormat.compact(10_000, locale: zhTW), "1萬")

        XCTAssertEqual(CountFormat.compact(12_345_678, locale: zhTW), "1,234.6萬")
        XCTAssertEqual(CountFormat.compact(1_234_567, locale: zhTW), "123.5萬")
        XCTAssertEqual(CountFormat.compact(120_000_000, locale: zhTW), "1.2億")
        // Taiwan stops the ladder at 兆 rather than 万亿.
        XCTAssertEqual(CountFormat.compact(1_200_000_000_000, locale: zhTW), "1.2兆")
        XCTAssertEqual(CountFormat.compact(34_000_000_000_000, locale: zhTW), "34兆")
    }

    // MARK: - The SI ladder

    func testEnglishCountsUseKMBT() {
        XCTAssertEqual(CountFormat.compact(999, locale: en), "999")
        XCTAssertEqual(CountFormat.compact(1_000, locale: en), "1K")
        XCTAssertEqual(CountFormat.compact(12_345, locale: en), "12.3K")
        XCTAssertEqual(CountFormat.compact(1_234_567, locale: en), "1.2M")
        XCTAssertEqual(CountFormat.compact(560_000_000_000, locale: en), "560B")
        XCTAssertEqual(CountFormat.compact(4_000_000_000_000, locale: en), "4T")
    }

    func testEveryOtherLanguageStillReadsTheEnglishSILadder() {
        let others = [Locale(identifier: "ja-JP"), Locale(identifier: "ko-KR"),
                      Locale(identifier: "de-DE"), Locale(identifier: "ru-RU")]
        for locale in others {
            XCTAssertEqual(CountFormat.compact(12_345, locale: locale), "12.3K", locale.identifier)
            XCTAssertEqual(CountFormat.compact(1_234_567, locale: locale), "1.2M", locale.identifier)
            XCTAssertEqual(CountFormat.compact(560_000_000_000, locale: locale), "560B", locale.identifier)
            XCTAssertEqual(CountFormat.compact(34_000_000_000_000, locale: locale), "34T", locale.identifier)
            XCTAssertEqual(CountFormat.compact(12_000, locale: locale), "12K", locale.identifier)
            // The Chinese mantissa grouping stays inside Chinese.
            XCTAssertEqual(CountFormat.compact(12_345_000, locale: locale), "12.3M", locale.identifier)
        }
    }

    func testAWholeUnitDropsItsTrailingZero() {
        XCTAssertEqual(CountFormat.compact(4_000_000_000_000, locale: en), "4T")
        XCTAssertEqual(CountFormat.compact(560_000_000_000, locale: en), "560B")
        XCTAssertEqual(CountFormat.compact(1_200_000_000_000, locale: zh), "1.2万亿")
        XCTAssertEqual(CountFormat.compact(20_000, locale: zh), "2万")
        XCTAssertEqual(CountFormat.compact(560_000_000_000, locale: zh), "5,600亿")
    }

    // MARK: - The exact figure

    func testTheExactFigureIsGroupedInEnglishInEveryLanguage() {
        XCTAssertEqual(CountFormat.exact(1_234_567, locale: en), "1,234,567")
        XCTAssertEqual(CountFormat.exact(1_234_567, locale: zh), "1,234,567")
        XCTAssertEqual(CountFormat.exact(1_234_567, locale: Locale(identifier: "de-DE")), "1,234,567")
    }

    // MARK: - The optional shape the menu bar uses

    func testNilReadsAsAnEmDash() {
        XCTAssertEqual(CountFormat.compact(nil), "—")
        XCTAssertEqual(CountFormat.compact(Int?.none, locale: zh), "—")
    }

    func testNoCountEverReadsAsZero() {
        // A fresh stats payload holds zeros; a count of zero is a real count,
        // so it prints as one (unlike a measured speed, which prints an em dash).
        XCTAssertEqual(CountFormat.compact(0, locale: en), "0")
        XCTAssertEqual(CountFormat.compact(0, locale: zh), "0")
    }
}
