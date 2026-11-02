// Compact counts for the window and the menu bar. Chinese, Japanese and Korean
// use ten-thousand units (万/亿/万亿, 萬/億/兆, 万/億/兆, 만/억/조) like the web UI;
// other languages use K/M/B/T.

import Foundation

enum CountFormat {

    private static let si = Locale(identifier: "en_US")

    static func compact(_ value: Int?, locale: Locale = .current) -> String {
        guard let value else { return "—" }
        return compact(value, locale: locale)
    }

    /// 12,345 -> "1.2万" in Chinese, Japanese and Korean, "12.3K" elsewhere.
    static func compact(_ value: Int, locale: Locale = .current) -> String {
        guard let units = eastAsianUnits(locale) else {
            guard Double(abs(value)) >= 1_000 else { return exact(value, locale: locale) }
            return value.formatted(.number
                .notation(.compactName)
                .precision(.fractionLength(0...1))
                .locale(si))
        }
        guard Double(abs(value)) >= 10_000 else { return exact(value, locale: locale) }
        return eastAsian(value, units: units, locale: locale)
    }

    private static func exact(_ value: Int, locale: Locale) -> String {
        value.formatted(.number.grouping(.automatic).locale(isChinese(locale) ? locale : si))
    }

    /// The ten-thousand ladder for the languages the web console groups the
    /// same way, or nil when the language uses K/M/B/T.
    private static func eastAsianUnits(_ locale: Locale) -> [String]? {
        switch locale.language.languageCode?.identifier {
        case "zh":
            let traditional = locale.language.script?.identifier == "Hant"
            return traditional ? ["萬", "億", "兆"] : ["万", "亿", "万亿"]
        case "ja":
            return ["万", "億", "兆"]
        case "ko":
            return ["만", "억", "조"]
        default:
            return nil
        }
    }

    // Compact notation cannot group the mantissa (1,234.5万), so the ladder is built here.
    private static func eastAsian(_ value: Int, units: [String], locale: Locale) -> String {
        let scales: [Double] = [1e4, 1e8, 1e12]
        let magnitude = Double(abs(value))
        var rung = scales.lastIndex { magnitude >= $0 } ?? 0
        // The unit follows the printed mantissa: 10,000.0万 reads 1.0亿.
        if oneDecimal(magnitude / scales[rung]) >= 10_000, rung < scales.count - 1 {
            rung += 1
        }
        return (Double(value) / scales[rung]).formatted(.number
            .precision(.fractionLength(1))
            .grouping(.automatic)
            .locale(locale)) + units[rung]
    }

    private static func oneDecimal(_ value: Double) -> Double {
        Double(value.formatted(.number
            .precision(.fractionLength(0...1))
            .grouping(.never)
            .locale(Locale(identifier: "en_US_POSIX")))) ?? value
    }

    private static func isChinese(_ locale: Locale) -> Bool {
        locale.language.languageCode?.identifier == "zh"
    }
}
