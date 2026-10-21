// Compact counts for the window and the menu bar. Chinese uses 万/亿/万亿
// (萬/億/兆 in Traditional) like the web UI; other languages use K/M/B/T.

import Foundation

enum CountFormat {

    private static let si = Locale(identifier: "en_US")

    static func compact(_ value: Int?, locale: Locale = .current) -> String {
        guard let value else { return "—" }
        return compact(value, locale: locale)
    }

    /// 12,345 -> "1.2万" in Chinese, "12.3K" in other languages.
    static func compact(_ value: Int, locale: Locale = .current) -> String {
        guard isChinese(locale) else {
            guard Double(abs(value)) >= 1_000 else { return exact(value, locale: locale) }
            return value.formatted(.number
                .notation(.compactName)
                .precision(.fractionLength(0...1))
                .locale(si))
        }
        guard Double(abs(value)) >= 10_000 else { return exact(value, locale: locale) }
        return chinese(value, locale: locale)
    }

    private static func exact(_ value: Int, locale: Locale) -> String {
        value.formatted(.number.grouping(.automatic).locale(isChinese(locale) ? locale : si))
    }

    // Compact notation cannot group the mantissa (1,234.5万), so the ladder is built here.
    private static func chinese(_ value: Int, locale: Locale) -> String {
        let traditional = locale.language.script?.identifier == "Hant"
        let rungs: [(scale: Double, unit: String)] = [
            (1e4, traditional ? "萬" : "万"),
            (1e8, traditional ? "億" : "亿"),
            (1e12, traditional ? "兆" : "万亿"),
        ]
        let magnitude = Double(abs(value))
        var rung = rungs.lastIndex { magnitude >= $0.scale } ?? 0
        // The unit follows the printed mantissa: 10,000万 reads 1亿.
        if oneDecimal(magnitude / rungs[rung].scale) >= 10_000, rung < rungs.count - 1 {
            rung += 1
        }
        return (Double(value) / rungs[rung].scale).formatted(.number
            .precision(.fractionLength(0...1))
            .grouping(.automatic)
            .locale(locale)) + rungs[rung].unit
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
