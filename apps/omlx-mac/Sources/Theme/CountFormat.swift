// CountFormat — one place where a count becomes text, for the window and the
// menu bar both.
//
// Follows the ladder the web console's own formatter defines — so a Chinese
// count reads the same on both surfaces:
//
//   • below the first rung a count stays exact and grouped: 9,999;
//   • at or above it the number takes the language's unit — 万 / 亿 / 万亿 in
//     Simplified Chinese, 萬 / 億 / 兆 in Traditional, K / M / B / T for every
//     other language — one decimal at most, a trailing ".0" dropped.
//
// Chinese is the only language with a ladder of its own: it counts in
// ten-thousands, so it keeps the exact figure up to 9,999 and reads
// 万 / 亿 / 万亿 above that, its mantissa grouped like any other count
// (1,234.5万). Every other language reads the English SI ladder this app has
// always printed, whatever language it is showing.
//
// Byte sizes are a different family (1024-scaled, `formatByteCount`), and model
// parameter counts are a unit rather than a count — an 8B model is 8B in every
// language — so neither comes through here.

import Foundation

enum CountFormat {

    /// The ladder every language but Chinese reads, grouping included: a count
    /// is spelled out in English, never in the reader's own compact names
    /// ("1,2 Mio.", "12,3 тыс.", "1.9万").
    private static let si = Locale(identifier: "en_US")

    /// 12,345 → "1.2万" (zh-Hans) / "12.3K" (any other language). `nil` reads as
    /// an em dash.
    static func compact(_ value: Int?, locale: Locale = .current) -> String {
        guard let value else { return "—" }
        return compact(value, locale: locale)
    }

    /// 12,345 → "1.2万" (zh-Hans) / "12.3K" (any other language).
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

    /// The figure itself, grouped: 1,234,567. For titles and tooltips that must
    /// not round.
    static func exact(_ value: Int, locale: Locale = .current) -> String {
        value.formatted(.number.grouping(.automatic).locale(isChinese(locale) ? locale : si))
    }

    /// The Chinese ladder, spelled out here because the platform's compact
    /// notation cannot group the mantissa it produces: 12,345,000 → "1,234.5万".
    private static func chinese(_ value: Int, locale: Locale) -> String {
        let traditional = locale.language.script?.identifier == "Hant"
        let rungs: [(scale: Double, unit: String)] = [
            (1e4, traditional ? "萬" : "万"),
            (1e8, traditional ? "億" : "亿"),
            (1e12, traditional ? "兆" : "万亿"),
        ]
        let magnitude = Double(abs(value))
        var rung = rungs.lastIndex { magnitude >= $0.scale } ?? 0
        // The unit follows the printed mantissa rather than the raw magnitude:
        // one that rounds up to 10,000 has left its unit behind — 10,000万 is
        // 1亿 — so 99,999,999 reads 1亿, the vector the console's own
        // formatter pins for both surfaces.
        if oneDecimal(magnitude / rungs[rung].scale) >= 10_000, rung < rungs.count - 1 {
            rung += 1
        }
        return (Double(value) / rungs[rung].scale).formatted(.number
            .precision(.fractionLength(0...1))
            .grouping(.automatic)
            .locale(locale)) + rungs[rung].unit
    }

    /// What one decimal place actually prints, read back. The rung decision has
    /// to follow the digits on screen, not the quotient they came from.
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
