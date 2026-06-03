// GitHub Releases poller — replaces the Sparkle appcast wiring.
//
// Mirrors the PyObjC menubar app's update model: GitHub Releases is the
// single source of truth for distribution. No appcast XML, no EdDSA key
// management on the maintainer side. Pulls a page of releases, picks the
// latest stable PEP 440 tag, and selects the DMG asset whose filename
// embeds the current macOS major version (e.g. `-macos15-` or `-macos26-`).
//
// Channel handling: Stable accepts final tags only, Release Candidate also
// accepts rc tags, and Dev accepts dev/pre-release tags. GitHub's
// `prerelease` flag is treated as advisory because it can be set
// incorrectly; tag strings are checked client-side too.

import Foundation

struct GitHubRelease: Decodable, Sendable {
    let tagName: String
    let name: String?
    let body: String?
    let htmlURL: URL
    let prerelease: Bool
    let draft: Bool
    let assets: [Asset]

    struct Asset: Decodable, Sendable {
        let name: String
        let browserDownloadURL: URL
        let size: Int64

        enum CodingKeys: String, CodingKey {
            case name
            case browserDownloadURL = "browser_download_url"
            case size
        }
    }

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case name
        case body
        case htmlURL = "html_url"
        case prerelease
        case draft
        case assets
    }

}

struct AvailableRelease: Sendable, Equatable {
    let version: String
    let notes: String
    let htmlURL: URL
    let dmgURL: URL?
    let dmgSize: Int64?
}

enum ReleasesError: Error, CustomStringConvertible {
    case httpStatus(Int)
    case decode(String)
    case noMatchingDMG

    var description: String {
        switch self {
        case .httpStatus(let code): return "GitHub returned HTTP \(code)"
        case .decode(let msg): return "Failed to parse releases payload: \(msg)"
        case .noMatchingDMG: return "No matching DMG asset for this macOS version"
        }
    }
}

enum ReleasesChecker {
    static let releasesURL = URL(string: "https://api.github.com/repos/jundot/omlx/releases?per_page=20")!

    /// Fetches recent releases and picks the newest one that beats `currentVersion`
    /// under the given channel. Returns `nil` if we're already up to date.
    static func check(
        currentVersion: String,
        channel: UpdateChannel,
        session: URLSession = .shared
    ) async throws -> AvailableRelease? {
        var request = URLRequest(url: releasesURL, timeoutInterval: 10)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ReleasesError.decode("response is not HTTP")
        }
        guard http.statusCode == 200 else {
            throw ReleasesError.httpStatus(http.statusCode)
        }

        let releases: [GitHubRelease]
        do {
            releases = try JSONDecoder().decode([GitHubRelease].self, from: data)
        } catch {
            throw ReleasesError.decode(String(describing: error))
        }

        guard let best = selectLatest(releases, channel: channel) else { return nil }

        let latest = best.tagName.trimmingPrefix("v").trimmingPrefix("V")
        let latestStr = String(latest)
        guard isNewer(latestStr, than: currentVersion) else { return nil }

        let dmg = findMatchingDMG(assets: best.assets)
        return AvailableRelease(
            version: latestStr,
            notes: best.body ?? "",
            htmlURL: best.htmlURL,
            dmgURL: dmg?.browserDownloadURL,
            dmgSize: dmg?.size
        )
    }

    /// Pick the latest release allowed by the channel. Stable excludes
    /// drafts and prerelease tags, Release Candidate accepts rc tags, and
    /// Dev accepts all non-draft tags. Ties are broken by version order.
    static func selectLatest(
        _ releases: [GitHubRelease],
        channel: UpdateChannel
    ) -> GitHubRelease? {
        let allowed = releases.filter { r in
            let version = String(r.tagName.trimmingPrefix("v").trimmingPrefix("V"))
            if r.draft { return false }
            switch channel {
            case .stable:
                return !r.prerelease && !isPrereleaseVersion(version)
            case .releaseCandidate:
                return !isDevVersion(version) && !isAlphaBetaVersion(version)
            case .dev:
                return true
            }
        }
        return allowed.max { lhs, rhs in
            let l = String(lhs.tagName.trimmingPrefix("v").trimmingPrefix("V"))
            let r = String(rhs.tagName.trimmingPrefix("v").trimmingPrefix("V"))
            return compareVersions(l, r) == .orderedAscending
        }
    }

    /// PEP 440-style "is A newer than B" check. Falls back to lexicographic
    /// comparison if parsing fails.
    static func isNewer(_ a: String, than b: String) -> Bool {
        compareVersions(a, b) == .orderedDescending
    }

    static func compareVersions(_ a: String, _ b: String) -> ComparisonResult {
        let lhs = parseVersion(a)
        let rhs = parseVersion(b)
        let count = max(lhs.release.count, rhs.release.count)
        for i in 0..<count {
            let lv = i < lhs.release.count ? lhs.release[i] : 0
            let rv = i < rhs.release.count ? rhs.release[i] : 0
            if lv < rv { return .orderedAscending }
            if lv > rv { return .orderedDescending }
        }
        if lhs.phaseRank < rhs.phaseRank { return .orderedAscending }
        if lhs.phaseRank > rhs.phaseRank { return .orderedDescending }
        if lhs.phaseNumber < rhs.phaseNumber { return .orderedAscending }
        if lhs.phaseNumber > rhs.phaseNumber { return .orderedDescending }
        return .orderedSame
    }

    private struct ParsedVersion {
        let release: [Int]
        let phaseRank: Int
        let phaseNumber: Int
    }

    private static func parseVersion(_ version: String) -> ParsedVersion {
        let normalized = String(version.trimmingPrefix("v").trimmingPrefix("V")).lowercased()
        let phase = parsePhase(normalized)
        return ParsedVersion(
            release: parseVersionComponents(normalized),
            phaseRank: phase.rank,
            phaseNumber: phase.number
        )
    }

    /// Extracts the leading numeric components of a PEP 440-ish version.
    private static func parseVersionComponents(_ version: String) -> [Int] {
        let trimmed = version.split(whereSeparator: { !$0.isNumber && $0 != "." })
            .first.map(String.init) ?? version
        return trimmed.split(separator: ".").compactMap { Int($0) }
    }

    /// PEP 440-style prerelease ordering for oMLX tags:
    /// dev < alpha < beta < rc < final.
    private static func parsePhase(_ version: String) -> (rank: Int, number: Int) {
        if let n = numberAfterFullMarker("dev", in: version) {
            return (0, n)
        }
        if let n = numberAfterFullMarker("alpha", in: version)
            ?? numberAfterShortMarker("a", in: version) {
            return (1, n)
        }
        if let n = numberAfterFullMarker("beta", in: version)
            ?? numberAfterShortMarker("b", in: version) {
            return (2, n)
        }
        if let n = numberAfterFullMarker("rc", in: version) {
            return (3, n)
        }
        return (4, 0)
    }

    private static func numberAfterFullMarker(_ marker: String, in version: String) -> Int? {
        guard let range = version.range(of: marker) else { return nil }
        return parseTrailingPhaseNumber(String(version[range.upperBound...]))
    }

    private static func numberAfterShortMarker(_ marker: Character, in version: String) -> Int? {
        var idx = version.startIndex
        while idx < version.endIndex {
            guard version[idx] == marker else {
                idx = version.index(after: idx)
                continue
            }
            let prev = idx == version.startIndex ? nil : version[version.index(before: idx)]
            let next = version.index(after: idx)
            let nextChar = next < version.endIndex ? version[next] : nil
            let prevOK = prev == nil || prev!.isNumber || prev == "." || prev == "-"
            let nextOK = nextChar == nil || nextChar!.isNumber || nextChar == "." || nextChar == "-"
            if prevOK && nextOK {
                return parseTrailingPhaseNumber(String(version[next...]))
            }
            idx = version.index(after: idx)
        }
        return nil
    }

    private static func parseTrailingPhaseNumber(_ suffix: String) -> Int {
        let trimmed = suffix.drop(while: { $0 == "." || $0 == "-" })
        let digits = trimmed.prefix(while: { $0.isNumber })
        return Int(digits) ?? 0
    }

    /// GitHub's `prerelease` flag is release metadata and can be set
    /// incorrectly. Stable channel should still ignore PEP 440 prerelease
    /// tags such as `0.4.0rc1`, `0.4.0.dev1`, `0.4.0b1`, and `0.4.0a1`.
    private static func isPrereleaseVersion(_ version: String) -> Bool {
        isDevVersion(version) || isRCVersion(version) || isAlphaBetaVersion(version)
    }

    private static func isDevVersion(_ version: String) -> Bool {
        version.lowercased().contains("dev")
    }

    private static func isRCVersion(_ version: String) -> Bool {
        version.lowercased().contains("rc")
    }

    private static func isAlphaBetaVersion(_ version: String) -> Bool {
        let normalized = version.lowercased()
        return normalized.range(of: #"(?:^|[.\-])a\d+"#, options: .regularExpression) != nil
            || normalized.range(of: #"(?:^|[.\-])b\d+"#, options: .regularExpression) != nil
            || normalized.contains("alpha")
            || normalized.contains("beta")
    }

    /// Pick the DMG asset whose filename embeds the current macOS major
    /// version (e.g. `-macos15-` / `-macos26-` / `-macos15_`). Falls back
    /// to the single DMG when there's only one.
    static func findMatchingDMG(assets: [GitHubRelease.Asset]) -> GitHubRelease.Asset? {
        let dmgs = assets.filter { $0.name.lowercased().hasSuffix(".dmg") }
        guard !dmgs.isEmpty else { return nil }

        let osMajor = currentMacOSMajor()
        let tag = "macos\(osMajor)"
        if let exact = dmgs.first(where: { $0.name.contains("-\(tag)-") || $0.name.contains("-\(tag)_") }) {
            return exact
        }
        return dmgs.count == 1 ? dmgs[0] : nil
    }

    /// Reads the host macOS major version (e.g. "15", "26"). Uses
    /// `ProcessInfo.operatingSystemVersion.majorVersion` which is the
    /// public, documented surface.
    static func currentMacOSMajor() -> Int {
        ProcessInfo.processInfo.operatingSystemVersion.majorVersion
    }
}

private extension Substring {
    func trimmingPrefix(_ prefix: String) -> Substring {
        if hasPrefix(prefix) { return dropFirst(prefix.count) }
        return self
    }
}

private extension String {
    func trimmingPrefix(_ prefix: String) -> Substring {
        Substring(self).trimmingPrefix(prefix)
    }
}
