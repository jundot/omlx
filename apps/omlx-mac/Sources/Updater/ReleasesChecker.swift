// GitHub Releases poller — replaces the Sparkle appcast wiring.
//
// Mirrors the PyObjC menubar app's update model: GitHub Releases is the
// single source of truth for distribution. No appcast XML, no EdDSA key
// management on the maintainer side. Pulls a page of releases, picks the
// latest stable PEP 440 tag, and selects the DMG asset whose filename
// embeds the current macOS major version (e.g. `-macos15-` or `-macos26-`).
//
// Channel handling: GitHub Releases marks pre-1.0 / rc / beta tags via
// the `prerelease` flag on each release. `UpdateChannel.beta` opens up
// pre-releases; `.nightly` opens up everything including drafts.
// Filtering happens client-side via PEP 440 version parsing.

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

    enum AssetKeys: String, CodingKey {
        case name
        case browserDownloadURL = "browser_download_url"
        case size
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
    /// drafts + prereleases; beta accepts prereleases; nightly accepts
    /// drafts too. Ties broken by PEP 440 version order.
    static func selectLatest(
        _ releases: [GitHubRelease],
        channel: UpdateChannel
    ) -> GitHubRelease? {
        let allowed = releases.filter { r in
            if r.draft { return channel == .nightly }
            if r.prerelease { return channel != .stable }
            return true
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
        let lhs = parseVersionComponents(a)
        let rhs = parseVersionComponents(b)
        let count = max(lhs.count, rhs.count)
        for i in 0..<count {
            let lv = i < lhs.count ? lhs[i] : 0
            let rv = i < rhs.count ? rhs[i] : 0
            if lv < rv { return .orderedAscending }
            if lv > rv { return .orderedDescending }
        }
        return .orderedSame
    }

    /// Extracts the leading numeric components of a PEP 440 version. Drops
    /// `rc`, `dev`, `post`, build metadata. Good enough for ordering oMLX's
    /// own tags; the prerelease distinction is already handled upstream
    /// through GitHub's `prerelease` flag.
    private static func parseVersionComponents(_ version: String) -> [Int] {
        let trimmed = version.split(whereSeparator: { !$0.isNumber && $0 != "." })
            .first.map(String.init) ?? version
        return trimmed.split(separator: ".").compactMap { Int($0) }
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
