// PR 14 — DFlash in-memory cache size conversion.
//
// The server's `dflash_in_memory_cache_max_bytes` is an Int64 byte count.
// The HTML editor exposes it as a GiB value (the JS does the GiB↔bytes
// dance inline). This helper centralizes the conversion so the Swift VM
// and the test suite share one source of truth.

import Foundation

enum DflashByteSize {
    /// 1 GiB = 1024^3 bytes.
    static let bytesPerGiB: Int64 = 1024 * 1024 * 1024

    /// Server bytes → editor GiB. Returns `nil` when the server value
    /// is absent so the row's text field shows its placeholder ("8")
    /// rather than "0".
    static func bytesToGib(_ bytes: Int64?) -> Int? {
        guard let bytes, bytes > 0 else { return nil }
        // `round` matches the JS: `Math.round(bytes / 1024^3)`.
        let g = Double(bytes) / Double(bytesPerGiB)
        return Int(g.rounded())
    }

    /// Editor GiB → server bytes. Clamps to ≥ 1 GiB to match the input's
    /// `min="1"` attribute; the HTML editor sends 8 GiB as the default
    /// when the user enables the toggle without picking a size.
    static func gibToBytes(_ gib: Int?) -> Int64? {
        guard let gib else { return nil }
        let clamped = max(1, gib)
        return Int64(clamped) * bytesPerGiB
    }
}
