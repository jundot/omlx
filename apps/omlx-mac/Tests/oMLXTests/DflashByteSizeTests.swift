// DFlash's in-memory cache size lives in two units across the stack:
// the server stores raw bytes, the UI lets the user type GiB. These
// tests pin the conversion at the boundaries we care about so the
// editor doesn't silently submit a value that's three orders of
// magnitude off.

import XCTest
@testable import oMLX

final class DflashByteSizeTests: XCTestCase {

    func testNilBytesYieldsNilGib() {
        XCTAssertNil(DflashByteSize.bytesToGib(nil))
    }

    func testZeroBytesYieldsNilGib() {
        // 0 is the server's "unset" sentinel — the editor should fall
        // back to the placeholder, not display "0 GiB".
        XCTAssertNil(DflashByteSize.bytesToGib(0))
    }

    func testOneGibRoundTrip() {
        let bytes = DflashByteSize.gibToBytes(1)
        XCTAssertEqual(bytes, 1 * 1024 * 1024 * 1024)
        XCTAssertEqual(DflashByteSize.bytesToGib(bytes), 1)
    }

    func testEightGibRoundTrip() {
        let bytes = DflashByteSize.gibToBytes(8)
        XCTAssertEqual(bytes, 8 * 1024 * 1024 * 1024)
        XCTAssertEqual(DflashByteSize.bytesToGib(bytes), 8)
    }

    func testLargeValueRoundTrip() {
        let bytes = DflashByteSize.gibToBytes(64)
        XCTAssertEqual(bytes, 64 * 1024 * 1024 * 1024)
        XCTAssertEqual(DflashByteSize.bytesToGib(bytes), 64)
    }

    func testGibBelowOneIsClampedToOne() {
        // The HTML input has min="1"; the helper enforces the floor so
        // a stray 0 from the editor doesn't disable the cache by accident.
        XCTAssertEqual(DflashByteSize.gibToBytes(0), 1 * 1024 * 1024 * 1024)
        XCTAssertEqual(DflashByteSize.gibToBytes(-5), 1 * 1024 * 1024 * 1024)
    }

    func testNilGibYieldsNilBytes() {
        XCTAssertNil(DflashByteSize.gibToBytes(nil))
    }

    func testFractionalBytesRoundToNearestGib() {
        // 1.4 GiB → 1, 1.6 GiB → 2 (typical rounding).
        let lowFraction = Int64(1.4 * Double(DflashByteSize.bytesPerGiB))
        let highFraction = Int64(1.6 * Double(DflashByteSize.bytesPerGiB))
        XCTAssertEqual(DflashByteSize.bytesToGib(lowFraction), 1)
        XCTAssertEqual(DflashByteSize.bytesToGib(highFraction), 2)
    }
}
