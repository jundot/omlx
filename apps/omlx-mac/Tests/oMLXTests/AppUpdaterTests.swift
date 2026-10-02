import XCTest
@testable import oMLX

@MainActor
final class AppUpdaterTests: XCTestCase {
    private enum TestError: Error, Equatable {
        case detachFailed
        case removalFailed
    }

    func testReadyIsNotifiedOnlyAfterMountedResourcesAreReleased() throws {
        var events: [String] = []

        try AppUpdater.finishStagedUpdate(
            detach: { events.append("detach") },
            removeTemporaryFiles: { events.append("remove temporary files") },
            notifyReady: { events.append("ready") }
        )

        XCTAssertEqual(events, [
            "detach",
            "remove temporary files",
            "ready",
        ])
    }

    func testDetachFailurePreventsTemporaryRemovalAndReadyNotification() {
        var events: [String] = []

        XCTAssertThrowsError(
            try AppUpdater.finishStagedUpdate(
                detach: {
                    events.append("detach")
                    throw TestError.detachFailed
                },
                removeTemporaryFiles: { events.append("remove temporary files") },
                notifyReady: { events.append("ready") }
            )
        ) { error in
            XCTAssertEqual(error as? TestError, .detachFailed)
        }

        XCTAssertEqual(events, ["detach"])
    }

    func testTemporaryRemovalFailurePreventsReadyNotification() {
        var events: [String] = []

        XCTAssertThrowsError(
            try AppUpdater.finishStagedUpdate(
                detach: { events.append("detach") },
                removeTemporaryFiles: {
                    events.append("remove temporary files")
                    throw TestError.removalFailed
                },
                notifyReady: { events.append("ready") }
            )
        ) { error in
            XCTAssertEqual(error as? TestError, .removalFailed)
        }

        XCTAssertEqual(events, ["detach", "remove temporary files"])
    }
}
