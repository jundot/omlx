import XCTest
@testable import oMLX

@MainActor
final class IntegrationsScreenVMTests: XCTestCase {
    private var server: MarkItDownSettingsServer { MarkItDownSettingsURLProtocol.server }

    private func makeClient() throws -> (OMLXClient, URLSession) {
        try server.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MarkItDownSettingsURLProtocol.self]
        let session = URLSession(configuration: configuration)
        return (OMLXClient(port: 18161, session: session), session)
    }

    func testConfiguredSettingsAndOcrChoicesPreserveHiddenPreference() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations([
            "markitdown_enabled": false, "markitdown_expose_model": true,
            "markitdown_max_file_size_mb": 40, "markitdown_max_files_per_request": 9,
            "markitdown_pdf_processing_engine": "GLM-OCR"
        ])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertFalse(vm.markitdownModelVisible)
        vm.markitdownModelVisible = false
        XCTAssertTrue(vm.markitdownExposeModel)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "40")
        XCTAssertEqual(vm.markitdownMaxFilesText, "9")
        XCTAssertEqual(vm.markitdownPdfOptions.map(\.0), ["markitdown", "GLM-OCR", "DeepSeek-OCR"])
        XCTAssertNil(vm.markitdownPdfWarning)
        vm.markitdownEnabled = true
        XCTAssertTrue(vm.markitdownModelVisible)
    }

    func testMissingOcrSelectionIsRetainedWithBothWarnings() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations(["markitdown_pdf_processing_engine": "missing-ocr"])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        XCTAssertTrue(vm.markitdownOcrModelMissing)
        XCTAssertEqual(vm.markitdownPdfOptions.map(\.0), ["markitdown", "missing-ocr", "GLM-OCR", "DeepSeek-OCR"])
        let missingWarning = try XCTUnwrap(vm.markitdownPdfWarning)
        vm.markitdownPdfProcessingEngine = "markitdown"
        let warning = try XCTUnwrap(vm.markitdownPdfWarning)
        XCTAssertTrue(missingWarning.hasSuffix("\n\(warning)"))
        XCTAssertTrue(server.patches.isEmpty)
    }

    func testVisibilitySavesRefreshModelsAndSurviveRefreshFailure() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFilesText = "12"
        vm.markitdownModelVisible = true
        await vm.save(.markitdownExposeModel, client: client)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_expose_model"] as? Bool, true)

        vm.markitdownEnabled = false
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertFalse(vm.markitdownModelVisible)
        XCTAssertTrue(vm.markitdownExposeModel)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_enabled"] as? Bool, false)
        XCTAssertEqual(vm.markitdownMaxFilesText, "12")
        XCTAssertEqual(server.modelReads, 3)

        server.rejectModelReads(true)
        vm.markitdownEnabled = true
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertTrue(vm.markitdownEnabled)
        XCTAssertNotNil(vm.lastError)
    }

    func testLimitsAndEngineSaveOnlyTheirChangedFieldWhileDisabled() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations(["markitdown_enabled": false])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFileSizeText = " 40 "
        vm.markitdownMaxFilesText = "9"
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_max_file_size_mb"] as? Int, 40)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "40")
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        await vm.save(.markitdownMaxFiles, client: client)
        XCTAssertEqual(server.patches.last?["markitdown_max_files_per_request"] as? Int, 9)
        vm.markitdownPdfProcessingEngine = "GLM-OCR"
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_pdf_processing_engine"] as? String, "GLM-OCR")
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(server.patches.count, 3)
        XCTAssertEqual(server.modelReads, 1)
        XCTAssertNil(vm.lastError)
    }

    func testInvalidLimitsKeepDraftsAndSendNoRequests() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        for text in ["0", "1.5", "abc"] {
            vm.markitdownMaxFileSizeText = text
            await vm.save(.markitdownMaxFileSize, client: client)
            XCTAssertNotNil(vm.lastError, text)
            XCTAssertEqual(vm.markitdownMaxFileSizeText, text)
        }
        XCTAssertTrue(server.patches.isEmpty)
    }

    func testFailedSavesRestoreSwitchesAndPickerAndKeepNumericDraftsForRetry() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations(["markitdown_expose_model": true, "markitdown_pdf_processing_engine": "GLM-OCR"])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        server.rejectSaves(true)
        vm.markitdownEnabled = false
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertTrue(vm.markitdownEnabled)
        XCTAssertNotNil(vm.lastError)
        vm.markitdownModelVisible = false
        await vm.save(.markitdownExposeModel, client: client)
        XCTAssertTrue(vm.markitdownModelVisible)
        vm.markitdownPdfProcessingEngine = "markitdown"
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(vm.markitdownPdfProcessingEngine, "GLM-OCR")
        vm.markitdownMaxFileSizeText = "60"
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "60")
        XCTAssertTrue(vm.savingMarkitdownFields.isEmpty)

        server.rejectSaves(false)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.last?["markitdown_max_file_size_mb"] as? Int, 60)
        XCTAssertNil(vm.lastError)
    }

    func testDuplicateSubmitAndBlurSavesAreSuppressedWhileInFlight() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFileSizeText = "30"
        let started = XCTestExpectation(description: "Numeric save started")
        server.holdNextSave(started: started)
        let submit = Task { await vm.save(.markitdownMaxFileSize, client: client) }
        let result = await XCTWaiter.fulfillment(of: [started], timeout: 3)
        XCTAssertEqual(result, .completed)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        server.finishHeldSave()
        await submit.value
        XCTAssertTrue(vm.savingMarkitdownFields.isEmpty)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
    }
}

private final class MarkItDownSettingsServer: @unchecked Sendable {
    private let lock = NSLock()
    private var settings: [String: Any] = [:]
    private var recordedPatches: [[String: Any]] = []
    private var modelReadCount = 0
    private var savesRejected = false
    private var modelReadsRejected = false
    private var saveStarted: XCTestExpectation?
    private var heldSave: (@Sendable () -> Void)?

    var patches: [[String: Any]] { lock.withLock { recordedPatches } }
    var modelReads: Int { lock.withLock { modelReadCount } }

    func reset() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/global-settings.json")
        let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        lock.withLock {
            settings = fixture
            recordedPatches = []
            modelReadCount = 0
            savesRejected = false
            modelReadsRejected = false
            saveStarted = nil
            heldSave = nil
        }
    }

    func setIntegrations(_ values: [String: Any]) {
        lock.withLock {
            var integrations = settings["integrations"] as! [String: Any]
            integrations.merge(values) { _, new in new }
            settings["integrations"] = integrations
        }
    }

    func rejectSaves(_ reject: Bool) { lock.withLock { savesRejected = reject } }
    func rejectModelReads(_ reject: Bool) { lock.withLock { modelReadsRejected = reject } }
    func holdNextSave(started: XCTestExpectation) { lock.withLock { saveStarted = started } }

    func finishHeldSave() {
        let completion = lock.withLock {
            let completion = heldSave
            heldSave = nil
            return completion
        }
        completion?()
    }

    func respond(to request: URLRequest, completion: @escaping @Sendable (Int, Data) -> Void) throws {
        lock.lock()
        defer { lock.unlock() }
        var status = 200
        var body: [String: Any]
        switch (request.httpMethod, request.url?.path) {
        case ("GET", "/admin/api/global-settings"):
            body = settings
        case ("GET", "/admin/api/models"):
            modelReadCount += 1
            if modelReadsRejected {
                status = 503
                body = ["detail": "Model list unavailable"]
            } else {
                body = ["models": [
                    model("GLM-OCR", configType: "GLM_OCR"),
                    model("DeepSeek-OCR", configType: "deepseek_ocr2"),
                    model("ocr-in-name-only", configType: "qwen3_5"),
                    model("unknown", configType: nil)
                ]]
            }
        case ("POST", "/admin/api/global-settings"):
            let patch = try JSONSerialization.jsonObject(with: requestBody(request)) as! [String: Any]
            recordedPatches.append(patch)
            if savesRejected {
                status = 400
                body = ["detail": "Settings save rejected"]
            } else {
                var integrations = settings["integrations"] as! [String: Any]
                integrations.merge(patch) { _, new in new }
                settings["integrations"] = integrations
                body = ["success": true, "runtime_applied": ["integrations"]]
            }
        default:
            body = [:]
        }
        let data = try JSONSerialization.data(withJSONObject: body)
        if request.httpMethod == "POST", let started = saveStarted {
            saveStarted = nil
            let responseStatus = status
            heldSave = { completion(responseStatus, data) }
            started.fulfill()
        } else {
            completion(status, data)
        }
    }

    private func model(_ id: String, configType: String?) -> [String: Any] {
        var model: [String: Any] = ["id": id, "loaded": false, "is_loading": false, "estimated_size": 0]
        model["config_model_type"] = configType
        return model
    }

    private func requestBody(_ request: URLRequest) throws -> Data {
        if let data = request.httpBody { return data }
        let stream = try XCTUnwrap(request.httpBodyStream)
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 1024)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count < 0 { throw stream.streamError ?? URLError(.cannotDecodeRawData) }
            if count == 0 { break }
            data.append(contentsOf: buffer.prefix(count))
        }
        return data
    }
}

private final class MarkItDownSettingsURLProtocol: URLProtocol, @unchecked Sendable {
    static let server = MarkItDownSettingsServer()

    override class func canInit(with request: URLRequest) -> Bool { request.url?.port == 18161 }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        do {
            try Self.server.respond(to: request) { [self] status, data in
                let response = HTTPURLResponse(url: request.url!, statusCode: status,
                                               httpVersion: nil, headerFields: nil)!
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            }
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
