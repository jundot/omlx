import XCTest
@testable import oMLX

@MainActor
final class IntegrationsScreenVMTests: XCTestCase {
    private var server: IntegrationsSettingsServer { IntegrationsSettingsURLProtocol.server }

    private func makeClient() throws -> (OMLXClient, URLSession) {
        try server.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [IntegrationsSettingsURLProtocol.self]
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
        server.holdNextPost(started: started)
        let submit = Task { await vm.save(.markitdownMaxFileSize, client: client) }
        let result = await XCTWaiter.fulfillment(of: [started], timeout: 3)
        XCTAssertEqual(result, .completed)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        server.finishHeldPost()
        await submit.value
        XCTAssertTrue(vm.savingMarkitdownFields.isEmpty)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
    }

    func testWebSearchLoadsConfiguredValuesAndMissingFieldsUseDefaults() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.setIntegrations([
            "web_search_provider": "ddgs_custom", "web_search_brave_api_key": "saved-key",
            "web_search_searxng_url": "http://localhost:8888", "web_search_ddgs_backends": "mojeek,yahoo",
            "web_search_max_results": 7, "web_search_content_mode": "full",
            "web_search_content_truncate": false, "web_search_content_max_chars": 500
        ])
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        XCTAssertEqual(vm.webSearch, .init(provider: "ddgs_custom", braveApiKey: "saved-key",
                                         searxngUrl: "http://localhost:8888", ddgsBackends: "mojeek,yahoo",
                                         maxResults: "7", contentMode: "full", contentTruncate: false,
                                         contentMaxChars: "500"))
        XCTAssertTrue(vm.ddgsBackendChecked("mojeek"))
        server.omitWebSearchSettings()
        await vm.load(client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(vm.webSearch, .init(provider: "ddgs", braveApiKey: "", searxngUrl: "", ddgsBackends: "",
                                         maxResults: "3", contentMode: "snippet", contentTruncate: true,
                                         contentMaxChars: "20000"))
    }

    func testWebSearchSavesExactlyOneFieldAndPreservesOtherDrafts() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.webSearch = .init(provider: "brave", braveApiKey: " draft-key ",
                             searxngUrl: " http://localhost:8888/ ", ddgsBackends: "mojeek,yahoo",
                             maxResults: " 7 ", contentMode: "full", contentTruncate: false,
                             contentMaxChars: "500")
        let cases: [(IntegrationsScreenVM.WebSearchField, String, Any)] = [
            (.provider, "web_search_provider", "brave"),
            (.braveApiKey, "web_search_brave_api_key", "draft-key"),
            (.searxngUrl, "web_search_searxng_url", "http://localhost:8888"),
            (.ddgsBackends, "web_search_ddgs_backends", "mojeek,yahoo"),
            (.maxResults, "web_search_max_results", 7),
            (.contentMode, "web_search_content_mode", "full"),
            (.contentTruncate, "web_search_content_truncate", false),
            (.contentMaxChars, "web_search_content_max_chars", 500)
        ]
        for (field, key, value) in cases {
            let previousCount = server.patches.count
            await vm.saveWebSearch(field, client: client)
            XCTAssertNil(vm.lastError)
            XCTAssertEqual(server.patches.count, previousCount + 1)
            XCTAssertEqual(server.patches.last! as NSDictionary, [key: value] as NSDictionary)
            await vm.saveWebSearch(field, client: client)
            XCTAssertEqual(server.patches.count, previousCount + 1)
            if field == .provider { XCTAssertEqual(vm.webSearch.braveApiKey, " draft-key ") }
        }
        vm.webSearch.provider = "ddgs"
        await vm.saveWebSearch(.provider, client: client)
        XCTAssertEqual(vm.webSearch.braveApiKey, "draft-key")
        XCTAssertEqual(vm.webSearch.searxngUrl, "http://localhost:8888")
        XCTAssertEqual(vm.webSearch.ddgsBackends, "mojeek,yahoo")
        XCTAssertEqual(vm.webSearch.contentMaxChars, "500")
        vm.setDdgsBackend("mojeek", selected: false)
        vm.setDdgsBackend("brave", selected: true)
        vm.setDdgsBackend("brave", selected: true)
        XCTAssertEqual(vm.webSearch.ddgsBackends, "yahoo,brave")
        vm.webSearch.braveApiKey = ""
        await vm.saveWebSearch(.braveApiKey, client: client)
        XCTAssertEqual(server.patches.last?["web_search_brave_api_key"] as? String, "")
    }

    func testWebSearchRejectsInvalidNumbersAndURLsWithoutSaving() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        for (field, values) in [
            (IntegrationsScreenVM.WebSearchField.maxResults, ["", "0", "11", "1.5", "bad"]),
            (.contentMaxChars, ["", "0", "-1", "1.5", "999999999999999999999"]),
            (.searxngUrl, ["localhost:8888", "ftp://localhost", "https://", "https:///search", "https://bad host"])
        ] {
            for text in values {
                switch field {
                case .maxResults: vm.webSearch.maxResults = text
                case .contentMaxChars: vm.webSearch.contentMaxChars = text
                default: vm.webSearch.searxngUrl = text
                }
                await vm.saveWebSearch(field, client: client)
                XCTAssertNotNil(vm.lastError, text)
                XCTAssertTrue(server.patches.isEmpty, text)
            }
        }
        vm.webSearch.contentMaxChars = "1"
        await vm.saveWebSearch(.contentMaxChars, client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(server.patches.last?["web_search_content_max_chars"] as? Int, 1)
        vm.webSearch.searxngUrl = "http://[::1]:8888/search/"
        await vm.saveWebSearch(.searxngUrl, client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(vm.webSearch.searxngUrl, "http://[::1]:8888/search")
    }

    func testWebSearchFailedSavesRestoreControlsAndRetainTypedValuesForRetry() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        server.rejectSaves(true)
        vm.webSearch = .init(provider: "brave", braveApiKey: "retry-key",
                             searxngUrl: "http://localhost:8888", ddgsBackends: "yahoo",
                             maxResults: "8", contentMode: "full", contentTruncate: false,
                             contentMaxChars: "500")
        for field in IntegrationsScreenVM.WebSearchField.allCases {
            await vm.saveWebSearch(field, client: client)
            XCTAssertNotNil(vm.lastError)
        }
        XCTAssertEqual(vm.webSearch.provider, "ddgs")
        XCTAssertEqual(vm.webSearch.ddgsBackends, "")
        XCTAssertEqual(vm.webSearch.contentMode, "snippet")
        XCTAssertTrue(vm.webSearch.contentTruncate)
        XCTAssertEqual(vm.webSearch.braveApiKey, "retry-key")
        XCTAssertEqual(vm.webSearch.searxngUrl, "http://localhost:8888")
        XCTAssertEqual(vm.webSearch.maxResults, "8")
        XCTAssertEqual(vm.webSearch.contentMaxChars, "500")
        XCTAssertTrue(vm.savingWebSearchFields.isEmpty)
        server.rejectSaves(false)
        await vm.saveWebSearch(.braveApiKey, client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertEqual(server.patches.last?["web_search_brave_api_key"] as? String, "retry-key")
    }

    func testWebSearchSuppressesDuplicateSavesAndPreservesEditsDuringSave() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.webSearch.maxResults = "7"
        let started = XCTestExpectation(description: "Web search save started")
        server.holdNextPost(started: started)
        let submit = Task { await vm.saveWebSearch(.maxResults, client: client) }
        let result = await XCTWaiter.fulfillment(of: [started], timeout: 3)
        XCTAssertEqual(result, .completed)
        await vm.saveWebSearch(.maxResults, client: client)
        XCTAssertEqual(server.patches.count, 1)
        vm.webSearch.braveApiKey = "new-draft"
        server.finishHeldPost()
        await submit.value
        XCTAssertEqual(vm.webSearch.braveApiKey, "new-draft")
        XCTAssertTrue(vm.savingWebSearchFields.isEmpty)
        await vm.saveWebSearch(.maxResults, client: client)
        XCTAssertEqual(server.patches.count, 1)
    }

    func testWebSearchConnectivityUsesUnsavedValuesAndDoesNotPersist() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.webSearch = .init(provider: "brave", braveApiKey: "pending-key",
                             searxngUrl: "http://localhost:8888", ddgsBackends: "yahoo",
                             maxResults: "7", contentMode: "full", contentTruncate: false,
                             contentMaxChars: "500")
        await vm.testWebSearch(client: client)
        XCTAssertEqual(server.searchRequests.last! as NSDictionary, [
            "provider": "brave", "brave_api_key": "pending-key",
            "searxng_url": "http://localhost:8888", "ddgs_backends": "yahoo", "max_results": 7
        ] as NSDictionary)
        XCTAssertEqual(server.searchTimeouts, [60])
        XCTAssertTrue(server.patches.isEmpty)
        XCTAssertEqual(vm.webSearchTestOK, true)
        XCTAssertEqual(vm.webSearchTestMessage, "Search OK: 1 results")
        XCTAssertFalse(vm.webSearchTesting)
        vm.webSearch.contentMaxChars = "600"
        XCTAssertNil(vm.webSearchTestMessage)
        XCTAssertNil(vm.webSearchTestOK)
    }

    func testWebSearchConnectivityShowsProviderHTTPAndTransportFailures() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        server.setSearchResponse(["ok": false, "error": ["message": "Provider rate limited"]])
        await vm.testWebSearch(client: client)
        XCTAssertEqual(vm.webSearchTestMessage, "Provider rate limited")
        XCTAssertEqual(vm.webSearchTestOK, false)
        server.setSearchResponse(["detail": "Service unavailable"], status: 503)
        await vm.testWebSearch(client: client)
        XCTAssertTrue(vm.webSearchTestMessage?.contains("503") == true)
        XCTAssertEqual(vm.webSearchTestOK, false)
        server.setSearchResponse(["ok": false])
        await vm.testWebSearch(client: client)
        XCTAssertEqual(vm.webSearchTestMessage, "Search test failed")
        server.failSearchTransport(true)
        await vm.testWebSearch(client: client)
        XCTAssertEqual(vm.webSearchTestOK, false)
        XCTAssertNotNil(vm.webSearchTestMessage)
        XCTAssertFalse(vm.webSearchTesting)
        vm.webSearch.maxResults = "0"
        let previousCount = server.searchRequests.count
        await vm.testWebSearch(client: client)
        XCTAssertEqual(server.searchRequests.count, previousCount)
        XCTAssertEqual(vm.webSearchTestOK, false)
    }

    func testWebSearchIgnoresStaleResponsesAndSuppressesConcurrentTests() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        for failure in [false, true] {
            server.setSearchResponse(failure ? ["ok": false, "error": ["message": "old error"]]
                                             : ["ok": true, "results": []])
            let started = XCTestExpectation(description: "Search started")
            server.holdNextPost(started: started)
            let search = Task { await vm.testWebSearch(client: client) }
            let result = await XCTWaiter.fulfillment(of: [started], timeout: 3)
            XCTAssertEqual(result, .completed)
            XCTAssertTrue(vm.webSearchTesting)
            let previousCount = server.searchRequests.count
            await vm.testWebSearch(client: client)
            XCTAssertEqual(server.searchRequests.count, previousCount)
            vm.webSearch.provider = "brave"
            vm.webSearch.provider = "ddgs"
            server.finishHeldPost()
            await search.value
            XCTAssertNil(vm.webSearchTestMessage)
            XCTAssertNil(vm.webSearchTestOK)
            XCTAssertFalse(vm.webSearchTesting)
        }
    }
}

private final class IntegrationsSettingsServer: @unchecked Sendable {
    private let lock = NSLock()
    private var settings: [String: Any] = [:]
    private var recordedPatches: [[String: Any]] = []
    private var modelReadCount = 0
    private var savesRejected = false
    private var modelReadsRejected = false
    private var postStarted: XCTestExpectation?
    private var heldPost: (@Sendable () -> Void)?
    private var recordedSearchRequests: [[String: Any]] = []
    private var recordedSearchTimeouts: [TimeInterval] = []
    private var searchResponse: [String: Any] = [:]
    private var searchStatus = 200
    private var searchTransportFailed = false

    var searchRequests: [[String: Any]] { lock.withLock { recordedSearchRequests } }
    var searchTimeouts: [TimeInterval] { lock.withLock { recordedSearchTimeouts } }

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
            postStarted = nil
            heldPost = nil
            recordedSearchRequests = []
            recordedSearchTimeouts = []
            searchResponse = ["ok": true, "results": [["title": "Example", "url": "https://example.com", "snippet": "Example result"]]]
            searchStatus = 200
            searchTransportFailed = false
        }
    }

    func omitWebSearchSettings() {
        lock.withLock {
            let integrations = settings["integrations"] as! [String: Any]
            settings["integrations"] = integrations.filter { !$0.key.hasPrefix("web_search_") }
        }
    }

    func setSearchResponse(_ body: [String: Any], status: Int = 200) {
        lock.withLock { searchResponse = body; searchStatus = status }
    }

    func failSearchTransport(_ fail: Bool) { lock.withLock { searchTransportFailed = fail } }

    func setIntegrations(_ values: [String: Any]) {
        lock.withLock {
            var integrations = settings["integrations"] as! [String: Any]
            integrations.merge(values) { _, new in new }
            settings["integrations"] = integrations
        }
    }

    func rejectSaves(_ reject: Bool) { lock.withLock { savesRejected = reject } }
    func rejectModelReads(_ reject: Bool) { lock.withLock { modelReadsRejected = reject } }
    func holdNextPost(started: XCTestExpectation) { lock.withLock { postStarted = started } }

    func finishHeldPost() {
        let completion = lock.withLock {
            let completion = heldPost
            heldPost = nil
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
        case ("POST", "/admin/api/web-search/test"):
            recordedSearchRequests.append(try JSONSerialization.jsonObject(with: requestBody(request)) as! [String: Any])
            recordedSearchTimeouts.append(request.timeoutInterval)
            if searchTransportFailed { throw URLError(.timedOut) }
            status = searchStatus
            body = searchResponse
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
        if request.httpMethod == "POST", let started = postStarted {
            postStarted = nil
            let responseStatus = status
            heldPost = { completion(responseStatus, data) }
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

private final class IntegrationsSettingsURLProtocol: URLProtocol, @unchecked Sendable {
    static let server = IntegrationsSettingsServer()

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
