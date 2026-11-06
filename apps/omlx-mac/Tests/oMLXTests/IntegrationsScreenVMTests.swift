import AppKit
import SwiftUI
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

    func testSettingsDecodeAndPatchesUseExistingWireFormat() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = Data(#"{"markitdown_enabled":false,"markitdown_expose_model":true,"markitdown_max_file_size_mb":40,"markitdown_max_files_per_request":9,"markitdown_pdf_processing_engine":"GLM-OCR"}"#.utf8)
        let settings = try decoder.decode(GlobalSettingsDTO.IntegrationsSettings.self, from: data)
        XCTAssertEqual(settings.markitdownEnabled, false)
        XCTAssertEqual(settings.markitdownExposeModel, true)
        XCTAssertEqual(settings.markitdownMaxFileSizeMb, 40)
        XCTAssertEqual(settings.markitdownMaxFilesPerRequest, 9)
        XCTAssertEqual(settings.markitdownPdfProcessingEngine, "GLM-OCR")
        let legacy = try decoder.decode(GlobalSettingsDTO.IntegrationsSettings.self, from: Data("{}".utf8))
        XCTAssertNil(legacy.markitdownEnabled)
        XCTAssertNil(legacy.markitdownExposeModel)
        XCTAssertNil(legacy.markitdownMaxFileSizeMb)
        XCTAssertNil(legacy.markitdownMaxFilesPerRequest)
        XCTAssertNil(legacy.markitdownPdfProcessingEngine)

        var patch = GlobalSettingsPatch()
        patch.markitdownEnabled = false
        patch.markitdownExposeModel = false
        patch.markitdownMaxFileSizeMb = 40
        patch.markitdownMaxFilesPerRequest = 9
        patch.markitdownPdfProcessingEngine = "GLM-OCR"
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let body = try JSONSerialization.jsonObject(with: encoder.encode(patch)) as! [String: Any]
        XCTAssertEqual(body.count, 5)
        XCTAssertEqual(body["markitdown_enabled"] as? Bool, false)
        XCTAssertEqual(body["markitdown_expose_model"] as? Bool, false)
        XCTAssertEqual(body["markitdown_max_file_size_mb"] as? Int, 40)
        XCTAssertEqual(body["markitdown_max_files_per_request"] as? Int, 9)
        XCTAssertEqual(body["markitdown_pdf_processing_engine"] as? String, "GLM-OCR")
        let empty = try JSONSerialization.jsonObject(with: encoder.encode(GlobalSettingsPatch())) as! [String: Any]
        XCTAssertTrue(empty.isEmpty)
    }

    func testLegacySettingsLoadDefaultsWithoutSaving() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        server.removeMarkItDownSettings()
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertTrue(vm.markitdownEnabled)
        XCTAssertFalse(vm.markitdownExposeModel)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "25")
        XCTAssertEqual(vm.markitdownMaxFilesText, "5")
        XCTAssertEqual(vm.markitdownPdfProcessingEngine, "markitdown")
        await vm.save(.markitdownMaxFileSize, client: client)
        await vm.save(.markitdownMaxFiles, client: client)
        await vm.save(.markitdownEnabled, client: client)
        await vm.save(.markitdownExposeModel, client: client)
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertTrue(server.patches.isEmpty)
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
        XCTAssertFalse(vm.markitdownEnabled)
        XCTAssertTrue(vm.markitdownExposeModel)
        XCTAssertFalse(vm.markitdownModelVisible)
        vm.markitdownModelVisible = false
        XCTAssertTrue(vm.markitdownExposeModel)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "40")
        XCTAssertEqual(vm.markitdownMaxFilesText, "9")
        XCTAssertEqual(vm.markitdownPdfOptions.map(\.0), ["markitdown", "GLM-OCR", "DeepSeek-OCR"])
        XCTAssertFalse(vm.markitdownOcrModelMissing)
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
        XCTAssertEqual(vm.markitdownPdfProcessingEngine, "missing-ocr")
        let missingWarning = try XCTUnwrap(vm.markitdownPdfWarning)
        XCTAssertTrue(missingWarning.contains("\n"))
        vm.markitdownPdfProcessingEngine = "markitdown"
        let warning = try XCTUnwrap(vm.markitdownPdfWarning)
        XCTAssertFalse(warning.contains("\n"))
        XCTAssertTrue(missingWarning.hasSuffix(warning))
        XCTAssertTrue(server.patches.isEmpty)
    }

    func testVisibilityChangesRefreshModelsPreserveDraftsAndPersistOnReopen() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFilesText = "12"
        vm.markitdownModelVisible = true
        await vm.save(.markitdownExposeModel, client: client)
        XCTAssertTrue(vm.availableModels.contains("MarkItDown"))
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_expose_model"] as? Bool, true)

        vm.markitdownEnabled = false
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertFalse(vm.markitdownModelVisible)
        XCTAssertTrue(vm.markitdownExposeModel)
        XCTAssertFalse(vm.availableModels.contains("MarkItDown"))
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_enabled"] as? Bool, false)

        vm.markitdownEnabled = true
        await vm.save(.markitdownEnabled, client: client)
        XCTAssertTrue(vm.markitdownModelVisible)
        XCTAssertTrue(vm.availableModels.contains("MarkItDown"))
        XCTAssertEqual(vm.markitdownMaxFilesText, "12")
        XCTAssertEqual(server.settingsReads, 1)
        XCTAssertEqual(server.modelReads, 4)

        let reopened = IntegrationsScreenVM()
        await reopened.load(client: client)
        XCTAssertTrue(reopened.markitdownModelVisible)
        XCTAssertEqual(reopened.markitdownMaxFilesText, "5")
        reopened.markitdownModelVisible = false
        await reopened.save(.markitdownExposeModel, client: client)
        XCTAssertEqual(server.patches.last?["markitdown_expose_model"] as? Bool, false)
        XCTAssertFalse(reopened.availableModels.contains("MarkItDown"))
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
        XCTAssertEqual(vm.markitdownMaxFilesText, "9")
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        await vm.save(.markitdownMaxFiles, client: client)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_max_files_per_request"] as? Int, 9)
        await vm.save(.markitdownMaxFiles, client: client)
        XCTAssertEqual(server.patches.count, 2)
        vm.markitdownPdfProcessingEngine = "GLM-OCR"
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(server.patches.last?.count, 1)
        XCTAssertEqual(server.patches.last?["markitdown_pdf_processing_engine"] as? String, "GLM-OCR")
        await vm.save(.markitdownPdfProcessingEngine, client: client)
        XCTAssertEqual(server.patches.count, 3)
        XCTAssertEqual(server.modelReads, 1)
        XCTAssertNil(vm.lastError)

        let reopened = IntegrationsScreenVM()
        await reopened.load(client: client)
        XCTAssertEqual(reopened.markitdownMaxFileSizeText, "40")
        XCTAssertEqual(reopened.markitdownMaxFilesText, "9")
        XCTAssertEqual(reopened.markitdownPdfProcessingEngine, "GLM-OCR")
    }

    func testInvalidLimitsKeepDraftsAndSendNoRequests() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        for text in ["", " ", "0", "-1", "1.5", "25.0", "abc", String(repeating: "9", count: 40)] {
            vm.markitdownMaxFileSizeText = text
            await vm.save(.markitdownMaxFileSize, client: client)
            XCTAssertNotNil(vm.lastError, text)
            XCTAssertEqual(vm.markitdownMaxFileSizeText, text)
            vm.markitdownMaxFilesText = text
            await vm.save(.markitdownMaxFiles, client: client)
            XCTAssertNotNil(vm.lastError, text)
            XCTAssertEqual(vm.markitdownMaxFilesText, text)
        }
        XCTAssertTrue(server.patches.isEmpty)
        vm.markitdownMaxFilesText = "5"
        await vm.save(.markitdownMaxFiles, client: client)
        XCTAssertNil(vm.lastError)
        XCTAssertTrue(server.patches.isEmpty)
        vm.markitdownMaxFileSizeText = "1"
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.last?["markitdown_max_file_size_mb"] as? Int, 1)
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
        vm.markitdownMaxFilesText = "8"
        await vm.save(.markitdownMaxFiles, client: client)
        XCTAssertEqual(vm.markitdownMaxFileSizeText, "60")
        XCTAssertEqual(vm.markitdownMaxFilesText, "8")
        XCTAssertNotNil(vm.lastError)
        XCTAssertTrue(vm.savingMarkitdownFields.isEmpty)
        XCTAssertEqual(server.modelReads, 1)

        server.rejectSaves(false)
        await vm.save(.markitdownMaxFileSize, client: client)
        await vm.save(.markitdownMaxFiles, client: client)
        XCTAssertEqual(server.patches.count, 7)
        XCTAssertNil(vm.lastError)
        let reopened = IntegrationsScreenVM()
        await reopened.load(client: client)
        XCTAssertEqual(reopened.markitdownMaxFileSizeText, "60")
        XCTAssertEqual(reopened.markitdownMaxFilesText, "8")
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
        XCTAssertTrue(vm.savingMarkitdownFields.contains(.markitdownMaxFileSize))
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
        server.finishHeldSave()
        await submit.value
        XCTAssertTrue(vm.savingMarkitdownFields.isEmpty)
        await vm.save(.markitdownMaxFileSize, client: client)
        XCTAssertEqual(server.patches.count, 1)
    }

    func testVisibilitySaveLocksBothSwitchesAndPreservesSavedStateWhenRefreshFails() async throws {
        let (client, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        let vm = IntegrationsScreenVM()
        await vm.load(client: client)
        vm.markitdownMaxFilesText = "11"
        vm.markitdownEnabled = false
        let started = XCTestExpectation(description: "Visibility save started")
        server.holdNextSave(started: started)
        let save = Task { await vm.save(.markitdownEnabled, client: client) }
        let result = await XCTWaiter.fulfillment(of: [started], timeout: 3)
        XCTAssertEqual(result, .completed)
        XCTAssertTrue(vm.isSavingMarkitdownVisibility)
        await vm.save(.markitdownEnabled, client: client)
        await vm.save(.markitdownExposeModel, client: client)
        XCTAssertEqual(server.patches.count, 1)
        server.rejectModelReads(true)
        server.finishHeldSave()
        await save.value
        XCTAssertFalse(vm.isSavingMarkitdownVisibility)
        XCTAssertFalse(vm.markitdownEnabled)
        XCTAssertEqual(vm.markitdownMaxFilesText, "11")
        XCTAssertNotNil(vm.lastError)
        XCTAssertEqual(server.settingsReads, 1)
    }

    func testNativeLimitFieldsSaveOnFocusLossAndReturn() async throws {
        let (_, session) = try makeClient()
        defer { session.invalidateAndCancel() }
        var config = AppConfig.default
        config.port = 18161
        let services = AppServices(config: config, session: session)
        let suite = "IntegrationsScreenVMTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let hosting = NSHostingView(rootView:
            ScrollView { IntegrationsScreen() }
                .environment(services)
                .omlxThemed()
                .environment(\.colorScheme, .light)
                .defaultAppStorage(defaults)
                .background(Color(nsColor: .windowBackgroundColor))
        )
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 720, height: 1000),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .aqua)
        window.contentView = hosting
        window.makeKeyAndOrderFront(nil)
        defer { window.close() }
        let loadedDeadline = Date().addingTimeInterval(3)
        while server.modelReads == 0, Date() < loadedDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(server.modelReads, 1)
        hosting.layoutSubtreeIfNeeded()
        let fields = editableTextFields(in: hosting)
        let size = try XCTUnwrap(fields.first { $0.stringValue == "25" })
        let files = try XCTUnwrap(fields.first { $0.stringValue == "5" })

        XCTAssertTrue(window.makeFirstResponder(size))
        try await Task.sleep(for: .milliseconds(30))
        let sizeEditor = try XCTUnwrap(size.currentEditor() as? NSTextView)
        sizeEditor.string = "40"
        sizeEditor.didChangeText()
        XCTAssertTrue(window.makeFirstResponder(files))
        let blurDeadline = Date().addingTimeInterval(3)
        while server.patches.isEmpty, Date() < blurDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(server.patches.count, 1)
        XCTAssertEqual(server.patches.first?["markitdown_max_file_size_mb"] as? Int, 40)

        let filesEditor = try XCTUnwrap(files.currentEditor() as? NSTextView)
        filesEditor.string = "9"
        filesEditor.didChangeText()
        let enter = try XCTUnwrap(NSEvent.keyEvent(with: .keyDown, location: .zero,
                                                  modifierFlags: [], timestamp: 0,
                                                  windowNumber: window.windowNumber, context: nil,
                                                  characters: "\r", charactersIgnoringModifiers: "\r",
                                                  isARepeat: false, keyCode: 36))
        filesEditor.keyDown(with: enter)
        let submitDeadline = Date().addingTimeInterval(3)
        while server.patches.count < 2, Date() < submitDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(server.patches.count, 2)
        XCTAssertEqual(server.patches.last?["markitdown_max_files_per_request"] as? Int, 9)
        let savedDeadline = Date().addingTimeInterval(3)
        while !files.isEnabled, Date() < savedDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(files.isEnabled)
        window.makeFirstResponder(nil)
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(server.patches.count, 2)

        hosting.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        let url = URL(fileURLWithPath: "/tmp/omlx-markitdown-native.png")
        try png.write(to: url)
        let attachment = XCTAttachment(contentsOfFile: url)
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func editableTextFields(in view: NSView) -> [NSTextField] {
        let field = (view as? NSTextField).flatMap { $0.isEditable ? $0 : nil }
        return (field.map { [$0] } ?? []) + view.subviews.flatMap { editableTextFields(in: $0) }
    }
}

private final class MarkItDownSettingsServer: @unchecked Sendable {
    private let lock = NSLock()
    private var settings: [String: Any] = [:]
    private var recordedPatches: [[String: Any]] = []
    private var settingsReadCount = 0
    private var modelReadCount = 0
    private var savesRejected = false
    private var modelReadsRejected = false
    private var saveStarted: XCTestExpectation?
    private var heldSave: (@Sendable () -> Void)?

    var patches: [[String: Any]] { lock.withLock { recordedPatches } }
    var settingsReads: Int { lock.withLock { settingsReadCount } }
    var modelReads: Int { lock.withLock { modelReadCount } }

    func reset() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/global-settings.json")
        let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        lock.withLock {
            settings = fixture
            recordedPatches = []
            settingsReadCount = 0
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

    func removeMarkItDownSettings() {
        lock.withLock {
            let integrations = settings["integrations"] as! [String: Any]
            settings["integrations"] = integrations.filter { !$0.key.hasPrefix("markitdown_") }
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
            settingsReadCount += 1
            body = settings
        case ("GET", "/admin/api/models"):
            modelReadCount += 1
            if modelReadsRejected {
                status = 503
                body = ["detail": "Model list unavailable"]
            } else {
                var models = [
                    model("GLM-OCR", configType: "GLM_OCR"),
                    model("DeepSeek-OCR", configType: "deepseek_ocr2"),
                    model("ocr-in-name-only", configType: "qwen3_5"),
                    model("unknown", configType: nil)
                ]
                let integrations = settings["integrations"] as! [String: Any]
                if integrations["markitdown_enabled"] as? Bool == true,
                   integrations["markitdown_expose_model"] as? Bool == true {
                    models.append(model("MarkItDown", configType: "markitdown"))
                }
                body = ["models": models]
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
