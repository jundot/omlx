import XCTest
@testable import oMLX

final class CodexInterceptorDTODecodeTests: XCTestCase {
    func testDecodesLiveStatus() throws {
        let data = Data(#"""
        {
          "phase": "running",
          "running": true,
          "error": null,
          "session_id": "session-1",
          "started_at": 123.5,
          "model": "local/qwen",
          "active_model": "local/qwen",
          "active_context_window": 131072,
          "active_model_loaded": true,
          "active_model_loading": false,
          "pending_model": "local/next",
          "pending_context_window": 262144,
          "pending_model_loaded": true,
          "pending_model_loading": false,
          "model_switching": true,
          "model_switch_loading": false,
          "model_switch_error": null,
          "local_slot": "gpt-5.3-codex-spark",
          "project": "/tmp/project",
          "proxy_pid": 101,
          "codex_pid": 202,
          "codex_running": true,
          "active_local_requests": 1,
          "local_requests": 4,
          "cloud_requests": 2,
          "completed_requests": 3,
          "failed_requests": 0,
          "last_route": "local",
          "last_requested_model": "gpt-5.3-codex-spark",
          "last_effective_model": "local/qwen",
          "latest_metrics": {
            "first_visible_ms": 42,
            "tokens_per_second": 18.5,
            "connection_reused": true
          },
          "warmup_status": "ready",
          "warmup_model": "local/next",
          "warmup_model_loaded": true,
          "warmup_model_loading": false,
          "recent_events": [],
          "diagnostics_path": "/tmp/status.jsonl",
          "config_path": "/Users/test/.codex/config.toml",
          "config_modified": false
        }
        """#.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let status = try decoder.decode(CodexInterceptorStatusDTO.self, from: data)

        XCTAssertTrue(status.running)
        XCTAssertEqual(status.localRequests, 4)
        XCTAssertEqual(status.activeModel, "local/qwen")
        XCTAssertEqual(status.activeContextWindow, 131072)
        XCTAssertEqual(status.activeModelLoaded, true)
        XCTAssertEqual(status.pendingModel, "local/next")
        XCTAssertEqual(status.modelSwitching, true)
        XCTAssertEqual(status.lastEffectiveModel, "local/qwen")
        XCTAssertEqual(status.latestMetrics.firstVisibleMs, 42)
        XCTAssertEqual(status.latestMetrics.tokensPerSecond, 18.5)
        XCTAssertEqual(status.warmupStatus, "ready")
        XCTAssertFalse(status.configModified)
    }

    func testDecodesDoctorWithoutSecrets() throws {
        let data = Data(#"""
        {
          "ready": true,
          "codex_app_installed": true,
          "codex_app_path": "/Applications/ChatGPT.app",
          "codex_running": false,
          "mitmproxy_available": true,
          "mitmproxy_source": "bundled_python",
          "config_path": "/Users/test/.codex/config.toml",
          "config_will_be_modified": false
        }
        """#.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let doctor = try decoder.decode(CodexInterceptorDoctorDTO.self, from: data)

        XCTAssertTrue(doctor.ready)
        XCTAssertEqual(doctor.mitmproxySource, "bundled_python")
        XCTAssertFalse(doctor.configWillBeModified)
    }
}
