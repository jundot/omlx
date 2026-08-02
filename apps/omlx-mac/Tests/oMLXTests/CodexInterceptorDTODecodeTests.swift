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
          "latest_metrics": {
            "first_visible_ms": 42,
            "tokens_per_second": 18.5,
            "connection_reused": true
          },
          "warmup_status": "ready",
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
