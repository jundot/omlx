// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import oMLX

@MainActor
final class IntegrationsScreenVMTests: XCTestCase {
    func testDirectRecipeUsesPlaceholderAndConfiguredContext() {
        let recipe = IntegrationsScreenVM.codexDirectRecipe(
            model: "example-model", host: "127.0.0.1", port: 8000, contextWindow: 32768
        )
        XCTAssertTrue(recipe.hasPrefix("OMLX_API_KEY='your-omlx-api-key' codex -m 'example-model'"))
        XCTAssertTrue(recipe.contains("http://127.0.0.1:8000/v1"))
        XCTAssertTrue(recipe.contains("model_providers.omlx.wire_api=\"responses\""))
        XCTAssertTrue(recipe.contains("model_context_window=32768"))
        XCTAssertFalse(recipe.contains("ANTHROPIC_"))
    }

    func testDirectRecipeHandlesIPv6AndWildcardBinds() {
        let ipv6 = IntegrationsScreenVM.codexDirectRecipe(model: "", host: "::1", port: 8000)
        XCTAssertTrue(ipv6.contains("http://[::1]:8000/v1"))
        XCTAssertTrue(ipv6.contains("-m 'select-a-model'"))
        XCTAssertFalse(ipv6.contains("model_context_window"))
        for host in ["0.0.0.0", "::", "[::]", "0.0.0.0,::"] {
            XCTAssertTrue(IntegrationsScreenVM.codexDirectRecipe(
                model: "", host: host, port: 8000
            ).contains("http://127.0.0.1:8000/v1"))
        }
    }

    func testDirectRecipeQuotesModelAsOneShellArgument() {
        let model = "example'$(touch SHOULD_NOT_EXIST)"
        let recipe = IntegrationsScreenVM.codexDirectRecipe(model: model, host: "localhost", port: 8000)
        XCTAssertTrue(recipe.contains("-m 'example'\"'\"'$(touch SHOULD_NOT_EXIST)'"))
    }
}
