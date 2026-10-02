// Covers `ModelCardDTO`'s decode contract with both `/admin/api/hf/model-info`
// and `/admin/api/ms/model-info`. The DTO is a single shared shape across
// both surfaces because the server already strips YAML front-matter and
// returns the same `{model_card: "<markdown>"}` envelope from either route
// — so a decode regression here would silently break both Hugging Face and
// ModelScope model-card sheets at once.

import XCTest
@testable import oMLX

final class ModelCardDTODecodeTests: XCTestCase {

    /// Mirrors the `OMLXClient` decoder configuration so snake_case
    /// `model_card` maps to the camelCase `modelCard` property.
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    func testDecodeMarkdownBody() throws {
        let json = """
        {
            "model_card": "# Model\\n\\nSome description.\\n\\n- bullet\\n"
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertEqual(dto.modelCard, "# Model\n\nSome description.\n\n- bullet\n")
    }

    func testDecodeEmptyStringIsLegal() throws {
        // The server signals "no README on the upstream repo" with an
        // empty string rather than a 404 — the sheet's `.empty` branch
        // distinguishes it from `.failed`. If we changed decoding to
        // reject empty strings, every model without a README would
        // surface as an error instead of the friendly empty state.
        let json = #"{"model_card": ""}"#.data(using: .utf8)!

        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertEqual(dto.modelCard, "")
    }

    func testIgnoresExtraFields() throws {
        // Real responses include additional fields the DTO doesn't
        // currently consume (e.g. `tags`, `files`, `description`,
        // `created_at`). Swift's default Decodable behaviour ignores
        // unknown keys; assert it here so a future migration to
        // .failOnUnknownKey is caught.
        let json = """
        {
            "model_card": "Hello",
            "tags": ["text-generation", "mlx"],
            "files": [{"name": "README.md", "size": 1234}],
            "description": "A model",
            "created_at": "2026-01-01T00:00:00Z"
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertEqual(dto.modelCard, "Hello")
        XCTAssertNil(dto.downloads)
        XCTAssertNil(dto.likes)
    }

    func testDecodesFullHFMetadata() throws {
        // Production-shaped HF response — every metadata field populated.
        // Drives the sheet's badges (params/size/pipeline_tag), the
        // downloads / likes counters, and the LoRA-adapter warning.
        let json = """
        {
            "repo_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "name": "Llama 3.2",
            "model_card": "# Llama 3.2 3B Instruct",
            "pipeline_tag": "text-generation",
            "params": 3000000000,
            "params_formatted": "3B",
            "size": 4500000000,
            "size_formatted": "4.2 GB",
            "downloads": 12500,
            "likes": 234,
            "is_adapter": false
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertEqual(dto.pipelineTag, "text-generation")
        XCTAssertEqual(dto.paramsFormatted, "3B")
        XCTAssertEqual(dto.sizeFormatted, "4.2 GB")
        XCTAssertEqual(dto.downloads, 12500)
        XCTAssertEqual(dto.likes, 234)
        XCTAssertEqual(dto.isAdapter, false)
    }

    func testDecodesMSResponseWithoutHFOnlyFields() throws {
        // ModelScope returns the same envelope but omits `is_adapter`
        // entirely and always leaves `params`/`params_formatted` null
        // (see ms_downloader.py). Decoder must treat both as nil rather
        // than throwing.
        let json = """
        {
            "model_card": "## A ModelScope model",
            "pipeline_tag": null,
            "params": null,
            "params_formatted": null,
            "size_formatted": "8.1 GB",
            "downloads": 4200,
            "likes": 88
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertNil(dto.pipelineTag)
        XCTAssertNil(dto.paramsFormatted)
        XCTAssertEqual(dto.sizeFormatted, "8.1 GB")
        XCTAssertEqual(dto.downloads, 4200)
        XCTAssertNil(dto.isAdapter,
                     "MS responses omit is_adapter entirely; decoder must produce nil, not throw.")
    }

    func testDecodesLoraAdapter() throws {
        // Adapter repos surface with `is_adapter: true` — the sheet's
        // warning banner hinges on this. Test it as an isolated case
        // so a server-side rename catches here, not via the larger
        // metadata test.
        let json = #"{"model_card": "...", "is_adapter": true}"#.data(using: .utf8)!
        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertEqual(dto.isAdapter, true)
    }

    func testDecodesFilesList() throws {
        // Files tab payload — server emits one object per repo file with
        // `name`, raw `size` in bytes, and a `size_formatted` string we
        // surface directly. `size_formatted` may be empty when the file
        // size is unknown (server returns "" rather than null in that
        // case — see hf_downloader.py:504).
        let json = """
        {
            "model_card": "...",
            "files": [
                {"name": "README.md", "size": 12345, "size_formatted": "12 KB"},
                {"name": "model.safetensors", "size": 4500000000, "size_formatted": "4.2 GB"},
                {"name": "config.json", "size": 800, "size_formatted": ""}
            ]
        }
        """.data(using: .utf8)!

        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertEqual(dto.files?.count, 3)
        XCTAssertEqual(dto.files?[0].name, "README.md")
        XCTAssertEqual(dto.files?[1].size, 4_500_000_000)
        XCTAssertEqual(dto.files?[1].sizeFormatted, "4.2 GB")
        XCTAssertEqual(dto.files?[2].sizeFormatted, "")
    }

    func testDecodesTagsList() throws {
        // Tags tab payload — flat array of strings. MS encodes them as
        // comma-separated server-side but converts before serializing,
        // so the wire shape is identical to HF.
        let json = #"{"model_card": "...", "tags": ["text-generation", "mlx", "license:apache-2.0"]}"#
            .data(using: .utf8)!
        let dto = try decoder.decode(ModelCardDTO.self, from: json)
        XCTAssertEqual(dto.tags, ["text-generation", "mlx", "license:apache-2.0"])
    }

    func testMissingFieldThrows() throws {
        // The contract requires `model_card` to be present, even if
        // empty. If the server ever omits the field, we want a decode
        // error surfaced as `.failed` in the sheet, not a silent empty
        // body that masks a server-side regression.
        let json = #"{"unrelated": "value"}"#.data(using: .utf8)!

        XCTAssertThrowsError(try decoder.decode(ModelCardDTO.self, from: json))
    }
}
