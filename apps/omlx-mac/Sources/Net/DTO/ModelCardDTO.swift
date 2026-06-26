// PR — Downloads model card. Decodes `/admin/api/hf/model-info` and
// `/admin/api/ms/model-info`. Both endpoints share the same envelope
// (modelo card + metadata) so a single DTO handles both.
//
// All metadata fields are optional because:
//   • `model_card` is always present (server normalizes empty to "")
//   • MS doesn't surface `params` (server returns null for it)
//   • MS doesn't surface `is_adapter` at all (HF-only field)
//   • Even on HF, some repos omit pipeline_tag or have zero downloads/likes
//
// Future tabs (Files, Tags) would decode `files` and `tags` here — left
// out for now since v1 ships a single Card view.

import Foundation

struct ModelCardDTO: Decodable, Equatable, Sendable {
    /// Raw Markdown body of the model's README. May be empty when the
    /// upstream repo doesn't ship one.
    let modelCard: String

    /// HF-style pipeline tag, e.g. "text-generation", "automatic-speech-recognition".
    /// Renders as a colored pill in the sheet's metadata row.
    let pipelineTag: String?

    /// Server-formatted parameter count, e.g. "7B", "13B". Already
    /// display-ready — never reformat client-side.
    let paramsFormatted: String?

    /// Server-formatted total weight size on disk, e.g. "4.2 GB".
    let sizeFormatted: String?

    /// Lifetime download count from the upstream API.
    let downloads: Int?

    /// Lifetime like/star count from the upstream API.
    let likes: Int?

    /// HF-only: true when the repo is a LoRA / PEFT adapter rather than
    /// a full model. The sheet renders a warning banner and hides the
    /// Download button so the user doesn't accidentally pull an adapter
    /// they can't run standalone.
    let isAdapter: Bool?

    /// Per-file listing — name + size in bytes + server-formatted size.
    /// Drives the Files tab. Always present for HF; MS returns the same
    /// shape but the sheet hides the Files tab for MS to match the HTML
    /// admin's behavior (MS repos are usually opaque sharded blobs).
    let files: [ModelCardFile]?

    /// Free-form tag strings from the upstream API. Drives the Tags tab
    /// for both sources.
    let tags: [String]?
}

/// One entry in `ModelCardDTO.files`. `sizeFormatted` is already
/// display-ready ("4.2 GB", "1.5 MB"); never reformat client-side.
struct ModelCardFile: Decodable, Equatable, Sendable, Identifiable {
    let name: String
    let size: Int64
    let sizeFormatted: String?

    var id: String { name }
}
