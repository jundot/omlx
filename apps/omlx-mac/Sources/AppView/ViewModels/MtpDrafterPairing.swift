// Pairing logic for speculative-decoding drafters (MTP / Assistant).
//
// A drafter checkpoint can't serve requests standalone — the server
// rejects loading it as the main model. Attaching one means enabling VLM
// MTP on a chat model with the drafter as `vlm_mtp_draft_model`; this
// resolver figures out *which* chat model a drafter belongs to.

import Foundation

enum MtpDrafterPairing {
    /// Resolve the chat model `drafter` should attach to.
    ///
    /// Priority:
    ///   1. A chat model whose settings already reference the drafter as
    ///      its VLM MTP draft model (prior manual configuration).
    ///   2. Name heuristic — the drafter id with its `-MTP` / `-Assistant`
    ///      token removed exactly matches another model's id, e.g.
    ///      `mlx-community--Qwen3.8-27B-MTP-8bit` resolves to
    ///      `mlx-community--Qwen3.8-27B-8bit`. Source repo ids are tried
    ///      the same way as a fallback.
    ///
    /// Returns nil when no candidate matches or several different models
    /// do; the caller surfaces a "attach manually" message in that case.
    static func resolveTarget(drafter: ModelDTO, in models: [ModelDTO]) -> ModelDTO? {
        let chatModels = models.filter { !($0.isHelper ?? false) && $0.id != drafter.id }

        // 1. Existing settings reference (by model id, on-disk path, or
        //    Hub repo id — the server accepts all three).
        let refs = Set([drafter.id, drafter.modelPath, drafter.sourceRepoId].compactMap { $0 })
        let referenced = chatModels.filter { model in
            guard let draft = model.settings?.vlmMtpDraftModel, !draft.isEmpty else { return false }
            return refs.contains(draft)
        }
        if referenced.count == 1 { return referenced[0] }

        // 2. Name heuristic on the model id, then on the repo id.
        let idCandidates = candidateNames(for: drafter.id)
        let idMatches = chatModels.filter { idCandidates.contains($0.id) }
        if idMatches.count == 1 { return idMatches[0] }

        if let repoId = drafter.sourceRepoId {
            let repoCandidates = candidateNames(for: repoId)
            let repoMatches = chatModels.filter { model in
                guard let other = model.sourceRepoId else { return false }
                return repoCandidates.contains(other)
            }
            if repoMatches.count == 1 { return repoMatches[0] }
        }

        return nil
    }

    /// Names produced by dropping every `-MTP` / `-Assistant` token
    /// (case-insensitive) from `name`. Splitting on "-" keeps the "--"
    /// org/model separator of Hub-derived ids intact as an empty
    /// component, so rejoining reproduces it exactly.
    private static func candidateNames(for name: String) -> Set<String> {
        let tokens = name.components(separatedBy: "-")
        let stripped = tokens.filter {
            let t = $0.lowercased()
            return t != "mtp" && t != "assistant"
        }
        guard stripped.count != tokens.count else { return [] }
        return [stripped.joined(separator: "-")]
    }
}
