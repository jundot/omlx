// Finds model folders that already exist on this Mac, so the Welcome wizard can
// prefill the Model Directory instead of leaving the empty `<base>/models`
// default in place.
//
// Why this exists: a first run whose models already live in `~/Models` used to
// finish with a Model Directory pointing at an empty folder. The server then
// listed no models at all (measured 2026-11-16 on the acceptance cluster: three
// Macs with 902 GB of models reported `models: []`). The wizard must show where
// the models are, not ask the user to find them.
//
// The scan is deliberately bounded: a fixed list of well-known roots, one level
// deep, at most `maxEntries` children per root. It never walks the disk.

import Foundation

/// One directory that looks like it holds models.
struct DetectedModelDirectory: Equatable, Sendable {
    let path: String
    let modelCount: Int
    /// HuggingFace's `hub` layout is counted by `models--org--name` folders,
    /// which is also how the server names them.
    let isHuggingFaceCache: Bool

    /// `~/Models` instead of `/Users/name/Models`, for the row's one line.
    var displayPath: String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        guard !home.isEmpty, path == home || path.hasPrefix(home + "/") else {
            return path
        }
        return "~" + path.dropFirst(home.count)
    }
}

enum ModelDirectoryDetector {
    /// One level of children per root; a model folder never hides deeper than
    /// that and a runaway directory listing must not stall the wizard.
    static let maxEntries = 500

    /// Roots we look in, in the order they are preferred when two hold the same
    /// number of models. `~/Models` first because that is this project's own
    /// convention; the HuggingFace cache last because its entries are shards,
    /// not checkpoints.
    static func candidateRoots(home: URL) -> [(url: URL, isHuggingFaceCache: Bool)] {
        [
            (home.appendingPathComponent("Models", isDirectory: true), false),
            (home.appendingPathComponent("models", isDirectory: true), false),
            (home.appendingPathComponent("mlx-models", isDirectory: true), false),
            (home.appendingPathComponent("MlxModels", isDirectory: true), false),
            (home.appendingPathComponent("Documents/Models", isDirectory: true), false),
            (home.appendingPathComponent("Documents/models", isDirectory: true), false),
            (home.appendingPathComponent("Downloads/Models", isDirectory: true), false),
            (home.appendingPathComponent(".cache/huggingface/hub", isDirectory: true), true),
            (home.appendingPathComponent("Library/Caches/huggingface/hub", isDirectory: true), true),
        ]
    }

    /// Every root that holds at least one model, most models first, ties broken
    /// by the root order above.
    static func detect(
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        fileManager: FileManager = .default
    ) -> [DetectedModelDirectory] {
        var found: [(order: Int, directory: DetectedModelDirectory)] = []
        // `~/Models` and `~/models` are the same folder on macOS' default
        // case-insensitive volume, so the roots are deduplicated by their
        // resolved path — otherwise one folder is reported (and counted) twice.
        var seen: Set<String> = []
        for (order, root) in candidateRoots(home: home).enumerated() {
            let identity = root.url.resolvingSymlinksInPath().path.lowercased()
            guard seen.insert(identity).inserted else { continue }
            let count = countModels(
                in: root.url,
                isHuggingFaceCache: root.isHuggingFaceCache,
                fileManager: fileManager
            )
            guard count > 0 else { continue }
            found.append(
                (
                    order,
                    DetectedModelDirectory(
                        path: root.url.standardizedFileURL.path,
                        modelCount: count,
                        isHuggingFaceCache: root.isHuggingFaceCache
                    )
                )
            )
        }
        // A real model folder always beats a package cache, whatever the
        // counts say: the HuggingFace hub is shards keyed by repo name, and
        // adopting it over `~/Models` on 2026-11-16 pointed a first run at the
        // cache instead of the user's own checkpoints (measured on .11, which
        // listed more `models--*` entries than it had models).
        return found
            .sorted { left, right in
                if left.directory.isHuggingFaceCache != right.directory.isHuggingFaceCache {
                    return !left.directory.isHuggingFaceCache
                }
                if left.directory.modelCount != right.directory.modelCount {
                    return left.directory.modelCount > right.directory.modelCount
                }
                return left.order < right.order
            }
            .map(\.directory)
    }

    /// How many model folders one root holds (0 when the root is absent).
    static func countModels(
        in root: URL,
        isHuggingFaceCache: Bool,
        fileManager: FileManager = .default
    ) -> Int {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue,
              let children = try? fileManager.contentsOfDirectory(
                  at: root,
                  includingPropertiesForKeys: [.isDirectoryKey],
                  options: [.skipsHiddenFiles]
              )
        else { return 0 }

        var count = 0
        for child in children.prefix(maxEntries) {
            guard (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            else { continue }
            if isHuggingFaceCache {
                if child.lastPathComponent.hasPrefix("models--"),
                   fileManager.fileExists(atPath: child.appendingPathComponent("snapshots").path) {
                    count += 1
                }
            } else if looksLikeModel(child, fileManager: fileManager) {
                count += 1
            }
        }
        return count
    }

    /// A folder is a model when it carries a config, a weights file, or a
    /// tokenizer — the same three things the server's discovery needs.
    static func looksLikeModel(_ directory: URL, fileManager: FileManager = .default) -> Bool {
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            if fileManager.fileExists(atPath: directory.appendingPathComponent(name).path) {
                return true
            }
        }
        guard let entries = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return false }
        return entries.prefix(maxEntries).contains { entry in
            let ext = entry.pathExtension.lowercased()
            return ext == "safetensors" || ext == "gguf"
        }
    }

    /// Whether the wizard may replace the field's current value with a detected
    /// directory. It only ever replaces an empty field or the untouched
    /// `<base>/models` default that holds nothing — a folder the user chose
    /// (or one that already has models) is never silently changed.
    static func shouldAdopt(
        current: String,
        basePath: String,
        fileManager: FileManager = .default
    ) -> Bool {
        let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        let currentPath = URL(fileURLWithPath: trimmed).standardizedFileURL.path
        let defaultPath = URL(
            fileURLWithPath: AppConfig.defaultModelDir(forBasePath: basePath)
        ).standardizedFileURL.path
        guard currentPath == defaultPath else { return false }
        return countModels(in: URL(fileURLWithPath: currentPath),
                           isHuggingFaceCache: false,
                           fileManager: fileManager) == 0
    }
}