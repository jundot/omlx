// PR 6 — placeholder. HF downloader with 1Hz progress polling lands in PR 8.
// ModelScope downloads stay in the browser per plan §1 (out of scope).

import SwiftUI

struct DownloadsScreen: View {
    var body: some View {
        PlaceholderScreen(
            landsIn: .pr8,
            summary: "Hugging Face downloader: queued tasks, progress, cancel / retry / delete; recommended set."
        )
    }
}
