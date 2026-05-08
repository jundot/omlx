// PR 7 — typed wrappers + URL constants for the /admin/api/* surface used by
// PR 7 screens. Kept separate from `OMLXClient.swift` so PR 8/9 can grow this
// file without churning the client itself.
//
// The actual `getX()` methods live as extensions on `OMLXClient` (see
// `OMLXClient.swift`). This file is the canonical list of paths so future
// fixture-capture scripts (`Scripts/capture_fixtures.sh`) can read them.

import Foundation

enum AdminAPI {
    static let prefix = "/admin/api"

    static let login           = "\(prefix)/login"
    static let globalSettings  = "\(prefix)/global-settings"
    static let serverInfo      = "\(prefix)/server-info"
    static let stats           = "\(prefix)/stats"
    static let logs            = "\(prefix)/logs"
}
