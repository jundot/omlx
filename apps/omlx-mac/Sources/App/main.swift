// Classic AppKit entry point. This is a pure menubar (accessory) app with
// no SwiftUI views, so we drive NSApplication directly rather than wrapping
// it in a SwiftUI `App`. A SwiftUI `Settings`-only scene was found to veto
// termination (a Cocoa Quit returns userCanceled / -128) and interfere with
// the POSIX signal reaping in SignalHandlers — neither of which a pure
// AppKit run loop does. AppDelegate owns the status item, the server
// lifecycle, and the activation-policy dance (see AppDelegate.swift).

import AppKit

let delegate = AppDelegate()
let app = NSApplication.shared
app.delegate = delegate
app.run()
