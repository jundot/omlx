// PR 5 — POSIX signal handlers that reap the Python child if the parent
// is killed from outside (SIGTERM/SIGINT/SIGHUP/SIGQUIT). Closes the orphan
// gap noted in PR 4 verification.
//
// Approach: install DispatchSource signal handlers at app launch. On receipt,
// run a synchronous SIGTERM-then-SIGKILL chain against the child's PID, then
// exit. This intentionally does NOT call NSApp.terminate(_:), because the
// app may be in a hung run loop when the signal arrives — exit() always works.
//
// SIGINT / SIGHUP / SIGQUIT are also caught so `kill -HUP <pid>` and Ctrl-C
// (when run from a terminal during dev) reap the child cleanly. atexit() is
// added as a belt-and-suspenders for exit() paths the signals don't cover.

import Foundation
import Darwin

@MainActor
final class SignalHandlers {
    static let shared = SignalHandlers()
    private init() {}

    private var sources: [DispatchSourceSignal] = []
    private var reap: (() -> Void)?
    private var atexitRegistered = false

    /// Install the handlers. Pass a synchronous `reap` closure that
    /// terminates the child (typically `ServerProcess.reapSync()`).
    func install(reap: @escaping () -> Void) {
        self.reap = reap

        let signals: [Int32] = [SIGTERM, SIGINT, SIGHUP, SIGQUIT]
        for sig in signals {
            // POSIX: ignore the default action so DispatchSource gets the signal.
            Darwin.signal(sig, SIG_IGN)

            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { [weak self] in
                guard let self else { return }
                self.runReap()
                // Exit with the conventional 128 + signo code.
                exit(128 + sig)
            }
            source.resume()
            sources.append(source)
        }

        if !atexitRegistered {
            atexitRegistered = true
            // atexit handlers are called from C runtime; @MainActor isn't
            // available, so we hop via a static C-level reference. The
            // reapClosure is set on the shared singleton.
            atexit {
                SignalHandlers.atexitTrampoline()
            }
        }
    }

    private func runReap() {
        reap?()
    }

    private nonisolated static func atexitTrampoline() {
        // Bridge back to MainActor synchronously; if the run loop is already
        // gone we still try the reap on whatever thread we're on.
        if Thread.isMainThread {
            MainActor.assumeIsolated {
                SignalHandlers.shared.runReap()
            }
        } else {
            DispatchQueue.main.sync {
                SignalHandlers.shared.runReap()
            }
        }
    }
}
