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
    ///
    /// Safe to call more than once — calling again replaces the `reap`
    /// closure (e.g. when the welcome wizard finishes and the spawned
    /// ServerProcess becomes the one we want to clean up) and tears
    /// down any previously-registered DispatchSourceSignal handles
    /// before re-installing fresh ones so we never end up with stacked
    /// signal sources routing into stale closures.
    func install(reap: @escaping () -> Void) {
        self.reap = reap

        // Cancel any previously-registered signal sources before
        // re-installing. Without this, a second install() leaks a
        // parallel set of DispatchSourceSignal handles attached to the
        // same signals, all firing the (now-stale) closure.
        for source in sources {
            source.cancel()
        }
        sources.removeAll()

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
