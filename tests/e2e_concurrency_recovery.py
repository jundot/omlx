#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2E check for per-stream throughput after cancelling half of a batch.

Protocol: N streams run concurrently for --dwell seconds, --cancel of them are dropped, and the
survivors are sampled for --recovery seconds in --bin-second windows. Every rate is computed from
the text that arrived in the window, encoded with the checkpoint tokenizer, so a frame carrying
several tokens is not counted as one.

Usage:
    python tests/e2e_concurrency_recovery.py --model <checkpoint> --base-url http://127.0.0.1:8000
    python tests/e2e_concurrency_recovery.py --model <checkpoint> --streams 4 --cancel 0   # cold reference

What the output answers:

    * whether per-stream throughput rises immediately after the cancel, or only after minutes;
    * how far the survivors are from a cold run of the same width (--baseline-tok-s);
    * whether the drop is a per-stream effect or an aggregate one (both are printed).

Notes that make the numbers trustworthy:

    * the prompt carries a per-run nonce at the front, because the SSD prefix cache outlives the
      server process; without it a "cold" run can be served from cache;
    * ignore_eos is requested so streams do not end early; a stream that still ends is reported;
    * a window rate is only as good as the client's read: each stream's longest read stall is
      printed, and the final row repeats the rate over the last 300 s, which a short stall cannot
      move;
    * whole-run totals are printed too, and they do not depend on window placement.

Requires: Python 3.9+, `tokenizers` (already a dependency of omlx), and a running server.
"""
import argparse
import glob
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = (
    "river stone market lantern copper harvest signal garden window thunder orchard ribbon harbor "
    "meadow candle anchor velvet compass pebble violin saddle lagoon quartz bridge falcon ember"
).split()


def make_prompt(seed, words, nonce):
    """A story request long enough to decode for the whole run, salted at the front."""
    rnd = random.Random(seed)
    return (
        "[run %s stream %d] " % (nonce, seed)
        + " ".join(rnd.choice(WORDS) for _ in range(words))
        + "\n\nWrite a 20,000-word story about this town. Do not stop early; keep going to the end."
    )


class Stream:
    """One streaming request, read on its own thread, cancelled by closing the response."""

    def __init__(self, args, seed, nonce, t0):
        self.args = args
        self.seed = seed
        self.nonce = nonce
        self.t0 = t0
        self.txt = []
        self.ev = []  # (arrival offset in seconds, characters in that frame)
        self.error = None
        self.done = False
        self.stop = threading.Event()
        self.response = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.stop.set()
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass

    def _run(self):
        a = self.args
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": make_prompt(self.seed, a.words, self.nonce)}],
            "max_tokens": a.max_tokens,
            "temperature": a.temperature,
            "stream": True,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            a.base_url.rstrip("/") + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer " + a.api_key},
        )
        try:
            self.response = urllib.request.urlopen(req, timeout=a.timeout)
            for raw in self.response:
                if self.stop.is_set():
                    break
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if obj.get("error"):
                    self.error = "server error: " + json.dumps(obj["error"])[:200]
                    break
                choices = obj.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    self.txt.append(content)
                    self.ev.append((time.monotonic() - self.t0, len(content)))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode(errors="replace")[:300]
            except Exception:
                pass
            self.error = "HTTP %s: %s" % (exc.code, body)
        except Exception as exc:  # a cancelled stream raises here; that is expected
            if not self.stop.is_set():
                self.error = repr(exc)
        self.done = True


def tokens_between(stream, tokenizer, t_from, t_to):
    """Tokens of the text whose frames arrived inside [t_from, t_to)."""
    text = "".join(stream.txt)
    cumulative, acc = [], 0
    for _, chars in stream.ev:
        acc += chars
        cumulative.append(acc)

    def chars_before(x):
        lo, hi = 0, len(stream.ev)
        while lo < hi:
            mid = (lo + hi) // 2
            if stream.ev[mid][0] < x:
                lo = mid + 1
            else:
                hi = mid
        return cumulative[lo - 1] if lo else 0

    return len(tokenizer.encode(text[chars_before(t_from):chars_before(t_to)]).ids)


def build_windows(cancel_at, recovery, bin_s):
    """Half-open windows: the pre-cancel sample, the bins, and the long tail."""
    windows = [("last %.0f s before cancel" % bin_s, cancel_at - bin_s, cancel_at)]
    t = 0.0
    while t + bin_s <= recovery + 1e-9:
        windows.append(("+%.0f s" % (t + bin_s), cancel_at + t, cancel_at + t + bin_s))
        t += bin_s
    remainder = recovery - t
    if remainder >= 0.5 * bin_s:
        windows.append(("+%.0f s" % recovery, cancel_at + t, cancel_at + t + remainder))
    long_span = min(recovery, 300.0)
    windows.append(
        ("+%.0f s, last %.0f s" % (recovery, long_span), cancel_at + recovery - long_span,
         cancel_at + recovery)
    )
    return windows


def longest_silence(stream, until):
    """Longest gap between two frames, seen from this client."""
    ts = [t for t, _ in stream.ev] + [until]
    if len(ts) < 2:
        return until
    return max(b - a for a, b in zip(ts, ts[1:]))


def find_tokenizer(model):
    """tokenizer.json of a checkpoint, in the flat and the org/name layout."""
    for path in [os.path.expanduser("~/.omlx/models/%s/tokenizer.json" % model)] + sorted(
        glob.glob(os.path.expanduser("~/.omlx/models/*/%s/tokenizer.json" % model))
    ):
        if os.path.exists(path):
            return path
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint name as the server lists it")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json (default: ~/.omlx/models/<model>/tokenizer.json)")
    ap.add_argument("--api-key", default="test")
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--cancel", type=int, default=4, help="streams to drop after --dwell seconds")
    ap.add_argument("--words", type=int, default=16000, help="prompt words, roughly one token each")
    ap.add_argument("--dwell", type=float, default=240.0)
    ap.add_argument("--recovery", type=float, default=600.0)
    ap.add_argument("--bin", dest="bin_s", type=float, default=30.0)
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--nonce", default=None, help="prefix-cache salt (default: time based)")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--baseline-tok-s", type=float, default=None,
                    help="cold per-stream rate of a run with --streams equal to the survivors")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("tokenizers is required: pip install 'tokenizers>=0.19.0'")

    tok_path = args.tokenizer or find_tokenizer(args.model)
    if not tok_path:
        sys.exit("tokenizer.json not found for %s (pass --tokenizer)" % args.model)
    tokenizer = Tokenizer.from_file(tok_path)

    if args.cancel >= args.streams:
        sys.exit("--cancel must be smaller than --streams")
    nonce = args.nonce or str(int(time.time()))
    t0 = time.monotonic()
    streams = [Stream(args, i, nonce, t0) for i in range(args.streams)]
    print("%d streams, %d words, model %s, nonce %s" % (len(streams), args.words, args.model, nonce))
    for s in streams:
        s.start()

    time.sleep(args.dwell)
    cancel_at = time.monotonic() - t0
    for s in streams[: args.cancel]:
        s.cancel()
    print("%d of %d cancelled at t=%.1f s" % (args.cancel, args.streams, cancel_at))
    time.sleep(args.recovery)
    stop_at = time.monotonic() - t0
    for s in streams:
        s.cancel()
    for s in streams:
        s.thread.join(timeout=5)

    survivors = streams[args.cancel:]
    failed = [s for s in survivors if not s.txt]
    if failed:
        print("warning: %d survivor(s) produced nothing: %s"
              % (len(failed), [s.error for s in failed]))
        survivors = [s for s in survivors if s.txt]
    if not survivors:
        sys.exit("no survivor produced output")

    windows = build_windows(cancel_at, args.recovery, args.bin_s)
    rows = [[tokens_between(s, tokenizer, a, b) / (b - a) for _, a, b in windows] for s in survivors]
    aggregates = [sum(r[i] for r in rows) for i in range(len(windows))]
    print("\nper-stream tok/s")
    print("| window | " + " | ".join("stream %d" % (args.cancel + i) for i in range(len(rows)))
          + " | aggregate |")
    print("|" + "---|" * (len(rows) + 2))
    for j, (label, _, _) in enumerate(windows):
        print("| %s | " % label + " | ".join("%.1f" % r[j] for r in rows)
              + " | %.1f |" % aggregates[j])

    for i, s in enumerate(survivors):
        print("stream %d: %d frames, longest read stall %.2f s%s"
              % (args.cancel + i, len(s.ev), longest_silence(s, stop_at),
                 (" error: " + s.error) if s.error else ""))

    totals = [tokens_between(s, tokenizer, cancel_at, stop_at) for s in survivors]
    span = stop_at - cancel_at
    print("whole recovery: %s tokens, per stream %.1f tok/s"
          % (totals, sum(totals) / len(totals) / span))

    if args.baseline_tok_s:
        first = aggregates[1] / len(survivors) if len(aggregates) > 1 else 0.0
        last = aggregates[-1] / len(survivors)
        print("first bin %.1f %% of the cold reference, last row %.1f %%"
              % (100 * first / args.baseline_tok_s, 100 * last / args.baseline_tok_s))

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"model": args.model, "streams": args.streams, "cancel": args.cancel,
                       "dwell": args.dwell, "recovery": args.recovery,
                       "cancel_at": cancel_at, "stop_at": stop_at,
                       "windows": [w[0] for w in windows], "rows": rows,
                       "aggregates": aggregates,
                       "stalls": [longest_silence(s, stop_at) for s in survivors]}, fh, indent=2)
        print("wrote %s" % args.json_out)


if __name__ == "__main__":
    main()
