# Discussion draft — running a model that does not fit one Mac (2026-11-15)

Ready to post as a GitHub **Discussion** (or the opening comment of a design
issue) in `jundot/omlx`. It asks for appetite before eight PRs are opened. Keep
it short; the PR bodies carry the detail.

---

**Title:** Cluster Assistant: run a model that does not fit one Mac, inside the existing app

**Body:**

oMLX serves a model on one Mac very well. The case that does not work today is a
checkpoint that does not fit one Mac's memory but does fit two to four Macs
connected over Thunderbolt — or a model that fits but needs more context than one
Mac can hold.

We built that flow inside the existing app rather than next to it:

- **No new control plane.** It is a card in the existing Cluster tab of the
  dashboard, driven by the same admin API the tab already calls. Rank 0 is the
  elected coordinator and owns the endpoint, as oMLX already assumes.
- **Measure, then plan.** Each Mac reports its real admission and model
  inventory; a recommendation picks a model and context that fit; the plan picks
  the fewest Macs that work, the rank order and a resource policy, and shows the
  tradeoff when experts are offloaded (smaller context at full speed vs larger
  context with streaming). Nothing is deployed by measuring or planning.
- **Pipeline parallelism only.** The split is a pipeline across Macs; general
  tensor parallelism is deliberately out of scope for this round.
- **Apply and start reuse what is there.** Staging, deployment, launch, the SSH
  control plane and the endpoint stay oMLX's; the assistant only decides *where*
  and *how much*, then drives those paths.
- **Smaller cases are covered too.** When a model fits one Mac, the answer is
  "run it here" (single node), which is measurably faster than a cluster for the
  same checkpoint.

Along the way we found five unrelated defects worth fixing on their own, already
prepared as separate, self-contained PRs: a single-Mac load silently adopting a
stale cluster deployment, a test-collection collision, three macOS defects (the
theme colour assertion on macOS 27, a locale-dependent percent separator, a
menu-bar probe that raises a Full Disk Access dialog during tests), and a
discovery/inventory fallback that drops the peer's name, chip and RAM.

**What we would like to know before opening the big PRs:**

1. Is a cluster workflow inside oMLX in scope for you, and would you want it as
   one feature behind the existing Cluster tab?
2. Are you open to pipeline parallelism as the only mode for this round?
3. Would you prefer the assistant split into three PRs (plan / run / run-local)
   that build on each other, or a different shape?
4. Two of our earlier PRs (#5024, #5033) are part of the foundation; both are
   rebased on current `main` and green. Would you rather see those merged first?

We will not open the assistant PRs before this discussion settles. The five small
fixes are independent and can go first if you want them.

---

## Notes for us (not for posting)

- Merge order: F1 (`local-only-load`), F2 (`test-infra-conftest`),
  F4 (`macos-theme-locale-xctest`) → F5 (`cluster-discovery-inventory`) →
  #5024, #5033 → assistant `plan` → `run` → `run-local`.
- The optional macOS helper/DMG feature is **not** part of this discussion; it is
  an app proposal of its own and is not required for the cluster work.
- If the maintainers prefer to take only the five fixes, that is a complete and
  useful outcome; nothing here depends on the assistant landing.
- Screenshots for the UI PRs are still to be captured (finding R8 in
  `UPSTREAM-PR-REVIEW-2026-11-15.md`).