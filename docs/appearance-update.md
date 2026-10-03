# The appearance update

One vocabulary for how oMLX looks, so the app and the console cannot drift apart:
a single token file is the source for the type scale, the colours and the
spacing; the console gets a shared layer (one number formatter, one set of
interaction pieces, one component spec) and its screens are rebuilt on it; the
app stops typing its own point sizes and names the same six steps; and the
wording that had drifted between the two surfaces is aligned.

Nothing here touches model loading, scheduling, engines or the API protocols.

## What each open PR carries

| PR | In the series | What it does | Stands on |
|---|---|---|---|
| #3810 | no | the app's Logs screen reads as records: columns, one colour per level, repeats collapsed | `main` |
| #3824 | yes | a capital T on the token words and on the `PP` / `TG` labels in the app | #3810 |
| #3826 | yes — draft | count in 万 / 亿 in Chinese; every other language keeps the SI ladder | #3824 |
| #4082 | yes — the foundation | the app's token source: `tokens.json`, the generator, the generated `DesignTokens.swift`, and the drift guard | #3826 |
| #3847 | yes — the visible half | every `.omlxText` / `.omlxMono` / `.omlxDisplay` call names one of the six steps instead of a number | #4082 |
| #3848 | yes | the ten console catalogues: a capital T on the token words, the layout block translated, the values still in English filled in | #3847 |
| #3863 | yes | a hint under a control in the model sheet spans the card instead of wrapping inside the label column | #3848 |
| #3849 | yes — console foundation | the console's own token pipeline (colours, space, radius, layout, `tokens.css`) plus the shared layer | #3863 |
| #3850 | yes | the eight console screens rebuilt on that layer, and the 10.9 MB font payload removed | #3849 |
| #3851 | yes | the chat page's right panel becomes the console's shared drawer | #3850 |
| #4005 | yes — draft | every console visual value resolves through a token instead of a Tailwind default | #3851 |
| #3773 | no | one envelope builder for both response paths | `main` |
| #3915 | no | the menubar refresh interval actually defaults to 0.5 s, pinned where CI can see it | `main` |

## Order

```
main ──► #3810 ──► #3824 ──► #3826 ──► #4082 ──► #3847 ──► #3848 ──► #3863 ──► #3849 ──► #3850 ──► #3851 ──► #4005

#3773 · #3915 : on `main` alone, and they can land anywhere in that order
```

The eleven branches form one line: each carries the ones below it, so merging
left to right leaves every PR's own change to land and nothing to rebase.
#3810 is not part of the series, but the app branches from #3824 up and the
console column behind them are built on its files, so it goes first. The rows
marked **no** otherwise take nothing from here.
