# Pin-knee matrix (JANG 4S/4M, 2026-09-04)

Prediction under test: SCH knee at 16 slots/layer (see ../jang4s_decode.json)
implies pins at the knee beat page-cache-only decode.

Protocol: frozen bench (budget 0, short prompt, decode 96, --min-free-gb 12),
interleaved A (no pins) / C (knee pins, own v2 profile, PIN_SYNC=1, KEEP=512).
Arms: c_probe and 4m_c1 are single-run probes (window-drift artifacts);
decisions use only the interleaved triples.

| arm | pin budget | ~slots/layer | tok/s reps | median | delta vs A |
|---|---|---|---|---|---|
| 4S A | 0 | 0 | 2.862 / 2.858 / 2.864 | 2.862 | — |
| 4S C | 1.5 GiB | 16 | 2.852 / 2.856 / 2.831 | 2.852 | **−0.3% (null)** |
| 4M A | 0 | 0 | 2.087 / 1.974 / 1.977 | 1.977 | — |
| 4M C | 2.0 GiB | 16 | 1.873 / 1.978 / 1.972 | 1.972 | **−0.3% (null)** |

Verdict: the SCH knee is NOT a page-pin budget. L2-null (oQ4e) reproduces on
both JANGs — the page cache already serves the decode working set up to the
oracle ceiling on this box; mlock only trades evictable for wired. The knee
sizes the future slot-bank instead (device-side residency): 16 slots/layer
captures ~65% SCH, 64 captures the ~77.5–77.8% ceiling.

Transfer check: loading the 4S profile on 4M logs `fingerprint mismatch —
profile ignored` (config_sha + packing oQ4e3b vs oQ4e4b differ) and degrades
to in-run observation. Profiles do not transfer across packings, by design.

Profiles: profiles/decode_4s.json, profiles/decode_4m.json (v2, 48 layers,
fingerprint-matched). Learned-pin store machinery itself verified working
(observe → pin at load in 76–1076 ms).
