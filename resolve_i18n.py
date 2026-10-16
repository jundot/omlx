#!/usr/bin/env python3
"""Resolve omlx/admin/i18n/*.json merge conflicts per the validated recipe.

- en.json: shrug off HEAD's Dead v1-cluster ours-block; keep main's theirs-block
  + all auto-merged (region) content (full key-level union minus the dead block).
- non-en: same sentence: region + theirs (main's) + ours-kept (head-only keys,
  kept only if covered by merged en and non-redundant), then a global
  redundancy-drop pass (entries byte-identical to merged en get dropped,
  except REQUIRED_RAW keys demanded raw by auto-merged tests).
  Values: main's value wins when main has the key, else head's.
"""
import json, os, re, sys
from pathlib import Path

TMP = '/var/folders/j_/7b1r_x0n21sfh2pbc1m57r0c0000gn/T/'
I18N = Path('omlx/admin/i18n')
LOCALES = ['es', 'fr', 'ja', 'ko', 'pt-BR', 'ru', 'zh', 'zh-TW', 'cs']

REQUIRED_RAW = {
    "settings.mcp.expose_tools", "settings.mcp.expose_tools_hint",
    "settings.usage.section_label", "settings.usage.history",
    "settings.usage.history_hint", "usage.disabled", "usage.open_settings",
    "settings.auth.skip_verification_warning", "js.error.api_key_required_network",
}


def split_conflict(path):
    """-> ordered list of ('region', lines) / ('hunk', ours_lines, theirs_lines)."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    segs, st = [], 'region'
    region, ours, theirs = [], [], []
    for l in lines:
        if l.startswith('<<<<<<<'):
            if region:
                segs.append(('region', region)); region = []
            ours, theirs = [], []
            st = 'ours'
        elif l == '=======':
            st = 'theirs'
        elif l.startswith('>>>>>>>'):
            segs.append(('hunk', ours, theirs))
            ours, theirs = [], []
            st = 'region'
        else:
            (region if st == 'region' else ours if st == 'ours' else theirs).append(l)
    if region:
        segs.append(('region', region))
    return segs


def piece_pairs(lines):
    """Parse an entry list (region or one side of a hunk) into ordered pairs."""
    txt = '\n'.join(lines)
    if not txt.strip():
        return []
    # strip outer braces at piece edges
    txt = txt.strip()
    if txt.startswith('{'):
        txt = txt[1:]
    if txt.rstrip().endswith('}'):
        txt = txt.rstrip()[:-1]
    txt = txt.strip()
    if txt.endswith(','):
        txt = txt[:-1]
    txt = txt.strip()
    if not txt:
        return []
    obj = json.loads('{' + txt + '}')
    # recover insertion order (json.loads via dict preserves it)
    return list(obj.items())


def write_locale(path, ordered):
    lines = ['{']
    for i, (k, v) in enumerate(ordered):
        tail = ',' if i < len(ordered) - 1 else ''
        lines.append(f'  {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}{tail}')
    lines.append('}')
    text = '\n'.join(lines) + '\n'
    json.loads(text)          # validate
    Path(path).write_text(text, encoding='utf-8')
    return text


def flatten(segs):
    """-> ordered (key, value, piece_src) triples."""
    out = []
    for seg in segs:
        if seg[0] == 'region':
            for k, v in piece_pairs(seg[1]):
                out.append((k, v, 'region'))
        else:
            for k, v in piece_pairs(seg[2]):
                out.append((k, v, 'theirs'))
            for k, v in piece_pairs(seg[1]):
                out.append((k, v, 'ours'))
    return out


# ---------- en.json ----------
en_segs = split_conflict(I18N / 'en.json')
en_head = json.load(open(TMP + 'fk_en.json'))
en_main = json.load(open(TMP + 'mn_en.json'))

triples = flatten(en_segs)
seen, ordered = set(), []
ours_seen = []
for k, v, src in triples:
    if src == 'ours':
        ours_seen.append(k)
        continue  # en.json: ours hunks are the dead v1-cluster block -> drop
    if k in seen:
        continue
    seen.add(k)
    # value precedence: main's value when main has the key, else head's
    v_final = en_main.get(k, en_head.get(k, v))
    ordered.append((k, v_final))
out = []
write_locale(I18N / 'en.json', ordered)
en_merged = dict(ordered)
print(f"en.json: {len(ordered)} keys (head {len(en_head)} / main {len(en_main)}), ours-block dropped: {len(ours_seen)}")

# reference check: only the dead v1-cluster block (head-only cluster.*) and the
# 4 settings.resource.guard_tier.* keys main deleted cleanly should be gone
fk_keys, mn_keys = set(en_head), set(en_main)
missing = (fk_keys | mn_keys) - set(en_merged)
assert not (set(en_merged) - (fk_keys | mn_keys)), "en.json gained unknown keys!"
unexpected = [k for k in missing if not (k.startswith('cluster.') or k.startswith('settings.resource.guard_tier.'))]
assert not unexpected, f"unexpected keys lost: {unexpected[:5]}"
print(f"en.json dropped vs union (v1 cluster + main-deleted guard_tier): {len(missing)}")

# ---------- non-en locales ----------
for loc in LOCALES:
    head_d = {}
    if os.path.exists(TMP + f'fk_{loc}.json') and os.path.getsize(TMP + f'fk_{loc}.json'):
        head_d = json.load(open(TMP + f'fk_{loc}.json'))
    main_d = {}
    if os.path.exists(TMP + f'mn_{loc}.json') and os.path.getsize(TMP + f'mn_{loc}.json'):
        main_d = json.load(open(TMP + f'mn_{loc}.json'))
    triples = flatten(split_conflict(I18N / f'{loc}.json'))
    seen, kept, drop_counts = set(), [], {'redundant': 0, 'orphan': 0, 'dead_ours': 0}
    for k, v, src in triples:
        if k in seen:
            drop_counts['dup'] = drop_counts.get('dup', 0) + 1
            continue
        v_final = main_d.get(k, head_d.get(k, v))
        if k not in en_merged:
            drop_counts['orphan'] += 1
            continue
        if en_merged[k] == v_final and k not in REQUIRED_RAW:
            drop_counts['redundant'] += 1
            continue
        seen.add(k)
        kept.append((k, v_final))
    write_locale(I18N / f'{loc}.json', kept)
    print(f"{loc}: kept {len(kept)} (head {len(head_d)} / main {len(main_d)}), dropped {drop_counts}")

# enforce required-raw presence in every locale (restore from merged en value)
for loc in ['en'] + LOCALES:
    path = I18N / (loc if loc != 'en' else 'en') .__str__() if False else I18N / f'{loc}.json'
    d = json.load(open(path))
    missing_req = [k for k in REQUIRED_RAW if not d.get(k)]
    if missing_req:
        items = list(d.items())
        items += [(k, en_merged[k]) for k in missing_req]
        write_locale(path, items)
        print(f"{loc}: restored required-raw {missing_req}")
print("done")
