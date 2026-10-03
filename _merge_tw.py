import re

def split_atoms(css):
    """Split a (minified) stylesheet body into ordered top-level atoms.

    Atom = selector{body} | @media(@anchor;@archive){...} | @keyframes name{...}.
    Returns list of dicts: {'sel': selector, 'body': body, 'text': full_text}
    For nesting (media/keyframes), body is content between outer braces.
    """
    atoms = []
    i = 0
    n = len(css)
    while i < n:
        brace = css.find('{', i)
        if brace < 0:
            break
        # sel is css[i:brace] — normalize whitespace and skip junk between rules
        sel = css[i:brace].strip()
        if not sel:
            i = brace + 1
            continue
        # find matching close brace
        depth = 1
        j = brace + 1
        while j < n and depth:
            if css[j] == '{':
                depth += 1
            elif css[j] == '}':
                depth -= 1
            j += 1
        body = css[brace + 1:j - 1]
        text = css[i:j]
        atoms.append({'sel': re.sub(r'\s+', '', sel), 'raw_sel': sel, 'body': body, 'text': text})
        i = j
    return atoms

def parse(text):
    return split_atoms(text)

def merge_atoms(theirs_list, ours_list):
    """Union: theirs order/base, theirs wins shared keys, ours-only anchored in."""
    out = list(theirs_list)
    anchors = [a['sel'] for a in ours_list]
    ours_only = [a for a in ours_list if a['sel'] not in {b['sel'] for b in theirs_list}]
    for atom in ours_only:
        idx = anchors.index(atom['sel'])
        pos = None
        for prev in reversed(anchors[:idx]):
            match = [(k, x) for k, x in enumerate(out) if x['sel'] == prev]
            if match:
                pos = match[0][0] + 1
                break
        if pos is None:
            for nxt in anchors[idx + 1:]:
                match = [(k, x) for k, x in enumerate(out) if x['sel'] == nxt]
                if match:
                    pos = match[0][0]
                    break
        # media queries: merge bodies if the same @media key exists later in theirs
        if atom['sel'].startswith('@media'):
            anchor = next((x for x in out if x['sel'] == atom['sel']), None)
            if anchor is not None:
                # merge nested atoms of the media body
                anchor['body'] = merge_atoms(
                    split_atoms(anchor['body']), split_atoms(atom['body'])
                )
                anchor['text'] = anchor['raw_sel'] + '{' + ''.join(x['text'] if isinstance(x, dict) and 'text' in x else x.raw_sel + '{' + x['body'] + '}' for x in anchor['body']) + '}'
                continue
        if pos is None:
            out.append(atom)
        else:
            out.insert(pos, atom)
    return out

def serialize(atoms):
    parts = []
    for a in atoms:
        if a['sel'].startswith('@media') or a['sel'].startswith('@keyframes') or a['sel'].startswith('@supports'):
            parts.append(a['raw_sel'] + '{' + serialize(a.get('children', [])) + '}')
        else:
            parts.append(a['sel'].replace(' ', '') + '{' + a['body'] + '}')
    return ''.join(parts)

def split_with_children(css):
    """Like split_atoms but media/keyframes atoms get children parsed."""
    atoms = []
    i, n = 0, len(css)
    while i < n:
        brace = css.find('{', i)
        if brace < 0:
            break
        sel = re.sub(r'\s+', '', css[i:brace].strip())
        if not sel:
            i = brace + 1
            continue
        depth, j = 1, brace + 1
        while j < n and depth:
            if css[j] == '{':
                depth += 1
            elif css[j] == '}':
                depth -= 1
            j += 1
        body = css[brace + 1:j - 1]
        atom = {'sel': sel, 'raw_sel': css[i:brace].strip(), 'body': body}
        if sel.startswith(('@media', '@keyframes', '@supports')):
            atom['children'] = split_with_children(body)
        atoms.append(atom)
        i = j
    return atoms

def union(old_css, new_css):
    """theirs=new wins on shared keys; old-only atoms are anchored in."""
    new = split_with_children(new_css)
    old = split_with_children(old_css)
    new_keys = {a['sel'] for a in new}
    old_keys = [a['sel'] for a in old]
    ours_only = [a for a in old if a['sel'] not in new_keys]

    def find(lst, key, start=0):
        for k in range(start, len(lst)):
            if lst[k]['sel'] == key:
                return k
        return None

    def insert_atom(atom):
        idx = old_keys.index(atom['sel'])
        pos = None
        for k in range(idx - 1, -1, -1):
            p = find(new, old_keys[k])
            if p is not None:
                pos = p + 1
                break
        if pos is None:
            for k in range(idx + 1, len(old_keys)):
                p = find(new, old_keys[k])
                if p is not None:
                    pos = p
                    break
        if pos is None:
            new.append(atom)
        else:
            new.insert(pos, atom)

    for atom in ours_only:
        if atom['sel'].startswith(('@media', '@supports')):
            p = find(new, atom['sel'])
            if p is not None:
                # merge media bodies recursively
                target_children = {c['sel']: c for c in new[p]['children']}
                for child in atom['children']:
                    if child['sel'] not in target_children:
                        new[p]['children'].append(child)
                continue
        if atom['sel'].startswith('@keyframes') and atom['sel'] in new_keys:
            continue
        insert_atom(atom)

    return new

def serialize2(atoms):
    parts = []
    for a in atoms:
        if a['sel'].startswith(('@media', '@supports')):
            parts.append(a['raw_sel'] + '{' + serialize2(a['children']) + '}')
        else:
            parts.append(a['raw_sel'] + '{' + a['body'] + '}')
    return ''.join(parts)

def flatten(atoms, out):
    for a in atoms:
        out.append(a)
        if 'children' in a:
            flatten(a['children'], out)
    return out

ours = open('_tw_ours.css').read()
theirs = open('_tw_theirs.css').read()
merged = union(ours, theirs)

# Validation 1: every old (ours) atom exists in merged by selector-with-body
def atom_key(a):
    return a['raw_sel'] + '{' + a['body'] + '}'

merged_flat = serialize2(merged)
merged_tree = split_with_children(merged_flat)
merged_keys = set()
def collect(atoms, acc):
    for a in atoms:
        acc.add((a['sel'], a['body']))
        if 'children' in a:
            collect(a['children'], acc)
    return acc
mk = collect(merged_tree, set())
ok = collect(split_with_children(ours), set())
tk = collect(split_with_children(theirs), set())
missing_from_ours = ok - mk
missing_from_theirs = tk - mk
print('ours atoms missing:', len(missing_from_ours))
print('theirs atoms missing:', len(missing_from_theirs))
for k in list(missing_from_ours)[:5]:
    print('  ours-miss:', k[0][:80], k[1][:60])
for k in list(missing_from_theirs)[:5]:
    print('  theirs-miss:', k[0][:80], k[1][:60])

# Validation 2: inspect the media merge —した bodies joined sequentially (valid CSS)
assert len(missing_from_ours) == 0, 'ours atom lost'
assert len(missing_from_theirs) == 0, 'theirs atom lost'

# Validation 3: brace balance
assert merged_flat.count('{') == merged_flat.count('}'), 'brace imbalance'
open('omlx/admin/static/css/tailwind.css', 'w', encoding='utf-8').write(merged_flat + '\n')
print('written, bytes =', len(merged_flat) + 1)
# test contract strings
for s in ['.max-h-40{', '.z-\\[200\\]{']:
    assert s in merged_flat, s
print('test-contract strings OK')
