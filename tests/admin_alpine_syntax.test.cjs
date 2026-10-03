// SPDX-License-Identifier: Apache-2.0
// Every Alpine expression in the templates must parse. The dashboard is one
// 1 MB HTML document whose attributes carry real JavaScript; a typo in one of
// them (which the Jinja tests cannot see) breaks the whole page at runtime.
// Run with: node --test tests/admin_alpine_syntax.test.cjs
const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const TEMPLATES = path.join(root, 'omlx/admin/templates');

function* templates(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) yield* templates(full);
        else if (entry.name.endsWith('.html')) yield full;
    }
}

const ATTRIBUTE = /([\w:@.-]+)="([^"]*)"/g;
// Alpine attributes hold JavaScript; the exception is the transition family,
// which holds class lists.
const ALPINE = /^(x-|:|@)/;
const NOT_JAVASCRIPT = /^(x-transition|:(enter|leave))/;
const JINJA = /\{\{[\s\S]*?\}\}|\{%[\s\S]*?%\}/g;

function expressions(text) {
    const found = [];
    for (const match of text.matchAll(ATTRIBUTE)) {
        const name = match[1];
        const raw = match[2];
        if (!ALPINE.test(name) || NOT_JAVASCRIPT.test(name)) continue;
        // Jinja-built values (and the truncated captures they can leave
        // behind) are rendered before Alpine sees them.
        if (JINJA.test(raw) || raw.includes('{{') || raw.includes('{%')) continue;
        const line = text.slice(0, match.index).split('\n').length;
        found.push({ value: raw, line, name });
    }
    return found;
}

test('every Alpine attribute holds a parseable expression', () => {
    const failures = [];
    let checked = 0;
    for (const file of templates(TEMPLATES)) {
        const text = fs.readFileSync(file, 'utf8');
        for (const { value, line, name } of expressions(text)) {
            const body = value.trim();
            if (!body) continue;
            checked += 1;
            try {
                // Alpine evaluates a statement list, and object literals
                // (:style / :class) need the expression form.
                try {
                    new Function(body); // eslint-disable-line no-new-func
                } catch (statementError) {
                    new Function(`return (${body});`); // eslint-disable-line no-new-func
                }
            } catch (error) {
                failures.push(`${path.relative(root, file)}:${line} ${name} ${error.message}\n    ${body}`);
            }
        }
    }
    assert.ok(checked > 500, `expected to check the whole dashboard, saw ${checked}`);
    assert.equal(failures.length, 0, failures.join('\n'));
});
