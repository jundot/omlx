/*
 * Log structure for the admin console's Logs tab.
 *
 * `GET /admin/api/logs` hands back raw text, so every structure on that screen
 * is derived here rather than asked of the server:
 *
 *   parseLogText()     one record per header line
 *                      (`%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s`),
 *                      with the continuation lines of a traceback or a
 *                      parameter list attached to the record they belong to. A
 *                      window can start mid-record, and that fragment is kept
 *                      as a headerless record rather than thrown away.
 *   mergeLogText()     what one poll added to the previous one. The API returns
 *                      the last N lines, so a refresh both drops lines off the
 *                      front and appends new ones; matching the overlap by line
 *                      is what lets the viewer append instead of rebuild (a
 *                      rebuild loses the scroll position).
 *   aggregateLogRows() consecutive identical warnings and errors as one row
 *                      with every occurrence behind it. INFO and below stay
 *                      one row per line: a run of INFO is not a repetition.
 *   memoryGuardFor()   the numbers a memory-guard line carries, so the row can
 *                      show usage/watermark/ceiling chips and the two inline
 *                      remedies instead of a paragraph of advice.
 *   visibleRange()     which rows to mount for a scroll offset; the viewer
 *                      keeps only the window (plus overscan) in the DOM.
 *   occurrenceWindow()  how many occurrences of a repeated line the detail
 *                      panel lists — the same windowing idea, for the list
 *                      behind a row's ×N badge.
 *
 * Pure functions, no DOM and no translation: tests/admin_logs.test.cjs runs
 * them directly under node.
 */
(function (global) {
    'use strict';

    var LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
    // Only WARNING and above repeat enough to be worth collapsing.
    var AGGREGATE_FROM = LEVELS.indexOf('WARNING');

    var HEADER = new RegExp(
        '^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2},\\d{3}) - (.*?) - ' +
        '(TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL) - (?:\\[(.*?)\\] - )?([\\s\\S]*)$'
    );

    // The four memory-guard advisories the server emits, plus the remedy text
    // the guard appends to any of them.
    var MEMORY_GUARD = [
        /process memory limit exceeded/,
        /Prefill would require/,
        /Hard memory pressure/,
        /Aborted \d+ requests? due to memory pressure/,
        /raise memory_guard_tier/,
    ];

    /* Index into the six levels; -1 for a line that is not a record header. */
    function levelRank(level) {
        return LEVELS.indexOf(String(level == null ? '' : level).toUpperCase());
    }

    /* Character offsets of each line in `text` (a trailing newline terminates
       the last line rather than starting an empty one). */
    function lineOffsets(text) {
        var offsets = [0];
        for (var i = 0; i < text.length; i++) {
            if (text.charCodeAt(i) === 10) offsets.push(i + 1);
        }
        if (offsets.length > 1 && offsets[offsets.length - 1] === text.length) offsets.pop();
        return offsets;
    }

    function lineAt(text, offsets, index) {
        var start = offsets[index];
        var end = index + 1 < offsets.length ? offsets[index + 1] - 1 : text.length;
        // The line terminator belongs to the separator, not to the line; a
        // rotated file copied off another platform can carry CRLF.
        if (end > start && text.charCodeAt(end - 1) === 10) end -= 1;
        if (end > start && text.charCodeAt(end - 1) === 13) end -= 1;
        return text.slice(start, end);
    }

    /* Every line of `text` as a record. `baseIndex` keeps `index` (and with it
       the row keys the viewer keys its DOM on) stable across appends. */
    function parseLogText(text, baseIndex) {
        var source = String(text == null ? '' : text);
        var offsets = lineOffsets(source);
        var base = baseIndex || 0;
        var records = [];

        for (var i = 0; i < offsets.length; i++) {
            var line = lineAt(source, offsets, i);
            var match = line.match(HEADER);
            if (match) {
                records.push({
                    index: base + records.length,
                    time: match[1],
                    module: match[2],
                    level: match[3],
                    requestId: match[4] === undefined ? '-' : match[4],
                    message: match[5],
                    lines: 1,
                    continuation: false,
                });
                continue;
            }
            var open = records[records.length - 1];
            if (!open) {
                // The window starts inside a record (a traceback cut in half).
                // Keep the fragment and say so instead of dropping it.
                records.push({
                    index: base,
                    time: '',
                    module: '',
                    level: '',
                    requestId: '-',
                    message: line,
                    lines: 1,
                    continuation: true,
                });
                continue;
            }
            open.message += '\n' + line;
            open.lines += 1;
        }
        return records;
    }

    /* A record that was still being written when we last polled gains its
       remaining lines in the next one; they belong to it, not to a new row. */
    function absorbContinuation(record, appended) {
        if (!record || !appended || !appended.lines) return false;
        record.message += '\n' + appended.message;
        record.lines += appended.lines;
        return true;
    }

    /* What changed between two polls of the same file. */
    function mergeLogText(previous, incoming) {
        var before = String(previous == null ? '' : previous);
        var next = String(incoming == null ? '' : incoming);
        if (!before || !next) {
            return { reset: true, dropped: 0, overlap: 0, appended: next };
        }
        var beforeOffsets = lineOffsets(before);
        if (before === next) {
            return { reset: false, dropped: 0, overlap: beforeOffsets.length, appended: '' };
        }

        var nextOffsets = lineOffsets(next);
        var head = lineAt(next, nextOffsets, 0);
        if (!head) return { reset: true, dropped: 0, overlap: 0, appended: next };

        // A header line is unique in practice (it carries a timestamp), so
        // matching the first new line and walking back from there is enough;
        // the overlap is then verified line by line before it is trusted.
        for (var start = beforeOffsets.length - 1; start >= 0; start--) {
            if (lineAt(before, beforeOffsets, start) !== head) continue;
            var overlap = beforeOffsets.length - start;
            if (overlap > nextOffsets.length) continue;
            var same = true;
            for (var i = 1; i < overlap; i++) {
                if (lineAt(before, beforeOffsets, start + i) !== lineAt(next, nextOffsets, i)) {
                    same = false;
                    break;
                }
            }
            if (!same) continue;
            return {
                reset: false,
                dropped: start,
                overlap: overlap,
                appended: overlap < nextOffsets.length ? next.slice(nextOffsets[overlap]) : '',
            };
        }
        // No shared line: the file rotated or was truncated.
        return { reset: true, dropped: 0, overlap: 0, appended: next };
    }

    /* Consecutive identical warnings/errors collapse; everything else is one
       row per record. `previousRows` are reused by key — and only written to
       when a value really changed — so an append patches the rows the viewer
       already has instead of replacing them, and a poll with no new lines
       touches nothing in the DOM at all. */
    function aggregateLogRows(records, minLevel, previousRows) {
        var minimum = levelRank(minLevel);
        var reusable = {};
        (previousRows || []).forEach(function (row) {
            reusable[row.key] = row;
        });

        var rows = [];
        var run = null;
        var times = null;

        function closeRun() {
            if (!run) return;
            if (run.count !== times.length) run.count = times.length;
            // Occurrences only ever grow at the tail of a run, so a matching
            // length means a matching list; the key cannot be recycled either,
            // because record indices are never reused.
            if (!run.occurrences || run.occurrences.length !== times.length) {
                run.occurrences = times;
            }
        }

        for (var i = 0; i < (records || []).length; i++) {
            var record = records[i];
            var rank = levelRank(record.level);
            if (minimum > 0 && rank >= 0 && rank < minimum) {
                // A filtered-out line ends a run: what is left is not consecutive.
                closeRun();
                run = null;
                continue;
            }
            if (
                run &&
                rank >= AGGREGATE_FROM &&
                run.rank >= AGGREGATE_FROM &&
                record.level === run.level &&
                record.module === run.module &&
                record.message === run.message
            ) {
                times.push({ time: record.time, requestId: record.requestId });
                continue;
            }
            closeRun();

            var key = 'r' + record.index;
            var row = reusable[key] || { key: key };
            if (row.time !== record.time) row.time = record.time;
            if (row.level !== record.level) row.level = record.level;
            if (row.rank !== rank) row.rank = rank;
            if (row.module !== record.module) row.module = record.module;
            if (row.message !== record.message) row.message = record.message;
            if (row.lines !== record.lines) row.lines = record.lines;
            if (row.continuation !== record.continuation) row.continuation = record.continuation;
            if (row.memoryMessage !== record.message) {
                row.memoryMessage = record.message;
                row.memory = memoryGuardFor(record.message);
            }
            run = row;
            times = [{ time: record.time, requestId: record.requestId }];
            // The counter and the occurrence list are written once, when the
            // run closes: writing them here as well would touch every row of a
            // repeated group on every poll.
            rows.push(row);
        }
        closeRun();
        return rows;
    }

    function numberField(text, pattern) {
        var match = text.match(pattern);
        return match ? { value: parseFloat(match[1]), text: match[1] } : null;
    }

    /* The numbers behind a memory-guard line, or null when it is not one. Both
       shapes the guard emits are covered: the abort (measured usage against the
       hard watermark and the dynamic ceiling) and the prefill rejection (a
       predicted peak against the ceiling). */
    function memoryGuardFor(message) {
        var text = String(message == null ? '' : message);
        var matched = false;
        for (var i = 0; i < MEMORY_GUARD.length; i++) {
            if (MEMORY_GUARD[i].test(text)) {
                matched = true;
                break;
            }
        }
        if (!matched) return null;

        var current = numberField(text, /\(current ([\d.]+) GB/);
        var ceiling = text.match(/\b([a-z_/]+) ceiling (?:is )?([\d.]+) GB/);
        var aborted = text.match(/[Aa]borted (\d+) requests?/);
        var model = text.match(/on '([^']+)' and kept model loaded/);
        return {
            kind: /Prefill would require/.test(text)
                ? 'prefill'
                : /process memory limit exceeded/.test(text)
                    ? 'abort'
                    : 'pressure',
            usage: numberField(text, /\(usage ([\d.]+) GB/) || current,
            watermark: numberField(text, /abort threshold \(hard watermark\) ([\d.]+) GB/),
            ceiling: ceiling ? { value: parseFloat(ceiling[2]), text: ceiling[2] } : null,
            ceilingBinding: ceiling ? ceiling[1] : '',
            peak: numberField(text, /require ~?([\d.]+) GB peak/),
            current: current,
            kv: numberField(text, /\+ KV\+SDPA ([\d.]+) GB/),
            staticCap: numberField(text, /static cap is ([\d.]+) GB/),
            reclaimable: numberField(text, /only ([\d.]+) GB is reclaimable/),
            aborted: aborted ? parseInt(aborted[1], 10) : null,
            model: model ? model[1] : '',
        };
    }

    /* The guard's own numbers as chips; the text is the server's, so a chip
       never re-rounds a value the log printed. `labels` are translated by the
       caller. */
    function memoryGuardChips(guard, labels) {
        if (!guard) return [];
        var caption = labels || {};
        var fields = [
            ['usage', guard.usage],
            ['watermark', guard.watermark],
            ['ceiling', guard.ceiling],
            ['peak', guard.peak],
        ];
        var chips = [];
        for (var i = 0; i < fields.length; i++) {
            var name = fields[i][0];
            var field = fields[i][1];
            if (!field || !caption[name]) continue;
            chips.push({ key: name, label: caption[name], value: field.text + ' GB' });
        }
        return chips;
    }

    /* How many occurrences of one repeated line the detail panel lists. A
       polling loop can repeat a warning tens of thousands of times; one element
       per occurrence froze the tab (a refresh of 20,000 lines painted 20,000
       nodes), so the panel lists the first window of them and says how many are
       behind it. The count on the badge stays the full one either way. */
    var OCCURRENCE_WINDOW = 200;

    function occurrenceWindow(occurrences, limit) {
        var list = occurrences || [];
        var max = limit > 0 ? limit : OCCURRENCE_WINDOW;
        return {
            shown: list.slice(0, max),
            hidden: Math.max(0, list.length - max),
        };
    }

    /* Rows to mount for a scroll offset: the visible slice plus overscan. */
    function visibleRange(total, scrollTop, viewportHeight, rowHeight, overscan) {
        var count = Math.max(0, total || 0);
        var pitch = rowHeight > 0 ? rowHeight : 1;
        var padding = overscan > 0 ? overscan : 0;
        var perScreen = Math.max(1, Math.ceil((viewportHeight > 0 ? viewportHeight : 0) / pitch));
        // Clamp to the last screenful: a filter can shrink the list under a
        // scroll position the browser has not corrected yet.
        var maxStart = Math.max(0, count - perScreen);
        var start = Math.min(maxStart, Math.max(0, Math.floor(Math.max(0, scrollTop) / pitch) - padding));
        return { start: start, end: Math.min(count, start + perScreen + padding * 2) };
    }

    var api = {
        LEVELS: LEVELS,
        levelRank: levelRank,
        parseLogText: parseLogText,
        absorbContinuation: absorbContinuation,
        mergeLogText: mergeLogText,
        aggregateLogRows: aggregateLogRows,
        memoryGuardFor: memoryGuardFor,
        memoryGuardChips: memoryGuardChips,
        visibleRange: visibleRange,
        occurrenceWindow: occurrenceWindow,
        OCCURRENCE_WINDOW: OCCURRENCE_WINDOW,
    };

    global.OmlxLogs = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
