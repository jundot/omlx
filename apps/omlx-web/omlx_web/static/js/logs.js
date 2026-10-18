/*
 * Record view of the Logs tab. Pure functions with no DOM access, so the tests
 * run them under node (tests/test_admin_logs_viewer.py).
 *
 * The logs API returns the last N lines of a file. Every poll parses that
 * window again and keys each record by its absolute line number in the file:
 * a key stays the same while its record is in the window and never moves to
 * another record.
 */
(function (global) {
    'use strict';

    var LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
    // Only WARNING and above repeat often enough to collapse.
    var AGGREGATE_FROM = LEVELS.indexOf('WARNING');

    // `%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s`
    var HEADER = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.*?) - (TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL) - (?:\[.*?\] - )?([\s\S]*)$/;

    // Memory-guard advisories the server logs.
    var MEMORY_GUARD = [
        /process memory limit exceeded/,
        /Prefill would require/,
        /Hard memory pressure/,
        /Aborted \d+ requests? due to memory pressure/,
        /raise memory_guard_tier/,
    ];

    // A warning can repeat tens of thousands of times; the detail panel lists
    // only the first ones.
    var OCCURRENCE_WINDOW = 200;

    function levelRank(level) {
        return LEVELS.indexOf(String(level == null ? '' : level).toUpperCase());
    }

    /* One record per header line. Lines without a header (tracebacks) join the
       record above them. `totalLines` is the line count of the whole file that
       `text` is the tail of. */
    function parseLogText(text, totalLines) {
        var lines = String(text == null ? '' : text).split('\n');
        if (lines.length && lines[lines.length - 1] === '') lines.pop();
        var first = Math.max(0, (totalLines || 0) - lines.length);
        var records = [];
        for (var i = 0; i < lines.length; i++) {
            var line = lines[i].replace(/\r$/, '');
            var match = HEADER.exec(line);
            var open = records[records.length - 1];
            if (!match && open) {
                open.message += '\n' + line;
                open.lines += 1;
                continue;
            }
            // Without a header this is the tail of a record cut by the window.
            records.push({
                key: 'L' + (first + i),
                time: match ? match[1] : '',
                module: match ? match[2] : '',
                level: match ? match[3] : '',
                message: match ? match[4] : line,
                lines: 1,
                continuation: !match,
            });
        }
        return records;
    }

    /* Consecutive identical warnings and errors become one row with a count.
       Rows from `previousRows` are reused by key and only written when a value
       changed, so a poll patches the rows on screen instead of replacing them. */
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
            if (!run.occurrences || run.occurrences.length !== times.length) run.occurrences = times;
        }

        for (var i = 0; i < (records || []).length; i++) {
            var record = records[i];
            var rank = levelRank(record.level);
            if (minimum > 0 && rank >= 0 && rank < minimum) {
                // A hidden line ends a run.
                closeRun();
                run = null;
                continue;
            }
            if (run && rank >= AGGREGATE_FROM && run.rank === rank
                && record.module === run.module && record.message === run.message) {
                times.push(record.time);
                continue;
            }
            closeRun();
            var row = reusable[record.key] || { key: record.key };
            if (row.time !== record.time) row.time = record.time;
            if (row.level !== record.level) row.level = record.level;
            if (row.rank !== rank) row.rank = rank;
            if (row.module !== record.module) row.module = record.module;
            if (row.message !== record.message) {
                row.message = record.message;
                row.memory = memoryGuardFor(record.message);
            }
            if (row.lines !== record.lines) row.lines = record.lines;
            if (row.continuation !== record.continuation) row.continuation = record.continuation;
            run = row;
            times = [record.time];
            rows.push(row);
        }
        closeRun();
        return rows;
    }

    function numberField(text, pattern) {
        var match = text.match(pattern);
        return match ? { value: parseFloat(match[1]), text: match[1] } : null;
    }

    /* The numbers in a memory-guard line, or null for any other line. */
    function memoryGuardFor(message) {
        var text = String(message == null ? '' : message);
        if (!MEMORY_GUARD.some(function (pattern) { return pattern.test(text); })) return null;
        var current = numberField(text, /\(current ([\d.]+) GB/);
        var ceiling = text.match(/\b[a-z_/]+ ceiling (?:is )?([\d.]+) GB/);
        return {
            usage: numberField(text, /\(usage ([\d.]+) GB/) || current,
            watermark: numberField(text, /abort threshold \(hard watermark\) ([\d.]+) GB/),
            ceiling: ceiling ? { value: parseFloat(ceiling[1]), text: ceiling[1] } : null,
            peak: numberField(text, /require ~?([\d.]+) GB peak/),
        };
    }

    /* Chips for the guard's numbers. The value text is the server's own, so a
       chip never rounds differently from the log. */
    function memoryGuardChips(guard, labels) {
        if (!guard) return [];
        var caption = labels || {};
        return ['usage', 'watermark', 'ceiling', 'peak']
            .filter(function (name) { return guard[name] && caption[name]; })
            .map(function (name) {
                return { key: name, label: caption[name], value: guard[name].text + ' GB' };
            });
    }

    function occurrenceWindow(occurrences, limit) {
        var list = occurrences || [];
        var max = limit > 0 ? limit : OCCURRENCE_WINDOW;
        return { shown: list.slice(0, max), hidden: Math.max(0, list.length - max) };
    }

    /* Rows to mount for a scroll offset: the visible slice plus overscan. */
    function visibleRange(total, scrollTop, viewportHeight, rowHeight, overscan) {
        var count = Math.max(0, total || 0);
        var pitch = rowHeight > 0 ? rowHeight : 1;
        var padding = overscan > 0 ? overscan : 0;
        var perScreen = Math.max(1, Math.ceil((viewportHeight > 0 ? viewportHeight : 0) / pitch));
        // A filter can shrink the list under a scroll offset the browser has
        // not corrected yet.
        var maxStart = Math.max(0, count - perScreen);
        var start = Math.min(maxStart, Math.max(0, Math.floor(Math.max(0, scrollTop) / pitch) - padding));
        return { start: start, end: Math.min(count, start + perScreen + padding * 2) };
    }

    var api = {
        LEVELS: LEVELS,
        OCCURRENCE_WINDOW: OCCURRENCE_WINDOW,
        levelRank: levelRank,
        parseLogText: parseLogText,
        aggregateLogRows: aggregateLogRows,
        memoryGuardFor: memoryGuardFor,
        memoryGuardChips: memoryGuardChips,
        occurrenceWindow: occurrenceWindow,
        visibleRange: visibleRange,
    };
    global.OmlxLogs = api;
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
