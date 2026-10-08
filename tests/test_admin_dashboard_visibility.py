"""Static checks for the dashboard's visibility resume wiring (#2868).

CI does not run JavaScript, so the shipped script is asserted the way the
other admin UI tests do it: slice the region out of ``dashboard.js`` and
check the wiring is present.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "omlx/admin/static/js/dashboard.js"


def _script() -> str:
    return DASHBOARD.read_text()


def _region(script: str, start: str, end: str, offset: int = 0) -> str:
    begin = script.index(start, offset)
    return script[begin : script.index(end, begin)]


def test_bench_listener_reattaches_the_open_sub_tab():
    """Returning to the tab re-attaches the stream of the open bench sub-tab.

    Only the throughput sub-tab was covered before, so a context bench that
    lost its stream had no recovery path at all.
    """
    listener = _region(
        _script(),
        "// When the user returns to this browser tab",
        "this.$watch('hfMlxOnly'",
    )

    assert "document.visibilityState !== 'visible'" in listener
    assert "if (this.mainTab !== 'bench') return;" in listener

    # The handle has to be dropped first: loadBenchState()/loadCtxBenchState()
    # return early while one is still set.
    assert "this.benchEventSource.close();" in listener
    assert "this.benchEventSource = null;" in listener
    assert "this.loadBenchState();" in listener
    assert "this.ctxBenchEventSource.close();" in listener
    assert "this.ctxBenchEventSource = null;" in listener
    assert "this.loadCtxBenchState();" in listener

    # Accuracy keeps its own 3 s poller, which reconnects by itself.
    assert "accEventSource" not in listener


def test_resume_machinery_that_the_pollers_never_needed_is_gone():
    """The pollers are not stopped while hidden, so resuming them is a no-op."""
    script = _script()

    assert "resumeVisibleRefreshers" not in script
    assert "addEventListener('pageshow'" not in script


def test_stats_listener_still_resumes_only_the_status_tab():
    listener = _region(
        _script(),
        "// Pause stats polling when tab is hidden",
        "async handleMainTabChange",
    )

    assert "this.stopStatsRefresh();" in listener
    assert "this.mainTab === 'status'" in listener
    assert "this.loadStats();" in listener
    assert "this.startStatsRefresh();" in listener


def test_stream_errors_only_apply_to_the_current_handle():
    """A stream dropped on foreground must not tear down its replacement."""
    script = _script()

    bench = _region(script, "es.onerror = () => {", "async cancelBenchmark")
    assert "if (this.benchEventSource !== es) return;" in bench
    assert bench.index("this.benchEventSource !== es") < bench.index("es.close();")

    connect_ctx = script.index("connectContextBenchSSE(benchId)")
    ctx = _region(
        script, "es.onerror = () => {", "async cancelContextBenchmark", connect_ctx
    )
    assert "if (this.ctxBenchEventSource !== es) return;" in ctx
    assert ctx.index("this.ctxBenchEventSource !== es") < ctx.index("es.close();")
