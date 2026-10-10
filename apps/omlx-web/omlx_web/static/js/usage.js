/* Numbers follow the UI language. Intl maps unknown tags to the OS locale, so fall back to 'en'. */
function usageLocale() {
    try {
        const lang = document.documentElement.lang;
        return Intl.NumberFormat.supportedLocalesOf(lang).length ? lang : 'en';
    } catch (error) {
        return 'en';
    }
}

/* Local serving history; independent of the high-frequency live stats poll. */
function usageHistory() {
    return {
        range: 'today', model: '', models: [], data: null, error: '', disabled: false, loading: false, peak: 1, displayedQuery: '',
        timer: null, request: null,
        init() {
            this.$watch('mainTab', tab => {
                if (tab === 'status') this.load();
            });
            if (this.mainTab === 'status') this.load();
            this.timer = setInterval(() => {
                if (this.mainTab === 'status' && !document.hidden) this.load();
            }, 15000);
        },
        destroy() { clearInterval(this.timer); this.request?.abort(); },
        async load() {
            this.request?.abort();
            const request = new AbortController();
            this.request = request;
            this.loading = true;
            try {
                const params = new URLSearchParams({range: this.range, model: this.model});
                if (this.displayedQuery !== params.toString()) this.data = null;
                this.displayedQuery = params.toString();
                const response = await fetch('/admin/api/usage?' + params, {signal: request.signal});
                if (!response.ok) throw new Error('unavailable');
                const data = await response.json();
                if (request.signal.aborted) return;
                // Recording switched off in Settings: a distinct state, not a storage failure.
                this.disabled = data.enabled === false;
                if (this.disabled) {
                    this.data = null;
                    this.models = [];
                    this.error = '';
                    return;
                }
                this.data = data;
                this.peak = Math.max(1, ...data.heatmap.flatMap(day => day.tokens));
                if (!this.model) this.models = data.models.map(row => row.model_id);
                this.error = data.available && !data.dropped_requests ? '' : window.t('usage.delayed');
            } catch (error) {
                if (error.name === 'AbortError' || request.signal.aborted) return;
                this.data = null;
                this.disabled = false;
                this.error = window.t('usage.unavailable');
            } finally {
                if (this.request === request) this.loading = false;
            }
        },
        // Totals per hour of day across every day in the range.
        hourlyTotals() {
            const totals = new Array(24).fill(0);
            (this.data?.heatmap || []).forEach(day => {
                (day.tokens || []).forEach((tokens, hour) => { totals[hour] += tokens || 0; });
            });
            return totals;
        },
        hourlyPeak() { return Math.max(...this.hourlyTotals()); },
        hourlyBarStyle(total) {
            const peak = this.hourlyPeak();
            const ratio = peak > 0 ? total / peak : 0;
            return `height: ${total > 0 ? Math.max(4, Math.round(ratio * 100)) : 0}%;`;
        },
        shade(tokens) {
            return tokens ? `rgb(var(--palette-green-600) / ${0.2 + 0.8 * Math.sqrt(tokens / this.peak)})` : 'rgb(var(--palette-mid-gray) / 0.12)';
        },
        number(value) { return new Intl.NumberFormat(usageLocale(), {notation: 'compact', maximumFractionDigits: 1}).format(value || 0); },
        speed(value) { return value == null ? '—' : value.toFixed(1); },
    };
}
