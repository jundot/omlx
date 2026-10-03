/* Local serving history; independent of the high-frequency live stats poll. */
function usageHistory() {
    return {
        range: 'today', model: '', models: [], data: null, error: '', errorTone: 'neutral', notice: null,
        disabled: false, loading: false, peak: 1, displayedQuery: '',
        timer: null, request: null,
        init() {
            this.$watch('mainTab', tab => {
                if (tab === 'status') this.load();
            });
            // The failure line used to sit inside the block; it is a toast now,
            // updated in place so a poll that keeps failing cannot stack.
            this.$watch('error', value => this.reportError(value));
            if (this.mainTab === 'status') this.load();
            this.timer = setInterval(() => {
                if (this.mainTab === 'status' && !document.hidden) this.load();
            }, 15000);
        },
        destroy() { clearInterval(this.timer); this.request?.abort(); },
        reportError(value) {
            if (typeof window.omlxToast !== 'function') return;
            if (value) {
                this.notice = window.omlxToast({
                    id: 'usage-error',
                    tone: this.errorTone,
                    title: window.t('toast.usage_notice'),
                    message: value,
                });
            } else if (this.notice) {
                this.notice.close();
                this.notice = null;
            }
        },
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
                    this.errorTone = 'neutral';
                    this.error = '';
                    return;
                }
                this.data = data;
                this.peak = Math.max(1, ...data.heatmap.flatMap(day => day.tokens));
                if (!this.model) this.models = data.models.map(row => row.model_id);
                this.errorTone = 'orange';
                this.error = data.available && !data.dropped_requests ? '' : window.t('usage.delayed');
            } catch (error) {
                if (error.name === 'AbortError' || request.signal.aborted) return;
                this.data = null;
                this.disabled = false;
                this.errorTone = 'red';
                this.error = window.t('usage.unavailable');
            } finally {
                if (this.request === request) this.loading = false;
            }
        },
        // Hourly totals across the selected range, so the strip above the
        // heatmap shows the shape of a day rather than a single date.
        hourlyTotals() {
            const totals = new Array(24).fill(0);
            (this.data?.heatmap || []).forEach(day => {
                (day.tokens || []).forEach((tokens, hour) => { totals[hour] += tokens || 0; });
            });
            return totals;
        },
        hourlyPeak() { return Math.max(1, ...this.hourlyTotals()); },
        hourlyBarStyle(total) {
            const ratio = total / this.hourlyPeak();
            return `height: ${Math.max(2, Math.round(ratio * 100))}%; opacity: ${(0.25 + 0.75 * Math.sqrt(ratio)).toFixed(2)};`;
        },
        shade(tokens) {
            return tokens ? `rgba(22, 163, 74, ${0.2 + 0.8 * Math.sqrt(tokens / this.peak)})` : 'rgba(128, 128, 128, 0.12)';
        },
        speed(value) { return value == null ? '—' : value.toFixed(1); },
    };
}
