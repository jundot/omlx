/* Local serving history; independent of the high-frequency live stats poll. */
function usageHistory() {
    return {
        range: 'today', model: '', models: [], data: null, error: '', loading: false, peak: 1, displayedQuery: '',
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
                this.data = data;
                this.peak = Math.max(1, ...data.heatmap.flatMap(day => day.tokens));
                if (!this.model) this.models = data.models.map(row => row.model_id);
                this.error = data.available && !data.dropped_requests ? '' : window.t('usage.delayed');
            } catch (error) {
                if (error.name === 'AbortError' || request.signal.aborted) return;
                this.data = null;
                this.error = window.t('usage.unavailable');
            } finally {
                if (this.request === request) this.loading = false;
            }
        },
        shade(tokens) {
            return tokens ? `rgba(22, 163, 74, ${0.2 + 0.8 * Math.sqrt(tokens / this.peak)})` : 'rgba(128, 128, 128, 0.12)';
        },
        number(value) { return new Intl.NumberFormat(undefined, {notation: 'compact', maximumFractionDigits: 1}).format(value || 0); },
        speed(value) { return value == null ? '—' : value.toFixed(1); },
    };
}
