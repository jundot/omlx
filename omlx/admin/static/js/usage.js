/* Local serving history; independent of the high-frequency live stats poll. */
function usageHistory() {
    return {
        range: 'today', model: '', clientView: 'all', models: [], data: null, error: '', disabled: false, loading: false, peak: 1, displayedQuery: '',
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
        shade(tokens) {
            return tokens ? `rgba(22, 163, 74, ${0.2 + 0.8 * Math.sqrt(tokens / this.peak)})` : 'rgba(128, 128, 128, 0.12)';
        },
        number(value) { return new Intl.NumberFormat(undefined, {notation: 'compact', maximumFractionDigits: 1}).format(value || 0); },
        speed(value) { return value == null ? '—' : value.toFixed(1); },
        // Rows for the selected Clients tab: each key+IP pair, per key, or per IP.
        clientRows() {
            if (!this.data) return [];
            if (this.clientView === 'ip') {
                return (this.data.clients_by_ip || []).map(row => ({...row, id: row.client_ip, label: row.client_ip, detail: ''}));
            }
            if (this.clientView === 'key') {
                return (this.data.clients_by_key || []).map(row => ({...row, id: row.key_kind + ':' + row.key_id, label: this.keyLabel(row), detail: this.keyDetail(row)}));
            }
            return (this.data.clients || []).map(row => ({
                ...row, id: row.key_kind + ':' + row.key_id + '@' + row.client_ip,
                label: this.keyLabel(row), detail: '· ' + row.client_ip,
            }));
        },
        // Sub key name, or a fixed label for the main key and keyless requests.
        keyLabel(row) {
            if (row.key_kind === 'main_key') return window.t('usage.client_main_key');
            if (row.key_kind === 'none') return window.t('usage.client_no_key');
            return row.key_id;
        },
        keyDetail(row) { return row.key_kind === 'sub_key' ? window.t('usage.client_sub_key') : ''; },
    };
}
