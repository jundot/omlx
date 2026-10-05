/* Locale resolution rule (#4227): the oMLX UI language (<html lang>, rendered
   from ui.language by base.html) is used only when Intl both accepts the tag and
   recognises it. supportedLocalesOf throws RangeError for structurally invalid
   tags ('en_US', 'None', '') and returns [] for structurally valid but
   unresolvable ones ('bogus') — the dangerous case, because Intl.NumberFormat
   silently resolves those to the RUNTIME locale. Both fall back to 'en', which
   is what _load_locale() uses for the panel labels; `undefined` is never passed,
   since that is precisely the runtime-locale fallback this fixes. */
function usageLocale() {
    const requested = (document?.documentElement?.lang || '').trim() || 'en';
    try {
        return Intl.NumberFormat.supportedLocalesOf([requested]).length ? requested : 'en';
    } catch (error) {
        return 'en';
    }
}

function usageFormat(value, options) {
    try {
        return new Intl.NumberFormat(usageLocale(), options).format(value || 0);
    } catch (error) {
        // Backstop: usageLocale() only returns usable tags, but keep the panel
        // rendering rather than propagate, and never fall back to the runtime
        // locale.
        return new Intl.NumberFormat('en', options).format(value || 0);
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
        shade(tokens) {
            return tokens ? `rgba(22, 163, 74, ${0.2 + 0.8 * Math.sqrt(tokens / this.peak)})` : 'rgba(128, 128, 128, 0.12)';
        },
        locale() { return usageLocale(); },
        number(value) {
            // Compact units follow the oMLX UI language, exposed as <html lang>
            // from ui.language by base.html. An undefined locale would fall back
            // to the runtime locale, so an English UI rendered 3332萬 on a
            // zh-TW machine — labels and units from two different languages.
            return usageFormat(value, {notation: 'compact', maximumFractionDigits: 1});
        },
        exact(value) {
            // Sibling of number() for the :title tooltips and the heatmap
            // aria-label: non-compact, but the same resolved locale, so an
            // English UI on a de machine pairs 1.5K with a 1,500 tooltip
            // instead of a 1.500 one.
            return usageFormat(value, {});
        },
        speed(value) { return value == null ? '—' : value.toFixed(1); },
    };
}
