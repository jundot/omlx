    function dashboard() {
        return {
            // Theme
            theme: localStorage.getItem('omlx-chat-theme') || 'light',

            // Main tab state (Status, Settings, or Logs)
            mainTab: 'status',

            activeTab: 'global',
            settingsDropdown: false,

            // Global settings
            globalSettings: {
                base_path: '',
                server: { host: '127.0.0.1', port: 8000, log_level: 'info' },
                model: { model_dir: '', max_model_memory: '' },
                scheduler: { max_num_seqs: 8, prefill_batch_size: 8, completion_batch_size: 8 },
                cache: { enabled: true, ssd_cache_dir: '', ssd_cache_max_size: 'auto' },
                sampling: { max_context_window: 32768, max_tokens: 32768, temperature: 1.0, top_p: 0.95, top_k: 40, repetition_penalty: 1.0 },
                mcp: { config_path: '' },
                auth: { api_key_set: false, api_key: '' },
                claude_code: { context_scaling_enabled: false, target_context_size: 200000 },
                system: { total_memory_bytes: 0, total_memory: '', auto_model_memory: '', ssd_total_bytes: 0, ssd_total: '', ssd_free_bytes: 0, ssd_free: '' },
            },

            // Memory slider (0-100%)
            memoryPercent: 80,
            // Cache slider (0-100%)
            cachePercent: 10,
            editingCache: false,

            // Models
            models: [],
            loadingModels: false,
            sortBy: 'id',
            sortOrder: 'asc',

            // Auth UI state
            showApiKey: false,

            // Saving state
            saving: false,
            saveSuccess: false,
            saveMessage: '',
            saveError: '',

            // Model settings modal
            showModelSettingsModal: false,
            selectedModel: null,
            modelSettings: {
                max_context_window: null,
                max_tokens: null,
                temperature: null,
                top_p: null,
                top_k: null,
                repetition_penalty: null,
                force_sampling: false,
                enableToolResultLimit: false,
                max_tool_result_tokens: null,
            },
            savingModelSettings: false,
            loadingGenDefaults: false,

            // Status tab state
            stats: {
                total_prompt_tokens: 0,
                total_cached_tokens: 0,
                cache_efficiency: 0.0,
                avg_prefill_tps: 0.0,
                avg_generation_tps: 0.0,
                total_requests: 0,
                port: 8000,
                api_key: '',
                engines: {},
            },
            selectedClaudeModel: '',
            selectedStatsModel: '',
            _statsRefreshTimer: null,

            // Log viewer state
            logContent: '',
            logLines: 500,
            logRefreshInterval: 5,  // seconds, 0 = disabled
            logAutoRefresh: false,
            logAutoScroll: true,
            logLoading: false,
            logError: '',
            logFile: 'server.log',
            logAvailableFiles: ['server.log'],
            logTotalLines: 0,
            logLastUpdated: '',
            _logRefreshTimer: null,

            // HF Downloader state
            hfRepoId: '',
            hfToken: '',
            hfDownloading: false,
            hfTasks: [],
            hfModels: [],
            hfModelsLoaded: false,
            hfError: '',
            hfSuccess: '',
            _hfRefreshTimer: null,
            hfDeleteConfirm: null,

            // Recommended models state
            hfRecommended: { trending: [], popular: [] },
            hfRecommendedLoaded: false,
            hfRecommendedLoading: false,
            hfRecommendedTab: 'trending',

            // Benchmark state
            benchModelId: '',
            benchPromptLengths: { 1024: true, 4096: true, 8192: false, 16384: false, 32768: false, 65536: false },
            benchBatchSizes: { 2: true, 4: true, 8: false },
            benchRunning: false,
            benchBenchId: null,
            benchProgress: null,
            benchSingleResults: [],
            benchBatchSameResults: [],
            benchBatchDiffResults: [],
            benchError: '',
            benchEventSource: null,
            benchShowMetrics: false,
            benchShowText: false,
            benchCopied: false,

            async init() {
                // Apply theme
                this.applyTheme();

                await Promise.all([
                    this.loadGlobalSettings(),
                    this.loadModels()
                ]);
                this.$nextTick(() => {
                    lucide.createIcons();
                });

                // Load stats and start polling if starting on status tab
                if (this.mainTab === 'status') {
                    await this.loadStats();
                    this.startStatsRefresh();
                }

                // Watch for main tab changes to manage refresh timers
                this.$watch('mainTab', (value) => {
                    if (value === 'status') {
                        this.loadStats();
                        this.startStatsRefresh();
                    } else {
                        this.stopStatsRefresh();
                    }
                    if (value === 'logs') {
                        this.loadLogs();
                        this.startLogRefresh();
                    } else {
                        this.stopLogRefresh();
                    }
                    if (value === 'models') {
                        this.loadHFModels();
                        this.loadHFTasks();
                        if (!this.hfRecommendedLoaded) this.loadRecommendedModels();
                        const hasActive = this.hfTasks.some(t =>
                            t.status === 'pending' || t.status === 'downloading');
                        if (hasActive) this.startHFRefresh();
                    } else {
                        this.stopHFRefresh();
                    }
                    this.$nextTick(() => lucide.createIcons());
                });
            },

            async loadGlobalSettings() {
                try {
                    const response = await fetch('/admin/api/global-settings');
                    if (response.ok) {
                        const data = await response.json();
                        // Deep merge to preserve defaults for missing fields
                        this.globalSettings = {
                            ...this.globalSettings,
                            ...data,
                            server: { ...this.globalSettings.server, ...data.server },
                            model: { ...this.globalSettings.model, ...data.model },
                            scheduler: { ...this.globalSettings.scheduler, ...data.scheduler },
                            cache: { ...this.globalSettings.cache, ...data.cache },
                            sampling: { ...this.globalSettings.sampling, ...data.sampling },
                            mcp: { ...this.globalSettings.mcp, ...data.mcp },
                            auth: { ...this.globalSettings.auth, ...data.auth },
                            claude_code: { ...this.globalSettings.claude_code, ...data.claude_code },
                            system: { ...this.globalSettings.system, ...data.system },
                        };

                        // Calculate memory percent from stored value
                        this.memoryPercent = this.parseMemoryToPercent(
                            this.globalSettings.model.max_model_memory,
                            this.globalSettings.system.total_memory_bytes
                        );
                        // Sync the memory string value from percent
                        this.updateMemoryFromSlider();

                        // Calculate cache percent from stored value (based on free space)
                        this.cachePercent = this.parseCacheToPercent(
                            this.globalSettings.cache.ssd_cache_max_size,
                            this.globalSettings.system.ssd_free_bytes
                        );
                        // Sync the cache string value from percent
                        this.updateCacheFromSlider();
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load global settings:', err);
                }
            },

            async saveGlobalSettings() {
                this.saving = true;
                this.saveSuccess = false;
                this.saveError = '';

                // Validate required fields
                const errors = [];
                const s = this.globalSettings;
                if (!s.server.host) errors.push('Host');
                if (!s.server.port) errors.push('Port');
                if (!s.model.model_dir) errors.push('Model Directory');
                if (!s.scheduler.max_num_seqs) errors.push('Max Sequences');
                if (!s.scheduler.prefill_batch_size) errors.push('Prefill Batch Size');
                if (!s.scheduler.completion_batch_size) errors.push('Completion Batch Size');
                if (!s.cache.ssd_cache_max_size) errors.push('Max Cache Size');
                if (!s.sampling.max_context_window) errors.push('Max Context Window');
                if (!s.sampling.max_tokens) errors.push('Max Tokens');

                if (errors.length > 0) {
                    this.saveError = `Required fields cannot be empty: ${errors.join(', ')}`;
                    this.saving = false;
                    return;
                }

                // Validate API key if provided
                if (s.auth.api_key) {
                    if (s.auth.api_key.length < 4) {
                        this.saveError = 'API key must be at least 4 characters';
                        this.saving = false;
                        return;
                    }
                    if (/\s/.test(s.auth.api_key)) {
                        this.saveError = 'API key must not contain whitespace';
                        this.saving = false;
                        return;
                    }
                }

                try {
                    const response = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            host: this.globalSettings.server.host,
                            port: this.globalSettings.server.port,
                            log_level: this.globalSettings.server.log_level,
                            model_dir: this.globalSettings.model.model_dir,
                            max_model_memory: this.globalSettings.model.max_model_memory,
                            max_num_seqs: this.globalSettings.scheduler.max_num_seqs,
                            prefill_batch_size: this.globalSettings.scheduler.prefill_batch_size,
                            completion_batch_size: this.globalSettings.scheduler.completion_batch_size,
                            cache_enabled: this.globalSettings.cache.enabled,
                            ssd_cache_dir: this.globalSettings.cache.ssd_cache_dir,
                            ssd_cache_max_size: this.globalSettings.cache.ssd_cache_max_size,
                            sampling_max_context_window: this.globalSettings.sampling.max_context_window,
                            sampling_max_tokens: this.globalSettings.sampling.max_tokens,
                            sampling_temperature: this.globalSettings.sampling.temperature,
                            sampling_top_p: this.globalSettings.sampling.top_p,
                            sampling_top_k: this.globalSettings.sampling.top_k,
                            sampling_repetition_penalty: this.globalSettings.sampling.repetition_penalty,
                            mcp_config: this.globalSettings.mcp.config_path,
                            ...(this.globalSettings.auth.api_key ? { api_key: this.globalSettings.auth.api_key } : {}),
                        }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        this.saveSuccess = true;
                        this.saveMessage = data.message || 'Settings saved successfully';
                        this.$nextTick(() => lucide.createIcons());
                        // Refresh stats so Claude Code command reflects new API key
                        await this.loadStats();
                        setTimeout(() => { this.saveSuccess = false; }, 5000);
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        this.saveError = Array.isArray(data.detail) ? data.detail.join(', ') : (data.detail || 'Failed to save settings');
                        // Reload settings to revert to server values
                        await this.loadGlobalSettings();
                        this.$nextTick(() => lucide.createIcons());
                    }
                } catch (err) {
                    console.error('Failed to save global settings:', err);
                    this.saveError = 'Failed to save settings';
                    // Reload settings to revert to server values
                    await this.loadGlobalSettings();
                    this.$nextTick(() => lucide.createIcons());
                } finally {
                    this.saving = false;
                }
            },

            async loadModels() {
                this.loadingModels = true;
                try {
                    const response = await fetch('/admin/api/models');
                    if (response.ok) {
                        const data = await response.json();
                        this.models = data.models || [];
                        this.$nextTick(() => lucide.createIcons());
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load models:', err);
                } finally {
                    this.loadingModels = false;
                }
            },

            async updateModelSetting(modelId, field, value) {
                try {
                    const response = await fetch(`/admin/api/models/${encodeURIComponent(modelId)}/settings`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ [field]: value }),
                    });

                    if (response.ok) {
                        if (field === 'is_default' && value === true) {
                            this.models.forEach(m => { m.is_default = (m.id === modelId); });
                        } else if (field === 'is_pinned') {
                            const model = this.models.find(m => m.id === modelId);
                            if (model) model.pinned = value;
                        }
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || 'Failed to update model setting');
                        await this.loadModels();
                    }
                } catch (err) {
                    console.error('Failed to update model setting:', err);
                    alert('Failed to update model setting');
                    await this.loadModels();
                }
            },

            openModelSettings(model) {
                this.selectedModel = model;
                // Load existing settings if available
                const settings = model.settings || {};
                this.modelSettings = {
                    max_context_window: settings.max_context_window || null,
                    max_tokens: settings.max_tokens || null,
                    temperature: settings.temperature ?? null,
                    top_p: settings.top_p ?? null,
                    top_k: settings.top_k ?? null,
                    repetition_penalty: settings.repetition_penalty ?? null,
                    force_sampling: settings.force_sampling || false,
                    enableToolResultLimit: !!(settings.max_tool_result_tokens),
                    max_tool_result_tokens: settings.max_tool_result_tokens || null,
                };
                this.showModelSettingsModal = true;
                this.$nextTick(() => lucide.createIcons());
            },

            async saveModelSettings() {
                if (!this.selectedModel) return;

                this.savingModelSettings = true;
                try {
                    const response = await fetch(`/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/settings`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            max_context_window: this.modelSettings.max_context_window || null,
                            max_tokens: this.modelSettings.max_tokens || null,
                            temperature: Number.isFinite(this.modelSettings.temperature) ? this.modelSettings.temperature : null,
                            top_p: Number.isFinite(this.modelSettings.top_p) ? this.modelSettings.top_p : null,
                            top_k: Number.isFinite(this.modelSettings.top_k) ? this.modelSettings.top_k : null,
                            repetition_penalty: Number.isFinite(this.modelSettings.repetition_penalty) ? this.modelSettings.repetition_penalty : null,
                            force_sampling: this.modelSettings.force_sampling,
                            max_tool_result_tokens: this.modelSettings.enableToolResultLimit
                                ? (this.modelSettings.max_tool_result_tokens || null)
                                : 0,
                        }),
                    });

                    if (response.ok) {
                        // Update local model data from server response
                        const data = await response.json();
                        const model = this.models.find(m => m.id === this.selectedModel.id);
                        if (model) {
                            model.settings = data.settings || {};
                        }
                        this.showModelSettingsModal = false;
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || 'Failed to save model settings');
                    }
                } catch (err) {
                    console.error('Failed to save model settings:', err);
                    alert('Failed to save model settings');
                } finally {
                    this.savingModelSettings = false;
                }
            },

            async loadGenerationDefaults() {
                if (!this.selectedModel) return;
                this.loadingGenDefaults = true;
                try {
                    const response = await fetch(`/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/generation_config`);
                    if (response.ok) {
                        const data = await response.json();
                        // Set values from config, clear everything else to Default (null)
                        this.modelSettings.max_context_window = data.max_context_window ?? null;
                        this.modelSettings.temperature = data.temperature ?? null;
                        this.modelSettings.top_p = data.top_p ?? null;
                        this.modelSettings.top_k = data.top_k ?? null;
                        this.modelSettings.repetition_penalty = data.repetition_penalty ?? null;
                    } else if (response.status === 404) {
                        alert('No config defaults found for this model');
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || 'Failed to load generation config');
                    }
                } catch (err) {
                    console.error('Failed to load generation config:', err);
                    alert('Failed to load generation config');
                } finally {
                    this.loadingGenDefaults = false;
                }
            },

            // Status tab functions
            get llmModels() {
                return this.models.filter(m => m.model_type === 'llm' || !m.model_type);
            },

            get claudeCodeCommand() {
                const model = this.selectedClaudeModel || 'select-a-model';
                const port = this.stats.port || 8000;
                const parts = [];
                parts.push(`ANTHROPIC_BASE_URL=http://localhost:${port}`);
                if (this.stats.api_key) {
                    parts.push(`ANTHROPIC_AUTH_TOKEN=${this.stats.api_key}`);
                }
                parts.push(`ANTHROPIC_MODEL=${model}`);
                parts.push(`ANTHROPIC_DEFAULT_SONNET_MODEL=${model}`);
                parts.push(`ANTHROPIC_DEFAULT_OPUS_MODEL=${model}`);
                parts.push(`ANTHROPIC_DEFAULT_HAIKU_MODEL=${model}`);
                parts.push('API_TIMEOUT_MS=3000000');
                parts.push('CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1');
                parts.push('claude');
                return parts.join(' ');
            },

            async saveClaudeCodeSettings() {
                try {
                    const response = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            claude_code_context_scaling_enabled: this.globalSettings.claude_code.context_scaling_enabled,
                            claude_code_target_context_size: this.globalSettings.claude_code.target_context_size,
                        }),
                    });
                    if (!response.ok) {
                        console.error('Failed to save Claude Code settings');
                    }
                } catch (err) {
                    console.error('Failed to save Claude Code settings:', err);
                }
            },

            async loadStats() {
                try {
                    const params = new URLSearchParams();
                    if (this.selectedStatsModel) {
                        params.set('model', this.selectedStatsModel);
                    }
                    const url = '/admin/api/stats' + (params.toString() ? '?' + params : '');
                    const response = await fetch(url);
                    if (response.ok) {
                        const data = await response.json();
                        this.stats = { ...this.stats, ...data };
                        // Set default selected model if not set
                        if (!this.selectedClaudeModel && this.llmModels.length > 0) {
                            this.selectedClaudeModel = this.llmModels[0].id;
                        }
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load stats:', err);
                }
            },

            async clearStats() {
                try {
                    await fetch('/admin/api/stats/clear', { method: 'POST' });
                    await this.loadStats();
                } catch (err) {
                    console.error('Failed to clear stats:', err);
                }
            },

            startStatsRefresh() {
                this.stopStatsRefresh();
                this._statsRefreshTimer = setInterval(() => {
                    this.loadStats();
                }, 1000);
            },

            stopStatsRefresh() {
                if (this._statsRefreshTimer) {
                    clearInterval(this._statsRefreshTimer);
                    this._statsRefreshTimer = null;
                }
            },

            formatNumber(num) {
                if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
                if (num >= 10000000) return (num / 1000000).toFixed(1) + 'M';
                return num.toLocaleString();
            },

            getStatFontClass(value) {
                if (value >= 1000000000) return 'text-2xl';
                if (value >= 1000000) return 'text-3xl';
                return 'text-5xl';
            },

            copyToClipboard(text) {
                navigator.clipboard.writeText(text).then(() => {
                    // Could add visual feedback here
                }).catch(err => {
                    console.error('Failed to copy:', err);
                });
            },

            async logout() {
                try {
                    await fetch('/admin/api/logout', { method: 'POST' });
                } catch (err) {
                    console.error('Logout error:', err);
                } finally {
                    window.location.href = '/admin';
                }
            },

            // Benchmark functions
            async startBenchmark() {
                if (!this.benchModelId) return;

                // Collect selected prompt lengths
                const promptLengths = Object.entries(this.benchPromptLengths)
                    .filter(([_, v]) => v)
                    .map(([k, _]) => parseInt(k));

                if (promptLengths.length === 0) {
                    this.benchError = 'Select at least one prompt length';
                    return;
                }

                // Collect selected batch sizes
                const batchSizes = Object.entries(this.benchBatchSizes)
                    .filter(([_, v]) => v)
                    .map(([k, _]) => parseInt(k));

                // Reset state
                this.benchRunning = true;
                this.benchProgress = null;
                this.benchSingleResults = [];
                this.benchBatchSameResults = [];
                this.benchBatchDiffResults = [];
                this.benchError = '';
                this.benchBenchId = null;

                try {
                    const response = await fetch('/admin/api/bench/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: this.benchModelId,
                            prompt_lengths: promptLengths,
                            generation_length: 128,
                            batch_sizes: batchSizes,
                        }),
                    });

                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }

                    if (!response.ok) {
                        const data = await response.json();
                        this.benchError = data.detail || 'Failed to start benchmark';
                        this.benchRunning = false;
                        return;
                    }

                    const data = await response.json();
                    this.benchBenchId = data.bench_id;
                    this.connectBenchSSE(data.bench_id);
                } catch (err) {
                    console.error('Failed to start benchmark:', err);
                    this.benchError = 'Failed to start benchmark: ' + err.message;
                    this.benchRunning = false;
                }
            },

            connectBenchSSE(benchId) {
                if (this.benchEventSource) {
                    this.benchEventSource.close();
                }

                const es = new EventSource(`/admin/api/bench/${benchId}/stream`);
                this.benchEventSource = es;

                es.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);

                        if (data.type === 'progress') {
                            this.benchProgress = {
                                phase: data.phase,
                                message: data.message,
                                current: data.current,
                                total: data.total,
                            };
                        } else if (data.type === 'result') {
                            if (data.data.test_type === 'single') {
                                this.benchSingleResults = [...this.benchSingleResults, data.data];
                            } else if (data.data.test_type === 'batch_same') {
                                this.benchBatchSameResults = [...this.benchBatchSameResults, data.data];
                            } else if (data.data.test_type === 'batch_diff') {
                                this.benchBatchDiffResults = [...this.benchBatchDiffResults, data.data];
                            }
                        } else if (data.type === 'done') {
                            this.benchRunning = false;
                            this.benchProgress = null;
                            es.close();
                            this.benchEventSource = null;
                            // Refresh model list to show updated load states
                            this.loadModels();
                        } else if (data.type === 'error') {
                            this.benchError = data.message;
                            this.benchRunning = false;
                            this.benchProgress = null;
                            es.close();
                            this.benchEventSource = null;
                            this.loadModels();
                        }

                        this.$nextTick(() => lucide.createIcons());
                    } catch (err) {
                        console.error('Failed to parse SSE event:', err);
                    }
                };

                es.onerror = () => {
                    if (this.benchRunning) {
                        this.benchError = 'Connection to benchmark lost';
                        this.benchRunning = false;
                        this.benchProgress = null;
                    }
                    es.close();
                    this.benchEventSource = null;
                };
            },

            async cancelBenchmark() {
                if (!this.benchBenchId) return;
                try {
                    await fetch(`/admin/api/bench/${this.benchBenchId}/cancel`, { method: 'POST' });
                } catch (err) {
                    console.error('Failed to cancel benchmark:', err);
                }
                // SSE handler will update state when error/done event arrives
            },

            benchGetSpeedup(batchResult) {
                const baseline = this.benchSingleResults.find(r => r.pp === 1024);
                if (!baseline || !baseline.gen_tps || baseline.gen_tps <= 0) return 0;
                return batchResult.tg_tps / baseline.gen_tps;
            },

            benchFormatMemory(bytes) {
                if (!bytes || bytes === 0) return '-';
                const gb = bytes / (1024 * 1024 * 1024);
                if (gb >= 1) return gb.toFixed(2) + ' GB';
                const mb = bytes / (1024 * 1024);
                return mb.toFixed(0) + ' MB';
            },

            benchBuildText() {
                const pad = (s, w) => s.toString().padStart(w);
                const rpad = (s, w) => s.toString().padEnd(w);
                let lines = [];

                lines.push(`oMLX Benchmark — ${this.benchModelId}`);
                lines.push('='.repeat(80));

                // Single Request Results
                if (this.benchSingleResults.length > 0) {
                    lines.push('');
                    lines.push('Single Request Results');
                    lines.push('-'.repeat(80));
                    const hdr = [rpad('Test', 16), pad('TTFT(ms)', 10), pad('TPOT(ms)', 10), pad('tg TPS', 12), pad('pp TPS', 12), pad('E2E(s)', 10), pad('Throughput', 12), pad('Peak Mem', 10)];
                    lines.push(hdr.join('  '));
                    for (const r of this.benchSingleResults) {
                        const row = [
                            rpad(`pp${r.pp}/tg${r.tg}`, 16),
                            pad(r.ttft_ms.toFixed(1), 10),
                            pad(r.tpot_ms.toFixed(2), 10),
                            pad(r.gen_tps.toFixed(1) + ' tok/s', 12),
                            pad(r.processing_tps.toFixed(1) + ' tok/s', 12),
                            pad(r.e2e_latency_s.toFixed(3), 10),
                            pad(r.total_throughput.toFixed(1) + ' tok/s', 12),
                            pad(this.benchFormatMemory(r.peak_memory_bytes), 10),
                        ];
                        lines.push(row.join('  '));
                    }
                }

                // Helper for batch table text
                const buildBatchText = (title, subtitle, results) => {
                    if (results.length === 0) return;
                    const baseline = this.benchSingleResults.find(r => r.pp === 1024);
                    lines.push('');
                    lines.push(`${title}`);
                    lines.push(subtitle);
                    lines.push('-'.repeat(80));
                    const hdr = [rpad('Batch', 8), pad('tg TPS', 12), pad('Speedup', 8), pad('pp TPS', 12), pad('pp TPS/req', 12), pad('TTFT(ms)', 10), pad('E2E(s)', 10)];
                    lines.push(hdr.join('  '));
                    if (baseline) {
                        const row = [
                            rpad('1x', 8),
                            pad(baseline.gen_tps.toFixed(1) + ' tok/s', 12),
                            pad('1.00x', 8),
                            pad(baseline.processing_tps.toFixed(1) + ' tok/s', 12),
                            pad(baseline.processing_tps.toFixed(1) + ' tok/s', 12),
                            pad(baseline.ttft_ms.toFixed(1), 10),
                            pad(baseline.e2e_latency_s.toFixed(3), 10),
                        ];
                        lines.push(row.join('  '));
                    }
                    for (const r of results) {
                        const speedup = baseline && baseline.gen_tps > 0 ? (r.tg_tps / baseline.gen_tps).toFixed(2) + 'x' : '-';
                        const row = [
                            rpad(r.batch_size + 'x', 8),
                            pad(r.tg_tps.toFixed(1) + ' tok/s', 12),
                            pad(speedup, 8),
                            pad(r.pp_tps.toFixed(1) + ' tok/s', 12),
                            pad((r.pp_tps / r.batch_size).toFixed(1) + ' tok/s', 12),
                            pad(r.avg_ttft_ms.toFixed(1), 10),
                            pad(r.e2e_latency_s.toFixed(3), 10),
                        ];
                        lines.push(row.join('  '));
                    }
                };

                buildBatchText(
                    'Continuous Batching — Same Prompt',
                    'pp1024 / tg128 · prefix cache reuse',
                    this.benchBatchSameResults
                );
                buildBatchText(
                    'Continuous Batching — Different Prompts',
                    'pp1024 / tg128 · no cache reuse',
                    this.benchBatchDiffResults
                );

                return lines.join('\n');
            },

            benchCopyText() {
                const text = this.benchBuildText();
                navigator.clipboard.writeText(text).then(() => {
                    this.benchCopied = true;
                    setTimeout(() => { this.benchCopied = false; }, 2000);
                }).catch(err => {
                    console.error('Failed to copy:', err);
                });
            },

            // Log viewer functions
            async loadLogs() {
                this.logLoading = true;
                this.logError = '';

                try {
                    const params = new URLSearchParams({
                        lines: this.logLines.toString(),
                    });
                    if (this.logFile && this.logFile !== 'server.log') {
                        params.append('file', this.logFile);
                    }

                    const response = await fetch(`/admin/api/logs?${params}`);

                    if (response.ok) {
                        const data = await response.json();
                        this.logContent = data.logs;
                        this.logTotalLines = data.total_lines;
                        this.logAvailableFiles = data.available_files || ['server.log'];
                        this.logLastUpdated = new Date().toLocaleTimeString();

                        // Auto-scroll to bottom
                        if (this.logAutoScroll) {
                            this.$nextTick(() => {
                                const textarea = this.$refs.logTextarea;
                                if (textarea) {
                                    textarea.scrollTop = textarea.scrollHeight;
                                }
                            });
                        }
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        this.logError = data.detail || 'Failed to load logs';
                    }
                } catch (err) {
                    console.error('Failed to load logs:', err);
                    this.logError = 'Failed to load logs';
                } finally {
                    this.logLoading = false;
                }
            },

            startLogRefresh() {
                this.stopLogRefresh();  // Clear existing timer

                if (this.logRefreshInterval > 0) {
                    this.logAutoRefresh = true;
                    this._logRefreshTimer = setInterval(() => {
                        this.loadLogs();
                    }, this.logRefreshInterval * 1000);
                }
            },

            stopLogRefresh() {
                if (this._logRefreshTimer) {
                    clearInterval(this._logRefreshTimer);
                    this._logRefreshTimer = null;
                }
                this.logAutoRefresh = false;
            },

            restartLogRefresh() {
                if (this.mainTab === 'logs') {
                    this.startLogRefresh();
                }
            },

            // Parse stored memory value (e.g., "102GB") to percent of usable memory
            parseMemoryToPercent(memoryStr, totalBytes) {
                const usableBytes = Math.max(0, totalBytes - 8 * 1024 * 1024 * 1024);
                if (!memoryStr || !usableBytes || usableBytes === 0) {
                    return 80; // Default 80%
                }

                // Parse memory string like "102GB", "50GB", etc.
                const match = memoryStr.match(/^(\d+(?:\.\d+)?)\s*(GB|MB|TB)?$/i);
                if (!match) {
                    return 80; // Default if not parseable
                }

                let bytes = parseFloat(match[1]);
                const unit = (match[2] || 'GB').toUpperCase();

                if (unit === 'TB') bytes *= 1024 * 1024 * 1024 * 1024;
                else if (unit === 'GB') bytes *= 1024 * 1024 * 1024;
                else if (unit === 'MB') bytes *= 1024 * 1024;

                const percent = Math.round((bytes / usableBytes) * 100);
                return Math.min(100, Math.max(0, percent));
            },

            // Get max usable memory (total - 8GB reserved for system)
            get maxUsableMemoryBytes() {
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                const reservedBytes = 8 * 1024 * 1024 * 1024; // 8GB
                return Math.max(0, totalBytes - reservedBytes);
            },

            // Convert percent to memory string (e.g., 80 -> "102GB")
            // Percent is based on usable memory (total - 8GB)
            percentToMemoryString(percent, totalBytes) {
                const usableBytes = Math.max(0, totalBytes - 8 * 1024 * 1024 * 1024);
                if (!usableBytes || usableBytes === 0) return 'auto';
                const bytes = Math.floor((percent / 100) * usableBytes);
                const gb = Math.floor(bytes / (1024 * 1024 * 1024));
                return `${gb}GB`;
            },

            // Get formatted memory for display
            getMemoryDisplay() {
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                if (!totalBytes) return '-';
                const usableBytes = Math.max(0, totalBytes - 8 * 1024 * 1024 * 1024);
                const bytes = Math.floor((this.memoryPercent / 100) * usableBytes);
                const gb = Math.round(bytes / (1024 * 1024 * 1024));
                return `${gb}GB`;
            },

            // Update memory value when slider changes
            updateMemoryFromSlider() {
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                this.globalSettings.model.max_model_memory = this.percentToMemoryString(this.memoryPercent, totalBytes);
            },

            // Parse cache size string (e.g., "10GB") to percent of SSD free space
            parseCacheToPercent(cacheStr, freeBytes) {
                if (!cacheStr || cacheStr === 'auto' || !freeBytes || freeBytes === 0) {
                    return 10; // Default 10%
                }

                const match = cacheStr.match(/^(\d+(?:\.\d+)?)\s*(GB|MB|TB)?$/i);
                if (!match) return 10;

                let bytes = parseFloat(match[1]);
                const unit = (match[2] || 'GB').toUpperCase();

                if (unit === 'TB') bytes *= 1024 * 1024 * 1024 * 1024;
                else if (unit === 'GB') bytes *= 1024 * 1024 * 1024;
                else if (unit === 'MB') bytes *= 1024 * 1024;

                const percent = Math.round((bytes / freeBytes) * 100);
                return percent; // Allow >100% for manual input
            },

            // Convert percent to cache size string
            percentToCacheString(percent, freeBytes) {
                if (!freeBytes || freeBytes === 0) return 'auto';
                const bytes = Math.floor((percent / 100) * freeBytes);
                const gb = Math.floor(bytes / (1024 * 1024 * 1024));
                return `${gb}GB`;
            },

            // Computed cache size in GB (for manual input)
            get cacheSizeGB() {
                const freeBytes = this.globalSettings.system?.ssd_free_bytes || 0;
                if (!freeBytes) return 0;
                const bytes = Math.floor((this.cachePercent / 100) * freeBytes);
                return Math.round(bytes / (1024 * 1024 * 1024));
            },

            // Update cache from slider
            updateCacheFromSlider() {
                const freeBytes = this.globalSettings.system?.ssd_free_bytes || 0;
                this.globalSettings.cache.ssd_cache_max_size = this.percentToCacheString(this.cachePercent, freeBytes);
            },

            // Update cache from manual GB input
            updateCacheFromInput(gbValue) {
                const gb = parseInt(gbValue) || 0;
                this.globalSettings.cache.ssd_cache_max_size = `${gb}GB`;

                // Update percent slider (allow >100%)
                const freeBytes = this.globalSettings.system?.ssd_free_bytes || 0;
                if (freeBytes > 0) {
                    const bytes = gb * 1024 * 1024 * 1024;
                    this.cachePercent = Math.round((bytes / freeBytes) * 100);
                }
            },

            // Sort models
            get sortedModels() {
                return [...this.models].sort((a, b) => {
                    let aVal, bVal;

                    switch (this.sortBy) {
                        case 'id':
                            aVal = (a.id || '').toLowerCase();
                            bVal = (b.id || '').toLowerCase();
                            break;
                        case 'type':
                            aVal = (a.model_type || 'llm').toLowerCase();
                            bVal = (b.model_type || 'llm').toLowerCase();
                            break;
                        case 'size':
                            aVal = a.estimated_size || 0;
                            bVal = b.estimated_size || 0;
                            break;
                        case 'loaded':
                            aVal = a.loaded ? 1 : 0;
                            bVal = b.loaded ? 1 : 0;
                            break;
                        case 'pinned':
                            aVal = a.pinned ? 1 : 0;
                            bVal = b.pinned ? 1 : 0;
                            break;
                        case 'is_default':
                            aVal = a.is_default ? 1 : 0;
                            bVal = b.is_default ? 1 : 0;
                            break;
                        default:
                            return 0;
                    }

                    if (aVal < bVal) return this.sortOrder === 'asc' ? -1 : 1;
                    if (aVal > bVal) return this.sortOrder === 'asc' ? 1 : -1;
                    return 0;
                });
            },

            toggleSort(column) {
                if (this.sortBy === column) {
                    this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
                } else {
                    this.sortBy = column;
                    this.sortOrder = 'asc';
                }
            },

            // Theme toggle
            toggleTheme() {
                this.theme = this.theme === 'light' ? 'dark' : 'light';
                localStorage.setItem('omlx-chat-theme', this.theme);
                this.applyTheme();
                this.$nextTick(() => lucide.createIcons());
            },

            applyTheme() {
                document.documentElement.setAttribute('data-theme', this.theme);
            },

            // =================================================================
            // HuggingFace Downloader Functions
            // =================================================================

            async startHFDownload() {
                const repoId = this.hfRepoId.trim();
                if (!repoId) return;

                this.hfError = '';
                this.hfSuccess = '';
                this.hfDownloading = true;

                try {
                    const response = await fetch('/admin/api/hf/download', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            repo_id: repoId,
                            hf_token: this.hfToken,
                        }),
                    });

                    if (response.ok) {
                        this.hfSuccess = `Download started: ${repoId}`;
                        this.hfRepoId = '';
                        await this.loadHFTasks();
                        this.startHFRefresh();
                        setTimeout(() => { this.hfSuccess = ''; }, 5000);
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        this.hfError = data.detail || 'Failed to start download';
                    }
                } catch (err) {
                    console.error('Failed to start download:', err);
                    this.hfError = 'Failed to start download. Check server connection.';
                } finally {
                    this.hfDownloading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            async loadHFTasks() {
                try {
                    const response = await fetch('/admin/api/hf/tasks');
                    if (response.ok) {
                        const data = await response.json();
                        this.hfTasks = data.tasks || [];

                        // Stop refresh if no active downloads
                        const hasActive = this.hfTasks.some(t =>
                            t.status === 'pending' || t.status === 'downloading');
                        if (!hasActive) {
                            this.stopHFRefresh();
                            // Refresh model lists when all downloads finish
                            if (this.hfTasks.some(t => t.status === 'completed')) {
                                await this.loadHFModels();
                                await this.loadModels();
                            }
                        }

                        this.$nextTick(() => lucide.createIcons());
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load HF tasks:', err);
                }
            },

            async loadHFModels() {
                try {
                    const response = await fetch('/admin/api/hf/models');
                    if (response.ok) {
                        const data = await response.json();
                        this.hfModels = data.models || [];
                        this.hfModelsLoaded = true;
                        this.$nextTick(() => lucide.createIcons());
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load HF models:', err);
                }
            },

            async cancelHFDownload(taskId) {
                try {
                    const response = await fetch(`/admin/api/hf/cancel/${taskId}`, {
                        method: 'POST',
                    });
                    if (response.ok) {
                        await this.loadHFTasks();
                    }
                } catch (err) {
                    console.error('Failed to cancel download:', err);
                }
            },

            async removeHFTask(taskId) {
                try {
                    const response = await fetch(`/admin/api/hf/task/${taskId}`, {
                        method: 'DELETE',
                    });
                    if (response.ok) {
                        await this.loadHFTasks();
                    }
                } catch (err) {
                    console.error('Failed to remove task:', err);
                }
            },

            async deleteHFModel(modelName) {
                this.hfDeleteConfirm = null;
                try {
                    const response = await fetch(`/admin/api/hf/models/${encodeURIComponent(modelName)}`, {
                        method: 'DELETE',
                    });
                    if (response.ok) {
                        await this.loadHFModels();
                        await this.loadModels();
                    } else {
                        const data = await response.json();
                        this.hfError = data.detail || 'Failed to delete model';
                        setTimeout(() => { this.hfError = ''; }, 5000);
                    }
                } catch (err) {
                    console.error('Failed to delete model:', err);
                    this.hfError = 'Failed to delete model. Check server connection.';
                    setTimeout(() => { this.hfError = ''; }, 5000);
                }
            },

            startHFRefresh() {
                this.stopHFRefresh();
                this._hfRefreshTimer = setInterval(() => {
                    this.loadHFTasks();
                }, 2000);
            },

            stopHFRefresh() {
                if (this._hfRefreshTimer) {
                    clearInterval(this._hfRefreshTimer);
                    this._hfRefreshTimer = null;
                }
            },

            formatProgress(task) {
                const pct = Math.round(task.progress || 0);
                const dlGB = (task.downloaded_size / (1024 ** 3)).toFixed(1);
                const totalGB = (task.total_size / (1024 ** 3)).toFixed(1);
                return `${pct}% \u00b7 ${dlGB} GB / ${totalGB} GB`;
            },

            // =================================================================
            // Recommended Models Functions
            // =================================================================

            async loadRecommendedModels() {
                this.hfRecommendedLoading = true;
                try {
                    const response = await fetch('/admin/api/hf/recommended');
                    if (response.ok) {
                        this.hfRecommended = await response.json();
                        this.hfRecommendedLoaded = true;
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load recommended models:', err);
                } finally {
                    this.hfRecommendedLoading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            downloadRecommended(repoId) {
                this.hfRepoId = repoId;
                this.startHFDownload();
            },

            getMemoryFitStatus(sizeBytes) {
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                if (!totalBytes || !sizeBytes) return 'safe';
                const ratio = sizeBytes / totalBytes;
                if (ratio > 0.95) return 'danger';
                if (ratio > 0.80) return 'warning';
                return 'safe';
            },

            formatDownloads(count) {
                if (count >= 1000000) return (count / 1000000).toFixed(1) + 'M';
                if (count >= 1000) return (count / 1000).toFixed(1) + 'K';
                return count.toString();
            },
        }
    }
