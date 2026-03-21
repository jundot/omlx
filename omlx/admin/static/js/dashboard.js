    // OCR model types that require temperature=0.0 (deterministic output)
    const OCR_CONFIG_MODEL_TYPES = new Set([
        'deepseekocr', 'deepseekocr_2', 'dots_ocr', 'glm_ocr',
    ]);
    const DSA_MODEL_TYPES = new Set([
        'deepseek_v32', 'glm_moe_dsa',
    ]);
    const DASHBOARD_MAIN_TABS = new Set(['status', 'settings', 'models', 'logs', 'bench']);
    const DASHBOARD_SETTINGS_TABS = new Set(['global', 'models']);
    const DASHBOARD_MODELS_TABS = new Set(['manager', 'downloader', 'quantizer']);
    const DASHBOARD_BENCH_TABS = new Set(['throughput', 'accuracy']);

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
                model: { model_dirs: [''], max_model_memory: '' },
                memory: { max_process_memory: 'auto', prefill_memory_guard: true },
                scheduler: { max_num_seqs: 8, completion_batch_size: 8 },
                cache: { enabled: true, ssd_cache_dir: '', ssd_cache_max_size: 'auto', hot_cache_max_size: '0', initial_cache_blocks: 256 },
                sampling: { max_context_window: 32768, max_tokens: 32768, temperature: 1.0, top_p: 0.95, top_k: 0, repetition_penalty: 1.0 },
                mcp: { config_path: '' },
                huggingface: { endpoint: '' },
                auth: { api_key_set: false, api_key: '', skip_api_key_verification: false, sub_keys: [] },
                claude_code: { context_scaling_enabled: false, target_context_size: 200000, mode: 'cloud', opus_model: null, sonnet_model: null, haiku_model: null },
                integrations: { codex_model: null, opencode_model: null, openclaw_model: null, openclaw_tools_profile: 'full' },
                ui: { language: 'en' },
                system: { total_memory_bytes: 0, total_memory: '', auto_model_memory: '', ssd_total_bytes: 0, ssd_total: '' },
            },

            // Process memory slider (10-99%)
            processMemoryPercent: 90,
            processMemoryAuto: true,
            // Memory slider (0-100%)
            memoryPercent: 80,
            modelMemoryAuto: true,
            // Cache slider (0-100%)
            cachePercent: 10,
            editingCache: false,
            // Hot cache slider (0-50%)
            hotCachePercent: 0,
            // Editing states for direct GB input
            editingProcessMemory: false,
            editingModelMemory: false,
            editingHotCache: false,

            // Models
            models: [],
            loadingModels: false,
            reloading: false,
            sortBy: 'id',
            sortOrder: 'asc',

            // Auth UI state
            showApiKey: false,
            // Sub key management
            newSubKeyValue: '',
            newSubKeyName: '',
            showNewSubKeyForm: false,
            showNewSubKey: false,
            subKeyError: '',
            showSubKeys: {},

            // Saving state
            saving: false,
            saveSuccess: false,
            saveMessage: '',
            saveError: '',

            // Model settings modal
            showModelSettingsModal: false,
            selectedModel: null,
            modelSettings: {
                model_alias: '',
                model_type_override: '',
                max_context_window: null,
                max_tokens: null,
                temperature: null,
                top_p: null,
                top_k: null,
                repetition_penalty: null,
                min_p: null,
                presence_penalty: null,
                force_sampling: false,
                enableToolResultLimit: false,
                max_tool_result_tokens: null,
                ctKwargEntries: [],
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
                host: '127.0.0.1',
                port: 8000,
                api_key: '',
                engines: {},
                active_models: {
                    models: [],
                    model_memory_used: 0,
                    model_memory_max: 0,
                    total_active_requests: 0,
                    total_waiting_requests: 0,
                },
                runtime_cache: {
                    base_path: '',
                    ssd_cache_dir: '',
                    response_state_dir: '',
                    models: [],
                    total_num_files: 0,
                    total_size_bytes: 0,
                    effective_block_sizes: [],
                },
            },
            alltimeStats: {
                total_prompt_tokens: 0,
                total_cached_tokens: 0,
                cache_efficiency: 0.0,
                avg_prefill_tps: 0.0,
                avg_generation_tps: 0.0,
                total_requests: 0,
            },
            statsScope: 'session',
            selectedStatsModel: '',
            showClearStatsConfirm: false,
            showClearAlltimeConfirm: false,
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

            // Models sub-tab state
            modelsTab: 'manager',
            modelsDropdown: false,

            // HF Mirror settings modal
            showHfMirrorModal: false,
            hfMirrorEndpoint: '',
            hfMirrorSaving: false,

            // Update check state
            updateAvailable: false,
            latestVersion: null,
            releaseUrl: null,
            versionHover: false,
            _updateCheckTimer: null,

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
            hfMlxOnly: true,

            // Pagination state
            hfPage: { trending: 1, popular: 1, search: 1 },
            hfPageSize: 10,

            // Search state
            hfSearchQuery: '',
            hfSearchSort: 'downloads',
            hfSearchResults: [],
            hfSearchLoading: false,
            hfSearchLoaded: false,
            hfSearchDebounceTimer: null,

            // Search history
            hfSearchHistory: JSON.parse(localStorage.getItem('hfSearchHistory') || '[]'),
            hfSearchHistoryOpen: false,

            // Model detail modal
            hfModelDetail: null,
            hfModelDetailLoading: false,

            // ModelScope Downloader state
            downloaderSource: 'hf',
            msAvailable: false,
            msInitialized: false,
            msRepoId: '',
            msToken: '',
            msDownloading: false,
            msTasks: [],
            msError: '',
            msSuccess: '',
            _msRefreshTimer: null,

            // MS Recommended models state
            msRecommended: { trending: [], popular: [] },
            msRecommendedLoaded: false,
            msRecommendedLoading: false,
            msRecommendedTab: 'trending',
            msMlxOnly: true,

            // MS Pagination state
            msPage: { trending: 1, popular: 1, search: 1 },
            msPageSize: 10,

            // MS Search state
            msSearchQuery: '',
            msSearchSort: 'trending',
            msSearchResults: [],
            msSearchLoading: false,
            msSearchLoaded: false,
            msSearchHistory: JSON.parse(localStorage.getItem('msSearchHistory') || '[]'),
            msSearchHistoryOpen: false,
            msSearchDebounceTimer: null,

            // MS Model detail modal
            msModelDetail: null,
            msModelDetailLoading: false,

            // oQ Quantizer state
            oqModels: [],
            oqModelsLoaded: false,
            oqSelectedModelPath: '',
            oqLevel: 4,
            oqStarting: false,
            oqTasks: [],
            oqError: '',
            oqSuccess: '',
            _oqRefreshTimer: null,
            // oQ Advanced Settings
            oqAdvancedOpen: false,
            oqEnableClip: false,
            oqGroupSize: 64,
            oqClipSamples: 128,
            oqClipSeqLen: 512,
            oqCalibDataset: 'code_multilingual',
            oqClipBatchSize: 1024,

            // Benchmark state
            benchModelId: '',
            benchPromptLengths: { 1024: true, 4096: true, 8192: false, 16384: false, 32768: false, 65536: false, 131072: false, 200000: false },
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
            benchTip: null,
            benchIncludeImage: false,
            benchDeviceInfo: null,
            benchUploadResults: [],
            benchUploadDone: null,
            benchUploading: false,

            // Bench sub-tab & dropdown
            benchTab: 'throughput',
            benchDropdown: false,

            // Accuracy benchmark state
            accModelId: '',
            accBenchmarks: { mmlu: true, hellaswag: true, truthfulqa: true, gsm8k: false, livecodebench: false },
            accSampleSizes: { mmlu: 300, hellaswag: 200, truthfulqa: 200, gsm8k: 100, livecodebench: 100 },
            accBenchmarkList: [
                { key: 'mmlu', label: 'MMLU', desc: 'Knowledge · 57 subjects' },
                { key: 'hellaswag', label: 'HellaSwag', desc: 'Commonsense reasoning' },
                { key: 'truthfulqa', label: 'TruthfulQA', desc: 'Truthfulness' },
                { key: 'gsm8k', label: 'GSM8K', desc: 'Math reasoning' },
                { key: 'livecodebench', label: 'LiveCodeBench', desc: 'Code generation' },
            ],
            accBatchSize: 1,
            accRunning: false,
            accCurrentModel: '',
            accCurrentBenchId: null,
            accProgress: null,
            accAllResults: [],   // accumulated across all models
            accQueue: [],        // server queue mirror
            accError: '',
            accEventSource: null,
            accShowText: false,
            accCopied: false,

            async init() {
                // Apply theme
                this.applyTheme();
                this.applyTabStateFromUrl();

                await Promise.all([
                    this.loadGlobalSettings(),
                    this.loadModels(),
                    this.checkForUpdate()
                ]);
                this.$nextTick(() => {
                    lucide.createIcons();
                });

                this.startUpdateCheckTimer();

                await this.handleMainTabChange(this.mainTab);

                // Watch for main tab changes to manage refresh timers
                this.$watch('mainTab', (value) => {
                    this.handleMainTabChange(value);
                });

                this.$watch('hfMlxOnly', () => {
                    this.hfRecommended = { trending: [], popular: [] };
                    this.hfRecommendedLoaded = false;
                    this.hfSearchResults = [];
                    this.hfSearchLoaded = false;
                    this.loadRecommendedModels();
                    if (this.hfSearchQuery.trim()) {
                        this.searchHFModels();
                    }
                });

                this.$watch('msMlxOnly', () => {
                    this.msRecommended = { trending: [], popular: [] };
                    this.msRecommendedLoaded = false;
                    this.msSearchResults = [];
                    this.msSearchLoaded = false;
                    this.loadMsRecommendedModels();
                    if (this.msSearchQuery.trim()) {
                        this.searchMSModels();
                    }
                });

                window.addEventListener('popstate', () => {
                    this.applyTabStateFromUrl();
                });
            },

            async handleMainTabChange(value) {
                if (value === 'status') {
                    await this.loadStats();
                    this.startStatsRefresh();
                } else {
                    this.stopStatsRefresh();
                }
                if (value === 'logs') {
                    await this.loadLogs();
                    this.startLogRefresh();
                } else {
                    this.stopLogRefresh();
                }
                if (value === 'models') {
                    const loads = [this.loadHFModels(), this.loadHFTasks(), this.loadOQTasks()];
                    if (this.modelsTab === 'downloader' && !this.hfRecommendedLoaded) {
                        loads.push(this.loadRecommendedModels());
                    }
                    if (this.modelsTab === 'quantizer') {
                        loads.push(this.loadOQModels());
                    }
                    if (this.msInitialized && this.msAvailable) {
                        loads.push(this.loadMSTasks());
                    }
                    await Promise.all(loads);
                    const hasActive = this.hfTasks.some(t =>
                        t.status === 'pending' || t.status === 'downloading');
                    if (hasActive) this.startHFRefresh();
                    const hasMsActive = this.msTasks.some(t =>
                        t.status === 'pending' || t.status === 'downloading');
                    if (hasMsActive) this.startMSRefresh();
                    const hasOqActive = this.oqTasks.some(t =>
                        ['pending', 'loading', 'quantizing', 'saving'].includes(t.status));
                    if (hasOqActive) this.startOQRefresh();
                } else {
                    this.stopHFRefresh();
                    this.stopMSRefresh();
                    this.stopOQRefresh();
                }
                if (value === 'bench') {
                    if (!this.benchDeviceInfo) await this.loadBenchDeviceInfo();
                    await this.loadAccState();
                }
                this.$nextTick(() => lucide.createIcons());
            },

            applyTabStateFromUrl() {
                const params = new URLSearchParams(window.location.search);
                const mainTab = params.get('tab');
                const settingsTab = params.get('settingsTab');
                const modelsTab = params.get('modelsTab');

                const benchTab = params.get('benchTab');

                this.mainTab = DASHBOARD_MAIN_TABS.has(mainTab) ? mainTab : 'status';
                this.activeTab = DASHBOARD_SETTINGS_TABS.has(settingsTab) ? settingsTab : 'global';
                this.modelsTab = DASHBOARD_MODELS_TABS.has(modelsTab) ? modelsTab : 'manager';
                this.benchTab = DASHBOARD_BENCH_TABS.has(benchTab) ? benchTab : 'throughput';
            },

            syncTabStateToUrl() {
                const url = new URL(window.location.href);
                url.searchParams.set('tab', this.mainTab);

                if (this.mainTab === 'settings') {
                    url.searchParams.set('settingsTab', this.activeTab);
                } else {
                    url.searchParams.delete('settingsTab');
                }

                if (this.mainTab === 'models') {
                    url.searchParams.set('modelsTab', this.modelsTab);
                } else {
                    url.searchParams.delete('modelsTab');
                }

                if (this.mainTab === 'bench') {
                    url.searchParams.set('benchTab', this.benchTab);
                } else {
                    url.searchParams.delete('benchTab');
                }

                window.history.replaceState({}, '', url);
            },

            setMainTab(tab) {
                if (!DASHBOARD_MAIN_TABS.has(tab)) return;
                this.mainTab = tab;
                this.syncTabStateToUrl();
            },

            setSettingsTab(tab) {
                if (!DASHBOARD_SETTINGS_TABS.has(tab)) return;
                this.activeTab = tab;
                this.mainTab = 'settings';
                this.syncTabStateToUrl();
            },

            setModelsTab(tab) {
                if (!DASHBOARD_MODELS_TABS.has(tab)) return;
                this.modelsTab = tab;
                this.mainTab = 'models';
                this.syncTabStateToUrl();
                if (tab === 'quantizer') {
                    this.loadOQModels();
                }
            },

            async checkForUpdate() {
                try {
                    const resp = await fetch('/admin/api/update-check');
                    if (resp.ok) {
                        const data = await resp.json();
                        this.updateAvailable = data.update_available;
                        this.latestVersion = data.latest_version;
                        this.releaseUrl = data.release_url;
                    }
                } catch (e) {
                    // Silently ignore - not critical
                }
            },

            startUpdateCheckTimer() {
                this._updateCheckTimer = setInterval(() => this.checkForUpdate(), 3600000);
            },

            stopUpdateCheckTimer() {
                if (this._updateCheckTimer) {
                    clearInterval(this._updateCheckTimer);
                    this._updateCheckTimer = null;
                }
            },

            async loadGlobalSettings() {
                try {
                    const response = await fetch('/admin/api/global-settings');
                    if (response.ok) {
                        const data = await response.json();
                        // Deep merge to preserve defaults for missing fields
                        // Handle model_dirs: prefer list, fallback to single model_dir
                        const modelDirs = data.model?.model_dirs?.length
                            ? data.model.model_dirs
                            : (data.model?.model_dir ? [data.model.model_dir] : ['']);
                        this.globalSettings = {
                            ...this.globalSettings,
                            ...data,
                            server: { ...this.globalSettings.server, ...data.server },
                            model: { ...this.globalSettings.model, ...data.model, model_dirs: modelDirs },
                            memory: { ...this.globalSettings.memory, ...data.memory },
                            scheduler: { ...this.globalSettings.scheduler, ...data.scheduler },
                            cache: { ...this.globalSettings.cache, ...data.cache },
                            sampling: { ...this.globalSettings.sampling, ...data.sampling },
                            mcp: { ...this.globalSettings.mcp, ...data.mcp },
                            huggingface: { ...this.globalSettings.huggingface, ...data.huggingface },
                            auth: { ...this.globalSettings.auth, ...data.auth },
                            claude_code: { ...this.globalSettings.claude_code, ...data.claude_code },
                            integrations: { ...this.globalSettings.integrations, ...data.integrations },
                            system: { ...this.globalSettings.system, ...data.system },
                        };
                        this.globalSettings.ui = data.ui || { language: 'en' };

                        // Calculate memory percent from stored value
                        if (this.globalSettings.model.max_model_memory === 'auto') {
                            this.modelMemoryAuto = true;
                            this.memoryPercent = 90;
                        } else if (this.globalSettings.model.max_model_memory === 'disabled') {
                            this.modelMemoryAuto = false;
                            this.memoryPercent = 0;
                        } else {
                            this.modelMemoryAuto = false;
                            this.memoryPercent = this.parseMemoryToPercent(
                                this.globalSettings.model.max_model_memory,
                                this.globalSettings.system.total_memory_bytes
                            );
                        }
                        // Sync the memory string value from percent
                        this.updateMemoryFromSlider();

                        // Calculate process memory slider state from stored value
                        const pmState = this.parseProcessMemoryToState(
                            this.globalSettings.memory.max_process_memory,
                            this.globalSettings.system.total_memory_bytes
                        );
                        this.processMemoryAuto = pmState.auto;
                        this.processMemoryPercent = pmState.percent;


                        // Calculate cache percent from stored value (based on total capacity)
                        this.cachePercent = this.parseCacheToPercent(
                            this.globalSettings.cache.ssd_cache_max_size,
                            this.globalSettings.system.ssd_total_bytes
                        );
                        // Sync the cache string value from percent
                        this.updateCacheFromSlider();

                        // Calculate hot cache percent from stored value
                        this.hotCachePercent = this.parseHotCacheToPercent(
                            this.globalSettings.cache.hot_cache_max_size,
                            this.globalSettings.system.total_memory_bytes
                        );
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
                if (!s.model.model_dirs || !s.model.model_dirs.some(d => d.trim())) errors.push('Model Directory');
                if (!s.scheduler.max_num_seqs) errors.push('Max Sequences');
                if (!s.scheduler.completion_batch_size) errors.push('Completion Batch Size');
                if (!s.cache.ssd_cache_max_size) errors.push('Max Cache Size');
                if (!s.sampling.max_context_window) errors.push('Max Context Window');
                if (!s.sampling.max_tokens) errors.push('Max Tokens');

                if (errors.length > 0) {
                    this.saveError = window.t('js.error.required_fields').replace('{fields}', errors.join(', '));
                    this.saving = false;
                    return;
                }

                // Validate API key if provided
                if (s.auth.api_key) {
                    if (s.auth.api_key.length < 4) {
                        this.saveError = window.t('js.error.api_key_min_length');
                        this.saving = false;
                        return;
                    }
                    if (/\s/.test(s.auth.api_key)) {
                        this.saveError = window.t('js.error.api_key_no_whitespace');
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
                            model_dirs: this.globalSettings.model.model_dirs.filter(d => d.trim()),
                            max_model_memory: this.globalSettings.model.max_model_memory,
                            model_fallback: this.globalSettings.model.model_fallback,
                            max_process_memory: this.globalSettings.memory.max_process_memory,
                            memory_prefill_memory_guard: this.globalSettings.memory.prefill_memory_guard,
                            max_num_seqs: this.globalSettings.scheduler.max_num_seqs,
                            completion_batch_size: this.globalSettings.scheduler.completion_batch_size,
                            cache_enabled: this.globalSettings.cache.enabled,
                            ssd_cache_dir: this.globalSettings.cache.ssd_cache_dir,
                            ssd_cache_max_size: this.globalSettings.cache.ssd_cache_max_size,
                            hot_cache_max_size: this.globalSettings.cache.hot_cache_max_size,
                            initial_cache_blocks: this.globalSettings.cache.initial_cache_blocks,
                            sampling_max_context_window: this.globalSettings.sampling.max_context_window,
                            sampling_max_tokens: this.globalSettings.sampling.max_tokens,
                            sampling_temperature: this.globalSettings.sampling.temperature,
                            sampling_top_p: this.globalSettings.sampling.top_p,
                            sampling_top_k: this.globalSettings.sampling.top_k,
                            sampling_repetition_penalty: this.globalSettings.sampling.repetition_penalty,
                            mcp_config: this.globalSettings.mcp.config_path,
                            ...(this.globalSettings.auth.api_key ? { api_key: this.globalSettings.auth.api_key } : {}),
                            skip_api_key_verification: this.globalSettings.auth.skip_api_key_verification,
                        }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        this.saveSuccess = true;
                        this.saveMessage = data.message || 'Settings saved successfully';
                        this.$nextTick(() => lucide.createIcons());
                        // Refresh stats and model list (cache changes unload models)
                        await this.loadStats();
                        await this.loadModels();
                        setTimeout(() => { this.saveSuccess = false; }, 5000);
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        this.saveError = Array.isArray(data.detail) ? data.detail.join(', ') : (data.detail || window.t('js.error.save_settings_failed'));
                        // Reload settings to revert to server values
                        await this.loadGlobalSettings();
                        this.$nextTick(() => lucide.createIcons());
                    }
                } catch (err) {
                    console.error('Failed to save global settings:', err);
                    this.saveError = window.t('js.error.save_settings_failed');
                    // Reload settings to revert to server values
                    await this.loadGlobalSettings();
                    this.$nextTick(() => lucide.createIcons());
                } finally {
                    this.saving = false;
                }
            },

            // Sub key management
            generateSubKey() {
                const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
                const rand = Array.from(crypto.getRandomValues(new Uint8Array(16)))
                    .map(b => chars[b % chars.length]).join('');
                this.newSubKeyValue = 'omlx-' + rand;
                this.showNewSubKey = true;
            },

            async createSubKey() {
                this.subKeyError = '';
                if (!this.newSubKeyValue || this.newSubKeyValue.length < 4) {
                    this.subKeyError = window.t('js.error.api_key_min_length');
                    return;
                }
                if (/\s/.test(this.newSubKeyValue)) {
                    this.subKeyError = window.t('js.error.api_key_no_whitespace');
                    return;
                }
                try {
                    const response = await fetch('/admin/api/sub-keys', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ key: this.newSubKeyValue, name: this.newSubKeyName }),
                    });
                    if (response.ok) {
                        this.newSubKeyValue = '';
                        this.newSubKeyName = '';
                        this.showNewSubKeyForm = false;
                        this.showNewSubKey = false;
                        await this.loadGlobalSettings();
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        this.subKeyError = data.detail || window.t('js.error.save_settings_failed');
                    }
                } catch (err) {
                    this.subKeyError = window.t('js.error.save_settings_failed');
                }
            },

            async deleteSubKey(key) {
                if (!confirm(window.t('settings.auth.sub_keys_delete_confirm'))) return;
                try {
                    const response = await fetch('/admin/api/sub-keys', {
                        method: 'DELETE',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ key }),
                    });
                    if (response.ok) {
                        await this.loadGlobalSettings();
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to delete sub key:', err);
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

            async reloadModels() {
                if (this.reloading) return;
                this.reloading = true;
                try {
                    const response = await fetch('/admin/api/reload', { method: 'POST' });
                    if (response.ok) {
                        await Promise.all([this.loadModels(), this.loadHFModels()]);
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || window.t('js.error.reload_failed'));
                    }
                } catch (err) {
                    console.error('Failed to reload models:', err);
                    alert(window.t('js.error.reload_failed'));
                } finally {
                    this.reloading = false;
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
                        alert(data.detail || window.t('js.error.update_model_setting_failed'));
                        await this.loadModels();
                    }
                } catch (err) {
                    console.error('Failed to update model setting:', err);
                    alert(window.t('js.error.update_model_setting_failed'));
                    await this.loadModels();
                }
            },

            async loadModel(modelId) {
                const model = this.models.find(m => m.id === modelId);
                if (model) model.is_loading = true;
                try {
                    const response = await fetch(`/admin/api/models/${encodeURIComponent(modelId)}/load`, {
                        method: 'POST',
                    });
                    if (response.ok) {
                        await this.loadModels();
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || window.t('js.error.load_model_failed'));
                        await this.loadModels();
                    }
                } catch (err) {
                    console.error('Failed to load model:', err);
                    alert(window.t('js.error.load_model_failed'));
                    await this.loadModels();
                }
            },

            async unloadModel(modelId) {
                try {
                    const response = await fetch(`/admin/api/models/${encodeURIComponent(modelId)}/unload`, {
                        method: 'POST',
                    });
                    if (response.ok) {
                        const model = this.models.find(m => m.id === modelId);
                        if (model) model.loaded = false;
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || window.t('js.error.unload_model_failed'));
                    }
                    await this.loadModels();
                } catch (err) {
                    console.error('Failed to unload model:', err);
                    alert(window.t('js.error.unload_model_failed'));
                    await this.loadModels();
                }
            },

            openModelSettings(model) {
                this.selectedModel = model;
                // Load existing settings if available
                const settings = model.settings || {};
                // Parse chat_template_kwargs into ctKwargEntries
                const ctk = settings.chat_template_kwargs || {};
                const forcedKeys = new Set(settings.forced_ct_kwargs || []);
                const ctKwargEntries = [];
                for (const [key, value] of Object.entries(ctk)) {
                    if (key === 'enable_thinking') {
                        ctKwargEntries.push({type: 'enable_thinking', value: String(value), force: forcedKeys.has('enable_thinking')});
                    } else if (key === 'reasoning_effort') {
                        ctKwargEntries.push({type: 'reasoning_effort', value: String(value), force: forcedKeys.has('reasoning_effort')});
                    } else {
                        ctKwargEntries.push({type: 'custom', key, value: String(value), force: forcedKeys.has(key)});
                    }
                }
                const isOcr = OCR_CONFIG_MODEL_TYPES.has(model.config_model_type || '');
                this.modelSettings = {
                    model_alias: settings.model_alias || '',
                    model_type_override: settings.model_type_override || '',
                    max_context_window: settings.max_context_window || null,
                    max_tokens: settings.max_tokens || null,
                    temperature: isOcr ? 0.0 : (settings.temperature ?? null),
                    top_p: settings.top_p ?? null,
                    top_k: settings.top_k ?? null,
                    repetition_penalty: settings.repetition_penalty ?? null,
                    min_p: settings.min_p ?? null,
                    presence_penalty: settings.presence_penalty ?? null,
                    force_sampling: settings.force_sampling || false,
                    enableThinkingBudget: !!(settings.thinking_budget_tokens),
                    thinking_budget_tokens: settings.thinking_budget_tokens || null,
                    enableToolResultLimit: !!(settings.max_tool_result_tokens),
                    max_tool_result_tokens: settings.max_tool_result_tokens || null,
                    ttl_seconds: settings.ttl_seconds ?? null,
                    enableIndexCache: !!(settings.index_cache_freq),
                    index_cache_freq: settings.index_cache_freq || null,
                    specprefill_enabled: settings.specprefill_enabled || false,
                    specprefill_draft_model: settings.specprefill_draft_model || '',
                    specprefill_keep_pct: settings.specprefill_keep_pct ? String(settings.specprefill_keep_pct) : '0.2',
                    specprefill_threshold: settings.specprefill_threshold || null,
                    ctKwargEntries,
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
                        body: JSON.stringify((() => {
                            // Build chat_template_kwargs and forced_ct_kwargs from ctKwargEntries
                            const chatTemplateKwargs = {};
                            const forcedCtKwargs = [];
                            for (const entry of this.modelSettings.ctKwargEntries) {
                                if (entry.type === 'enable_thinking') {
                                    chatTemplateKwargs.enable_thinking = entry.value === 'true';
                                    if (entry.force) forcedCtKwargs.push('enable_thinking');
                                } else if (entry.type === 'reasoning_effort') {
                                    chatTemplateKwargs.reasoning_effort = entry.value;
                                    if (entry.force) forcedCtKwargs.push('reasoning_effort');
                                } else if (entry.type === 'custom' && entry.key && entry.key.trim()) {
                                    let val = entry.value;
                                    if (val === 'true') val = true;
                                    else if (val === 'false') val = false;
                                    else if (!isNaN(Number(val)) && val.trim() !== '') val = Number(val);
                                    const key = entry.key.trim();
                                    chatTemplateKwargs[key] = val;
                                    if (entry.force) forcedCtKwargs.push(key);
                                }
                            }
                            return {
                                model_alias: this.modelSettings.model_alias?.trim() || null,
                                model_type_override: this.modelSettings.model_type_override || null,
                                max_context_window: this.modelSettings.max_context_window || null,
                                max_tokens: this.modelSettings.max_tokens || null,
                                temperature: Number.isFinite(this.modelSettings.temperature) ? this.modelSettings.temperature : null,
                                top_p: Number.isFinite(this.modelSettings.top_p) ? this.modelSettings.top_p : null,
                                top_k: Number.isFinite(this.modelSettings.top_k) ? this.modelSettings.top_k : null,
                                repetition_penalty: Number.isFinite(this.modelSettings.repetition_penalty) ? this.modelSettings.repetition_penalty : null,
                                min_p: Number.isFinite(this.modelSettings.min_p) ? this.modelSettings.min_p : null,
                                presence_penalty: Number.isFinite(this.modelSettings.presence_penalty) ? this.modelSettings.presence_penalty : null,
                                force_sampling: this.modelSettings.force_sampling,
                                ttl_seconds: this.modelSettings.ttl_seconds || null,
                                index_cache_freq: this.modelSettings.enableIndexCache
                                    ? (this.modelSettings.index_cache_freq || 4)
                                    : 0,
                                thinking_budget_enabled: this.modelSettings.enableThinkingBudget,
                                thinking_budget_tokens: this.modelSettings.enableThinkingBudget
                                    ? (this.modelSettings.thinking_budget_tokens || null)
                                    : 0,
                                max_tool_result_tokens: this.modelSettings.enableToolResultLimit
                                    ? (this.modelSettings.max_tool_result_tokens || null)
                                    : 0,
                                chat_template_kwargs: Object.keys(chatTemplateKwargs).length > 0
                                    ? chatTemplateKwargs : null,
                                forced_ct_kwargs: forcedCtKwargs.length > 0
                                    ? forcedCtKwargs : null,
                                specprefill_enabled: this.modelSettings.specprefill_enabled,
                                specprefill_draft_model: this.modelSettings.specprefill_draft_model || null,
                                specprefill_keep_pct: this.modelSettings.specprefill_enabled
                                    ? parseFloat(this.modelSettings.specprefill_keep_pct) || 0.2
                                    : null,
                                specprefill_threshold: this.modelSettings.specprefill_enabled
                                    ? (this.modelSettings.specprefill_threshold || null)
                                    : null,
                            };
                        })()),
                    });

                    if (response.ok) {
                        // Update local model data from server response
                        const data = await response.json();
                        const model = this.models.find(m => m.id === this.selectedModel.id);
                        if (model) {
                            model.settings = data.settings || {};
                            // Update effective model_type/engine_type from server
                            if (data.model_type) {
                                model.model_type = data.model_type;
                            }
                            if (data.engine_type) {
                                model.engine_type = data.engine_type;
                            }
                        }
                        this.showModelSettingsModal = false;
                        if (data.requires_reload) {
                            alert(window.t('js.info.model_type_reload_required'));
                        }
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || window.t('js.error.save_model_settings_failed'));
                    }
                } catch (err) {
                    console.error('Failed to save model settings:', err);
                    alert(window.t('js.error.save_model_settings_failed'));
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
                        alert(window.t('js.error.no_config_defaults'));
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(data.detail || window.t('js.error.load_generation_config_failed'));
                    }
                } catch (err) {
                    console.error('Failed to load generation config:', err);
                    alert(window.t('js.error.load_generation_config_failed'));
                } finally {
                    this.loadingGenDefaults = false;
                }
            },

            // Status tab functions
            get displayHost() {
                const host = this.stats.host || '127.0.0.1';
                if (host === '0.0.0.0') return 'your-ip-address';
                if (host === 'localhost') return '127.0.0.1';
                return host;
            },

            get llmModels() {
                return this.models.filter(m => m.model_type === 'llm' || m.model_type === 'vlm' || !m.model_type);
            },

            shellQuote(value) {
                const s = String(value ?? '');
                if (!s) return "''";
                return `'${s.replace(/'/g, `'"'"'`)}'`;
            },

            shellEnvAssign(name, value) {
                return `${name}=${this.shellQuote(value)}`;
            },

            get claudeCodeCommand() {
                const mode = this.globalSettings.claude_code.mode;
                if (mode === 'cloud') {
                    return 'env -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_DEFAULT_OPUS_MODEL -u ANTHROPIC_DEFAULT_SONNET_MODEL -u ANTHROPIC_DEFAULT_HAIKU_MODEL -u API_TIMEOUT_MS -u CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC claude';
                }
                // Local mode
                const port = this.stats.port || 8000;
                const opusModel = this.globalSettings.claude_code.opus_model || 'select-a-model';
                const sonnetModel = this.globalSettings.claude_code.sonnet_model || 'select-a-model';
                const haikuModel = this.globalSettings.claude_code.haiku_model || 'select-a-model';
                const parts = [];
                parts.push(this.shellEnvAssign('ANTHROPIC_BASE_URL', `http://${this.displayHost}:${port}`));
                if (this.stats.api_key) {
                    parts.push(this.shellEnvAssign('ANTHROPIC_AUTH_TOKEN', this.stats.api_key));
                }
                parts.push(this.shellEnvAssign('ANTHROPIC_DEFAULT_OPUS_MODEL', opusModel));
                parts.push(this.shellEnvAssign('ANTHROPIC_DEFAULT_SONNET_MODEL', sonnetModel));
                parts.push(this.shellEnvAssign('ANTHROPIC_DEFAULT_HAIKU_MODEL', haikuModel));
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
                            claude_code_mode: this.globalSettings.claude_code.mode,
                            claude_code_opus_model: this.globalSettings.claude_code.opus_model,
                            claude_code_sonnet_model: this.globalSettings.claude_code.sonnet_model,
                            claude_code_haiku_model: this.globalSettings.claude_code.haiku_model,
                        }),
                    });
                    if (!response.ok) {
                        console.error('Failed to save Claude Code settings');
                    }
                } catch (err) {
                    console.error('Failed to save Claude Code settings:', err);
                }
            },

            get codexCommand() {
                const cli = this.stats.cli_prefix || 'omlx';
                const model = this.globalSettings.integrations.codex_model || 'select-a-model';
                const parts = [`${this.shellQuote(cli)} launch codex --model ${this.shellQuote(model)}`];
                if (this.stats.api_key) {
                    parts.push(`--api-key ${this.shellQuote(this.stats.api_key)}`);
                }
                return parts.join(' ');
            },

            get opencodeCommand() {
                const cli = this.stats.cli_prefix || 'omlx';
                const model = this.globalSettings.integrations.opencode_model || 'select-a-model';
                const parts = [`${this.shellQuote(cli)} launch opencode --model ${this.shellQuote(model)}`];
                if (this.stats.api_key) {
                    parts.push(`--api-key ${this.shellQuote(this.stats.api_key)}`);
                }
                return parts.join(' ');
            },

            get openclawCommand() {
                const cli = this.stats.cli_prefix || 'omlx';
                const model = this.globalSettings.integrations.openclaw_model || 'select-a-model';
                const profile = this.globalSettings.integrations.openclaw_tools_profile || 'full';
                const parts = [`${this.shellQuote(cli)} launch openclaw --model ${this.shellQuote(model)}`];
                if (this.stats.api_key) {
                    parts.push(`--api-key ${this.shellQuote(this.stats.api_key)}`);
                }
                parts.push(`--tools-profile ${this.shellQuote(profile)}`);
                return parts.join(' ');
            },

            async saveIntegrationSettings() {
                try {
                    const response = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            integrations_codex_model: this.globalSettings.integrations.codex_model,
                            integrations_opencode_model: this.globalSettings.integrations.opencode_model,
                            integrations_openclaw_model: this.globalSettings.integrations.openclaw_model,
                            integrations_openclaw_tools_profile: this.globalSettings.integrations.openclaw_tools_profile,
                        }),
                    });
                    if (!response.ok) {
                        console.error('Failed to save integration settings');
                    }
                } catch (err) {
                    console.error('Failed to save integration settings:', err);
                }
            },

            async saveLanguage(lang) {
                try {
                    const response = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ui_language: lang })
                    });
                    if (response.ok) {
                        location.reload();
                    } else {
                        console.error('Failed to save language');
                    }
                } catch (e) {
                    console.error('Failed to save language:', e);
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
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }

                    // Load all-time stats
                    const alltimeParams = new URLSearchParams({ scope: 'alltime' });
                    if (this.selectedStatsModel) {
                        alltimeParams.set('model', this.selectedStatsModel);
                    }
                    const alltimeUrl = '/admin/api/stats?' + alltimeParams;
                    const alltimeResponse = await fetch(alltimeUrl);
                    if (alltimeResponse.ok) {
                        const alltimeData = await alltimeResponse.json();
                        this.alltimeStats = { ...this.alltimeStats, ...alltimeData };
                    }
                    this.$nextTick(() => lucide.createIcons());
                } catch (err) {
                    console.error('Failed to load stats:', err);
                }
            },

            async clearStats() {
                try {
                    await fetch('/admin/api/stats/clear', { method: 'POST' });
                    this.showClearStatsConfirm = false;
                    await this.loadStats();
                } catch (err) {
                    console.error('Failed to clear stats:', err);
                    this.showClearStatsConfirm = false;
                }
            },

            async clearAlltimeStats() {
                try {
                    await fetch('/admin/api/stats/clear-alltime', { method: 'POST' });
                    this.showClearAlltimeConfirm = false;
                    await this.loadStats();
                } catch (err) {
                    console.error('Failed to clear all-time stats:', err);
                    this.showClearAlltimeConfirm = false;
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

            formatSizeBytes(bytes) {
                if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
                if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(0) + ' MB';
                return '0';
            },

            formatTokenCount(n) {
                if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
                if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
                return String(n);
            },

            get activeModelsMemoryPercent() {
                const am = this.stats.active_models;
                if (!am || !am.model_memory_max) return 0;
                return Math.min(100, (am.model_memory_used / am.model_memory_max) * 100);
            },

            copyToClipboard(text) {
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text).catch(() => {
                        this._copyFallback(text);
                    });
                } else {
                    this._copyFallback(text);
                }
            },

            _copyFallback(text) {
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                try {
                    document.execCommand('copy');
                } catch (err) {
                    console.error('Failed to copy:', err);
                }
                document.body.removeChild(textarea);
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
                    this.benchError = window.t('js.error.select_prompt_length');
                    return;
                }

                // Collect selected batch sizes
                const batchSizes = Object.entries(this.benchBatchSizes)
                    .filter(([_, v]) => v)
                    .map(([k, _]) => parseInt(k));

                // Load device info if not loaded yet
                if (!this.benchDeviceInfo) {
                    this.loadBenchDeviceInfo();
                }

                // Reset state
                this.benchRunning = true;
                this.benchProgress = null;
                this.benchSingleResults = [];
                this.benchBatchSameResults = [];
                this.benchBatchDiffResults = [];
                this.benchError = '';
                this.benchBenchId = null;
                this.benchUploadResults = [];
                this.benchUploadDone = null;
                this.benchUploading = false;

                try {
                    const response = await fetch('/admin/api/bench/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: this.benchModelId,
                            prompt_lengths: promptLengths,
                            generation_length: 128,
                            batch_sizes: batchSizes,
                            include_image: this.benchIncludeImage,
                        }),
                    });

                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }

                    if (!response.ok) {
                        const data = await response.json();
                        this.benchError = data.detail || window.t('js.error.start_benchmark_failed');
                        this.benchRunning = false;
                        return;
                    }

                    const data = await response.json();
                    this.benchBenchId = data.bench_id;
                    this.connectBenchSSE(data.bench_id);
                } catch (err) {
                    console.error('Failed to start benchmark:', err);
                    this.benchError = window.t('js.error.start_benchmark_error').replace('{message}', err.message);
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
                            // Benchmark tests done, uploading starts
                            this.benchUploading = true;
                            this.benchProgress = {
                                phase: 'upload',
                                message: 'Uploading to community benchmarks...',
                                current: 0,
                                total: 0,
                            };
                            this.loadModels();
                        } else if (data.type === 'upload') {
                            this.benchUploadResults = [...this.benchUploadResults, data.data];
                        } else if (data.type === 'upload_done') {
                            this.benchUploadDone = data.data;
                            this.benchUploading = false;
                            this.benchRunning = false;
                            this.benchProgress = null;
                            es.close();
                            this.benchEventSource = null;
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
                        this.benchError = window.t('js.error.benchmark_connection_lost');
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

                lines.push('oMLX - LLM inference, optimized for your Mac');
                lines.push('https://github.com/jundot/omlx');
                lines.push(`Benchmark Model: ${this.benchModelId}`);
                lines.push('='.repeat(80));

                // Single Request Results
                if (this.benchSingleResults.length > 0) {
                    lines.push('');
                    lines.push('Single Request Results');
                    lines.push('-'.repeat(80));
                    const hdr = [rpad('Test', 16), pad('TTFT(ms)', 10), pad('TPOT(ms)', 10), pad('pp TPS', 12), pad('tg TPS', 12), pad('E2E(s)', 10), pad('Throughput', 12), pad('Peak Mem', 10)];
                    lines.push(hdr.join('  '));
                    for (const r of this.benchSingleResults) {
                        const row = [
                            rpad(`pp${r.pp}/tg${r.tg}`, 16),
                            pad(r.ttft_ms.toFixed(1), 10),
                            pad(r.tpot_ms.toFixed(2), 10),
                            pad(r.processing_tps.toFixed(1) + ' tok/s', 12),
                            pad(r.gen_tps.toFixed(1) + ' tok/s', 12),
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

                const imgSuffix = this.benchIncludeImage ? ' + image tokens' : '';
                buildBatchText(
                    'Continuous Batching — Same Prompt',
                    `pp1024${imgSuffix} / tg128 · partial prefix cache hit`,
                    this.benchBatchSameResults
                );
                buildBatchText(
                    'Continuous Batching — Different Prompts',
                    `pp1024${imgSuffix} / tg128 · no cache reuse`,
                    this.benchBatchDiffResults
                );

                return lines.join('\n');
            },

            benchCopyText() {
                const text = this.benchBuildText();
                const onSuccess = () => {
                    this.benchCopied = true;
                    setTimeout(() => { this.benchCopied = false; }, 2000);
                };
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(text).then(onSuccess).catch(() => {
                        this._copyFallback(text);
                        onSuccess();
                    });
                } else {
                    this._copyFallback(text);
                    onSuccess();
                }
            },

            async loadBenchDeviceInfo() {
                try {
                    const resp = await fetch('/admin/api/device-info');
                    if (resp.ok) {
                        this.benchDeviceInfo = await resp.json();
                    }
                } catch (err) {
                    console.error('Failed to load device info:', err);
                }
            },

            // Bench sub-tab
            setBenchTab(tab) {
                if (!DASHBOARD_BENCH_TABS.has(tab)) return;
                this.benchTab = tab;
                this.mainTab = 'bench';
                this.syncTabStateToUrl();
                if (tab === 'throughput') {
                    this.loadBenchDeviceInfo();
                }
            },

            // Accuracy benchmark functions

            async loadAccState() {
                // Load accumulated results + queue status from server (page load / tab switch)
                try {
                    const resp = await fetch('/admin/api/bench/accuracy/results');
                    if (resp.ok) {
                        const data = await resp.json();
                        this.accAllResults = (data.results || []).map(r => ({ ...r, _showCategories: false }));
                        this.accRunning = data.running || false;
                        this.accCurrentModel = data.current_model || '';
                        if (data.current_bench_id && data.running) {
                            this.accCurrentBenchId = data.current_bench_id;
                            this.connectAccSSE(data.current_bench_id);
                        }
                    }
                } catch (err) {
                    console.error('Failed to load accuracy state:', err);
                }
                await this.loadAccQueueStatus();
            },

            async loadAccQueueStatus() {
                try {
                    const resp = await fetch('/admin/api/bench/accuracy/queue/status');
                    if (resp.ok) {
                        const data = await resp.json();
                        this.accQueue = data.queue || [];
                        this.accRunning = data.running || false;
                        this.accCurrentModel = data.current_model || '';
                        if (data.current_bench_id) {
                            this.accCurrentBenchId = data.current_bench_id;
                        }
                        // Restore last progress for reconnect
                        if (data.last_progress && data.running) {
                            this.accProgress = data.last_progress;
                        }
                    }
                } catch (err) {
                    console.error('Failed to load queue status:', err);
                }
            },

            async addToAccQueue() {
                if (!this.accModelId) return;
                const selected = Object.entries(this.accBenchmarks)
                    .filter(([_, v]) => v)
                    .map(([k]) => k);
                if (selected.length === 0) return;

                this.accError = '';

                try {
                    const resp = await fetch('/admin/api/bench/accuracy/queue/add', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: this.accModelId,
                            benchmarks: Object.fromEntries(
                                selected.map(k => [k, this.accSampleSizes[k]])
                            ),
                            batch_size: this.accBatchSize,
                        }),
                    });
                    if (!resp.ok) {
                        const err = await resp.json();
                        throw new Error(err.detail || 'Failed to add to queue');
                    }
                    const data = await resp.json();
                    this.accQueue = data.queue || [];
                    this.accRunning = data.running || false;
                    this.accCurrentModel = data.current_model || '';
                    if (data.last_progress) this.accProgress = data.last_progress;

                    // Connect SSE to current run
                    if (data.current_bench_id) {
                        this.accCurrentBenchId = data.current_bench_id;
                        this.connectAccSSE(data.current_bench_id);
                    }
                } catch (err) {
                    this.accError = err.message;
                }
            },

            async removeFromAccQueue(idx) {
                try {
                    await fetch(`/admin/api/bench/accuracy/queue/${idx}`, { method: 'DELETE' });
                    await this.loadAccQueueStatus();
                } catch (err) {
                    console.error('Failed to remove from queue:', err);
                }
            },

            connectAccSSE(benchId) {
                if (this.accEventSource) {
                    this.accEventSource.close();
                }
                this._stopAccPolling();

                const es = new EventSource(`/admin/api/bench/accuracy/${benchId}/stream`);
                this.accEventSource = es;

                es.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        switch (data.type) {
                            case 'progress':
                                this.accProgress = data;
                                this.accCurrentModel = data.model_id || this.accCurrentModel;
                                break;
                            case 'result':
                                data.data._showCategories = false;
                                this.accAllResults.push(data.data);
                                break;
                            case 'done':
                                this.accProgress = null;
                                es.close();
                                this.accEventSource = null;
                                // Check for next in queue
                                this._pollForNextRun();
                                break;
                            case 'error':
                                this.accError = data.message;
                                this.accProgress = null;
                                es.close();
                                this.accEventSource = null;
                                this.loadAccQueueStatus();
                                break;
                        }
                    } catch (err) {
                        console.error('SSE parse error:', err);
                    }
                };

                es.onerror = () => {
                    es.close();
                    this.accEventSource = null;
                    // SSE disconnected — fall back to polling
                    this._startAccPolling();
                };
            },

            _startAccPolling() {
                this._stopAccPolling();
                this._accPollTimer = setInterval(async () => {
                    await this.loadAccQueueStatus();
                    // Load latest results too
                    try {
                        const resp = await fetch('/admin/api/bench/accuracy/results');
                        if (resp.ok) {
                            const data = await resp.json();
                            this.accAllResults = (data.results || []).map(r => ({ ...r, _showCategories: false }));
                        }
                    } catch (e) {}
                    // Try to reconnect SSE if running
                    if (this.accRunning && this.accCurrentBenchId && !this.accEventSource) {
                        this._stopAccPolling();
                        this.connectAccSSE(this.accCurrentBenchId);
                    }
                    if (!this.accRunning) {
                        this._stopAccPolling();
                    }
                }, 3000);
            },

            _stopAccPolling() {
                if (this._accPollTimer) {
                    clearInterval(this._accPollTimer);
                    this._accPollTimer = null;
                }
            },

            _pollForNextRun() {
                // After a run completes, poll briefly for the next run to start
                let attempts = 0;
                const poll = setInterval(async () => {
                    attempts++;
                    await this.loadAccQueueStatus();
                    if (this.accCurrentBenchId && this.accRunning) {
                        clearInterval(poll);
                        this.connectAccSSE(this.accCurrentBenchId);
                    } else if (!this.accRunning || attempts > 10) {
                        clearInterval(poll);
                    }
                }, 1000);
            },

            async cancelAccuracyBenchmark() {
                try {
                    await fetch('/admin/api/bench/accuracy/cancel', { method: 'POST' });
                } catch (err) {
                    console.error('Cancel error:', err);
                }
                this.accRunning = false;
                this.accProgress = null;
                this.accQueue = [];
                this.accCurrentModel = '';
                if (this.accEventSource) {
                    this.accEventSource.close();
                    this.accEventSource = null;
                }
            },

            async resetAccResults() {
                try {
                    await fetch('/admin/api/bench/accuracy/results/reset', { method: 'POST' });
                    this.accAllResults = [];
                } catch (err) {
                    console.error('Reset error:', err);
                }
            },

            accBuildText() {
                if (this.accAllResults.length === 0) return '';
                const pad = (s, w) => s.toString().padStart(w);
                const rpad = (s, w) => s.toString().padEnd(w);

                // Group by model
                const models = [...new Set(this.accAllResults.map(r => r.model_id))];
                const benchmarks = [...new Set(this.accAllResults.map(r => r.benchmark))];

                // Build lookup: model -> benchmark -> accuracy
                const lookup = {};
                for (const r of this.accAllResults) {
                    if (!lookup[r.model_id]) lookup[r.model_id] = {};
                    lookup[r.model_id][r.benchmark] = r;
                }

                // Determine column widths
                const modelWidth = Math.max(12, ...models.map(m => m.length + 2));
                const benchWidth = Math.max(14, ...benchmarks.map(b => b.length + 2));

                let lines = [];
                lines.push('Intelligence Benchmark Comparison');
                lines.push('');

                // Header row
                let header = rpad('', benchWidth);
                for (const m of models) header += pad(m, modelWidth);
                lines.push(header);
                lines.push('-'.repeat(benchWidth + models.length * modelWidth));

                // Data rows
                for (const b of benchmarks) {
                    let row = rpad(b.toUpperCase(), benchWidth);
                    for (const m of models) {
                        const r = lookup[m]?.[b];
                        row += pad(r ? (r.accuracy * 100).toFixed(1) + '%' : '-', modelWidth);
                    }
                    lines.push(row);
                }

                // Detail section per model
                lines.push('');
                lines.push('--- Detail ---');
                for (const m of models) {
                    lines.push('');
                    lines.push('Model: ' + m);
                    lines.push(rpad('Benchmark', 16) + pad('Accuracy', 10) + pad('Correct', 10) + pad('Total', 8) + pad('Time(s)', 10));
                    lines.push('-'.repeat(54));
                    for (const r of this.accAllResults.filter(r => r.model_id === m)) {
                        lines.push(
                            rpad(r.benchmark.toUpperCase(), 16) +
                            pad((r.accuracy * 100).toFixed(1) + '%', 10) +
                            pad(r.correct, 10) +
                            pad(r.total, 8) +
                            pad(r.time_s, 10)
                        );
                    }
                }

                return lines.join('\n');
            },

            accCopyText() {
                const text = this.accBuildText();
                const onSuccess = () => {
                    this.accCopied = true;
                    setTimeout(() => { this.accCopied = false; }, 2000);
                };
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(onSuccess).catch(() => {
                        const ta = document.getElementById('accTextarea');
                        if (ta) { ta.select(); document.execCommand('copy'); onSuccess(); }
                    });
                } else {
                    const ta = document.getElementById('accTextarea');
                    if (ta) { ta.select(); document.execCommand('copy'); onSuccess(); }
                }
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
                        this.logError = data.detail || window.t('js.error.load_logs_failed');
                    }
                } catch (err) {
                    console.error('Failed to load logs:', err);
                    this.logError = window.t('js.error.load_logs_failed');
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

            // Parse process memory setting to slider state
            parseProcessMemoryToState(value, totalBytes) {
                if (!value || value === 'disabled') {
                    return { auto: false, percent: 0 };
                }
                if (value === 'auto') {
                    // auto = (total - 8GB) / total * 100
                    if (!totalBytes || totalBytes === 0) return { auto: true, percent: 90 };
                    const reserved = 8 * 1024 * 1024 * 1024;
                    const autoBytes = Math.max(totalBytes - reserved, totalBytes * 0.1);
                    const percent = Math.round((autoBytes / totalBytes) * 100);
                    return { auto: true, percent: Math.min(99, Math.max(10, percent)) };
                }
                // Handle GB/TB/MB format (e.g., "69GB")
                const sizeMatch = value.match(/^(\d+(?:\.\d+)?)\s*(GB|MB|TB)$/i);
                if (sizeMatch && totalBytes > 0) {
                    let bytes = parseFloat(sizeMatch[1]);
                    const unit = sizeMatch[2].toUpperCase();
                    if (unit === 'TB') bytes *= 1024 * 1024 * 1024 * 1024;
                    else if (unit === 'GB') bytes *= 1024 * 1024 * 1024;
                    else if (unit === 'MB') bytes *= 1024 * 1024;
                    const percent = Math.round((bytes / totalBytes) * 100);
                    return { auto: false, percent: Math.min(99, Math.max(1, percent)) };
                }
                // Handle percent format (e.g., "69%")
                const percent = parseInt(value.replace('%', ''));
                if (isNaN(percent)) return { auto: false, percent: 90 };
                return { auto: false, percent: Math.min(99, Math.max(0, percent)) };
            },

            // Update process memory setting from slider
            updateProcessMemoryFromSlider() {
                if (this.processMemoryPercent === 0) {
                    this.globalSettings.memory.max_process_memory = 'disabled';
                } else {
                    this.globalSettings.memory.max_process_memory =
                        this.processMemoryPercent + '%';
                }
            },

            // Get formatted process memory for display
            getProcessMemoryDisplay() {
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                if (!totalBytes) return '-';
                if (this.processMemoryPercent === 0 && !this.processMemoryAuto) return '-';
                if (this.processMemoryAuto) {
                    const reserved = 8 * 1024 * 1024 * 1024;
                    const autoBytes = Math.max(totalBytes - reserved, totalBytes * 0.1);
                    const gb = Math.round(autoBytes / (1024 * 1024 * 1024));
                    return `${gb}GB`;
                }
                const bytes = Math.floor((this.processMemoryPercent / 100) * totalBytes);
                const gb = Math.round(bytes / (1024 * 1024 * 1024));
                return `${gb}GB`;
            },

            // Parse stored memory value (e.g., "102GB") to percent of usable memory
            parseMemoryToPercent(memoryStr, totalBytes) {
                if (memoryStr === 'disabled') return 0;
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
                if (this.memoryPercent === 0 && !this.modelMemoryAuto) return '-';
                const usableBytes = Math.max(0, totalBytes - 8 * 1024 * 1024 * 1024);
                const bytes = Math.floor((this.memoryPercent / 100) * usableBytes);
                const gb = Math.round(bytes / (1024 * 1024 * 1024));
                return `${gb}GB`;
            },

            // Update memory value when slider changes
            updateMemoryFromSlider() {
                if (this.modelMemoryAuto) {
                    this.globalSettings.model.max_model_memory = 'auto';
                } else if (this.memoryPercent === 0) {
                    this.globalSettings.model.max_model_memory = 'disabled';
                } else {
                    const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                    this.globalSettings.model.max_model_memory = this.percentToMemoryString(this.memoryPercent, totalBytes);
                }
            },

            // Parse cache size string (e.g., "10GB") to percent of SSD total capacity
            parseCacheToPercent(cacheStr, totalBytes) {
                if (!cacheStr || cacheStr === 'auto' || !totalBytes || totalBytes === 0) {
                    return 10; // Default 10%
                }

                const match = cacheStr.match(/^(\d+(?:\.\d+)?)\s*(GB|MB|TB)?$/i);
                if (!match) return 10;

                let bytes = parseFloat(match[1]);
                const unit = (match[2] || 'GB').toUpperCase();

                if (unit === 'TB') bytes *= 1024 * 1024 * 1024 * 1024;
                else if (unit === 'GB') bytes *= 1024 * 1024 * 1024;
                else if (unit === 'MB') bytes *= 1024 * 1024;

                const percent = Math.round((bytes / totalBytes) * 100);
                return Math.min(100, percent);
            },

            // Convert percent to cache size string
            percentToCacheString(percent, totalBytes) {
                if (!totalBytes || totalBytes === 0) return 'auto';
                const bytes = Math.floor((percent / 100) * totalBytes);
                const gb = Math.floor(bytes / (1024 * 1024 * 1024));
                return `${gb}GB`;
            },

            // Helper: parse GB from a settings string like "68GB", "1TB", "512MB"
            _parseSettingsGB(val) {
                if (!val) return null;
                const match = val.match(/^(\d+(?:\.\d+)?)\s*(GB|MB|TB)?$/i);
                if (!match) return null;
                let num = parseFloat(match[1]);
                const unit = (match[2] || 'GB').toUpperCase();
                if (unit === 'TB') return Math.round(num * 1024);
                if (unit === 'MB') return Math.round(num / 1024);
                return Math.round(num);
            },

            // Computed process memory size in GB (for manual input)
            // Reads from settings value directly to avoid percent round-trip precision loss
            get processMemorySizeGB() {
                const val = this.globalSettings.memory?.max_process_memory;
                // If stored as GB/TB/MB, parse directly
                const parsed = this._parseSettingsGB(val);
                if (parsed !== null) return parsed;
                // Otherwise derive from percent (for "XX%" format or auto)
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                if (!totalBytes) return 0;
                if (this.processMemoryAuto) {
                    const reserved = 8 * 1024 * 1024 * 1024;
                    const autoBytes = Math.max(totalBytes - reserved, totalBytes * 0.1);
                    return Math.round(autoBytes / (1024 * 1024 * 1024));
                }
                if (this.processMemoryPercent === 0) return 0;
                const bytes = Math.floor((this.processMemoryPercent / 100) * totalBytes);
                return Math.round(bytes / (1024 * 1024 * 1024));
            },

            // Update process memory from manual GB input
            updateProcessMemoryFromInput(gbValue) {
                const gb = parseInt(gbValue) || 0;
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                if (gb === 0) {
                    this.processMemoryPercent = 0;
                    this.globalSettings.memory.max_process_memory = 'disabled';
                } else {
                    // Store as GB string (backend supports parse_size fallback)
                    this.globalSettings.memory.max_process_memory = `${gb}GB`;
                    // Sync slider position
                    if (totalBytes > 0) {
                        const bytes = gb * 1024 * 1024 * 1024;
                        this.processMemoryPercent = Math.min(99, Math.max(1, Math.round((bytes / totalBytes) * 100)));
                    }
                }
            },

            // Computed model memory size in GB (for manual input)
            get modelMemorySizeGB() {
                const val = this.globalSettings.model?.max_model_memory;
                if (val === 'disabled') return 0;
                const parsed = this._parseSettingsGB(val);
                if (parsed !== null) return parsed;
                // Fallback: derive from percent
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                if (!totalBytes) return 0;
                const usableBytes = Math.max(0, totalBytes - 8 * 1024 * 1024 * 1024);
                const bytes = Math.floor((this.memoryPercent / 100) * usableBytes);
                return Math.round(bytes / (1024 * 1024 * 1024));
            },

            // Update model memory from manual GB input
            updateModelMemoryFromInput(gbValue) {
                const gb = parseInt(gbValue) || 0;
                if (gb === 0) {
                    this.memoryPercent = 0;
                    this.globalSettings.model.max_model_memory = 'disabled';
                    return;
                }
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                const usableBytes = Math.max(0, totalBytes - 8 * 1024 * 1024 * 1024);
                if (usableBytes > 0) {
                    const bytes = gb * 1024 * 1024 * 1024;
                    this.memoryPercent = Math.min(100, Math.max(0, Math.round((bytes / usableBytes) * 100)));
                }
                this.globalSettings.model.max_model_memory = `${gb}GB`;
            },

            // Computed hot cache size in GB (for manual input)
            get hotCacheSizeGB() {
                const val = this.globalSettings.cache?.hot_cache_max_size;
                if (val && val !== '0') {
                    const parsed = this._parseSettingsGB(val);
                    if (parsed !== null) return parsed;
                }
                if (this.hotCachePercent === 0) return 0;
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                const bytes = Math.floor((this.hotCachePercent / 100) * totalBytes);
                return Math.floor(bytes / (1024 * 1024 * 1024));
            },

            // Update hot cache from manual GB input
            updateHotCacheFromInput(gbValue) {
                const gb = parseInt(gbValue) || 0;
                if (gb === 0) {
                    this.hotCachePercent = 0;
                    this.globalSettings.cache.hot_cache_max_size = '0';
                } else {
                    const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                    if (totalBytes > 0) {
                        const bytes = gb * 1024 * 1024 * 1024;
                        this.hotCachePercent = Math.min(50, Math.max(1, Math.round((bytes / totalBytes) * 100)));
                    }
                    this.globalSettings.cache.hot_cache_max_size = `${gb}GB`;
                }
            },

            // Computed cache size in GB (for manual input)
            get cacheSizeGB() {
                const val = this.globalSettings.cache?.ssd_cache_max_size;
                if (val && val !== 'auto') {
                    const parsed = this._parseSettingsGB(val);
                    if (parsed !== null) return parsed;
                }
                const totalBytes = this.globalSettings.system?.ssd_total_bytes || 0;
                if (!totalBytes) return 0;
                const bytes = Math.floor((this.cachePercent / 100) * totalBytes);
                return Math.round(bytes / (1024 * 1024 * 1024));
            },

            // Update cache from slider
            updateCacheFromSlider() {
                const totalBytes = this.globalSettings.system?.ssd_total_bytes || 0;
                this.globalSettings.cache.ssd_cache_max_size = this.percentToCacheString(this.cachePercent, totalBytes);
            },

            // Update cache from manual GB input
            updateCacheFromInput(gbValue) {
                const gb = parseInt(gbValue) || 0;
                this.globalSettings.cache.ssd_cache_max_size = `${gb}GB`;

                // Update percent slider
                const totalBytes = this.globalSettings.system?.ssd_total_bytes || 0;
                if (totalBytes > 0) {
                    const bytes = gb * 1024 * 1024 * 1024;
                    this.cachePercent = Math.min(100, Math.round((bytes / totalBytes) * 100));
                }
            },

            // Parse hot cache size string to percent of total memory
            parseHotCacheToPercent(hotCacheStr, totalBytes) {
                if (!hotCacheStr || hotCacheStr === '0' || !totalBytes || totalBytes === 0) {
                    return 0;
                }
                const match = hotCacheStr.match(/^(\d+(?:\.\d+)?)\s*(GB|MB|TB)?$/i);
                if (!match) return 0;

                let bytes = parseFloat(match[1]);
                const unit = (match[2] || 'GB').toUpperCase();
                if (unit === 'TB') bytes *= 1024 * 1024 * 1024 * 1024;
                else if (unit === 'GB') bytes *= 1024 * 1024 * 1024;
                else if (unit === 'MB') bytes *= 1024 * 1024;

                const percent = Math.round((bytes / totalBytes) * 100);
                return Math.min(50, Math.max(0, percent));
            },

            // Update hot cache setting from slider
            updateHotCacheFromSlider() {
                if (this.hotCachePercent === 0) {
                    this.globalSettings.cache.hot_cache_max_size = '0';
                } else {
                    const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                    const bytes = Math.floor((this.hotCachePercent / 100) * totalBytes);
                    const gb = Math.floor(bytes / (1024 * 1024 * 1024));
                    this.globalSettings.cache.hot_cache_max_size = gb > 0 ? `${gb}GB` : '0';
                }
            },

            // Get formatted hot cache size for display
            getHotCacheDisplay() {
                if (this.hotCachePercent === 0) return '0GB';
                const totalBytes = this.globalSettings.system?.total_memory_bytes || 0;
                const bytes = Math.floor((this.hotCachePercent / 100) * totalBytes);
                const gb = Math.floor(bytes / (1024 * 1024 * 1024));
                return `${gb}GB`;
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
            // HuggingFace Mirror Settings
            // =================================================================

            openHfMirrorModal() {
                this.hfMirrorEndpoint = this.globalSettings.huggingface.endpoint || '';
                this.showHfMirrorModal = true;
            },

            async saveHfMirrorEndpoint() {
                this.hfMirrorSaving = true;
                try {
                    const response = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ hf_endpoint: this.hfMirrorEndpoint }),
                    });
                    if (response.ok) {
                        this.globalSettings.huggingface.endpoint = this.hfMirrorEndpoint;
                        this.showHfMirrorModal = false;
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        alert(Array.isArray(data.detail) ? data.detail.join(', ') : (data.detail || 'Failed to save'));
                    }
                } catch (err) {
                    console.error('Failed to save HF mirror endpoint:', err);
                } finally {
                    this.hfMirrorSaving = false;
                }
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
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 15000);

                try {
                    const response = await fetch('/admin/api/hf/download', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            repo_id: repoId,
                            hf_token: this.hfToken,
                        }),
                        signal: controller.signal,
                    });

                    if (response.ok) {
                        this.hfSuccess = window.t('js.success.download_started').replace('{repo_id}', repoId);
                        this.hfRepoId = '';
                        await this.loadHFTasks();
                        this.startHFRefresh();
                        setTimeout(() => { this.hfSuccess = ''; }, 5000);
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.hfError = data.detail || window.t('js.error.start_download_failed');
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.hfError = 'HuggingFace request timed out. The service may be unavailable.';
                    } else {
                        this.hfError = window.t('js.error.start_download_connection');
                    }
                    console.error('Failed to start download:', err);
                } finally {
                    clearTimeout(timeoutId);
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

            async retryHFDownload(taskId) {
                try {
                    const response = await fetch(`/admin/api/hf/retry/${taskId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ hf_token: this.hfToken || '' }),
                    });
                    if (response.ok) {
                        await this.loadHFTasks();
                        this.startHFRefresh();
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.hfError = data.detail || 'Retry failed';
                        setTimeout(() => { this.hfError = ''; }, 5000);
                    }
                } catch (err) {
                    console.error('Failed to retry download:', err);
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
                        this.hfError = data.detail || window.t('js.error.delete_model_failed');
                        setTimeout(() => { this.hfError = ''; }, 5000);
                    }
                } catch (err) {
                    console.error('Failed to delete model:', err);
                    this.hfError = window.t('js.error.delete_model_connection');
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
            // oQ Quantization Functions
            // =================================================================

            async loadOQModels() {
                try {
                    const response = await fetch('/admin/api/oq/models');
                    if (response.ok) {
                        const data = await response.json();
                        this.oqModels = data.models || [];
                        this.oqModelsLoaded = true;
                    }
                } catch (err) {
                    console.error('Failed to load quantizable models:', err);
                }
            },

            async startOQQuantization() {
                if (!this.oqSelectedModelPath || this.oqStarting) return;
                this.oqError = '';
                this.oqSuccess = '';
                this.oqStarting = true;
                try {
                    const response = await fetch('/admin/api/oq/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_path: this.oqSelectedModelPath,
                            oq_level: this.oqLevel,
                            enable_clip: this.oqEnableClip,
                            group_size: this.oqGroupSize,
                            clip_num_samples: this.oqClipSamples,
                            clip_seq_length: this.oqClipSeqLen,
                            calib_dataset: this.oqCalibDataset,
                            clip_batch_size: this.oqClipBatchSize,
                        }),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (response.ok) {
                        const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                        const name = model ? model.name : this.oqSelectedModelPath;
                        this.oqSuccess = `Quantization started: ${name} → oQ${this.oqLevel}`;
                        await this.loadOQTasks();
                        this.startOQRefresh();
                        setTimeout(() => { this.oqSuccess = ''; }, 5000);
                    } else {
                        this.oqError = data.detail || 'Failed to start quantization';
                    }
                } catch (err) {
                    this.oqError = 'Connection error. Server may be unavailable.';
                } finally {
                    this.oqStarting = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            async loadOQTasks() {
                try {
                    const response = await fetch('/admin/api/oq/tasks');
                    if (response.ok) {
                        const data = await response.json();
                        this.oqTasks = data.tasks || [];
                        const hasActive = this.oqTasks.some(t =>
                            ['pending', 'loading', 'quantizing', 'saving'].includes(t.status));
                        if (!hasActive) {
                            this.stopOQRefresh();
                            if (this.oqTasks.some(t => t.status === 'completed')) {
                                await this.loadHFModels();
                                await this.loadModels();
                                await this.loadOQModels();
                            }
                        }
                        this.$nextTick(() => lucide.createIcons());
                    }
                } catch (err) {
                    console.error('Failed to load oQ tasks:', err);
                }
            },

            async cancelOQTask(taskId) {
                try {
                    await fetch(`/admin/api/oq/cancel/${taskId}`, { method: 'POST' });
                    await this.loadOQTasks();
                } catch (err) {
                    console.error('Failed to cancel oQ task:', err);
                }
            },

            async removeOQTask(taskId) {
                try {
                    await fetch(`/admin/api/oq/task/${taskId}`, { method: 'DELETE' });
                    await this.loadOQTasks();
                } catch (err) {
                    console.error('Failed to remove oQ task:', err);
                }
            },

            startOQRefresh() {
                this.stopOQRefresh();
                this._oqRefreshTimer = setInterval(() => {
                    this.loadOQTasks();
                }, 2000);
            },

            stopOQRefresh() {
                if (this._oqRefreshTimer) {
                    clearInterval(this._oqRefreshTimer);
                    this._oqRefreshTimer = null;
                }
            },

            formatOQProgress(task) {
                const pct = Math.round(task.progress || 0);
                return `${pct}% · ${task.phase || task.status}`;
            },

            formatOQElapsed(task) {
                if (!task.started_at) return '';
                const now = task.completed_at || (Date.now() / 1000);
                const elapsed = now - task.started_at;
                const mins = Math.floor(elapsed / 60);
                const secs = Math.floor(elapsed % 60);
                return `${mins}:${String(secs).padStart(2, '0')}`;
            },

            oqSelectedModelSupportsClip() {
                const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                return model?.supports_clip || false;
            },

            oqEstimatedMemory() {
                // Use precise estimate from API if available
                if (this.oqEstimate) {
                    if (this.oqEnableClip) {
                        return this.oqEstimate.memory_clip_formatted || '';
                    }
                    return this.oqEstimate.memory_streaming_formatted || '';
                }
                // Fallback to rough model-level estimate
                const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                if (!model) return '';
                if (this.oqEnableClip) {
                    return model.memory_clip?.peak_formatted || '';
                }
                return model.memory_streaming?.peak_formatted || '';
            },

            oqEstimate: null,
            _oqEstimateTimer: null,

            oqEstimatedBpw() {
                return this.oqEstimate?.effective_bpw?.toFixed(1) || '';
            },

            oqEstimatedOutputSize() {
                return this.oqEstimate?.output_size_formatted || '';
            },

            oqRefreshEstimate() {
                // Debounce: wait 300ms after last change
                if (this._oqEstimateTimer) clearTimeout(this._oqEstimateTimer);
                if (!this.oqSelectedModelPath) {
                    this.oqEstimate = null;
                    return;
                }
                this._oqEstimateTimer = setTimeout(async () => {
                    try {
                        const params = new URLSearchParams({
                            model_path: this.oqSelectedModelPath,
                            oq_level: this.oqLevel,
                        });
                        const resp = await fetch(`/admin/api/oq/estimate?${params}`);
                        if (resp.ok) {
                            this.oqEstimate = await resp.json();
                        }
                    } catch (e) {
                        console.error('Failed to estimate oQ:', e);
                    }
                }, 300);
            },

            // =================================================================
            // Recommended Models Functions
            // =================================================================

            async loadRecommendedModels() {
                this.hfRecommendedLoading = true;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 15000);
                try {
                    const response = await fetch(`/admin/api/hf/recommended?mlx_only=${this.hfMlxOnly}`, { signal: controller.signal });
                    if (response.ok) {
                        this.hfRecommended = await response.json();
                        this.hfRecommendedLoaded = true;
                        this.hfPage.trending = 1;
                        this.hfPage.popular = 1;
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.hfError = data.detail || 'Failed to load recommended models';
                        setTimeout(() => { this.hfError = ''; }, 5000);
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.hfError = 'HuggingFace request timed out. The service may be unavailable.';
                    } else {
                        this.hfError = 'Failed to connect to HuggingFace.';
                    }
                    setTimeout(() => { this.hfError = ''; }, 5000);
                    console.error('Failed to load recommended models:', err);
                } finally {
                    clearTimeout(timeoutId);
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

            // Pagination helpers
            getPagedModels(tab) {
                const page = this.hfPage[tab] || 1;
                const size = this.hfPageSize;
                let list;
                if (tab === 'trending') list = this.hfRecommended.trending || [];
                else if (tab === 'popular') list = this.hfRecommended.popular || [];
                else list = this.hfSearchResults || [];
                return list.slice((page - 1) * size, page * size);
            },

            getTotalPages(tab) {
                let total;
                if (tab === 'trending') total = (this.hfRecommended.trending || []).length;
                else if (tab === 'popular') total = (this.hfRecommended.popular || []).length;
                else total = (this.hfSearchResults || []).length;
                const maxPages = tab === 'search' ? 10 : 5;
                return Math.min(Math.ceil(total / this.hfPageSize), maxPages);
            },

            setPage(tab, page) {
                this.hfPage[tab] = page;
                this.$nextTick(() => lucide.createIcons());
            },

            // Search
            async searchHFModels() {
                if (!this.hfSearchQuery.trim()) return;
                this.hfSearchLoading = true;
                this.hfRecommendedTab = 'search';
                this.hfPage.search = 1;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 15000);
                try {
                    const params = new URLSearchParams({
                        q: this.hfSearchQuery,
                        sort: this.hfSearchSort,
                        limit: '100',
                        mlx_only: this.hfMlxOnly,
                    });
                    const response = await fetch(`/admin/api/hf/search?${params}`, { signal: controller.signal });
                    if (response.ok) {
                        const data = await response.json();
                        this.hfSearchResults = data.models || [];
                        this.hfSearchLoaded = true;
                        // Save to search history
                        this.addSearchHistory(this.hfSearchQuery.trim());
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.hfError = data.detail || 'Search failed';
                        setTimeout(() => { this.hfError = ''; }, 5000);
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.hfError = 'HuggingFace request timed out. The service may be unavailable.';
                    } else {
                        this.hfError = 'Failed to connect to HuggingFace.';
                    }
                    setTimeout(() => { this.hfError = ''; }, 5000);
                    console.error('Search failed:', err);
                } finally {
                    clearTimeout(timeoutId);
                    this.hfSearchLoading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            debounceSearch() {
                clearTimeout(this.hfSearchDebounceTimer);
                if (!this.hfSearchQuery.trim()) return;
                this.hfSearchDebounceTimer = setTimeout(() => this.searchHFModels(), 500);
            },

            immediateSearch() {
                clearTimeout(this.hfSearchDebounceTimer);
                this.searchHFModels();
            },

            formatParamCount(params) {
                if (!params) return null;
                if (params >= 1e12) return (params / 1e12).toFixed(1) + 'T';
                if (params >= 1e9) return (params / 1e9).toFixed(1) + 'B';
                if (params >= 1e6) return (params / 1e6).toFixed(1) + 'M';
                return params.toString();
            },

            // Search history
            addSearchHistory(query) {
                let history = this.hfSearchHistory.filter(h => h !== query);
                history.unshift(query);
                history = history.slice(0, 5);
                this.hfSearchHistory = history;
                localStorage.setItem('hfSearchHistory', JSON.stringify(history));
            },

            selectSearchHistory(query) {
                this.hfSearchQuery = query;
                this.hfSearchHistoryOpen = false;
                this.searchHFModels();
            },

            closeSearchHistory() {
                setTimeout(() => { this.hfSearchHistoryOpen = false; }, 150);
            },

            // Model detail modal
            async openModelDetail(repoId) {
                this.hfModelDetailLoading = true;
                this.hfModelDetail = null;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 15000);
                try {
                    const params = new URLSearchParams({ repo_id: repoId });
                    const response = await fetch(`/admin/api/hf/model-info?${params}`, { signal: controller.signal });
                    if (response.ok) {
                        this.hfModelDetail = await response.json();
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.hfError = data.detail || 'Failed to fetch model info';
                        setTimeout(() => { this.hfError = ''; }, 5000);
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.hfError = 'HuggingFace request timed out. The service may be unavailable.';
                    } else {
                        this.hfError = 'Failed to connect to HuggingFace.';
                    }
                    setTimeout(() => { this.hfError = ''; }, 5000);
                    console.error('Failed to fetch model info:', err);
                } finally {
                    clearTimeout(timeoutId);
                    this.hfModelDetailLoading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            closeModelDetail() {
                this.hfModelDetail = null;
                this.hfModelDetailLoading = false;
                this.msModelDetail = null;
                this.msModelDetailLoading = false;
            },

            formatFileSize(bytes) {
                if (!bytes) return '';
                if (bytes < 1024) return bytes + ' B';
                if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
                if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
                return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
            },

            // =================================================================
            // ModelScope Downloader Functions
            // =================================================================

            async initMsDownloader() {
                if (this.msInitialized) return;
                this.msInitialized = true;
                try {
                    const response = await fetch('/admin/api/ms/status');
                    if (response.ok) {
                        const data = await response.json();
                        this.msAvailable = data.available === true;
                    } else {
                        this.msAvailable = false;
                    }
                } catch (err) {
                    this.msAvailable = false;
                    console.error('Failed to check MS status:', err);
                }
                if (this.msAvailable) {
                    await this.loadMSTasks();
                }
                this.$nextTick(() => lucide.createIcons());
            },

            async startMSDownload() {
                let repoId = this.msRepoId.trim();
                if (!repoId) return;

                // Default owner to mlx-community if not specified
                if (!repoId.includes('/')) {
                    repoId = 'mlx-community/' + repoId;
                }

                this.msError = '';
                this.msSuccess = '';
                this.msDownloading = true;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 20000);

                try {
                    const response = await fetch('/admin/api/ms/download', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: repoId,
                            ms_token: this.msToken || '',
                        }),
                        signal: controller.signal,
                    });

                    if (response.ok) {
                        this.msSuccess = window.t('js.success.download_started').replace('{repo_id}', repoId);
                        this.msRepoId = '';
                        await this.loadMSTasks();
                        this.startMSRefresh();
                        setTimeout(() => { this.msSuccess = ''; }, 5000);
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.msError = data.detail || window.t('js.error.start_download_failed');
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.msError = 'ModelScope request timed out. The service may be unavailable.';
                    } else {
                        this.msError = window.t('js.error.start_download_connection');
                    }
                    console.error('Failed to start MS download:', err);
                } finally {
                    clearTimeout(timeoutId);
                    this.msDownloading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            async loadMSTasks() {
                try {
                    const response = await fetch('/admin/api/ms/tasks');
                    if (response.ok) {
                        const data = await response.json();
                        this.msTasks = data.tasks || [];

                        const hasActive = this.msTasks.some(t =>
                            t.status === 'pending' || t.status === 'downloading');
                        if (!hasActive) {
                            this.stopMSRefresh();
                            if (this.msTasks.some(t => t.status === 'completed')) {
                                await this.loadHFModels();
                                await this.loadModels();
                            }
                        }

                        this.$nextTick(() => lucide.createIcons());
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load MS tasks:', err);
                }
            },

            async cancelMSDownload(taskId) {
                try {
                    const response = await fetch(`/admin/api/ms/cancel/${taskId}`, {
                        method: 'POST',
                    });
                    if (response.ok) {
                        await this.loadMSTasks();
                    }
                } catch (err) {
                    console.error('Failed to cancel MS download:', err);
                }
            },

            async retryMSDownload(taskId) {
                try {
                    const response = await fetch(`/admin/api/ms/retry/${taskId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ms_token: this.msToken || null }),
                    });
                    if (response.ok) {
                        await this.loadMSTasks();
                        this.startMSRefresh();
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.msError = data.detail || 'Retry failed';
                        setTimeout(() => { this.msError = ''; }, 5000);
                    }
                } catch (err) {
                    console.error('Failed to retry MS download:', err);
                }
            },

            async removeMSTask(taskId) {
                try {
                    const response = await fetch(`/admin/api/ms/task/${taskId}`, {
                        method: 'DELETE',
                    });
                    if (response.ok) {
                        await this.loadMSTasks();
                    }
                } catch (err) {
                    console.error('Failed to remove MS task:', err);
                }
            },

            startMSRefresh() {
                this.stopMSRefresh();
                this._msRefreshTimer = setInterval(() => {
                    this.loadMSTasks();
                }, 2000);
            },

            stopMSRefresh() {
                if (this._msRefreshTimer) {
                    clearInterval(this._msRefreshTimer);
                    this._msRefreshTimer = null;
                }
            },

            downloadMsModel(repoId) {
                this.msRepoId = repoId;
                this.startMSDownload();
            },

            // MS Recommended models
            async loadMsRecommendedModels() {
                this.msRecommendedLoading = true;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 20000);
                try {
                    const response = await fetch(`/admin/api/ms/recommended?mlx_only=${this.msMlxOnly}`, { signal: controller.signal });
                    if (response.ok) {
                        const data = await response.json();
                        this.msRecommended = data;
                        this.msRecommendedLoaded = true;
                        this.msPage.trending = 1;
                        this.msPage.popular = 1;
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.msError = data.detail || 'Failed to load recommended models';
                        setTimeout(() => { this.msError = ''; }, 5000);
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.msError = 'ModelScope request timed out. The service may be unavailable.';
                    } else {
                        this.msError = 'Failed to connect to ModelScope.';
                    }
                    setTimeout(() => { this.msError = ''; }, 5000);
                    console.error('Failed to load MS recommended models:', err);
                } finally {
                    clearTimeout(timeoutId);
                    this.msRecommendedLoading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            // MS Pagination helpers
            getMsPagedModels(tab) {
                const page = this.msPage[tab] || 1;
                const size = this.msPageSize;
                let list;
                if (tab === 'trending') list = (this.msRecommended.trending || []);
                else if (tab === 'popular') list = (this.msRecommended.popular || []);
                else list = this.msSearchResults || [];
                return list.slice((page - 1) * size, page * size);
            },

            getMsTotalPages(tab) {
                let total;
                if (tab === 'trending') total = (this.msRecommended.trending || []).length;
                else if (tab === 'popular') total = (this.msRecommended.popular || []).length;
                else total = (this.msSearchResults || []).length;
                const maxPages = tab === 'search' ? 10 : 5;
                return Math.min(Math.ceil(total / this.msPageSize), maxPages);
            },

            setMsPage(tab, page) {
                this.msPage[tab] = page;
                this.$nextTick(() => lucide.createIcons());
            },

            // MS Search
            async searchMSModels() {
                if (!this.msSearchQuery.trim()) return;
                this.msSearchLoading = true;
                this.msRecommendedTab = 'search';
                this.msPage.search = 1;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 20000);
                try {
                    const params = new URLSearchParams({
                        q: this.msSearchQuery,
                        sort: this.msSearchSort,
                        limit: '50',
                        mlx_only: this.msMlxOnly,
                    });
                    const response = await fetch(`/admin/api/ms/search?${params}`, { signal: controller.signal });
                    if (response.ok) {
                        const data = await response.json();
                        this.msSearchResults = data.models || [];
                        this.msSearchLoaded = true;
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.msError = data.detail || 'Search failed';
                        setTimeout(() => { this.msError = ''; }, 5000);
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.msError = 'ModelScope request timed out. The service may be unavailable.';
                    } else {
                        this.msError = 'Failed to connect to ModelScope.';
                    }
                    setTimeout(() => { this.msError = ''; }, 5000);
                    console.error('MS search failed:', err);
                } finally {
                    clearTimeout(timeoutId);
                    this.msSearchLoading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },

            mImmediateSearch() {
                const query = this.msSearchQuery.trim();
                if (query) {
                    // Save to search history
                    this.msSearchHistory = [query, ...this.msSearchHistory.filter(h => h !== query)].slice(0, 10);
                    localStorage.setItem('msSearchHistory', JSON.stringify(this.msSearchHistory));
                }
                this.msSearchHistoryOpen = false;
                this.searchMSModels();
            },

            msDebounceSearch() {
                clearTimeout(this.msSearchDebounceTimer);
                this.msSearchDebounceTimer = setTimeout(() => {
                    if (this.msSearchQuery.trim()) {
                        this.searchMSModels();
                    }
                }, 500);
            },

            closeMsSearchHistory() {
                setTimeout(() => { this.msSearchHistoryOpen = false; }, 200);
            },

            selectMsSearchHistory(item) {
                this.msSearchQuery = item;
                this.msSearchHistoryOpen = false;
                this.mImmediateSearch();
            },

            // MS Model detail modal
            async openMsModelDetail(repoId) {
                this.msModelDetailLoading = true;
                this.msModelDetail = null;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 20000);
                try {
                    const params = new URLSearchParams({ model_id: repoId });
                    const response = await fetch(`/admin/api/ms/model-info?${params}`, { signal: controller.signal });
                    if (response.ok) {
                        this.msModelDetail = await response.json();
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.msError = data.detail || 'Failed to fetch model info';
                        setTimeout(() => { this.msError = ''; }, 5000);
                    }
                } catch (err) {
                    if (err.name === 'AbortError') {
                        this.msError = 'ModelScope request timed out. The service may be unavailable.';
                    } else {
                        this.msError = 'Failed to connect to ModelScope.';
                    }
                    setTimeout(() => { this.msError = ''; }, 5000);
                    console.error('Failed to fetch MS model info:', err);
                } finally {
                    clearTimeout(timeoutId);
                    this.msModelDetailLoading = false;
                    this.$nextTick(() => lucide.createIcons());
                }
            },
        }
    }
