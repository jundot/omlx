    // OCR model types that require temperature=0.0 (deterministic output)
    const OCR_CONFIG_MODEL_TYPES = new Set([
        'deepseekocr', 'deepseekocr_2', 'dots_ocr', 'glm_ocr',
    ]);
    const DSA_MODEL_TYPES = new Set([
        'deepseek_v32', 'glm_moe_dsa',
    ]);
    const QWEN35_ANE_CONFIG_PREFIXES = ['qwen3_5', 'qwen3_6', 'qwen3_8'];
    const DIFFUSION_CONFIG_MODEL_TYPES = new Set([
        'diffusion_gemma',
    ]);
    const DIFFUSION_UNSUPPORTED_PROFILE_FIELDS = new Set([
        'top_p',
        'top_k',
        'min_p',
        'repetition_penalty',
        'presence_penalty',
        'force_sampling',
        'enable_thinking',
        'preserve_thinking',
        'thinking_budget_enabled',
        'thinking_budget_tokens',
        'reasoning_parser',
        'guided_grammar_enabled',
        'guided_grammar',
        'max_tool_result_tokens',
        'index_cache_freq',
        'turboquant_kv_enabled',
        'turboquant_kv_bits',
        'turboquant_skip_last',
        'qwen35_ane_prefill_enabled',
        'qwen35_ane_prefill_sequence_length',
        'qwen35_ane_prefill_tail_padding_min_tokens',
        'qwen35_ane_prefill_fraction',
        'qwen35_ane_prefill_fused_down',
        'qwen35_ane_prefill_max_layers',
        'qwen35_ane_prefill_dual_ane',
        'qwen35_ane_prefill_gdn',
        'qwen35_ane_prefill_gdn_fraction',
        'qwen35_ane_prefill_gdn_max_layers',
        'qwen35_ane_prefill_cpu_enabled',
        'qwen35_ane_prefill_cpu_fraction',
        'qwen35_ane_prefill_cpu_down_fraction',
        'qwen35_ane_prefill_cpu_gdn_fraction',
        'qwen35_ane_prefill_cpu_threads',
        'qwen35_ane_prefill_cpu_shared_resource',
        'specprefill_enabled',
        'specprefill_draft_model',
        'specprefill_keep_pct',
        'specprefill_threshold',
        'dflash_enabled',
        'dflash_draft_model',
        'dflash_draft_quant_enabled',
        'dflash_draft_quant_weight_bits',
        'dflash_draft_quant_activation_bits',
        'dflash_draft_quant_group_size',
        'dflash_max_ctx',
        'dflash_in_memory_cache',
        'dflash_in_memory_cache_max_entries',
        'dflash_in_memory_cache_max_bytes',
        'dflash_ssd_cache',
        'dflash_ssd_cache_max_bytes',
        'dflash_draft_window_size',
        'dflash_draft_sink_size',
        'dflash_block_size',
        'dflash_verify_mode',
        'mtp_enabled',
        'vlm_mtp_enabled',
        'vlm_mtp_draft_model',
        'vlm_mtp_draft_block_size',
    ]);
    const DIFFUSION_UNSUPPORTED_CT_KWARGS = new Set([
        'enable_thinking',
        'reasoning_effort',
        'preserve_thinking',
    ]);
    const REASONING_EFFORT_PRESETS = new Set([
        'low', 'medium', 'high', 'xhigh', 'max',
    ]);
    const VLM_MTP_DRAFTER_CONFIG_MODEL_TYPES = new Set([
        'gemma4_assistant',
        'gemma4_unified_assistant',
        'qwen3_5_mtp',
    ]);
    // DFlash drafters that carry no "dflash" name token. Meta ships the Muse
    // Glimmer DFlash drafter as "-assistant", which oMLX's name heuristics
    // would otherwise route to the MTP/spec-prefill buckets.
    const DFLASH_DRAFTER_CONFIG_MODEL_TYPES = new Set([
        'muse_glimmer_assistant',
    ]);
    const DASHBOARD_MAIN_TABS = new Set(['status', 'cluster', 'settings', 'models', 'logs', 'bench']);
    const DASHBOARD_SETTINGS_TABS = new Set(['global', 'integrations', 'models']);
    const DASHBOARD_MODELS_TABS = new Set(['manager', 'downloader', 'quantizer', 'uploader']);
    const DASHBOARD_BENCH_TABS = new Set(['throughput', 'accuracy', 'context']);
    const THEME_STORAGE_KEY = 'omlx-chat-theme';
    const ENHANCED_READABILITY_KEY = 'omlx-enhanced-readability';

    // Default sort for the settings and manager model tables. Also the target
    // state for the "reset sort" action.
    const MODELS_SORT_DEFAULT = { by: 'id', order: 'asc' };
    const MANAGER_SORT_DEFAULT = { by: 'name', order: 'asc' };

    function dashboard() {
        return {
            // Theme
            theme: localStorage.getItem(THEME_STORAGE_KEY) || 'auto',
            activeTheme: 'light', // Will be updated by applyTheme
            systemThemeListener: null,
            enhancedReadability: localStorage.getItem(ENHANCED_READABILITY_KEY) === 'on',

            // Mobile menu
            mobileMenuOpen: false,

            // Main tab state (Status, Settings, or Logs)
            mainTab: 'status',

            activeTab: 'global',
            settingsDropdown: false,
            themeDropdown: false,

            // Global settings
            globalSettings: {
                base_path: '',
                server: { host: '127.0.0.1', port: 8000, log_level: 'info', sse_keepalive_mode: 'chunk', burst_decode_mode: 'balanced', preserve_mid_system_cache: true, distributed_inference_enabled: false, distributed_inference_active: false, max_audio_upload_size: '100MB' },
                model: { model_dirs: [''], model_fallback: false, hide_helper_models: false },
                memory: { prefill_memory_guard: true, memory_guard_tier: 'balanced', memory_guard_custom_ceiling_gb: 0 },
                scheduler: { max_concurrent_requests: 8, embedding_batch_size: 32, chunked_prefill: false, prefill_priority: 'context', decode_fairness: true },
                cache: { enabled: true, ssd_cache_dir: '', ssd_cache_max_size: 'auto', hot_cache_max_size: '0', hot_cache_write_through: false, ane_compile_cache: false, initial_cache_blocks: 256, hot_cache_only: false, gdn_snapshot_storage: 'auto', gdn_ssd_split_enabled: true, gdn_ssd_pending_max_size: '512MB', gdn_sidecar_precision: 'fp32' },
                sampling: { max_context_window: 32768, max_context_window_policy: null, max_tokens: 32768, temperature: 1.0, top_p: 0.95, top_k: 0, repetition_penalty: 1.0 },
                mcp: { config_path: '', expose_tools: true },
                huggingface: { endpoint: '', hf_cache_enabled: true, hf_cache_path: '' },
                network: { http_proxy: '', https_proxy: '', no_proxy: '', ca_bundle: '' },
                auth: { api_key_set: false, api_key: '', skip_api_key_verification: false, sub_keys: [] },
                claude_code: { mode: 'cloud', opus_model: null, sonnet_model: null, haiku_model: null },
                integrations: {
                    copilot_model: null,
                    codex_model: null,
                    opencode_model: null,
                    openclaw_model: null,
                    hermes_model: null,
                    pi_model: null,
                    openclaw_tools_profile: 'full',
                    markitdown_enabled: true,
                    markitdown_expose_model: false,
                    markitdown_max_file_size_mb: 25,
                    markitdown_max_files_per_request: 5,
                    markitdown_pdf_processing_engine: 'markitdown',
                    web_search_provider: 'ddgs',
                    web_search_brave_api_key: '',
                    web_search_searxng_url: '',
                    web_search_ddgs_backends: '',
                    web_search_max_results: 3,
                    web_search_content_mode: 'snippet',
                    web_search_content_truncate: true,
                    web_search_content_max_chars: 20000,
                },
                ui: { language: 'en' },
                idle_timeout: { idle_timeout_seconds: null },
                system: { total_memory_bytes: 0, total_memory: '', auto_model_memory: '', ssd_total_bytes: 0, ssd_total: '' },
            },

            // Web search "Test search" button state
            webSearchTest: { running: false, ok: null, message: '' },
            // Engines selectable for the DDGS Custom provider (ddgs 9.14.1 text registry)
            ddgsBackendList: ['brave', 'duckduckgo', 'grokipedia', 'mojeek', 'wikipedia', 'yahoo', 'yandex'],

            // Cache slider (0-100%)
            cachePercent: 10,
            editingCache: false,
            // Hot cache slider (0-50%)
            hotCachePercent: 0,
            // Editing state for direct GB input
            editingHotCache: false,

            // Idle timeout string value for select binding (null ↔ '')
            idleTimeoutValue: '',

            // Models
            models: [],
            loadingModels: false,
            reloading: false,
            // Sort state persists across refreshes/restarts via localStorage.
            sortBy: localStorage.getItem('omlx_models_sort_by') || MODELS_SORT_DEFAULT.by,
            sortOrder: localStorage.getItem('omlx_models_sort_order') || MODELS_SORT_DEFAULT.order,
            modelSearch: '',
            // Manager tab (Browse Models > Local) sort + search state.
            managerSortBy: localStorage.getItem('omlx_manager_sort_by') || MANAGER_SORT_DEFAULT.by,
            managerSortOrder: localStorage.getItem('omlx_manager_sort_order') || MANAGER_SORT_DEFAULT.order,
            managerSearch: '',

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
                is_diffusion_model: false,
                qwen35_ane_prefill_enabled: false,
                qwen35_ane_prefill_sequence_length: 2048,
                qwen35_ane_prefill_tail_padding_min_tokens: 0,
                qwen35_ane_prefill_fraction: 0.53,
                qwen35_ane_prefill_fused_down: false,
                qwen35_ane_prefill_max_layers: 64,
                qwen35_ane_prefill_dual_ane: true,
                qwen35_ane_prefill_gdn: true,
                qwen35_ane_prefill_gdn_fraction: 0.5,
                qwen35_ane_prefill_gdn_max_layers: 48,
                qwen35_ane_prefill_cpu_enabled: false,
                qwen35_ane_prefill_cpu_fraction: 0.135,
                qwen35_ane_prefill_cpu_down_fraction: 0,
                qwen35_ane_prefill_cpu_gdn_fraction: 0,
                qwen35_ane_prefill_cpu_threads: 8,
                qwen35_ane_prefill_cpu_shared_resource: true,
                trust_remote_code: false,
            },
            savingModelSettings: false,
            importingMtplx: false,
            loadingGenDefaults: false,
            reasoningParsers: [],
            aneTuning: {
                tuningId: null,
                modelId: null,
                running: false,
                cancelling: false,
                applying: false,
                applied: false,
                total: 0,
                status: null,
                error: '',
            },
            aneTuningOverrides: {
                allowCpu: true,
                allowCpuGate: true,
                allowCpuDown: true,
                allowAneGdn: true,
                allowCpuGdn: true,
                allowCpuSharedResource: true,
            },
            _aneTuningPollTimer: null,

            // Profile / template / preset state
            profiles: [],                // per-model profiles for selectedModel
            templates: [],               // global templates
            presets: [],                 // curated presets (bundled + remote refresh)
            profileFields: { universal: [], model_specific: [] },  // loaded from /api/profile-fields
            profileScope: 'model',       // 'preset' | 'global' | 'model'
            refreshingPresets: false,
            activeProfileName: null,     // currently-active profile for the form
            profilesDrift: false,        // true if form values differ from active profile
            _applySeq: 0,               // monotonic counter for apply race guard
            profileError: '',
            showNewProfileForm: false,
            newProfile: { display_name: '', api_name: '', api_name_touched: false, description: '', also_as_template: false },
            showNewTemplateForm: false,
            newTemplate: { name: '', display_name: '', description: '' },
            editingProfile: null,        // profile name being edited inline
            editingTemplate: null,       // template name being edited inline
            profileDeleteConfirm: null,
            templateDeleteConfirm: null,

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
                    memory_pressure: {
                        enabled: false,
                        current_bytes: 0,
                        soft_bytes: 0,
                        hard_bytes: 0,
                        pressure_level: 'ok',
                    },
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
                    hot_cache_size_bytes: 0,
                    hot_cache_entries: 0,
                    hot_cache_max_bytes: 0,
                    disk_max_bytes: 0,
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
            // Server connectivity info (from /admin/api/server-info)
            serverAliases: [],
            selectedAlias: '',

            // Distributed cluster prototype
            clusterStatus: null,
            clusterLoading: false,
            clusterError: '',
            // Connection failures outlive plan invalidation. Automatic memory,
            // model and fabric refreshes rebuild the plan in the background;
            // treating that as permission to unmount an SSH error made the
            // whole page jump on every retry.
            clusterConnectionError: '',
            clusterRouteTo: '',
            clusterWorkerRunning: false,
            clusterWorkerResult: null,
            clusterCollectiveRunning: false,
            clusterCollectiveResult: null,
            clusterPipelineRunning: false,
            clusterPipelineResult: null,
            clusterPlanMode: 'estimate',
            clusterPlanModelSizeGiB: 300,
            clusterPlanLayerCount: 80,
            clusterPlanModelPath: '',
            clusterPlanModelSource: '127.0.0.1',
            clusterPlanModelSourcePython: '',
            clusterTargetContextTokens: 8192,
            // Automatic follows the highest context the current model, split,
            // and per-Mac memory allowances can safely serve. Manual keeps a
            // deliberately lower choice until it no longer fits.
            clusterContextMode: 'auto',
            clusterPlanTensorParallelSize: 1,
            clusterPeerHealth: null,
            clusterPeerHealthLoading: false,
            // Server-owned incident feed. Polls merge by id and never delete:
            // the only paths that change a row are new server records (via the
            // since cursor) and an explicit dismissal round-trip.
            clusterIncidents: [],
            _clusterIncidentSeq: 0,
            _clusterIncidentsById: null,
            _clusterIncidentEpoch: '',
            clusterStagingResult: null,
            clusterStagingLoading: false,
            clusterGuidance: null,
            clusterLastGoodConfig: null,
            clusterShowAdvanced: false,
            clusterActivationProgress: '',
            clusterCatalogue: null,
            clusterCatalogueLoading: false,
            clusterCatalogueError: '',
            clusterModelInventory: null,
            clusterModelInventoryLoading: false,
            clusterModelInventoryError: '',
            clusterCatalogueDir: '~/.omlx/models',
            clusterModelSearch: '',
            // The normal cluster flow is deliberately small: confirm the two
            // Macs, choose a downloaded model, start. The planner and link
            // diagnostics remain available without competing with that path.
            clusterShowModelPicker: false,
            clusterShowSetupDetails: false,
            clusterDiagnosticsLoading: false,
            clusterLinkStatus: null,
            clusterLinkStatusLoading: false,
            clusterLinkSetupLoading: false,
            // One "Copied!" affordance for every block of commands the page
            // shows, keyed by panel. A second flag per panel is how two copy
            // buttons drift apart.
            clusterCommandsCopied: {},
            clusterTransports: null,
            clusterTransportsLoading: false,
            clusterTransportsError: '',
            clusterAutoconfigure: null,
            clusterAutoconfigureLoading: false,
            clusterAutoconfigureError: '',
            clusterAutoconfigurePrefer: 'speed',
            clusterStrategy: 'auto',
            clusterStrategyOptions: [
                { key: 'auto', label: 'Automatic',
                  summary: 'Pick the best split for this model and link',
                  detail: 'Uses tensor parallelism when the model can be split evenly and the link is fast enough, otherwise falls back to pipeline stages.' },
                { key: 'tensor', label: 'Tensor — faster responses',
                  summary: 'Every accelerator works on every token',
                  detail: 'Splits every layer across the selected accelerators so they compute each token together. Needs a verified fast link and a model whose attention heads divide evenly.' },
                { key: 'pipeline', label: 'Pipeline — bigger models',
                  summary: 'Each accelerator holds different layers',
                  detail: 'Gives each accelerator a slice of the layers, so a model too large for one device fits across several. Works on slower links when the model supports pipeline execution.' },
            ],
            clusterPlanNodes: [
                { key: 1, node_id: 'this-mac', capacity_gib: 128, reserve_gib: 8, role: 'workstation' },
                { key: 2, node_id: 'peer-mac', capacity_gib: 256, reserve_gib: 8, role: 'headless' },
            ],
            clusterNodeRoles: [],
            clusterRoleTooltip: '',
            clusterMemoryLimitsManual: false,
            clusterMemoryAllowancesGiB: {},
            _clusterMemoryAllowancesLoaded: false,
            clusterBudgetsLoading: false,
            clusterBudgetsError: '',
            // Legacy two-node split value retained for stored dashboard state.
            // New plans use one soft target per node so the same control works
            // for two, three, or more Macs.
            clusterSplitGiB: null,
            clusterWeightTargetsGiB: {},
            clusterSplitBusy: false,
            clusterPlan: null,
            clusterPlanLoading: false,
            clusterPlanError: '',
            _clusterPlanSignature: '',
            _clusterAutoPlanKey: '',
            _clusterPlanRevision: 0,
            _clusterNodeKey: 2,
            _clusterDefaultsApplied: false,
            clusterPeerSsh: '',
            // Every discovered worker selected for the automatic cluster.
            // clusterPeerSsh remains the first worker for the legacy pair
            // diagnostics, while planning and activation use this full list.
            clusterSelectedPeers: [],
            clusterLocalIp: '',
            clusterPeerIp: '',
            // Both addresses are read off the live interfaces of both Macs.
            // Typing over them is allowed and remembered, because a person who
            // knows their fabric should not be argued with — but a typed
            // address is the field that took a launch down, so it is never the
            // default.
            clusterIpsOverridden: false,
            clusterFabric: null,
            clusterFabricLoading: false,
            clusterFabricError: '',
            clusterCudaFabricVerificationLoading: '',
            clusterCudaFabricVerificationError: '',
            clusterCudaFabricVerifications: {},
            clusterKeychainStoring: false,
            clusterExecutionProfile: 'balanced',
            clusterAutoTune: true,
            clusterSamplingRankOnly: true,
            clusterAsyncOverlap: true,
            clusterCacheAffinity: true,
            clusterMaxKvSize: '',
            clusterRingConnectionsPerIp: 2,
            clusterPeerProbeLoading: false,
            clusterPeerProbe: null,
            // Hardware belongs to a peer, not to its position in the ring.
            // Keep every probe so three-or-more Mac clusters can render the
            // real chip and physical memory for every selected worker.
            clusterPeerProbes: {},
            clusterShowPeerAdvanced: false,
            clusterDiscoveryLoading: false,
            clusterDiscoveredPeers: null,
            clusterDiscoveryWarning: '',
            _clusterKnownNodesHydrated: false,
            _clusterKnownNodesNeedsSync: false,
            clusterPairingToken: null,
            clusterPairingTokenLoading: false,
            clusterPairingSecret: '',
            clusterPairingSecretCopied: false,
            clusterJoinControllerIp: '',
            clusterJoinCommand: '',
            clusterJoinId: '',
            clusterJoinExpiresAt: 0,
            clusterJoinKeys: [],
            clusterJoinedNodes: [],
            clusterJoinLoading: false,
            clusterJoinError: '',
            clusterSshKey: null,
            clusterSshKeyLoading: false,
            clusterSshKeyGenerating: false,
            clusterExchangeToken: null,
            clusterExchangeTokenLoading: false,
            clusterExchangeTokenCopied: false,
            clusterPeerExchangeToken: '',
            clusterKeyExchangeLoading: false,
            clusterKeyExchangeResult: null,
            clusterDeployments: [],
            clusterDeploymentsError: '',
            clusterActivationLoading: false,
            clusterActivationResult: null,
            // What automatic tuning proposed after measuring the fabric. The
            // signed placement still wins when the proposal moves layers.
            clusterPlanChanges: null,
            clusterDeactivatingId: '',
            _clusterRefreshTimer: null,
            _clusterActivationProgressTimer: null,
            _clusterDiscoveryRefreshCounter: 0,

            // Server-restart state machine (driven by Settings > Server > Restart).
            // status transitions: idle → restarting → waiting → idle (success)
            //                   |                   |
            //                   |                   └─→ error (timeout / non-200)
            //                   └─→ unsupported (no menubar supervisor)
            //                   └─→ error (POST failed)
            restartServer: {
                status: 'idle',
                message: '',
            },

            statsScope: 'session',
            selectedStatsModel: '',
            showClearStatsConfirm: false,
            showClearAlltimeConfirm: false,
            showClearSsdCacheConfirm: false,
            showClearHotCacheConfirm: false,
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
            logMinLevel: 'TRACE',
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
            hfTokenInvalid: false,
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
            // Search filters
            hfSearchFiltersOpen: false,
            hfSearchMinParams: '',
            hfSearchMaxParams: '',
            hfSearchMaxSize: '',
            hfSearchMinSize: '',
            // Table sort state for Browse Models
            hfTableSort: 'downloads',
            hfTableSortDir: 'desc',

            // Computed: check if any filters are active
            get hfSearchFiltersActive() {
                return this.hfSearchMinParams || this.hfSearchMaxParams || this.hfSearchMaxSize || this.hfSearchMinSize;
            },

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
            oqAllModels: [],
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
            oqTextOnly: false,
            oqDtype: 'bfloat16',
            oqSensitivityModelPath: '',
            oqPreserveMtp: false,
            oqMtpAssistantPath: '',
            oqEnhanced: false,
            oqeReuseImatrixCache: true,
            oqeImatrixCachePath: '',
            oqeStrictImatrix: false,

            // oQ Uploader state
            uploadHfToken: localStorage.getItem('omlx-hf-upload-token') || '',
            uploadHfUsername: '',
            uploadHfOrgs: [],
            uploadHfNamespace: '',
            uploadTokenValidated: false,
            uploadTokenValidating: false,
            uploadOqModels: [],
            uploadAllModels: [],
            uploadOqModelsLoaded: false,
            uploadTasks: [],
            uploadError: '',
            uploadSuccess: '',
            _uploadRefreshTimer: null,
            // Upload modal
            uploadModalOpen: false,
            uploadModalModelPath: '',
            uploadModalModelName: '',
            uploadModalRepoId: '',
            uploadReadmeSource: '',
            uploadAutoReadme: true,
            uploadPrivate: false,
            uploadStarting: false,

            // Benchmark state
            benchModelId: '',
            benchContextProfile: 'code_python',
            benchAlignPromptToAne: false,
            benchPromptLengths: { 1024: true, 4096: true, 8192: false, 16384: false, 32768: false, 65536: false, 131072: false, 200000: false },
            benchBatchSizes: { 2: true, 4: true, 8: false },
            benchForceLmEngine: false,
            benchAdvancedOptionsOpen: false,
            benchExternalEnabled: false,
            // Shared external endpoint settings (persisted in localStorage,
            // used by both the throughput and accuracy bench tabs)
            externalBaseUrl: localStorage.getItem('omlx_bench_external_base_url') || '',
            externalApiKey: localStorage.getItem('omlx_bench_external_api_key') || '',
            externalModel: localStorage.getItem('omlx_bench_external_model') || '',
            // { base_url, model } snapshot of the current run when external
            // (no API key — used for the text export header)
            benchRunExternal: null,
            benchRunning: false,
            benchBenchId: null,
            benchProgress: null,
            benchSingleResults: [],
            benchBatchResults: [],
            benchError: '',
            benchEventSource: null,
            benchShowMetrics: false,
            benchShowText: false,
            benchCopied: false,
            benchTip: null,
            benchDeviceInfo: null,
            benchUploadResults: [],
            benchUploadDone: null,
            benchUploading: false,
            benchUploadSkipped: null,  // { reason } — only external-endpoint runs skip now
            benchUploadFlags: [],      // [{key, label}] acceleration active during the run
            // { bench_id, model_id } when the server reports a running bench
            // that is NOT the one this tab is displaying. Drives the "another
            // bench is running" banner + disables Start so the user doesn't
            // race a 409 on the server.
            benchOtherActive: null,

            // Bench sub-tab & dropdown
            benchTab: 'throughput',
            benchDropdown: false,

            // Context benchmark state
            ctxBenchModelId: '',
            ctxBenchTarget: 131072,
            ctxBenchRunning: false,
            ctxBenchBenchId: null,
            ctxBenchProgress: null,   // { phase, progress, message }
            ctxBenchResult: null,
            ctxBenchError: '',
            ctxBenchEventSource: null,

            // Accuracy benchmark state
            accModelId: '',
            accBenchmarks: { mmlu: true, mmlu_pro: false, kmmlu: false, cmmlu: false, jmmlu: false, hellaswag: false, truthfulqa: true, arc_challenge: false, winogrande: false, gsm8k: false, mathqa: false, humaneval: true, mbpp: false, livecodebench: false, bbq: false, safetybench: false },
            accSampleSizes: { mmlu: 1000, mmlu_pro: 300, kmmlu: 300, cmmlu: 300, jmmlu: 300, hellaswag: 200, truthfulqa: 0, arc_challenge: 300, winogrande: 300, gsm8k: 100, mathqa: 300, humaneval: 0, mbpp: 200, livecodebench: 100, bbq: 300, safetybench: 300 },
            accBenchmarkGroups: [
                {
                    name: 'Knowledge',
                    benchmarks: [
                        { key: 'mmlu', label: 'MMLU', desc: 'Knowledge · 57 subjects', fullSize: 14042, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                        { key: 'mmlu_pro', label: 'MMLU-Pro', desc: 'Hard knowledge · 14 subjects (10-way)', fullSize: 12032, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                        { key: 'kmmlu', label: 'KMMLU', desc: '한국어 지식 · 45 과목', fullSize: 35030, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                        { key: 'cmmlu', label: 'CMMLU', desc: '中文知识 · 67 科目', fullSize: 11582, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                        { key: 'jmmlu', label: 'JMMLU', desc: '日本語知識 · 112 科目', fullSize: 7536, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                    ],
                },
                {
                    name: 'Commonsense & Reasoning',
                    benchmarks: [
                        { key: 'hellaswag', label: 'HellaSwag', desc: 'Commonsense reasoning', fullSize: 10042, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                        { key: 'arc_challenge', label: 'ARC-C', desc: 'Science reasoning', fullSize: 1172, sizes: [30, 50, 100, 200, 300] },
                        { key: 'winogrande', label: 'Winogrande', desc: 'Coreference resolution', fullSize: 1267, sizes: [30, 50, 100, 200, 300] },
                        { key: 'truthfulqa', label: 'TruthfulQA', desc: 'Truthfulness', fullSize: 817, sizes: [30, 50, 100, 200, 300] },
                    ],
                },
                {
                    name: 'Math',
                    benchmarks: [
                        { key: 'gsm8k', label: 'GSM8K', desc: 'Math reasoning', fullSize: 1319, sizes: [30, 50, 100, 200, 300] },
                        { key: 'mathqa', label: 'MathQA', desc: 'Quantitative reasoning · 5-way', fullSize: 2985, sizes: [30, 50, 100, 200, 300, 500, 1000] },
                    ],
                },
                {
                    name: 'Coding',
                    benchmarks: [
                        { key: 'humaneval', label: 'HumanEval', desc: 'Function completion', fullSize: 164, sizes: [30, 50, 100] },
                        { key: 'mbpp', label: 'MBPP', desc: 'Python problems', fullSize: 500, sizes: [30, 50, 100, 200, 300] },
                        { key: 'livecodebench', label: 'LiveCodeBench', desc: 'Code generation', fullSize: 1055, sizes: [30, 50, 100, 200, 300] },
                    ],
                },
                {
                    name: 'Safety & Alignment',
                    benchmarks: [
                        { key: 'bbq', label: 'BBQ', desc: 'Social bias · 11 categories', fullSize: 10864, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                        { key: 'safetybench', label: 'SafetyBench', desc: 'Safety · 7 categories', fullSize: 11435, sizes: [30, 50, 100, 200, 300, 500, 1000, 2000] },
                    ],
                },
            ],
            accBatchSize: 1,
            accEnableThinking: false,
            accSamplingProfile: 'deterministic',
            accAdvancedOptionsOpen: false,
            accExternalEnabled: false,
            // Provider-specific JSON is intentionally session-only.
            accExternalExtraBody: '',
            // Accuracy-only max_tokens floor; persisted like the shared
            // endpoint settings because it is a simple number the user sets
            // once per endpoint (thinking models need a larger budget).
            accExternalMaxTokens: localStorage.getItem('omlx_acc_external_max_tokens') || '',
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
                    this.loadServerInfo(),
                    this.loadProfileFields(),
                    this.loadPresets(),
                    this.checkForUpdate()
                ]);

                this.startUpdateCheckTimer();

                await this.handleMainTabChange(this.mainTab);

                // Watch for main tab changes to manage refresh timers
                this.$watch('mainTab', (value) => {
                    this.handleMainTabChange(value);
                });

                // When the user returns to this browser tab after looking
                // elsewhere, re-check whether a different bench just started
                // in another tab. Fires the banner without requiring an
                // in-app tab switch.
                document.addEventListener('visibilitychange', () => {
                    if (document.visibilityState !== 'visible') return;
                    if (this.mainTab === 'bench' && this.benchTab === 'throughput') {
                        this.loadBenchState();
                    }
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

                // Pause stats polling when tab is hidden to reduce server load
                document.addEventListener('visibilitychange', () => {
                    if (document.hidden) {
                        this.stopStatsRefresh();
                        this.stopClusterRefresh();
                    } else if (this.mainTab === 'status') {
                        this.loadStats();
                        this.startStatsRefresh();
                    } else if (this.mainTab === 'cluster') {
                        this.refreshClusterExperience();
                        this.startClusterRefresh();
                    }
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
                    await this.loadBenchState();
                    await this.loadAccState();
                    await this.loadCtxBenchState();
                }
                if (value === 'cluster') {
                    // Render remembered Macs before Bonjour pays its discovery
                    // timeout. Cached nodes are display hints only: the live
                    // peer probe below still gates planning and activation.
                    this.loadClusterKnownNodes();
                    await Promise.all([
                        this.clusterStatus ? Promise.resolve() : this.loadClusterStatus(),
                        this.loadClusterDeployments(),
                        this.loadClusterJoinStatus(),
                        this.clusterDiscoveredPeers === null
                            ? this.discoverClusterPeers()
                            : Promise.resolve(),
                    ]);
                    await this.initializeClusterSetup();
                    this.startClusterRefresh();
                } else {
                    this.stopClusterRefresh();
                }
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
                if (tab === 'cluster' && !this.globalSettings.server.distributed_inference_active) return;
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
                if (tab === 'uploader') {
                    if (!this.uploadOqModelsLoaded) this.loadUploadOqModels();
                    this.loadUploadTasks();
                }
            },

            async clusterResponseError(response, fallback) {
                const payload = await response.json().catch(() => ({}));
                const message = this.clusterErrorMessage(payload.detail, fallback);
                // Fetch recovery steps for the failure the user is about to see.
                // Fire-and-forget: guidance is an improvement on the raw message,
                // never a precondition for showing it.
                this.explainClusterError(message);
                return message;
            },

            clusterErrorMessage(detail, fallback = 'Something went wrong') {
                if (!Array.isArray(detail)) {
                    if (detail && typeof detail === 'object') {
                        return detail.msg || detail.message || JSON.stringify(detail);
                    }
                    return String(detail || fallback);
                }
                const messages = detail.map(item => {
                    if (!item || typeof item !== 'object') return String(item);
                    const location = Array.isArray(item.loc)
                        ? item.loc.filter(part => part !== 'body').join('.')
                        : '';
                    const message = item.msg || item.message || JSON.stringify(item);
                    return location ? `${location}: ${message}` : message;
                });
                return messages.filter(Boolean).join(', ') || fallback;
            },

            clusterDisplayedError() {
                return this.clusterConnectionError || this.clusterError;
            },

            async explainClusterError(message) {
                this.clusterGuidance = null;
                if (!message) return;
                try {
                    const response = await fetch('/admin/api/cluster/guidance', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: String(message).slice(0, 4096) }),
                    });
                    if (response.ok) {
                        this.clusterGuidance = await response.json();
                    }
                } catch (error) {
                    // Leaving guidance null just falls back to the raw message.
                }
            },

            dismissClusterGuidance() {
                this.clusterGuidance = null;
            },

            async openClusterPairingSetup() {
                this.clusterShowSetupDetails = true;
                this.clusterShowPeerAdvanced = true;
                if (!this.clusterSshKey) await this.loadClusterSshKey();
                await this.$nextTick();
                const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
                document.querySelector('[data-cluster-ssh-setup]')?.scrollIntoView({
                    behavior: reduced ? 'auto' : 'smooth',
                    block: 'center',
                });
            },

            clusterJoinSuggestedIp() {
                const explicit = String(this.clusterJoinControllerIp || '').trim();
                if (explicit) return explicit;
                const detected = String(this.clusterLocalIp || '').trim();
                if (detected) return detected;
                const browserHost = String(window.location.hostname || '').trim();
                if (/^(?:\d{1,3}\.){3}\d{1,3}$/.test(browserHost)
                    && !browserHost.startsWith('127.')) {
                    return browserHost;
                }
                return '';
            },

            clusterJoinLanWarning() {
                const host = String(window.location.hostname || '').toLowerCase();
                if (!['localhost', '127.0.0.1', '::1'].includes(host)) return '';
                return 'This dashboard is open through localhost. Enter the Studio LAN IP, '
                    + 'and make sure Server host is set to 0.0.0.0 in Settings before running '
                    + 'the command on a CUDA box.';
            },

            clusterJoinCommandState() {
                if (!this.clusterJoinId) return null;
                return (this.clusterJoinKeys || []).find(
                    item => item.join_id === this.clusterJoinId
                ) || null;
            },

            clusterJoinCommandUsable() {
                const state = this.clusterJoinCommandState();
                return Boolean(
                    this.clusterJoinCommand
                    && state?.status === 'pending'
                    && Number(state.expires_at || this.clusterJoinExpiresAt) * 1000 > Date.now()
                );
            },

            clusterJoinCommandStatusLabel() {
                if (!this.clusterJoinCommand) return '';
                const state = this.clusterJoinCommandState();
                if (state?.status === 'used') return 'Used · worker is finishing setup';
                const remaining = Math.ceil(
                    (Number(state?.expires_at || this.clusterJoinExpiresAt) * 1000 - Date.now())
                    / 60000
                );
                if (!state || remaining <= 0) return 'Expired · generate a new command';
                return `Single use · expires in ${remaining} min`;
            },

            clusterJoinedAtLabel(node) {
                const timestamp = Number(node?.joined_at || 0) * 1000;
                if (!timestamp) return 'Joined';
                return `Joined ${new Date(timestamp).toLocaleString()}`;
            },

            async loadClusterJoinStatus() {
                try {
                    const response = await fetch('/admin/api/cluster/join-status', {
                        cache: 'no-store',
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(
                            response,
                            'Could not read CUDA worker enrollment status',
                        ));
                    }
                    const result = await response.json();
                    const previousIds = new Set(
                        (this.clusterJoinedNodes || []).map(node => node.node_id)
                    );
                    const nodes = Array.isArray(result.nodes) ? result.nodes : [];
                    const discovered = new Map(
                        (this.clusterDiscoveredPeers || []).map(peer => [peer.ssh, peer])
                    );
                    const selected = new Map(
                        (this.clusterSelectedPeers || []).map(peer => [peer.ssh, peer])
                    );
                    nodes.forEach(node => {
                        const peer = {
                            ...discovered.get(node.ssh),
                            ssh: node.ssh,
                            name: node.hostname || node.node_id,
                            service: 'oMLX CUDA Worker',
                            transport: 'detecting',
                            accelerator: 'cuda',
                            accelerator_vendor: 'nvidia',
                            python_executable: node.python_executable || '',
                            addresses: node.addresses || [],
                            enrolled: true,
                            cached: false,
                        };
                        discovered.set(peer.ssh, peer);
                        // A newly completed GUI enrollment joins the active pool
                        // once. Manual de-selection later in this browser session
                        // is respected and polling will not add it back.
                        if (!previousIds.has(node.node_id) && !selected.has(peer.ssh)) {
                            selected.set(peer.ssh, peer);
                        }
                    });
                    this.clusterJoinKeys = Array.isArray(result.join_keys)
                        ? result.join_keys
                        : [];
                    this.clusterJoinedNodes = nodes;
                    this.clusterDiscoveredPeers = [...discovered.values()];
                    this.clusterSelectedPeers = [...selected.values()];
                    if (!this.clusterPeerSsh && this.clusterSelectedPeers.length) {
                        this.clusterPeerSsh = this.clusterSelectedPeers[0].ssh;
                    }
                    if (nodes.some(node => !previousIds.has(node.node_id))) {
                        this._clusterKnownNodesNeedsSync = true;
                        // A newly joined node changes the topology, so its
                        // budgets should be measured once on the next setup pass.
                        this._clusterBudgetsMeasured = false;
                        this.saveClusterKnownNodes();
                    }
                    this.clusterJoinError = result.load_error || '';
                } catch (error) {
                    this.clusterJoinError = error?.message
                        || 'Could not read CUDA worker enrollment status';
                }
            },

            async revokeClusterJoinCommand() {
                const joinId = String(this.clusterJoinId || '').trim();
                if (!joinId) return true;
                try {
                    const response = await fetch(
                        `/admin/api/cluster/join-keys/${encodeURIComponent(joinId)}`,
                        { method: 'DELETE' },
                    );
                    if (!response.ok && response.status !== 404) {
                        throw new Error(await this.clusterResponseError(
                            response,
                            'Could not revoke the join command',
                        ));
                    }
                } catch (error) {
                    this.clusterJoinError = error?.message
                        || 'Could not revoke the join command';
                    return false;
                }
                this.clusterJoinCommand = '';
                this.clusterJoinId = '';
                this.clusterJoinExpiresAt = 0;
                await this.loadClusterJoinStatus();
                return true;
            },

            async generateClusterCudaJoinCommand() {
                if (this.clusterJoinLoading) return;
                const controllerIp = this.clusterJoinSuggestedIp();
                if (!controllerIp) {
                    this.clusterJoinError = 'Enter the Studio LAN IPv4 address first.';
                    return;
                }
                this.clusterJoinControllerIp = controllerIp;
                this.clusterJoinLoading = true;
                this.clusterJoinError = '';
                if (
                    this.clusterJoinId
                    && this.clusterJoinCommandState()?.status === 'pending'
                ) {
                    const revoked = await this.revokeClusterJoinCommand();
                    if (!revoked) {
                        this.clusterJoinLoading = false;
                        return;
                    }
                } else {
                    // A claimed command owns a live installation session. Do
                    // not revoke it merely because the user is enrolling the
                    // second CUDA box.
                    this.clusterJoinCommand = '';
                    this.clusterJoinId = '';
                    this.clusterJoinExpiresAt = 0;
                }
                try {
                    const secure = window.location.protocol === 'https:';
                    const port = Number(window.location.port)
                        || (secure ? 443 : 80);
                    const response = await fetch('/admin/api/cluster/join-keys', {
                        method: 'POST',
                        cache: 'no-store',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            controller_ip: controllerIp,
                            controller_port: port,
                            scheme: secure ? 'https' : 'http',
                            ttl_seconds: 1800,
                        }),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(
                            response,
                            'Could not generate the CUDA join command',
                        ));
                    }
                    const result = await response.json();
                    this.clusterJoinCommand = result.command || '';
                    this.clusterJoinId = result.join_id || '';
                    this.clusterJoinExpiresAt = Number(result.expires_at || 0);
                    this.clusterJoinKeys = [
                        ...(this.clusterJoinKeys || []).filter(
                            item => item.join_id !== result.join_id
                        ),
                        result,
                    ];
                } catch (error) {
                    this.clusterJoinError = error?.message
                        || 'Could not generate the CUDA join command';
                } finally {
                    this.clusterJoinLoading = false;
                }
            },

            async loadClusterStatus() {
                if (this.clusterLoading) return;
                this.clusterLoading = true;
                this.clusterError = '';
                // Status polling is unrelated to an in-flight peer login. Keep
                // that login's recovery steps mounted until it succeeds or the
                // user changes the peer.
                if (!this.clusterConnectionError) this.dismissClusterGuidance();
                this.loadRememberedClusterConfig();
                try {
                    const params = new URLSearchParams();
                    const routeTo = this.clusterRouteTo.trim();
                    if (routeTo) params.set('route_to', routeTo);
                    const query = params.toString();
                    const response = await fetch(`/admin/api/cluster/status${query ? `?${query}` : ''}`);
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Failed to load cluster status'));
                    }
                    this.clusterStatus = await response.json();
                    if (!this._clusterDefaultsApplied) {
                        this.useLocalClusterNode(0);
                        this.clusterPlanNodes[0].role = 'workstation';
                        this._clusterDefaultsApplied = true;
                    }
                } catch (error) {
                    this.clusterError = error?.message || 'Failed to load cluster status';
                } finally {
                    this.clusterLoading = false;
                }
            },

            async loadClusterRuntime() {
                if (!this.clusterStatus) return;
                try {
                    const response = await fetch('/admin/api/cluster/runtime');
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (response.ok) {
                        this.clusterStatus.runtime_jobs = await response.json();
                    }
                } catch (_) {
                    // The full status refresh remains the visible error path.
                }
            },

            async downloadClusterDiagnostics() {
                if (this.clusterDiagnosticsLoading) return;
                this.clusterDiagnosticsLoading = true;
                try {
                    const response = await fetch('/admin/api/cluster/diagnostics');
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(
                            response,
                            'Could not build the diagnostic report',
                        ));
                    }
                    const report = await response.json();
                    const blob = new Blob(
                        [JSON.stringify(report, null, 2) + '\n'],
                        { type: 'application/json' },
                    );
                    const link = document.createElement('a');
                    const stamp = new Date().toISOString().replaceAll(':', '-');
                    const url = URL.createObjectURL(blob);
                    link.href = url;
                    link.download = `omlx-cluster-diagnostics-${stamp}.json`;
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                    URL.revokeObjectURL(url);
                } catch (error) {
                    this.clusterError = error?.message || 'Could not build the diagnostic report';
                    await this.explainClusterError(this.clusterError);
                } finally {
                    this.clusterDiagnosticsLoading = false;
                }
            },

            clusterActivationProgressFromRuntime() {
                const jobs = (this.clusterStatus?.runtime_jobs?.jobs || [])
                    .filter(job => job.live);
                if (!jobs.length) return 'Checking compatibility…';
                const validating = jobs.filter(
                    job => job.load_stage === 'validating'
                ).length;
                const ready = jobs.filter(job => job.phase === 'ready').length;
                const worldSize = Math.max(
                    ...jobs.map(job => Number(job.world_size) || 1),
                    1,
                );
                if (ready === worldSize) return 'Running readiness test…';
                if (validating) {
                    return `Validating loaded ranks · ${ready}/${worldSize} ready`;
                }
                const loadMemory = jobs.reduce(
                    (total, job) => total + Number(job.load_memory_bytes || 0),
                    0,
                );
                const memoryLabel = loadMemory > 0
                    ? ` · ${this.formatClusterGiB(loadMemory)} in use`
                    : '';
                const fixedJobs = jobs.filter(
                    job => job.load_stage === 'materializing_fixed'
                );
                if (fixedJobs.length) {
                    return `Preparing embeddings and shared weights · ${ready}/${worldSize} ranks ready${memoryLabel}`;
                }
                const layerJobs = jobs.filter(
                    job => Number(job.layers_total || 0) > 0
                );
                if (layerJobs.length) {
                    const layerTotal = layerJobs.reduce(
                        (total, job) => total + Number(job.layers_total || 0),
                        0,
                    );
                    const layersLoaded = layerJobs.reduce(
                        (total, job) => total + Math.min(
                            Number(job.layers_loaded || 0),
                            Number(job.layers_total || 0),
                        ),
                        0,
                    );
                    const percent = layerTotal > 0
                        ? Math.min(100, Math.floor((layersLoaded / layerTotal) * 100))
                        : 0;
                    const action = layerJobs.some(
                        job => job.load_stage === 'tensor_sharding'
                    ) ? 'Sharding model layers' : 'Loading model layers';
                    return `${action} · ${percent}% · ${ready}/${worldSize} ranks ready${memoryLabel}`;
                }
                return `Loading model weights · ${ready}/${worldSize} ranks ready${memoryLabel}`;
            },

            startClusterActivationProgress() {
                this.stopClusterActivationProgress();
                this.clusterActivationProgress = 'Checking compatibility…';
                const refresh = async () => {
                    await this.loadClusterRuntime();
                    if (this.clusterActivationLoading) {
                        this.clusterActivationProgress =
                            this.clusterActivationProgressFromRuntime();
                    }
                };
                refresh();
                this._clusterActivationProgressTimer = setInterval(refresh, 750);
            },

            stopClusterActivationProgress() {
                if (this._clusterActivationProgressTimer) {
                    clearInterval(this._clusterActivationProgressTimer);
                    this._clusterActivationProgressTimer = null;
                }
            },

            startClusterRefresh() {
                this.stopClusterRefresh();
                if (document.hidden || this.mainTab !== 'cluster') return;
                this._clusterDiscoveryRefreshCounter = 0;
                this._clusterRefreshTimer = setInterval(
                    () => this.refreshClusterExperience(),
                    2000,
                );
            },

            stopClusterRefresh() {
                if (this._clusterRefreshTimer) {
                    clearInterval(this._clusterRefreshTimer);
                    this._clusterRefreshTimer = null;
                }
            },

            // Runtime stays live every two seconds. Discovery repeats every ten
            // seconds even after the first worker appears, so three- and
            // four-node clusters grow automatically when another Mac starts.
            async refreshClusterExperience() {
                await this.loadClusterRuntime();
                await this.loadClusterIncidents();
                this._clusterDiscoveryRefreshCounter += 1;
                if (this._clusterDiscoveryRefreshCounter < 5) return;
                this._clusterDiscoveryRefreshCounter = 0;
                await Promise.all([
                    this.discoverClusterPeers(),
                    this.loadClusterJoinStatus(),
                ]);
                await this.initializeClusterSetup({ preview: false });
            },

            // Monotonic merge: the ?since= cursor means the server only ever
            // sends records this browser has not seen, and the merge below
            // only adds or updates Map entries — nothing on the poll path can
            // delete a row, so no refresh can wipe error state (#8). The only
            // removal is an explicit dismissal, which round-trips through the
            // server so it also survives reloads.
            async loadClusterIncidents() {
                try {
                    const since = this._clusterIncidentSeq || 0;
                    const response = await fetch(`/admin/api/cluster/incidents?since=${since}`);
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) return;
                    const payload = await response.json();
                    // The epoch names the seq numbering. A corrupt-log reset
                    // restarts seq at 1 under a new epoch; keeping the old
                    // cursor there would silence the feed for this tab
                    // forever, so restart the merge from scratch.
                    if (payload.epoch && payload.epoch !== this._clusterIncidentEpoch) {
                        if (this._clusterIncidentEpoch) {
                            this._clusterIncidentSeq = 0;
                            this._clusterIncidentsById = null;
                        }
                        this._clusterIncidentEpoch = payload.epoch;
                    }
                    const incidents = Array.isArray(payload.incidents) ? payload.incidents : [];
                    if (!this._clusterIncidentsById) this._clusterIncidentsById = new Map();
                    for (const incident of incidents) {
                        if (incident && incident.id) {
                            this._clusterIncidentsById.set(incident.id, incident);
                        }
                    }
                    if (typeof payload.latest_seq === 'number' && payload.latest_seq > since) {
                        this._clusterIncidentSeq = payload.latest_seq;
                    }
                    this.clusterIncidents = Array.from(this._clusterIncidentsById.values())
                        .sort((a, b) => (b.seq || 0) - (a.seq || 0));
                } catch (error) {
                    // A failed poll must leave the existing incident state alone.
                }
            },

            async dismissClusterIncident(incidentId) {
                if (!incidentId) return;
                try {
                    const response = await fetch(
                        `/admin/api/cluster/incidents/${encodeURIComponent(incidentId)}/dismiss`,
                        { method: 'POST' },
                    );
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) return;
                    // Reflect the server's decision locally: the record keeps
                    // its seq, so the cursor will not re-send it — update the
                    // merged copy rather than waiting for a full reload.
                    const existing = this._clusterIncidentsById
                        ? this._clusterIncidentsById.get(incidentId)
                        : null;
                    if (existing && !existing.dismissed_at) {
                        existing.dismissed_at = Date.now() / 1000;
                        this.clusterIncidents = Array.from(this._clusterIncidentsById.values())
                            .sort((a, b) => (b.seq || 0) - (a.seq || 0));
                    }
                } catch (error) {
                    // Leave the row visible; dismissal is retryable.
                }
            },

            clusterActiveIncidents() {
                return (this.clusterIncidents || []).filter((incident) => !incident.dismissed_at);
            },

            clusterIncidentAge(ts) {
                if (!ts) return '';
                const seconds = Math.max(0, Math.floor(Date.now() / 1000 - ts));
                if (seconds < 60) return `${seconds}s ago`;
                if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
                if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
                return `${Math.floor(seconds / 86400)}d ago`;
            },

            // The error-banner X posts a dismissal for the matching incident
            // (when one exists) instead of only blanking client state, so the
            // dismissal is server-owned and holds across browser reloads.
            async dismissClusterErrorBanner() {
                const displayed = this.clusterDisplayedError();
                this.clusterError = '';
                this.clusterConnectionError = '';
                this.dismissClusterGuidance();
                if (!displayed) return;
                const match = this.clusterActiveIncidents().find(
                    (incident) => incident.message === displayed,
                );
                if (match) await this.dismissClusterIncident(match.id);
            },

            async runClusterWorkerSmoke() {
                if (this.clusterWorkerRunning) return;
                this.clusterWorkerRunning = true;
                this.clusterWorkerResult = null;
                this.clusterError = '';
                try {
                    const response = await fetch('/admin/api/cluster/worker-smoke?timeout=5', {
                        method: 'POST',
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Worker check failed'));
                    }
                    this.clusterWorkerResult = await response.json();
                } catch (error) {
                    this.clusterError = error?.message || 'Worker check failed';
                } finally {
                    this.clusterWorkerRunning = false;
                }
            },

            async runClusterCollectiveSmoke() {
                if (this.clusterCollectiveRunning) return;
                this.clusterCollectiveRunning = true;
                this.clusterCollectiveResult = null;
                this.clusterError = '';
                try {
                    const response = await fetch('/admin/api/cluster/collective-smoke?timeout=20', {
                        method: 'POST',
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'MLX collective check failed'));
                    }
                    this.clusterCollectiveResult = await response.json();
                } catch (error) {
                    this.clusterError = error?.message || 'MLX collective check failed';
                } finally {
                    this.clusterCollectiveRunning = false;
                }
            },

            async runClusterPipelineSmoke() {
                if (this.clusterPipelineRunning) return;
                this.clusterPipelineRunning = true;
                this.clusterPipelineResult = null;
                this.clusterError = '';
                try {
                    const response = await fetch('/admin/api/cluster/pipeline-smoke?timeout=30', {
                        method: 'POST',
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Pipeline graph check failed'));
                    }
                    this.clusterPipelineResult = await response.json();
                } catch (error) {
                    this.clusterError = error?.message || 'Pipeline graph check failed';
                } finally {
                    this.clusterPipelineRunning = false;
                }
            },

            async loadClusterDeployments() {
                this.loadClusterPeerHealth();
                this.clusterDeploymentsError = '';
                try {
                    const response = await fetch('/admin/api/cluster/deployments');
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Could not load deployments'));
                    }
                    const payload = await response.json();
                    this.clusterDeployments = payload.deployments || [];
                    this.clusterDeploymentsError = payload.load_error || '';
                } catch (error) {
                    this.clusterDeploymentsError = error?.message || 'Could not load deployments';
                }
            },

            // A probe that failed keeps failing until someone installs keys or
            // plugs a cable back in. Retrying every ten seconds regardless
            // spams both Macs' SSH logs; back off instead, and let a manual
            // action or a completed pairing resume immediately.
            clusterProbeBackoffActive() {
                return Date.now() < Number(this._clusterProbeHoldUntilMs || 0);
            },

            resetClusterProbeBackoff() {
                this._clusterProbeFailureCount = 0;
                this._clusterProbeHoldUntilMs = 0;
            },

            _escalateClusterProbeBackoff() {
                const failures = Number(this._clusterProbeFailureCount || 0) + 1;
                this._clusterProbeFailureCount = failures;
                const delay = Math.min(60000, 10000 * 2 ** (failures - 1));
                this._clusterProbeHoldUntilMs = Date.now() + delay;
            },

            async probeClusterPeer() {
                if (this.clusterPeerProbeLoading) return;
                const ssh = this.clusterPeerSsh.trim();
                if (!ssh) {
                    this.clusterError = 'Enter the worker SSH name first';
                    return;
                }
                this.clusterPeerProbeLoading = true;
                this.clusterPeerProbe = null;
                try {
                    const request = { ssh };
                    if (this.clusterLocalIp.trim()) request.route_to = this.clusterLocalIp.trim();
                    const response = await fetch('/admin/api/cluster/peer-probe', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(request),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Peer probe failed'));
                    }
                    const result = await response.json();
                    // Keep the previous actionable failure mounted while an
                    // automatic retry is in flight. Clearing it before SSH
                    // answers made the whole page jump every ten seconds.
                    this.clusterError = '';
                    this.clusterConnectionError = '';
                    this.dismissClusterGuidance();
                    this.resetClusterProbeBackoff();
                    this.clusterPeerProbe = result;
                    this.clusterPeerProbes = {
                        ...(this.clusterPeerProbes || {}),
                        [ssh]: result,
                    };
                    this.saveClusterKnownNodes();
                    const status = result.status || {};
                    const node = status.node || {};
                    if (result.bootstrap_required) {
                        // Repeat what the probe measured. This used to assert
                        // "not installed" for every bootstrap_required result,
                        // including peers whose runtime it merely could not
                        // check — which is what surfaced the wrong guidance in
                        // #2680.
                        const measured = (result.runtime_mismatches || [])
                            .find((entry) => Boolean(entry));
                        this.clusterConnectionError =
                            `${node.hostname || ssh} is online, but `
                            + (measured || 'its oMLX worker runtime is not installed yet.');
                        this.explainClusterError(this.clusterConnectionError);
                    }
                    if (this.clusterPlanNodes[1]) {
                        const planned = this.clusterPlanNodes[1];
                        const previousId = String(planned.ssh || '').trim() === ssh
                            ? planned.node_id
                            : '';
                        const deployedId = (this.clusterDeployments?.[0]?.hosts || [])
                            .find(host => host.ssh === ssh)?.node_id;
                        const sshHostname = ssh.split('@').pop();
                        planned.ssh = ssh;
                        planned.node_id = this.clusterNodeId(
                            deployedId,
                            previousId,
                            node.hostname,
                            sshHostname,
                            'worker-1',
                        );
                        const exactCapacity = Number(
                            node.admission_ceiling_bytes
                            || node.recommended_working_set_bytes
                            || 0
                        );
                        if (exactCapacity > 0) {
                            this.clusterPlanNodes[1].capacity_gib = Number(
                                (exactCapacity / (1024 ** 3)).toFixed(2)
                            );
                            this.clusterPlanNodes[1].capacity_bytes = exactCapacity;
                        }
                    }
                    await this.$nextTick();
                    this.invalidateClusterPlan();
                    // Now that both ends are known, ask them where they answer
                    // rather than leaving two addresses to be typed.
                    await this.loadClusterFabric();
                } catch (error) {
                    const message = error?.message || 'Peer probe failed';
                    this.clusterError = message;
                    this.clusterConnectionError = message;
                    this._escalateClusterProbeBackoff();
                } finally {
                    this.clusterPeerProbeLoading = false;
                }
            },

            async loadClusterPeerHardware() {
                const peers = this.clusterWorkerPeers();
                if (!peers.length) return;
                const probes = { ...(this.clusterPeerProbes || {}) };
                if (this.clusterPeerProbe && this.clusterPeerSsh.trim()) {
                    probes[this.clusterPeerSsh.trim()] = this.clusterPeerProbe;
                }
                await Promise.all(peers.map(async peer => {
                    const ssh = String(peer?.ssh || '').trim();
                    // Browser-cached hardware is a display hint only. It
                    // intentionally omits runtime paths and must not suppress
                    // the live probe needed to launch a heterogeneous host.
                    if (
                        !ssh
                        || (probes[ssh]?.status?.node && !probes[ssh]?.cached)
                    ) return;
                    try {
                        const request = { ssh };
                        if (this.clusterLocalIp.trim()) {
                            request.route_to = this.clusterLocalIp.trim();
                        }
                        const response = await fetch('/admin/api/cluster/peer-probe', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(request),
                        });
                        if (!response.ok) return;
                        probes[ssh] = await response.json();
                    } catch (error) {
                        // Hardware identity is progressive enhancement. A peer
                        // health or launch check will surface an actual outage.
                        console.debug(`Could not read hardware for ${ssh}`, error);
                    }
                }));
                this.clusterPeerProbes = probes;
                this.saveClusterKnownNodes();
            },

            async discoverClusterPeers() {
                if (this.clusterDiscoveryLoading) return;
                this.clusterDiscoveryLoading = true;
                this.clusterDiscoveryWarning = '';
                try {
                    const response = await fetch('/admin/api/cluster/discover');
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Peer discovery failed'));
                    }
                    const result = await response.json();
                    const remembered = this.clusterDiscoveredPeers || [];
                    const merged = new Map(
                        remembered.map(peer => [
                            peer.ssh,
                            { ...peer, cached: true },
                        ])
                    );
                    (result.peers || []).forEach(peer => {
                        merged.set(peer.ssh, { ...merged.get(peer.ssh), ...peer, cached: false });
                    });
                    this.clusterDiscoveredPeers = [...merged.values()];
                    this.clusterDiscoveryWarning = result.warning || '';
                    this.saveClusterKnownNodes();
                    if (result.pairing_token) {
                        this.clusterPairingToken = result.pairing_token;
                    }
                } catch (error) {
                    if (this.clusterDiscoveredPeers === null) {
                        this.clusterDiscoveredPeers = [];
                    }
                    this.clusterDiscoveryWarning = error?.message || 'Peer discovery failed';
                } finally {
                    this.clusterDiscoveryLoading = false;
                }
            },

            syncClusterNodesFromPeers() {
                // Don't let a background status poll rebuild the node set (and
                // reset selections / invalidate the plan) while an activation is
                // in flight — startCluster() runs a multi-step autoconfigure →
                // stage → activate sequence and a mid-flight resync can discard
                // its own in-progress proposal.
                if (this.clusterAutoconfigureLoading
                    || this.clusterActivationLoading
                    || this.clusterLinkSetupLoading) {
                    // Report the skip so callers holding a pending-sync flag
                    // keep it armed for the next tick instead of dropping it.
                    return false;
                }
                this.clusterModelInventory = null;
                this.clusterCatalogue = null;
                const gib = 1024 ** 3;
                const localCapacityBytes = Number(
                    this.clusterStatus?.node?.admission_ceiling_bytes
                    || this.clusterStatus?.node?.recommended_working_set_bytes
                    || 0
                );
                const localCapacity = localCapacityBytes / gib;
                const existing = new Map(
                    (this.clusterPlanNodes || []).map(node => [node.node_id, node])
                );
                const existingBySsh = new Map(
                    (this.clusterPlanNodes || [])
                        .filter(node => String(node.ssh || '').trim())
                        .map(node => [String(node.ssh).trim(), node])
                );
                const deployedBySsh = new Map(
                    (this.clusterDeployments?.[0]?.hosts || [])
                        .filter(host => String(host.ssh || '').trim())
                        .map(host => [String(host.ssh).trim(), host])
                );
                const localName = this.clusterNodeId(
                    deployedBySsh.get('127.0.0.1')?.node_id,
                    existingBySsh.get('127.0.0.1')?.node_id,
                    this.clusterStatus?.node?.hostname,
                    'this-mac',
                );
                const local = existingBySsh.get('127.0.0.1')
                    || existing.get(localName)
                    || this.clusterPlanNodes?.[0]
                    || {};
                const nodes = [{
                    ...local,
                    key: local.key || 1,
                    node_id: localName,
                    ssh: '127.0.0.1',
                    capacity_gib: localCapacity > 0
                        ? Number(localCapacity.toFixed(2))
                        : Number(local.capacity_gib || 0),
                    capacity_bytes: localCapacityBytes > 0
                        ? localCapacityBytes
                        : Number(local.capacity_bytes || 0),
                    reserve_gib: Number(local.reserve_gib || 0),
                    role: local.role || 'workstation',
                    accelerator: this.clusterStatus?.node?.accelerator || 'metal',
                    accelerator_vendor:
                        this.clusterStatus?.node?.accelerator_vendor || 'apple',
                    memory_kind: this.clusterStatus?.node?.memory_kind || 'unified',
                    fabric_kind: this.clusterStatus?.node?.fabric_kind || '',
                    fabric_group_id:
                        this.clusterStatus?.node?.fabric_group_id || '',
                    fabric_verified: Boolean(
                        this.clusterStatus?.node?.fabric_verified
                    ),
                    python_executable:
                        this.clusterStatus?.runtime?.python_executable || '',
                }];
                this.clusterWorkerPeers().forEach((peer, index) => {
                    const previousBySsh = existingBySsh.get(peer.ssh);
                    const hardware = this.clusterPeerProbes?.[peer.ssh]?.status?.node || {};
                    const sshHostname = String(peer.ssh || '').trim().split('@').pop();
                    const nodeId = this.clusterNodeId(
                        deployedBySsh.get(peer.ssh)?.node_id,
                        peer.node_id,
                        previousBySsh?.node_id,
                        peer.name,
                        hardware.hostname,
                        sshHostname,
                        `worker-${index + 1}`,
                    );
                    const namedPrevious = existing.get(nodeId);
                    const previous = previousBySsh
                        || (
                            namedPrevious && !String(namedPrevious.ssh || '').trim()
                                ? namedPrevious
                                : {}
                        );
                    const peerRuntime = this.clusterPeerProbes?.[peer.ssh]?.status?.runtime || {};
                    const exactCapacityBytes = Number(
                        hardware.admission_ceiling_bytes
                        || peer.admission_ceiling_bytes
                        || previous.capacity_bytes
                        || 0
                    );
                    const displayCapacityBytes = exactCapacityBytes || Number(
                        hardware.recommended_working_set_bytes
                        || peer.recommended_working_set_bytes
                        || 0
                    );
                    nodes.push({
                        ...previous,
                        key: previous.key || (index + 2),
                        node_id: nodeId,
                        ssh: peer.ssh,
                        // Unknown is shown as unknown. The old 64 - 8 GiB
                        // placeholder became a confident-looking “56 GiB
                        // usable” warning whenever SSH had not answered yet.
                        capacity_gib: displayCapacityBytes > 0
                            ? Number((displayCapacityBytes / gib).toFixed(2))
                            : 0,
                        capacity_bytes: exactCapacityBytes,
                        reserve_gib: displayCapacityBytes > 0
                            ? Number(previous.reserve_gib || 0)
                            : 0,
                        role: 'headless',
                        accelerator: hardware.accelerator
                            || peer.accelerator
                            || 'metal',
                        accelerator_vendor: hardware.accelerator_vendor
                            || peer.accelerator_vendor
                            || 'apple',
                        memory_kind: hardware.memory_kind
                            || peer.memory_kind
                            || 'unified',
                        fabric_kind: hardware.fabric_kind
                            || peer.fabric_kind
                            || '',
                        fabric_group_id: hardware.fabric_group_id
                            || peer.fabric_group_id
                            || '',
                        fabric_verified: Boolean(
                            hardware.fabric_verified || peer.fabric_verified
                        ),
                        python_executable: peerRuntime.python_executable
                            || peer.python_executable
                            || previous.python_executable
                            || '',
                    });
                });
                const connectxCandidates = nodes.filter(node => (
                    node.accelerator === 'cuda'
                    && node.fabric_kind === 'connectx-7'
                    && !node.fabric_group_id
                ));
                if (connectxCandidates.length === 2) {
                    connectxCandidates.forEach(node => {
                        node.fabric_group_id = 'connectx-7-auto-pair';
                        node.fabric_verified = false;
                    });
                }
                // Only invalidate a built plan when the node set actually
                // changed. The 2s/10s cluster status poll calls this on every
                // tick; invalidating unconditionally nulled clusterPlan and its
                // signature, which dead-buttoned "Activate manual plan"
                // (clusterActivationReady requires a non-null plan whose
                // signature still matches) and erased error banners every poll.
                // #2721 stopped the poll from POSTing /plan but left this
                // client-side wipe in place.
                const nodesChanged =
                    JSON.stringify(nodes)
                    !== JSON.stringify(this.clusterPlanNodes ?? []);
                this.clusterPlanNodes = nodes;
                this._clusterNodeKey = Math.max(
                    this._clusterNodeKey,
                    ...nodes.map(node => Number(node.key) || 0),
                );
                this.normalizeClusterTensorParallelSize();
                if (nodesChanged) {
                    this.invalidateClusterPlan();
                }
                return true;
            },

            // Turn every unambiguous fast-link discovery into the default
            // cluster. A deployment remains authoritative on restart; without
            // one, all RDMA peers are selected instead of forcing a three-Mac
            // user to choose one and silently ignore the rest.
            async initializeClusterSetup({ preview = true } = {}) {
                const deployment = (this.clusterDeployments || [])[0] || null;
                if (
                    this._clusterKnownNodesNeedsSync
                    && this.clusterStatus
                    && this.clusterWorkerPeers().length
                ) {
                    // Clear the flag only when the sync actually ran: it
                    // skips itself during activation loads, and dropping the
                    // flag on a skipped run lost the joining node until the
                    // next join event re-armed it.
                    if (this.syncClusterNodesFromPeers() === true) {
                        this._clusterKnownNodesNeedsSync = false;
                    }
                }
                if (!this.clusterWorkerPeers().length) {
                    const deployedPeers = (deployment?.hosts || []).filter(
                        host => !['127.0.0.1', 'localhost', '::1'].includes(host.ssh)
                    );
                    const discovered = this.clusterDiscoveredPeers || [];
                    const fastPeers = discovered.filter(
                        peer => peer.rdma_available || peer.transport === 'rdma'
                    );
                    const omlxPeers = discovered.filter(
                        peer => peer.service === 'oMLX Distributed'
                    );
                    const automaticPeers = fastPeers.length
                        ? fastPeers
                        : (omlxPeers.length
                            ? omlxPeers
                            : (discovered.length === 1 ? discovered : []));
                    const preferred = deployedPeers.length
                        ? deployedPeers.map(host =>
                            discovered.find(peer => peer.ssh === host.ssh)
                            || { ssh: host.ssh, name: host.node_id })
                        : automaticPeers;
                    if (preferred.length) {
                        this.clusterSelectedPeers = preferred;
                        this.clusterPeerSsh = preferred[0].ssh;
                        this.syncClusterNodesFromPeers();
                        if (!this.clusterProbeBackoffActive()) {
                            await this.probeClusterPeer();
                            await this.loadClusterTransports();
                        }
                    }
                } else if (!this.clusterPeerProbe
                        && !this.clusterProbeBackoffActive()) {
                    await this.probeClusterPeer();
                    await this.loadClusterTransports();
                }
                await this.loadClusterPeerHardware();
                // The first synchronization can run before SSH answers. Apply
                // the now-live accelerator identity, ConnectX membership, and
                // per-host Python paths before memory/model API requests.
                if (this.clusterWorkerPeers().length) {
                    this.syncClusterNodesFromPeers();
                }
                // A remembered peer skips the selection branch above on page
                // reload. Its budgets still need to be measured: keeping the
                // old 8 GiB form defaults here made the workstation look as if
                // it could safely donate almost all of its memory.
                // While the probe is backing off after failures, budgets would
                // fail over the same SSH path, so they wait for the same hold.
                if (this.clusterWorkerPeers().length
                        && !this.clusterProbeBackoffActive()
                        && !this._clusterBudgetsMeasured) {
                    await this.measureClusterBudgets();
                }

                if (!this.clusterModelInventory
                    && !this.clusterModelInventoryLoading) {
                    await this.loadClusterModelInventory();
                }
                if (!this.clusterPlanModelPath.trim()) {
                    let path = deployment?.model || '';
                    let source = '127.0.0.1';
                    let targetContext = null;
                    let targetContextMode = null;
                    if (!path) {
                        try {
                            const saved = window.localStorage.getItem(
                                'omlx.cluster.selectedModel'
                            ) || '';
                            try {
                                const parsed = JSON.parse(saved);
                                path = parsed.model_path || '';
                                source = parsed.model_source || source;
                                targetContext = Number(
                                    parsed.target_context_tokens || 0
                                ) || null;
                                targetContextMode =
                                    parsed.context_mode
                                    || (targetContext ? 'manual' : null);
                            } catch (_) {
                                // Older builds stored only the path.
                                path = saved;
                            }
                        } catch (_) {
                            path = '';
                        }
                    }
                    const candidates = this.clusterModelCandidates();
                    const model = candidates.find(
                        item => item.model_path === path
                            && (!source || item.model_source === source)
                    ) || candidates.find(
                        item => item.model_path === path
                    ) || candidates.find(item => item.is_default);
                    if (model) {
                        this.selectClusterModel(
                            model,
                            targetContext,
                            targetContextMode,
                        );
                    }
                }
                if (this.clusterPeerSsh.trim()
                    && this.clusterModelCandidates().length
                    && !this.clusterCatalogue
                    && !this.clusterCatalogueLoading) {
                    await this.loadClusterCatalogue();
                }
                if (!this.clusterPlanModelPath.trim()) {
                    const recommended = this.clusterRecommendedModels()[0] || null;
                    if (recommended) this.selectClusterModel(recommended);
                }
                this.normalizeClusterTensorParallelSize();
                if (preview) await this.previewClusterWeightBalance();
            },

            // 2-second "Copied!" affordance on the exchange token, matching
            // the wired-limit copy button. Reset by the same setTimeout so
            // rapid clicks are harmless.
            copyClusterExchangeToken() {
                if (!this.clusterExchangeToken) return;
                this.copyToClipboard(this.clusterExchangeToken);
                this.clusterExchangeTokenCopied = true;
                setTimeout(() => { this.clusterExchangeTokenCopied = false; }, 2000);
            },

            generateClusterPairingSecret() {
                const bytes = new Uint8Array(18);
                window.crypto.getRandomValues(bytes);
                this.clusterPairingSecret = Array.from(
                    bytes,
                    value => value.toString(16).padStart(2, '0'),
                ).join('');
                // Tokens are bound to the secret that authenticated them.
                this.clusterPairingToken = null;
                this.clusterExchangeToken = null;
                this.clusterPairingSecretCopied = false;
            },

            copyClusterPairingSecret() {
                if (this.clusterPairingSecret.length < 16) return;
                this.copyToClipboard(this.clusterPairingSecret);
                this.clusterPairingSecretCopied = true;
                setTimeout(() => {
                    this.clusterPairingSecretCopied = false;
                }, 2000);
            },

            async generatePairingToken() {
                if (this.clusterPairingTokenLoading || this.clusterPairingSecret.length < 16) return;
                this.clusterPairingTokenLoading = true;
                try {
                    const response = await fetch('/admin/api/cluster/pairing-token', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ shared_secret: this.clusterPairingSecret }),
                    });
                    if (response.ok) {
                        const result = await response.json();
                        this.clusterPairingToken = result.pairing_token;
                    }
                } catch (error) {
                    console.error('Pairing token generation failed:', error);
                } finally {
                    this.clusterPairingTokenLoading = false;
                }
            },

            async verifyPairingToken(token) {
                try {
                    const response = await fetch('/admin/api/cluster/verify-pairing-token', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            token: token,
                            shared_secret: this.clusterPairingSecret,
                        }),
                    });
                    if (response.ok) {
                        const result = await response.json();
                        return result.valid;
                    }
                } catch (error) {
                    console.error('Pairing token verification failed:', error);
                }
                return false;
            },

            async loadClusterSshKey() {
                if (this.clusterSshKeyLoading) return;
                this.clusterSshKeyLoading = true;
                try {
                    const response = await fetch('/admin/api/cluster/ssh-key');
                    if (response.ok) {
                        this.clusterSshKey = await response.json();
                    } else {
                        this.clusterSshKey = { available: false };
                    }
                } catch (error) {
                    this.clusterSshKey = { available: false, error: error.message };
                }
                this.clusterSshKeyLoading = false;
            },

            async generateClusterSshKey() {
                if (this.clusterSshKeyGenerating) return;
                const overwrite = Boolean(this.clusterSshKey?.available);
                if (overwrite && !window.confirm(
                    'Regenerating this key disconnects every paired worker. '
                    + 'You will need to exchange keys again on every Mac. Continue?'
                )) return;
                this.clusterSshKeyGenerating = true;
                try {
                    const endpoint = '/admin/api/cluster/ssh-key/generate'
                        + (overwrite ? '?overwrite=true' : '');
                    const response = await fetch(endpoint, {
                        method: 'POST',
                    });
                    if (response.ok) {
                        this.clusterSshKey = await response.json();
                        this.clusterExchangeToken = null;
                        this.clusterPeerExchangeToken = '';
                        this.clusterKeyExchangeResult = null;
                        if (overwrite) this.invalidateClusterPeer(true);
                        this.showNotification(
                            overwrite
                                ? 'SSH key regenerated. Pair every worker again before reconnecting.'
                                : 'SSH key generated successfully',
                            'success'
                        );
                    } else {
                        const error = await response.json();
                        this.showNotification('SSH key generation failed: ' + (error.detail || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    this.showNotification('SSH key generation failed: ' + error.message, 'error');
                }
                this.clusterSshKeyGenerating = false;
            },

            async generateKeyExchangeToken(nodeId) {
                if (this.clusterExchangeTokenLoading || this.clusterPairingSecret.length < 16) return;
                this.clusterExchangeTokenLoading = true;
                try {
                    const response = await fetch('/admin/api/cluster/ssh-key/exchange-token', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            node_id: nodeId || '',
                            shared_secret: this.clusterPairingSecret,
                        }),
                    });
                    if (response.ok) {
                        const result = await response.json();
                        this.clusterExchangeToken = result.exchange_token;
                    } else {
                        const error = await response.json();
                        this.showNotification('Key exchange token generation failed: ' + (error.detail || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    this.showNotification('Key exchange token generation failed: ' + error.message, 'error');
                }
                this.clusterExchangeTokenLoading = false;
            },

            async exchangeKeysWithPeer(exchangeToken) {
                const token = (exchangeToken || '').trim();
                if (!token || this.clusterKeyExchangeLoading || this.clusterPairingSecret.length < 16) return;
                this.clusterKeyExchangeLoading = true;
                try {
                    const response = await fetch('/admin/api/cluster/ssh-key/exchange', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            exchange_token: token,
                            shared_secret: this.clusterPairingSecret,
                        }),
                    });
                    if (response.ok) {
                        this.clusterKeyExchangeResult = await response.json();
                        this.clusterPeerExchangeToken = '';
                        this.showNotification('SSH keys exchanged successfully', 'success');
                        // Pairing just changed what a probe would find; retry
                        // now rather than waiting out the failure hold.
                        this.resetClusterProbeBackoff();
                        // Reload SSH key info
                        await this.loadClusterSshKey();
                    } else {
                        const error = await response.json();
                        this.showNotification('Key exchange failed: ' + (error.detail || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    this.showNotification('Key exchange failed: ' + error.message, 'error');
                } finally {
                    this.clusterKeyExchangeLoading = false;
                }
            },

            async storeClusterKeyInKeychain() {
                this.clusterKeychainStoring = true;
                try {
                    const response = await fetch('/admin/api/cluster/ssh-key/store-keychain', {
                        method: 'POST',
                    });
                    if (response.ok) {
                        const result = await response.json();
                        if (result.stored) {
                            this.showNotification('Key fingerprint stored in macOS Keychain', 'success');
                        } else {
                            this.showNotification('Keychain storage unavailable on this system', 'warning');
                        }
                    } else {
                        const error = await response.json();
                        this.showNotification('Keychain storage failed: ' + (error.detail || 'Unknown error'), 'error');
                    }
                } catch (error) {
                    this.showNotification('Keychain storage failed: ' + error.message, 'error');
                }
                this.clusterKeychainStoring = false;
            },

            async selectClusterDiscoveredPeer(peer) {
                this.invalidateClusterPeer(true);
                // A deliberate click outranks the automatic retry hold.
                this.resetClusterProbeBackoff();
                this.clusterSelectedPeers = [peer];
                this.clusterPeerSsh = peer.ssh;
                this.syncClusterNodesFromPeers();
                await this.probeClusterPeer();
                await this.loadClusterTransports();
                await this.measureClusterBudgets();
                await this.loadClusterModelInventory();
                await this.loadClusterCatalogue();
            },

            invalidateClusterPlan() {
                this.clusterPlan = null;
                this.clusterPlanError = '';
                // Setup failures only describe the exact model, context,
                // topology and memory budgets that produced them. Once any
                // planning input changes, keeping that failure visible makes
                // the new selection look rejected before it has even been
                // checked (for example, a 128k refusal labelled as an 8k
                // refusal after the context button was changed).
                this.clusterAutoconfigureError = '';
                this.clusterError = '';
                this.clusterActivationResult = null;
                this.clusterPlanChanges = null;
                this._clusterPlanSignature = '';
                this._clusterPlanRevision =
                    Number(this._clusterPlanRevision || 0) + 1;
            },

            invalidateClusterPeer(clearNetwork = false) {
                this.clusterPeerProbe = null;
                if (this.clusterError === this.clusterConnectionError) {
                    this.clusterError = '';
                }
                this.clusterConnectionError = '';
                this.dismissClusterGuidance();
                if (clearNetwork) {
                    this.clusterPeerProbes = {};
                    // A different peer is a different fabric: its addresses and
                    // its RDMA matrix say nothing about the new one.
                    this.clusterPeerIp = '';
                    this.clusterFabric = null;
                    this.clusterFabricError = '';
                    this.clusterIpsOverridden = false;
                }
                // A different peer is a different machine: its budgets must be
                // re-measured once (the recurring poll skips already-measured
                // topologies to avoid jitter).
                this._clusterBudgetsMeasured = false;
                this.invalidateClusterPlan();
            },

            // Is every rank still answering? A collective cannot proceed
            // without all of them, so a peer that has gone away should show as a
            // stated failure rather than a request that never returns.
            async loadClusterPeerHealth() {
                if (this.clusterPeerHealthLoading) return;
                const peers = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ].filter(Boolean);
                if (peers.length < 2) { this.clusterPeerHealth = null; return; }
                this.clusterPeerHealthLoading = true;
                try {
                    const query = new URLSearchParams({
                        hosts: peers.join(','),
                        deployment_id: (this.clusterDeployments[0] || {}).deployment_id || '',
                    });
                    const response = await fetch(
                        `/admin/api/cluster/peer-health?${query}`
                    );
                    if (response.status === 401) { window.location.href = '/admin'; return; }
                    if (response.ok) this.clusterPeerHealth = await response.json();
                } catch (error) {
                    this.clusterPeerHealth = null;
                } finally {
                    this.clusterPeerHealthLoading = false;
                }
            },

            // Copy only the weight files this Mac's layers need. A pipeline rank
            // loads its own layers, but the files holding them must be on local
            // disk — and copying the whole model to every Mac wastes most of the
            // transfer (24 of 76 shards for a 78-layer model split 56/22).
            async stageClusterModel(proposal = null) {
                if (this.clusterStagingLoading) return;
                const configured = proposal || this.clusterAutoconfigure;
                const staging = configured && configured.staging;
                if (!staging || staging.ready) return;
                if (!configured.activation) {
                    this.clusterError = 'Automatic setup did not return a staging plan.';
                    return false;
                }
                this.clusterStagingLoading = true;
                this.clusterStagingResult = null;
                this.clusterActivationProgress = 'Preparing model files…';
                try {
                    const response = await fetch('/admin/api/cluster/stage', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            activation: configured.activation,
                        }),
                    });
                    if (response.status === 401) { window.location.href = '/admin'; return; }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Staging failed'));
                    }
                    let job = await response.json();
                    this.clusterStagingResult = job;
                    while (!['completed', 'failed'].includes(job.status)) {
                        const nodes = Object.values(job.nodes || {});
                        const complete = nodes.reduce(
                            (sum, node) => sum + Number(node.files_completed || 0), 0
                        );
                        const total = nodes.reduce(
                            (sum, node) => sum + Number(node.files_total || 0), 0
                        );
                        this.clusterActivationProgress = total
                            ? `Copying model files · ${complete} of ${total}`
                            : 'Checking model files on each worker…';
                        await new Promise(resolve => setTimeout(resolve, 500));
                        const status = await fetch(
                            `/admin/api/cluster/stage/${encodeURIComponent(job.job_id)}`
                        );
                        if (status.status === 401) {
                            window.location.href = '/admin';
                            return false;
                        }
                        if (!status.ok) {
                            throw new Error(await this.clusterResponseError(
                                status, 'Could not read staging progress'
                            ));
                        }
                        job = await status.json();
                        this.clusterStagingResult = job;
                    }
                    if (job.status !== 'completed' || !job.ready) {
                        throw new Error(job.error || 'Model staging failed');
                    }
                    await this.autoconfigureCluster();
                    return true;
                } catch (error) {
                    this.clusterError = error?.message || 'Staging failed';
                    return false;
                } finally {
                    this.clusterStagingLoading = false;
                    this.clusterActivationProgress = '';
                }
            },

            clusterStagingNodes() {
                return Object.values(this.clusterStagingResult?.nodes || {});
            },

            clusterStagingNodeProgress(node) {
                if (node.status === 'ready') return 'Ready';
                if (node.status === 'failed') return node.error || 'Copy failed';
                const copied = Number(node.bytes_completed || 0) / (1024 ** 3);
                const total = Number(node.bytes_total || 0) / (1024 ** 3);
                if (total > 0) {
                    return `${copied.toFixed(1)} of ${total.toFixed(1)} GiB`;
                }
                return node.status === 'copying' ? 'Checking files…' : 'Waiting';
            },

            formatClusterStaging(node) {
                const gib = b => (b / (1024 ** 3)).toFixed(1) + ' GiB';
                return `${node.missing_files} of ${node.required_files} files still needed `
                     + `(${gib(node.missing_bytes)}) — saves ${gib(node.saved_bytes)} vs copying the whole model`;
            },

            // Probe the fabric on its own, without planning anything. Fills the
            // server-side cache that peer discovery reads, so the peer list can
            // show transports without paying for SSH on its own request path.
            loadRememberedClusterConfig() {
                this.loadClusterKnownNodes();
                this.loadClusterMemoryAllowances();
                try {
                    const raw = window.localStorage.getItem('omlx.cluster.lastGoodConfig');
                    if (!raw) return null;
                    const saved = JSON.parse(raw);
                    this.clusterLastGoodConfig = saved.activation || null;
                    return saved;
                } catch (error) {
                    return null;
                }
            },

            loadClusterKnownNodes() {
                if (this._clusterKnownNodesHydrated) return;
                this._clusterKnownNodesHydrated = true;
                try {
                    let saved = null;
                    const raw = window.localStorage.getItem('omlx.cluster.knownNodes');
                    if (raw) saved = JSON.parse(raw);
                    if (!saved?.peers?.length) {
                        // Older builds already remembered the last safe
                        // activation. Reuse only its host identities, never its
                        // approval or runtime result.
                        const lastGoodRaw = window.localStorage.getItem(
                            'omlx.cluster.lastGoodConfig'
                        );
                        const lastGood = lastGoodRaw ? JSON.parse(lastGoodRaw) : null;
                        const hosts = lastGood?.activation?.hosts || [];
                        const peers = hosts.filter(host => ![
                            '127.0.0.1',
                            'localhost',
                            '::1',
                        ].includes(host.ssh)).map(host => ({
                            ssh: host.ssh,
                            name: host.node_id || host.ssh,
                            service: 'oMLX Distributed',
                            transport: 'detecting',
                            cached: true,
                        }));
                        if (peers.length) {
                            saved = {
                                peers,
                                selected_ssh: peers.map(peer => peer.ssh),
                                hardware: {},
                            };
                        }
                    }
                    const peers = (saved?.peers || []).filter(peer => (
                        typeof peer?.ssh === 'string'
                        && /^[A-Za-z0-9._@-]{1,255}$/.test(peer.ssh)
                    )).slice(0, 64).map(peer => ({ ...peer, cached: true }));
                    if (!peers.length) return;
                    this.clusterDiscoveredPeers = peers;
                    const bySsh = new Map(peers.map(peer => [peer.ssh, peer]));
                    const selected = (saved.selected_ssh || [])
                        .map(ssh => bySsh.get(ssh))
                        .filter(Boolean);
                    this.clusterSelectedPeers = selected.length ? selected : [peers[0]];
                    this.clusterPeerSsh = this.clusterSelectedPeers[0].ssh;
                    const probes = { ...(this.clusterPeerProbes || {}) };
                    Object.entries(saved.hardware || {}).forEach(([ssh, node]) => {
                        if (!bySsh.has(ssh) || !node || typeof node !== 'object') return;
                        probes[ssh] = {
                            cached: true,
                            status: {
                                node: {
                                    hostname: String(node.hostname || '').slice(0, 255),
                                    chip_name: String(node.chip_name || '').slice(0, 255),
                                    physical_memory_bytes: Number(
                                        node.physical_memory_bytes || 0
                                    ),
                                    recommended_working_set_bytes: Number(
                                        node.recommended_working_set_bytes || 0
                                    ),
                                    admission_ceiling_bytes: Number(
                                        node.admission_ceiling_bytes || 0
                                    ),
                                },
                            },
                        };
                    });
                    this.clusterPeerProbes = probes;
                    this._clusterKnownNodesNeedsSync = true;
                } catch (error) {
                    // A corrupt browser hint is ignored; live discovery remains
                    // the source of truth and will replace it.
                }
            },

            saveClusterKnownNodes() {
                try {
                    const cacheable = peer => ({
                        ssh: String(peer?.ssh || '').trim(),
                        name: String(peer?.name || '').slice(0, 255),
                        service: String(peer?.service || '').slice(0, 128),
                        transport: String(peer?.transport || 'detecting').slice(0, 32),
                        rdma_available: Boolean(peer?.rdma_available),
                        link_speed_gbps: Number(peer?.link_speed_gbps) || null,
                        recommended_working_set_bytes: Number(
                            peer?.recommended_working_set_bytes || 0
                        ),
                        admission_ceiling_bytes: Number(
                            peer?.admission_ceiling_bytes || 0
                        ),
                        physical_memory_bytes: Number(
                            peer?.physical_memory_bytes || 0
                        ),
                        chip_name: String(peer?.chip_name || '').slice(0, 255),
                    });
                    const peers = (this.clusterDiscoveredPeers || [])
                        .filter(peer => (
                            typeof peer?.ssh === 'string'
                            && /^[A-Za-z0-9._@-]{1,255}$/.test(peer.ssh)
                        ))
                        .slice(0, 64)
                        .map(cacheable);
                    if (!peers.length) return;
                    const hardware = {};
                    peers.forEach(peer => {
                        const node = this.clusterPeerProbes?.[peer.ssh]?.status?.node;
                        if (!node) return;
                        hardware[peer.ssh] = {
                            hostname: node.hostname || '',
                            chip_name: node.chip_name || '',
                            physical_memory_bytes: Number(
                                node.physical_memory_bytes || 0
                            ),
                            recommended_working_set_bytes: Number(
                                node.recommended_working_set_bytes || 0
                            ),
                            admission_ceiling_bytes: Number(
                                node.admission_ceiling_bytes || 0
                            ),
                        };
                    });
                    window.localStorage.setItem(
                        'omlx.cluster.knownNodes',
                        JSON.stringify({
                            version: 1,
                            saved_at: new Date().toISOString(),
                            peers,
                            selected_ssh: this.clusterWorkerPeers()
                                .map(peer => peer.ssh),
                            hardware,
                        })
                    );
                } catch (error) {
                    // Caching is a speed optimization. Discovery and the live
                    // compatibility probe remain fully functional without it.
                }
            },

            loadClusterMemoryAllowances() {
                if (this._clusterMemoryAllowancesLoaded) return;
                this._clusterMemoryAllowancesLoaded = true;
                try {
                    const raw = window.localStorage.getItem(
                        'omlx.cluster.memoryAllowances'
                    );
                    if (!raw) return;
                    const saved = JSON.parse(raw);
                    const source = saved?.allowances_gib || {};
                    this.clusterMemoryAllowancesGiB = Object.fromEntries(
                        Object.entries(source).filter(([, value]) => (
                            Number.isFinite(Number(value)) && Number(value) > 0
                        )).map(([nodeId, value]) => [nodeId, Number(value)])
                    );
                    this.clusterMemoryLimitsManual =
                        Object.keys(this.clusterMemoryAllowancesGiB).length > 0;
                } catch (error) {
                    this.clusterMemoryAllowancesGiB = {};
                    this.clusterMemoryLimitsManual = false;
                }
            },

            saveClusterMemoryAllowances() {
                try {
                    const allowances = this.clusterMemoryAllowancesGiB || {};
                    if (!Object.keys(allowances).length) {
                        window.localStorage.removeItem(
                            'omlx.cluster.memoryAllowances'
                        );
                        return;
                    }
                    window.localStorage.setItem(
                        'omlx.cluster.memoryAllowances',
                        JSON.stringify({
                            version: 1,
                            allowances_gib: allowances,
                        })
                    );
                } catch (error) {
                    // Private browsing may refuse persistence. The in-memory
                    // ceiling must still remain authoritative for this session.
                }
            },

            clusterManualMemoryAllowanceGiB(nodeId) {
                const key = String(nodeId || '').trim();
                const allowances = this.clusterMemoryAllowancesGiB || {};
                if (!Object.prototype.hasOwnProperty.call(allowances, key)) {
                    return null;
                }
                const value = Number(allowances[key]);
                return Number.isFinite(value) && value > 0 ? value : null;
            },

            // Is the fabric actually usable? Device presence is not readiness,
            // and each failure state has a different fix — one of which needs
            // admin rights we do not have, so it has to be shown, not hidden.
            async loadClusterLinkStatus() {
                if (this.clusterLinkStatusLoading) return this.clusterLinkStatus;
                const hosts = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ].filter(Boolean);
                if (hosts.length < 2) {
                    this.clusterLinkStatus = null;
                    return null;
                }
                this.clusterLinkStatusLoading = true;
                try {
                    const query = new URLSearchParams({ hosts: hosts.join(',') });
                    const response = await fetch(
                        `/admin/api/cluster/link-status?${query}`
                    );
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (response.ok) {
                        this.clusterLinkStatus = await response.json();
                    }
                } catch (error) {
                    this.clusterLinkStatus = null;
                } finally {
                    this.clusterLinkStatusLoading = false;
                }
                return this.clusterLinkStatus;
            },

            // The fast path is product setup, not a terminal prerequisite.
            // macOS owns the password dialog; this request contains only the
            // paired hosts and cannot smuggle command text to the server.
            async prepareClusterLink() {
                if (this.clusterLinkSetupLoading) return false;
                const hosts = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ].filter(Boolean);
                if (hosts.length < 2) return false;
                const pairs = [];
                const seen = new Set();
                for (const link of (this.clusterTransports?.transports || [])) {
                    if (!['thunderbolt', 'rdma'].includes(link.kind)) continue;
                    const endpoints = [link.source_node_id, link.peer_node_id];
                    if (endpoints.some(host => !hosts.includes(host))) continue;
                    const key = [...endpoints].sort().join('\u0000');
                    if (seen.has(key)) continue;
                    seen.add(key);
                    pairs.push(endpoints);
                }
                if (!pairs.length && hosts.length === 2) {
                    pairs.push([hosts[0], hosts[1]]);
                }
                if (!pairs.length) {
                    this.clusterAutoconfigureError =
                        'oMLX could not identify the physical links between these Macs.';
                    return false;
                }
                this.clusterLinkSetupLoading = true;
                this.clusterAutoconfigureError = '';
                this.clusterActivationProgress = 'Waiting for macOS approval…';
                try {
                    for (let index = 0; index < pairs.length; index += 1) {
                        this.clusterActivationProgress =
                            `Preparing link ${index + 1} of ${pairs.length}…`;
                        const response = await fetch('/admin/api/cluster/link-setup', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ hosts: pairs[index] }),
                        });
                        if (response.status === 401) {
                            window.location.href = '/admin';
                            return false;
                        }
                        if (!response.ok) {
                            throw new Error(await this.clusterResponseError(
                                response, 'Could not configure the Thunderbolt link'
                            ));
                        }
                        this.clusterLinkStatus = await response.json();
                    }
                    await this.loadClusterLinkStatus();
                    await this.loadClusterFabric();
                    return this.clusterLinkStatus.ready === true;
                } catch (error) {
                    this.clusterAutoconfigureError = error?.message
                        || 'Could not configure the Thunderbolt link';
                    return false;
                } finally {
                    this.clusterLinkSetupLoading = false;
                    this.clusterActivationProgress = '';
                }
            },

            clusterModelInventoryHosts() {
                const localName = this.clusterStatus?.node?.hostname || 'This Mac';
                const plannedBySsh = new Map(
                    (this.clusterPlanNodes || [])
                        .filter(node => String(node?.ssh || '').trim())
                        .map(node => [String(node.ssh).trim(), node])
                );
                return [
                    {
                        node_id: localName,
                        ssh: '127.0.0.1',
                        python_executable:
                            this.clusterStatus?.runtime?.python_executable || undefined,
                    },
                    ...this.clusterWorkerPeers().map((peer, index) => {
                        const planned = plannedBySsh.get(peer.ssh) || {};
                        const runtime = this.clusterPeerProbes?.[peer.ssh]
                            ?.status?.runtime || {};
                        return {
                            node_id: this.clusterFriendlyMacName(
                            peer.name || peer.ssh,
                            `Worker ${index + 1}`,
                        ),
                        ssh: peer.ssh,
                            python_executable: planned.python_executable
                                || runtime.python_executable
                                || undefined,
                        };
                    }),
                ];
            },

            async loadClusterModelInventory() {
                if (this.clusterModelInventoryLoading) return;
                const hosts = this.clusterModelInventoryHosts();
                if (!hosts.length) return;
                this.clusterModelInventoryLoading = true;
                this.clusterModelInventoryError = '';
                try {
                    const response = await fetch('/admin/api/cluster/models', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ hosts }),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(
                            response, 'Could not read models from the cluster'
                        ));
                    }
                    this.clusterModelInventory = await response.json();
                    const errors = this.clusterModelInventory?.errors || [];
                    this.clusterModelInventoryError = errors
                        .map(item => `${item.node_id}: ${item.detail}`)
                        .join(' · ');
                } catch (error) {
                    this.clusterModelInventoryError =
                        error?.message || 'Could not read models from the cluster';
                } finally {
                    this.clusterModelInventoryLoading = false;
                }
            },

            clusterAllModels() {
                const inventory = this.clusterModelInventory?.models;
                return Array.isArray(inventory) && inventory.length
                    ? inventory
                    : (this.models || []).map(model => ({
                        ...model,
                        model_source: '127.0.0.1',
                        source_node_id:
                            this.clusterStatus?.node?.hostname || 'This Mac',
                        locations: [{
                            node_id: this.clusterStatus?.node?.hostname || 'This Mac',
                            ssh: '127.0.0.1',
                            model_path: model.model_path,
                            estimated_size: model.estimated_size || 0,
                        }],
                        location_count: 1,
                    }));
            },

            clusterModelCandidates() {
                const query = String(this.clusterModelSearch || '').trim().toLowerCase();
                return this.clusterAllModels()
                    .filter(model => {
                        if (!model?.model_path) return false;
                        if (!['llm', 'vlm', null, undefined].includes(model.model_type)) {
                            return false;
                        }
                        if (!query) return true;
                        return `${model.id} ${model.model_path} ${model.model_type || ''} ${this.clusterModelHostsLabel(model)}`
                            .toLowerCase()
                            .includes(query);
                    });
            },

            clusterModelHostsLabel(model) {
                const names = (model?.locations || [])
                    .map(location => this.clusterFriendlyMacName(location.node_id))
                    .filter(Boolean);
                if (!names.length) {
                    return model?.source_node_id
                        ? `On ${this.clusterFriendlyMacName(model.source_node_id)}`
                        : 'On this Mac';
                }
                if (names.length === 1) return `On ${names[0]}`;
                if (names.length === this.clusterModelInventoryHosts().length) {
                    return 'On every Mac';
                }
                return `On ${names.join(' + ')}`;
            },

            // The server catalogue is already ordered by the strongest model
            // this exact pair can run. Keep that signal ahead of incidental
            // repository naming, then fold in the choices the user made in
            // the regular oMLX model manager.
            clusterModelOptions() {
                const fitOrder = new Map(
                    (this.clusterCatalogue?.models || []).map(
                        (fit, index) => [fit.model_path, index]
                    )
                );
                const score = model => {
                    const fit = this.clusterCatalogueFit(model.model_path);
                    if (fit?.fits === false) return -1000000;
                    let value = 0;
                    if (fit?.fits) {
                        value += 100000 - ((fitOrder.get(model.model_path) || 0) * 1000);
                    }
                    if (model.model_path === this.clusterPlanModelPath.trim()) value += 600;
                    if (model.is_default) value += 500;
                    if (model.is_favorite) value += 400;
                    if (model.loaded) value += 300;
                    if (!this.clusterCatalogue) {
                        value += Math.min(Number(model.estimated_size || 0) / (1024 ** 3), 250);
                    }
                    return value;
                };
                return this.clusterModelCandidates()
                    .sort((left, right) => {
                        const difference = score(right) - score(left);
                        if (difference) return difference;
                        return this.clusterModelDisplayName(left).localeCompare(
                            this.clusterModelDisplayName(right)
                        );
                    });
            },

            clusterRecommendedModels() {
                if (String(this.clusterModelSearch || '').trim()) return [];
                const options = this.clusterModelOptions();
                const hasFitData = Boolean((this.clusterCatalogue?.models || []).length);
                return options
                    .filter(model => {
                        const fit = this.clusterCatalogueFit(model.model_path);
                        return hasFitData ? fit?.fits === true : true;
                    })
                    .slice(0, 3);
            },

            clusterModelGroupLabel(index) {
                if (String(this.clusterModelSearch || '').trim()) {
                    return index === 0 ? 'Search results' : '';
                }
                const recommended = this.clusterRecommendedModels().length;
                if (!recommended) return index === 0 ? 'All models' : '';
                if (index === 0) return 'Recommended for this pool';
                if (index === recommended) return 'All other models';
                return '';
            },

            clusterModelDisplayName(model) {
                const raw = String(model?.display_name || model?.id || 'Unnamed model');
                const parts = raw.split('/');
                return parts[parts.length - 1] || raw;
            },

            clusterModelOwner(model) {
                const raw = String(model?.display_name || model?.id || '');
                const parts = raw.split('/');
                return parts.length > 1 ? parts.slice(0, -1).join('/') : '';
            },

            clusterModelFailureLabel(fit) {
                if (!fit || fit.fits !== false) return '';
                if (fit.failure_kind === 'single_node_only' && fit.standalone_node_id) {
                    return `${this.clusterFriendlyMacName(fit.standalone_node_id)} only`;
                }
                if (fit.failure_kind === 'cannot_split') return 'Cannot split';
                if (fit.failure_kind === 'model_unreadable') return 'Needs attention';
                return 'Does not fit';
            },

            clusterModelFailureDetail(model) {
                const fit = this.clusterCatalogueFit(model?.model_path);
                return fit?.fits === false ? String(fit.reason || '') : '';
            },

            clusterModelContextLabel(model) {
                const fit = this.clusterCatalogueFit(model?.model_path);
                const tokens = Number(fit?.max_context_tokens || 0);
                return fit?.fits === true && tokens > 0
                    ? `Up to ${this.clusterTokens(tokens)} context`
                    : '';
            },

            clusterModelTargetContext(model, requested = null) {
                const declared = Number(model?.model_context_length || 8192);
                const target = Number(requested || declared || 8192);
                const fit = this.clusterCatalogueFit(model?.model_path);
                const maximum = Number(fit?.max_context_tokens || 0);
                return fit?.fits === true && maximum > 0
                    ? Math.min(target, maximum)
                    : target;
            },

            clusterContextMaximumTokens() {
                const model = this.clusterSelectedModel();
                if (!model) return 0;
                const fit = this.clusterCatalogueFit(model.model_path);
                const declared = Number(
                    fit?.declared_context_tokens
                    || model.model_context_length
                    || 0
                );
                const maximum = Number(fit?.max_context_tokens || 0);
                if (fit?.fits !== true || maximum <= 0) {
                    // Catalogue assessment is asynchronous. Until it arrives,
                    // retain the last known-safe choice (8k on first use)
                    // instead of flashing the model's raw native ceiling and
                    // a false "does not fit" error.
                    const safePending = Number(
                        this.clusterTargetContextTokens || 8192
                    );
                    return Math.max(1, Math.min(
                        safePending,
                        declared > 0 ? declared : 8192,
                        1_048_576,
                    ));
                }
                return Math.max(1, Math.min(
                    declared > 0 ? declared : 1_048_576,
                    maximum,
                    1_048_576,
                ));
            },

            clusterContextOptions() {
                const model = this.clusterSelectedModel();
                if (!model) return [];
                const ceiling = this.clusterContextMaximumTokens();
                const current = Number(this.clusterTargetContextTokens || 8192);
                const presets = [
                    8192,
                    32768,
                    65536,
                    131072,
                    262144,
                    524288,
                    current,
                    ceiling,
                ];
                return [...new Set(
                    presets
                        .map(value => Math.round(Number(value)))
                        .filter(value => Number.isFinite(value) && value > 0 && value <= ceiling)
                )].sort((left, right) => left - right);
            },

            async clusterSetAutomaticContext() {
                const model = this.clusterSelectedModel();
                if (!model) return;
                const target = this.clusterContextMaximumTokens();
                if (!Number.isFinite(target) || target <= 0) return;
                const changed = target !== Number(this.clusterTargetContextTokens)
                    || this.clusterContextMode !== 'auto';
                this.clusterContextMode = 'auto';
                this.clusterTargetContextTokens = target;
                if (!changed) return;
                this.clusterMaxKvSize = '';
                try {
                    window.localStorage.setItem(
                        'omlx.cluster.selectedModel',
                        JSON.stringify({
                            model_path: model.model_path,
                            model_source:
                                model.model_source || this.clusterPlanModelSource || '127.0.0.1',
                            target_context_tokens: target,
                            context_mode: 'auto',
                        })
                    );
                } catch (_) {
                    // A private session can still use automatic context now.
                }
                this.clusterAutoconfigure = null;
                this._clusterAutoPlanKey = '';
                this.invalidateClusterPlan();
                await this.previewClusterWeightBalance();
            },

            async clusterSetTargetContext(tokens) {
                const model = this.clusterSelectedModel();
                if (!model) return;
                const target = this.clusterModelTargetContext(
                    model,
                    Math.round(Number(tokens)),
                );
                if (!Number.isFinite(target) || target <= 0) return;
                if (
                    target === Number(this.clusterTargetContextTokens)
                    && this.clusterContextMode === 'manual'
                ) return;
                this.clusterContextMode = 'manual';
                this.clusterTargetContextTokens = target;
                this.clusterMaxKvSize = '';
                try {
                    window.localStorage.setItem(
                        'omlx.cluster.selectedModel',
                        JSON.stringify({
                            model_path: model.model_path,
                            model_source:
                                model.model_source || this.clusterPlanModelSource || '127.0.0.1',
                            target_context_tokens: target,
                            context_mode: 'manual',
                        })
                    );
                } catch (_) {
                    // A private session can still use the selected context now.
                }
                // A context change alters every rank's KV reservation. Never
                // leave the previous autoconfigure result visible as though it
                // described the new selection.
                this.clusterAutoconfigure = null;
                this._clusterAutoPlanKey = '';
                this.invalidateClusterPlan();
                await this.previewClusterWeightBalance();
            },

            clusterContextMemorySummary() {
                const plan = this.clusterWeightPlan();
                if (!plan?.assignments?.length) return null;
                const assignments = plan.assignments || [];
                const weightBytes = assignments.reduce(
                    (sum, item) => sum
                        + Number(item.layer_weight_bytes || 0)
                        + Number(item.fixed_weight_bytes || 0),
                    0,
                );
                const kvBytes = assignments.reduce(
                    (sum, item) => sum + Number(item.kv_cache_bytes || 0),
                    0,
                );
                const usableBytes = assignments.reduce(
                    (sum, item) => sum + Math.max(
                        0,
                        Number(item.capacity_bytes || 0)
                            - Number(item.reserve_bytes || 0),
                    ),
                    0,
                );
                return {
                    targetTokens: Number(
                        plan.cluster?.target_context_tokens
                        || this.clusterTargetContextTokens
                        || 8192
                    ),
                    weightBytes,
                    kvBytes,
                    usedBytes: weightBytes + kvBytes,
                    usableBytes,
                    headroomBytes: Math.max(0, usableBytes - weightBytes - kvBytes),
                };
            },

            clusterContextMemoryNodes() {
                const plan = this.clusterWeightPlan();
                const quickNodes = this.clusterQuickNodes();
                return [...(plan?.assignments || [])]
                    .sort((left, right) => Number(left.rank) - Number(right.rank))
                    .map((assignment, index) => {
                        const weightBytes = Number(assignment.layer_weight_bytes || 0)
                            + Number(assignment.fixed_weight_bytes || 0);
                        const kvBytes = Number(assignment.kv_cache_bytes || 0);
                        const usableBytes = Math.max(
                            1,
                            Number(assignment.capacity_bytes || 0)
                                - Number(assignment.reserve_bytes || 0),
                        );
                        const usedBytes = weightBytes + kvBytes;
                        return {
                            ...assignment,
                            hardwareLabel: this.clusterNodeHardwareLabel(
                                quickNodes[Number(assignment.rank)] || {
                                    name: assignment.node_id,
                                }
                            ),
                            weightBytes,
                            kvBytes,
                            usedBytes,
                            usableBytes,
                            headroomBytes: Math.max(0, usableBytes - usedBytes),
                            weightPercent: Math.max(
                                0,
                                Math.min(100, weightBytes / usableBytes * 100),
                            ),
                            kvPercent: Math.max(
                                0,
                                Math.min(
                                    100 - (weightBytes / usableBytes * 100),
                                    kvBytes / usableBytes * 100,
                                ),
                            ),
                            layerLabel: Number(assignment.tensor_parallel_size || 1) > 1
                                ? `all layers · tensor ${Number(assignment.tensor_parallel_rank || 0) + 1}/${assignment.tensor_parallel_size}`
                                : `layers ${assignment.start_layer}–${assignment.end_layer}`,
                            index,
                        };
                    });
            },

            clusterContextStatusText() {
                if (this.clusterPlanLoading || this.clusterAutoconfigureLoading) {
                    return 'Calculating the exact split…';
                }
                if (this.clusterPlanError || this.clusterAutoconfigureError) {
                    return `Does not fit at ${this.clusterTokens(
                        this.clusterTargetContextTokens
                    )} — choose a smaller context or allow more memory.`;
                }
                const summary = this.clusterContextMemorySummary();
                if (!summary) return 'Checking model weights and KV cache…';
                if (
                    Number(summary.targetTokens)
                    !== Number(this.clusterTargetContextTokens)
                ) {
                    return 'Updating the memory split…';
                }
                return `${this.clusterGiB(summary.usedBytes)} reserved across `
                    + `${summary.usableBytes > 0
                        ? this.clusterGiB(summary.usableBytes)
                        : 'the cluster'} allowed`;
            },

            clusterModelBadge(model) {
                const fit = this.clusterCatalogueFit(model?.model_path);
                if (fit?.fits === false) return this.clusterModelFailureLabel(fit);
                if (model?.model_path === this.clusterPlanModelPath.trim()) return 'Selected';
                const recommended = this.clusterRecommendedModels();
                const recommendedIndex = recommended.findIndex(
                    item => item.model_path === model?.model_path
                );
                if (recommendedIndex === 0) return 'Best for this pool';
                if (fit?.nodes_required > 1) {
                    return `Uses ${fit.nodes_required} devices`;
                }
                if (model?.is_favorite) return 'Favourite';
                if (model?.loaded) return 'Loaded';
                if (model?.is_default) return 'Default';
                if (recommendedIndex > 0) return 'Recommended';
                return '';
            },

            clusterModelBadgeTone(model) {
                const badge = this.clusterModelBadge(model);
                const fit = this.clusterCatalogueFit(model?.model_path);
                if (model?.model_path === this.clusterPlanModelPath.trim()) {
                    return 'bg-white/10 text-white';
                }
                if (fit?.failure_kind === 'single_node_only') {
                    return 'bg-amber-100 text-amber-800';
                }
                if (fit?.fits === false) return 'bg-neutral-200 text-neutral-600';
                if (badge.startsWith('Uses ')) return 'bg-purple-50 text-purple-700';
                if (badge === 'Favourite') return 'bg-amber-50 text-amber-700';
                return 'bg-blue-50 text-blue-700';
            },

            clusterSelectedModelFitLabel() {
                const model = this.clusterSelectedModel();
                if (!model) return '';
                const fit = this.clusterCatalogueFit(model.model_path);
                if (fit?.fits === false) return this.clusterModelFailureLabel(fit);
                if (fit?.nodes_required > 1) {
                    return `Uses ${fit.nodes_required} devices`;
                }
                if (this.clusterRecommendedModels()[0]?.model_path === model.model_path) {
                    return 'Best for this pool';
                }
                if (fit?.fits) return 'Fits easily';
                return '';
            },

            clusterSelectedModel() {
                const path = this.clusterPlanModelPath.trim();
                const source = this.clusterPlanModelSource || '127.0.0.1';
                const candidates = this.clusterAllModels();
                return candidates.find(model =>
                    model.model_path === path
                    && (model.model_source || '127.0.0.1') === source
                ) || candidates.find(model => model.model_path === path) || null;
            },

            clusterPythonExecutableForSsh(ssh) {
                const target = String(ssh || '').trim();
                if (!target) return '';
                if (['127.0.0.1', 'localhost', '::1', 'local'].includes(target)) {
                    return this.clusterStatus?.runtime?.python_executable || '';
                }
                const planned = (this.clusterPlanNodes || []).find(
                    node => String(node?.ssh || '').trim() === target
                );
                if (planned?.python_executable) return planned.python_executable;
                return this.clusterPeerProbes?.[target]?.status?.runtime
                    ?.python_executable || '';
            },

            selectClusterModel(
                model,
                requestedContext = null,
                requestedContextMode = null,
            ) {
                if (!model?.model_path) return;
                this.clusterPlanMode = 'model';
                this.clusterPlanModelPath = model.model_path;
                this.clusterPlanModelSource =
                    model.model_source || '127.0.0.1';
                this.clusterPlanModelSourcePython =
                    model.model_source_python
                    || model.python_executable
                    || this.clusterPythonExecutableForSsh(
                        this.clusterPlanModelSource
                    );
                const fit = this.clusterCatalogueFit(model.model_path);
                if (fit?.fits === true) {
                    const tensorSize = this.clusterFitTensorParallelSize(fit);
                    if (this.clusterTensorParallelOptions().includes(tensorSize)) {
                        // The catalogue arrived at this topology using the
                        // same fit planner. Preview that proven topology rather
                        // than defaulting every model to pipeline-only — an
                        // architecture with tensor support but no pipeline
                        // path otherwise displayed a false refusal.
                        this.clusterPlanTensorParallelSize = tensorSize;
                    }
                }
                this.clusterContextMode =
                    requestedContextMode === 'manual' ? 'manual' : 'auto';
                const pendingAutomaticContext = Number(
                    requestedContext || 8192
                );
                this.clusterTargetContextTokens =
                    this.clusterContextMode === 'auto'
                        ? (
                            fit?.fits === true
                            && Number(fit.max_context_tokens || 0) > 0
                                ? this.clusterContextMaximumTokens()
                                : Math.max(1, Math.min(
                                    pendingAutomaticContext,
                                    Number(model.model_context_length || 8192),
                                    1_048_576,
                                ))
                        )
                        : this.clusterModelTargetContext(
                            model,
                            Number(
                                requestedContext
                                || model.model_context_length
                                || 8192
                            ),
                        );
                // The model-adjacent context picker is the single source of
                // truth. Clear the retired advanced override so an old value
                // cannot make the runtime cache differ from the visible plan.
                this.clusterMaxKvSize = '';
                this.clusterAutoconfigure = null;
                this.clusterCatalogueError = '';
                this.clusterShowModelPicker = false;
                this.clusterWeightTargetsGiB = {};
                this.invalidateClusterPlan();
                try {
                    window.localStorage.setItem(
                        'omlx.cluster.selectedModel',
                        JSON.stringify({
                            model_path: model.model_path,
                            model_source: this.clusterPlanModelSource,
                            model_source_python:
                                this.clusterPlanModelSourcePython || undefined,
                            target_context_tokens:
                                this.clusterTargetContextTokens,
                            context_mode: this.clusterContextMode,
                        })
                    );
                } catch (_) {
                    // Remembering the picker is a convenience only.
                }
                // On first load the catalogue has not yet selected the
                // architecture's viable tensor degree. The initializer will
                // preview immediately after that fit result arrives.
                if (this.clusterCatalogue) this.previewClusterWeightBalance();
            },

            clusterCatalogueFit(modelPath) {
                return (this.clusterCatalogue?.models || []).find(
                    fit => fit.model_path === modelPath
                ) || null;
            },

            clusterTensorParallelOptions() {
                const nodes = Math.max(1, this.clusterPlanNodes.length);
                return nodes > 1 ? [1, nodes] : [1];
            },

            // The catalogue fit is the FEWEST-node plan: a model that fits one
            // Mac reports tensor_parallel_size=1 even when two Macs are wired.
            // Adopting that 1 into a 2-node plan builder made every preview a
            // pipeline plan, which 400s for architectures without the MLX-LM
            // pipeline forward path. Translate the fit into the topology valid
            // for the nodes actually configured.
            clusterFitTensorParallelSize(fit) {
                const nodes = Math.max(1, this.clusterPlanNodes.length);
                const fitted = Number(fit?.tensor_parallel_size || 1);
                if (Number(fit?.nodes_required || 1) >= nodes) return fitted;
                // Force full tensor only when the architecture actually
                // supports it — both flags can be false (unsplittable), and
                // tp=nodes there turns the accurate "cannot combine Macs"
                // refusal into a confusing heads-divisibility error.
                if (
                    fit?.supports_pipeline === false
                    && fit?.supports_tensor_parallel === true
                ) return nodes;
                // Pipeline is possible too: keep whatever is selected.
                return Number(this.clusterPlanTensorParallelSize) || 1;
            },

            normalizeClusterTensorParallelSize() {
                const options = this.clusterTensorParallelOptions();
                const current = Number(this.clusterPlanTensorParallelSize);
                // Snap an out-of-range value to the largest available degree
                // (tensor parallelism) rather than 1 (pipeline) — several
                // architectures (VLMs) support tensor but not the MLX-LM
                // pipeline forward path, so 1 would make the only valid plan
                // fail. This must also fire when the cluster genuinely shrank
                // to one node (options === [1]); leaving a stale multi-way
                // size there sends /plan a world size that cannot divide. The
                // transient-poll wipe this used to cause is prevented
                // upstream: the poll skips resync mid-activation and only
                // rebuilds the node set when it actually changed.
                if (!options.includes(current)) {
                    this.clusterPlanTensorParallelSize =
                        options[options.length - 1];
                    this.invalidateClusterPlan();
                }
                // tp=1 on a multi-node cluster means pipeline. For a model whose
                // architecture cannot pipeline the only valid multi-node plan is
                // full tensor; leaving 1 here guarantees a 400 from /plan on
                // every preview.
                const fit = this.clusterCatalogueFit(
                    this.clusterPlanModelPath.trim()
                );
                if (
                    options.length > 1
                    && Number(this.clusterPlanTensorParallelSize) === 1
                    && fit?.fits === true
                    && fit.supports_pipeline === false
                    && fit.supports_tensor_parallel === true
                ) {
                    this.clusterPlanTensorParallelSize =
                        options[options.length - 1];
                    this.invalidateClusterPlan();
                }
            },

            clusterLiveJobs() {
                return (this.clusterStatus?.runtime_jobs?.jobs || []).filter(job => job.live);
            },

            clusterPrimaryDeployment() {
                const selected = this.clusterPlanModelPath.trim();
                return (this.clusterDeployments || []).find(
                    deployment => deployment.model === selected
                ) || (this.clusterDeployments || [])[0] || null;
            },

            clusterQuickStatus() {
                const selected = this.clusterSelectedModel();
                const deployment = this.clusterPrimaryDeployment();
                const liveJobs = this.clusterLiveJobs();
                const error = this.clusterAutoconfigureError
                    || this.clusterError
                    || this.clusterDeploymentsError;

                if (this.clusterDeactivatingId) {
                    return {
                        key: 'stopping',
                        label: 'Stopping…',
                        detail: 'The workers are finishing their current work safely.',
                        tone: 'blue',
                        busy: true,
                    };
                }
                if (this.clusterActivationLoading) {
                    return {
                        key: 'starting',
                        label: this.clusterActivationProgress || 'Preparing model…',
                        detail: `oMLX is starting the model on ${this.clusterQuickNodes().length} devices.`,
                        tone: 'blue',
                        busy: true,
                    };
                }
                if (this.clusterAutoconfigureLoading || this.clusterLinkSetupLoading) {
                    return {
                        key: 'preparing',
                        label: this.clusterLinkSetupLoading
                            ? 'Connecting the workers…'
                            : 'Preparing the model…',
                        detail: 'Connection, memory, and model split checks are automatic.',
                        tone: 'blue',
                        busy: true,
                    };
                }
                if (deployment) {
                    const deployedModel = this.clusterAllModels().find(
                        model => model.model_path === deployment.model
                    );
                    const name = deployedModel
                        ? this.clusterModelDisplayName(deployedModel)
                        : String(deployment.model || 'The selected model').split('/').pop();
                    if (liveJobs.length) {
                        return {
                            key: 'running',
                            label: `Running on ${this.clusterQuickNodes().length} devices`,
                            detail: `${name} is loaded and available through oMLX.`,
                            tone: 'green',
                            busy: false,
                        };
                    }
                    return {
                        key: 'stopped',
                        label: 'Cluster is stopped',
                        detail: `${name} is configured but its weights are not loaded.`,
                        tone: 'amber',
                        busy: false,
                    };
                }
                if (error) {
                    return {
                        key: 'error',
                        label: 'Needs attention',
                        detail: 'oMLX kept the cluster stopped. Open the message below for the fix.',
                        tone: 'red',
                        busy: false,
                    };
                }
                if (!this.clusterPeerSsh.trim()) {
                    return {
                        key: 'finding',
                        label: 'Finding workers…',
                        detail: 'Keep every worker awake and connected to the cluster network.',
                        tone: 'blue',
                        busy: this.clusterDiscoveryLoading,
                    };
                }
                if (this.clusterPeerProbeLoading
                    || this.clusterFabricLoading
                    || this.clusterLinkStatusLoading
                    || !this.clusterPeerProbe) {
                    return {
                        key: 'checking',
                        label: 'Checking the connection…',
                        detail: 'oMLX is confirming every worker and the fastest shared links.',
                        tone: 'blue',
                        busy: true,
                    };
                }
                // Strictly false, not merely not-true: a payload that predates
                // the runtime fields carries no verdict, and rendering it red
                // would invent a measurement nobody took.
                if (this.clusterPeerProbe.runtime_compatible === false) {
                    const mismatches = Array.isArray(
                        this.clusterPeerProbe.runtime_mismatches
                    ) ? this.clusterPeerProbe.runtime_mismatches.filter(Boolean) : [];
                    // bootstrap_required means oMLX is absent or unverifiable on
                    // that Mac, not that two runtimes disagree. The two need
                    // different actions, so they must not share a headline.
                    if (this.clusterPeerProbe.bootstrap_required) {
                        return {
                            key: 'bootstrap',
                            label: 'Worker runtime setup needed',
                            detail: this.clusterConnectionError
                                || mismatches[0]
                                || 'This worker is reachable, but its oMLX runtime could not be verified.',
                            tone: 'amber',
                            busy: false,
                        };
                    }
                    return {
                        key: 'runtime-mismatch',
                        label: 'Worker runtime mismatch',
                        detail: mismatches.join(' · ')
                            || 'The worker runtime differs from this Mac.',
                        tone: 'red',
                        busy: false,
                    };
                }
                if (this.clusterPeerProbe.runtime_compatible !== true) {
                    return {
                        key: 'runtime-unverified',
                        label: 'Worker runtime not verified',
                        detail: 'This worker is reachable, but oMLX could not verify its runtime yet.',
                        tone: 'amber',
                        busy: false,
                    };
                }
                if (this.clusterCatalogueLoading && !selected) {
                    return {
                        key: 'choosing',
                        label: 'Choosing a model…',
                        detail: 'oMLX is checking which downloaded models fit this accelerator pool.',
                        tone: 'blue',
                        busy: true,
                    };
                }
                if (!selected) {
                    return {
                        key: 'model',
                        label: 'Choose a model',
                        detail: 'Pick a recommendation or search all downloaded models.',
                        tone: 'amber',
                        busy: false,
                    };
                }
                const selectedFit = this.clusterCatalogueFit(selected.model_path);
                if (selectedFit?.fits === false) {
                    return {
                        key: 'model',
                        label: selectedFit.failure_kind === 'single_node_only'
                            ? `${this.clusterFriendlyMacName(
                                selectedFit.standalone_node_id
                            )} only`
                            : (selectedFit.failure_kind === 'cannot_split'
                                ? 'Cannot combine these devices'
                                : 'Choose another model'),
                        detail: selectedFit.reason
                            || 'The selected model cannot run with this cluster configuration.',
                        tone: 'amber',
                        busy: false,
                    };
                }
                return {
                    key: 'ready',
                    label: 'Ready',
                    detail: 'Every worker and the selected model passed the setup checks.',
                    tone: 'green',
                    busy: false,
                };
            },

            clusterQuickStatusTone() {
                return {
                    green: 'bg-green-50 text-green-700 border border-green-200',
                    red: 'bg-red-50 text-red-700 border border-red-200',
                    blue: 'bg-blue-50 text-blue-700 border border-blue-200',
                    amber: 'bg-amber-50 text-amber-700 border border-amber-200',
                }[this.clusterQuickStatus().tone]
                    || 'bg-neutral-50 text-neutral-600 border border-neutral-200';
            },

            clusterPrimaryActionDisabled() {
                const selected = this.clusterSelectedModel();
                const rejected = selected
                    && this.clusterCatalogueFit(selected.model_path)?.fits === false;
                const stopping = this.clusterPrimaryDeployment()
                    && this.clusterLiveJobs().length;
                return Boolean(
                    this.clusterDeactivatingId
                    || this.clusterActivationLoading
                    || this.clusterAutoconfigureLoading
                    || this.clusterLinkSetupLoading
                    || this.clusterPeerProbeLoading
                    || this.clusterFabricLoading
                    || this.clusterLinkStatusLoading
                    || (this.clusterDiscoveryLoading && !this.clusterPeerSsh.trim())
                    || (rejected && !stopping)
                );
            },

            clusterPrimaryBlockReason() {
                // Why the primary button will not act, in words. A disabled
                // button with no reason cost an hour of "it just seems to not
                // launch": the catalogue had rejected the plan and nothing
                // said so.
                if (this.clusterDeactivatingId) return 'Stopping the current deployment…';
                if (this.clusterActivationLoading) return 'Activation is already running.';
                const busy = [
                    ['clusterAutoconfigureLoading', 'measuring the workers'],
                    ['clusterLinkSetupLoading', 'setting up the link'],
                    ['clusterPeerProbeLoading', 'probing the worker'],
                    ['clusterFabricLoading', 'reading the fabric'],
                    ['clusterLinkStatusLoading', 'checking the link'],
                ].find(([flag]) => this[flag]);
                if (busy) return `Still ${busy[1]} — wait a moment, then try again.`;
                if (this.clusterDiscoveryLoading && !this.clusterPeerSsh.trim()) {
                    return 'Still discovering workers on the network.';
                }
                const selected = this.clusterSelectedModel();
                const fit = selected
                    ? this.clusterCatalogueFit(selected.model_path)
                    : null;
                if (fit?.fits === false && !(this.clusterPrimaryDeployment() && this.clusterLiveJobs().length)) {
                    return `This plan cannot launch — ${fit.reason || fit.summary || 'the catalogue rejected it'}. Adjust the split or memory settings.`;
                }
                return '';
            },

            clusterPrimaryActionLabel() {
                if (this.clusterDeactivatingId) return 'Stopping…';
                if (this.clusterActivationLoading) {
                    return this.clusterActivationProgress || 'Starting on every worker…';
                }
                if (this.clusterLinkSetupLoading) return 'Connecting the workers…';
                if (this.clusterAutoconfigureLoading) return 'Preparing the model…';
                if (this.clusterPrimaryDeployment() && this.clusterLiveJobs().length) {
                    return 'Stop';
                }
                if (!this.clusterAllModels().length) return 'Get a model';
                if (!this.clusterPeerSsh.trim()) {
                    return this.clusterDiscoveryLoading
                        ? 'Finding workers…'
                        : 'Find workers';
                }
                const model = this.clusterSelectedModel();
                if (!model) return 'Choose a model';
                if (this.clusterCatalogueFit(model.model_path)?.fits === false) {
                    return 'Choose a cluster-compatible model';
                }
                if (this.clusterError || this.clusterAutoconfigureError) {
                    return `Retry ${this.clusterModelDisplayName(model)} setup`;
                }
                return `Start ${this.clusterModelDisplayName(model)} on ${this.clusterQuickNodes().length} devices`;
            },

            async runClusterPrimaryAction() {
                if (this.clusterPrimaryActionDisabled()) {
                    // Never swallow a click: say what is blocking, in the same
                    // banner server errors use.
                    const reason = this.clusterPrimaryBlockReason();
                    if (reason) this.clusterError = reason;
                    return;
                }
                const deployment = this.clusterPrimaryDeployment();
                if (deployment && this.clusterLiveJobs().length) {
                    await this.deactivateClusterDeployment(deployment.deployment_id);
                    await this.loadClusterRuntime();
                    this.clusterActivationResult = null;
                    return;
                }
                if (!this.clusterAllModels().length) {
                    this.setMainTab('models');
                    return;
                }
                if (!this.clusterPeerSsh.trim()) {
                    this._clusterDiscoveryRefreshCounter = 0;
                    await this.discoverClusterPeers();
                    await this.initializeClusterSetup();
                    return;
                }
                if (!this.clusterSelectedModel()) {
                    if (!this.clusterCatalogue && !this.clusterCatalogueLoading) {
                        await this.loadClusterCatalogue();
                    }
                    this.clusterShowModelPicker = true;
                    return;
                }
                await this.startCluster();
            },

            clusterNodeId(...candidates) {
                const pattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
                for (const candidate of candidates) {
                    const value = String(candidate || '').trim();
                    if (pattern.test(value)) return value;
                }
                return '';
            },

            clusterFriendlyMacName(name, fallback = 'Mac') {
                let value = String(name || '').trim().replace(/\.local$/i, '');
                value = value.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();
                if (!value) return fallback;
                value = value.replace(/\bmacbook\b/i, 'MacBook');
                if (/^(?:omlx\s+on\s+)?(?:mac\s+)?studio$/i.test(value)) {
                    return 'Mac Studio';
                }
                return value;
            },

            clusterNodeHardwareLabel(node) {
                if (Number(node?.memberCount || 0) > 1) {
                    const members = Number(node.memberCount);
                    const memory = Number(node.physicalMemory || 0) / (1024 ** 3);
                    const parts = [
                        `${members}× ${node.accelerator === 'cuda' ? 'NVIDIA CUDA' : 'accelerator'} ${node.fabricVerified ? 'verified pair' : 'pair'}`,
                    ];
                    if (node.chip) parts.push(node.chip);
                    if (memory > 0) parts.push(`${Math.round(memory)} GB pooled`);
                    return parts.join(' · ');
                }
                const parts = [
                    this.clusterFriendlyMacName(node?.name || node?.ssh, 'Mac'),
                ];
                const chip = String(node?.chip || '')
                    .trim()
                    .replace(/^Apple\s+/i, '');
                if (chip) parts.push(chip);
                const physicalBytes = Number(node?.physicalMemory || 0);
                const physicalGiB = Number(node?.physicalGiB || 0);
                const memoryGiB = physicalBytes > 0
                    ? physicalBytes / (1024 ** 3)
                    : physicalGiB;
                if (Number.isFinite(memoryGiB) && memoryGiB > 0) {
                    parts.push(`${Math.round(memoryGiB)} GB`);
                }
                return parts.join(' · ');
            },

            // Kept as a compatibility seam for extensions that used the old
            // Mac-only helper. New UI code calls clusterNodeHardwareLabel.
            clusterMacHardwareLabel(node) {
                return this.clusterNodeHardwareLabel(node);
            },

            clusterNodeRankLabel(node) {
                const ranks = Array.isArray(node?.ranks) && node.ranks.length
                    ? node.ranks
                    : [Number(node?.rank || 0)];
                const rankLabel = ranks.length > 1
                    ? `R${Math.min(...ranks)}–R${Math.max(...ranks)}`
                    : `R${ranks[0]}`;
                if (node?.local) return `Coordinator · ${rankLabel}`;
                if (ranks.length > 1) {
                    return `${node?.fabricVerified ? 'Verified CUDA pair' : 'CUDA pair'} · ${rankLabel}`;
                }
                return `Worker · ${rankLabel}`;
            },

            clusterWorkerPeers() {
                const selected = this.clusterSelectedPeers || [];
                if (selected.length) return selected;
                const ssh = this.clusterPeerSsh.trim();
                if (!ssh) return [];
                const discovered = (this.clusterDiscoveredPeers || []).find(
                    peer => peer.ssh === ssh
                );
                return [discovered || { ssh, name: this.clusterPeerDisplayName() }];
            },

            clusterQuickNodes() {
                const localHardware = this.clusterStatus?.node || {};
                const plannedBySsh = new Map(
                    (this.clusterPlanNodes || [])
                        .filter(node => String(node?.ssh || '').trim())
                        .map(node => [String(node.ssh).trim(), node])
                );
                const localPlan = plannedBySsh.get('127.0.0.1')
                    || this.clusterPlanNodes?.[0]
                    || {};
                const local = {
                    rank: 0,
                    name: this.clusterFriendlyMacName(
                        this.clusterStatus?.node?.hostname,
                        'This Mac',
                    ),
                    ssh: '127.0.0.1',
                    local: true,
                    connected: true,
                    online: true,
                    chip: this.clusterStatus?.node?.chip_name || '',
                    memory: Number(
                        this.clusterStatus?.node?.recommended_working_set_bytes || 0
                    ),
                    physicalMemory: Number(
                        this.clusterStatus?.node?.physical_memory_bytes || 0
                    ),
                    accelerator: localHardware.accelerator || 'metal',
                    acceleratorVendor: localHardware.accelerator_vendor || 'apple',
                    memoryKind: localHardware.memory_kind || 'unified',
                    fabricKind: localPlan.fabric_kind || localHardware.fabric_kind || '',
                    fabricGroup: localPlan.fabric_group_id || localHardware.fabric_group_id || '',
                    fabricVerified: Boolean(
                        localPlan.fabric_verified || localHardware.fabric_verified
                    ),
                };
                const nodes = [local, ...this.clusterWorkerPeers().map((peer, index) => {
                    const ssh = String(peer?.ssh || '').trim();
                    const planned = plannedBySsh.get(ssh) || {};
                    const probe = this.clusterPeerProbes?.[ssh]
                        || (index === 0 ? this.clusterPeerProbe : null);
                    const hardware = probe?.status?.node || {};
                    const compatible = Boolean(probe?.runtime_compatible);
                    return {
                        rank: index + 1,
                        name: this.clusterFriendlyMacName(
                            hardware.hostname || peer.name || ssh,
                            `Worker ${index + 1}`,
                        ),
                        ssh,
                        local: false,
                        connected: Boolean(
                            peer.rdma_available
                            || (peer.transport && peer.transport !== 'detecting')
                            || probe?.ssh_reachable
                            || compatible
                        ),
                        online: (
                            this.clusterPeerHealth?.peers?.find(
                                item => Number(item.rank) === index + 1
                            )?.healthy
                        ) ?? Boolean(
                            peer.rdma_available
                            || (peer.transport && peer.transport !== 'detecting')
                            || probe?.ssh_reachable
                            || compatible
                        ),
                        chip: hardware.chip_name || peer.chip_name || '',
                        memory: Number(
                            hardware.recommended_working_set_bytes
                            || peer.recommended_working_set_bytes
                            || 0
                        ),
                        physicalMemory: Number(
                            hardware.physical_memory_bytes
                            || peer.physical_memory_bytes
                            || 0
                        ),
                        accelerator: hardware.accelerator
                            || peer.accelerator
                            || 'metal',
                        acceleratorVendor: hardware.accelerator_vendor
                            || peer.accelerator_vendor
                            || 'apple',
                        memoryKind: hardware.memory_kind
                            || peer.memory_kind
                            || 'unified',
                        fabricKind: planned.fabric_kind
                            || hardware.fabric_kind
                            || peer.fabric_kind
                            || '',
                        fabricGroup: planned.fabric_group_id
                            || hardware.fabric_group_id
                            || peer.fabric_group_id
                            || '',
                        fabricVerified: Boolean(
                            planned.fabric_verified
                            || hardware.fabric_verified
                            || peer.fabric_verified
                        ),
                        transport: peer.transport,
                        link_speed_gbps: peer.link_speed_gbps,
                    };
                })];
                // Two ConnectX-capable workers with no explicit name are the
                // common dual-Spark topology. Show them as one candidate pair,
                // but keep fabricVerified false until the direct-link probe
                // passes. Larger fabrics require an explicit group ID so
                // unrelated CUDA nodes are never silently collapsed together.
                const implicitConnectx = nodes.filter(node => (
                    node.accelerator === 'cuda'
                    && node.fabricKind === 'connectx-7'
                    && !node.fabricGroup
                ));
                if (implicitConnectx.length === 2) {
                    implicitConnectx.forEach(node => {
                        node.fabricGroup = 'connectx-7-auto-pair';
                    });
                }
                return nodes;
            },

            clusterLogicalNodes() {
                const logical = [];
                const groups = new Map();
                const physical = this.clusterQuickNodes();
                const candidateGroups = new Map();
                physical.forEach(node => {
                    if (!node.fabricGroup) return;
                    const members = candidateGroups.get(node.fabricGroup) || [];
                    members.push(node);
                    candidateGroups.set(node.fabricGroup, members);
                });
                const eligibleGroups = new Set(
                    [...candidateGroups.entries()]
                        .filter(([, members]) => (
                            members.length >= 2
                            && members.every(item => item.accelerator === 'cuda')
                        ))
                        .map(([group]) => group)
                );
                physical.forEach(node => {
                    if (!node.fabricGroup || !eligibleGroups.has(node.fabricGroup)) {
                        logical.push({ ...node, ranks: [node.rank], memberCount: 1 });
                        return;
                    }
                    let composite = groups.get(node.fabricGroup);
                    if (!composite) {
                        composite = {
                            ...node,
                            name: 'CUDA fabric pair',
                            local: false,
                            ranks: [],
                            members: [],
                            memberCount: 0,
                            memory: 0,
                            physicalMemory: 0,
                        };
                        groups.set(node.fabricGroup, composite);
                        logical.push(composite);
                    }
                    composite.ranks.push(node.rank);
                    composite.members.push(node);
                    composite.memberCount += 1;
                    composite.memory += Number(node.memory || 0);
                    composite.physicalMemory += Number(node.physicalMemory || 0);
                    composite.connected = composite.members.every(item => item.connected);
                    composite.online = composite.members.every(item => item.online);
                    composite.fabricVerified = composite.members.every(
                        item => item.fabricVerified
                    );
                    composite.chip = composite.members
                        .map(item => String(item.chip || '').replace(/^NVIDIA\s+/i, ''))
                        .filter(Boolean)
                        .filter((value, index, values) => values.indexOf(value) === index)
                        .join(' + ');
                    const verification = (this.clusterCudaFabricVerifications || {})[
                        this.clusterCudaFabricVerificationKey(composite)
                    ];
                    if (verification?.verified) {
                        composite.fabricVerified = true;
                        composite.fabricVerification = verification;
                        composite.name = 'Verified CUDA pair';
                    }
                });
                return logical;
            },

            clusterCudaFabricVerificationKey(node) {
                return (node?.members || [])
                    .map(member => String(member?.ssh || '').trim())
                    .filter(Boolean)
                    .sort()
                    .join('|');
            },

            clusterCanVerifyCudaSupernode(node) {
                return Boolean(
                    Number(node?.memberCount || 0) === 2
                    && node?.accelerator === 'cuda'
                    && (node.members || []).every(member => member.online)
                    && !this.clusterActivationLoading
                    && !this.clusterNeuralFabricJob()?.live
                );
            },

            clusterCudaFabricRateLabel(node) {
                const result = node?.fabricVerification
                    || (this.clusterCudaFabricVerifications || {})[
                        this.clusterCudaFabricVerificationKey(node)
                    ];
                const rate = Number(result?.payload_bytes_per_second || 0);
                return rate > 0
                    ? `${(rate / (1024 ** 3)).toFixed(1)} GiB/s NCCL`
                    : '';
            },

            async verifyCudaSupernode(node) {
                const key = this.clusterCudaFabricVerificationKey(node);
                if (!key || this.clusterCudaFabricVerificationLoading) return false;
                if (!this.clusterCanVerifyCudaSupernode(node)) {
                    this.clusterCudaFabricVerificationError =
                        'Both CUDA workers must be online before the direct link can be verified.';
                    return false;
                }
                this.clusterCudaFabricVerificationLoading = key;
                this.clusterCudaFabricVerificationError = '';
                try {
                    const response = await fetch('/admin/api/cluster/cuda-fabric/verify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            hosts: node.members.map(member => ({
                                node_id: member.name || member.ssh,
                                ssh: member.ssh,
                            })),
                        }),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return false;
                    }
                    if (!response.ok) {
                        throw new Error(
                            await this.clusterResponseError(
                                response,
                                'ConnectX verification failed'
                            )
                        );
                    }
                    const result = await response.json();
                    this.clusterCudaFabricVerifications = {
                        ...this.clusterCudaFabricVerifications,
                        [key]: result,
                    };
                    if (!result.verified) {
                        this.clusterCudaFabricVerificationError = result.reason
                            || 'NCCL ran, but the direct link was slower than expected.';
                        return false;
                    }
                    const memberSsh = new Set(node.members.map(member => member.ssh));
                    this.clusterPlanNodes.forEach(planNode => {
                        if (!memberSsh.has(String(planNode.ssh || '').trim())) return;
                        planNode.fabric_kind = 'connectx-7';
                        planNode.fabric_group_id = result.group_id;
                        planNode.fabric_verified = true;
                    });
                    this.invalidateClusterPlan();
                    return true;
                } catch (error) {
                    this.clusterCudaFabricVerificationError = error?.message
                        || 'ConnectX verification failed';
                    return false;
                } finally {
                    this.clusterCudaFabricVerificationLoading = '';
                }
            },

            async ensureCudaSupernodesVerified() {
                const candidates = this.clusterLogicalNodes().filter(node => (
                    node.memberCount === 2
                    && node.accelerator === 'cuda'
                ));
                for (const node of candidates) {
                    if (!await this.verifyCudaSupernode(node)) return false;
                }
                return true;
            },

            async prepareCudaSupernodesForActivation({ rebuildPlan = false } = {}) {
                const planRevision = Number(this._clusterPlanRevision || 0);
                if (
                    typeof this.ensureCudaSupernodesVerified === 'function'
                    && !await this.ensureCudaSupernodesVerified()
                ) {
                    const message = this.clusterCudaFabricVerificationError
                        || 'Verify the CUDA direct link before starting the cluster.';
                    if (rebuildPlan) this.clusterPlanError = message;
                    else this.clusterAutoconfigureError = message;
                    return false;
                }
                if (
                    rebuildPlan
                    && Number(this._clusterPlanRevision || 0) !== planRevision
                ) {
                    await this.runClusterPlan();
                    if (!this.clusterActivationReady()) {
                        this.clusterPlanError = this.clusterPlanError
                            || 'Rebuild the plan after verifying the CUDA direct link.';
                        return false;
                    }
                }
                return true;
            },

            clusterNeuralFabricJob() {
                const jobs = this.clusterStatus?.runtime_jobs?.jobs || [];
                return jobs.find(job => job.live) || null;
            },

            clusterNeuralFabricMode() {
                const job = this.clusterNeuralFabricJob();
                if (this.clusterDeactivatingId) return 'stopping';
                if (
                    this.clusterActivationLoading
                    || (job?.live && job.phase !== 'ready')
                ) {
                    return 'loading';
                }
                const activeRequests = Number(job?.metrics?.active_requests || 0);
                if (activeRequests > 0) {
                    const request = job?.metrics?.last_request;
                    if (
                        request?.prefill_progress?.active
                        || request?.ttft_seconds === null
                        || request?.ttft_seconds === undefined
                    ) {
                        return 'prefill';
                    }
                    if (Number(request?.completion_tokens || 0) > 0) {
                        return 'decode';
                    }
                    const batch = job?.metrics?.pipeline?.last_batch;
                    if (Number(batch?.generation_responses || 0) > 0) return 'decode';
                    if (Number(batch?.prompt_responses || 0) > 0) return 'prefill';
                    return 'inference';
                }
                if (job?.live && job.phase === 'ready') return 'ready';
                if (
                    this.clusterAutoconfigureError
                    || this.clusterError
                    || this.clusterDeploymentsError
                ) {
                    return 'attention';
                }
                if (this.clusterQuickNodes().length > 1) return 'connected';
                return 'discovering';
            },

            clusterNeuralFabricLabel() {
                return {
                    stopping: 'Draining requests safely',
                    loading: 'Loading model across the ring',
                    prefill: 'Prefilling prompt across every rank',
                    decode: 'Decoding tokens across every rank',
                    inference: 'Ranks are processing a live request',
                    ready: 'Cluster ready for requests',
                    attention: 'Cluster needs attention',
                    connected: 'Workers connected and standing by',
                    discovering: 'Looking for another worker',
                }[this.clusterNeuralFabricMode()] || 'Cluster standing by';
            },

            clusterNeuralFabricFiring() {
                return ['loading', 'prefill', 'decode', 'inference'].includes(
                    this.clusterNeuralFabricMode()
                );
            },

            clusterNeuralFabricNodes() {
                const nodes = this.clusterLogicalNodes();
                const job = this.clusterNeuralFabricJob();
                const assignments = job?.assignments || [];
                const ranks = job?.ranks || [];
                const memoryNodes = this.clusterMemoryNodes();
                const positions = nodes.length === 1
                    ? [{ x: 50, y: 50 }]
                    : (nodes.length === 2
                        ? [{ x: 21, y: 50 }, { x: 79, y: 50 }]
                        : nodes.map((_, index) => {
                            const angle = (-Math.PI / 2)
                                + ((Math.PI * 2 * index) / nodes.length);
                            return {
                                x: 50 + (Math.cos(angle) * 36),
                                y: 50 + (Math.sin(angle) * 34),
                            };
                        }));
                return nodes.map((node, index) => {
                    const memberRanks = Array.isArray(node.ranks)
                        ? node.ranks.map(Number)
                        : [Number(node.rank)];
                    const memberAssignments = assignments.filter(
                        item => memberRanks.includes(Number(item.rank))
                    );
                    const memberRuntimeRanks = ranks.filter(
                        item => memberRanks.includes(Number(item.rank))
                    );
                    const measured = memberRuntimeRanks.reduce(
                        (total, rank) => total + Number(rank?.measured_weight_bytes || 0),
                        0
                    );
                    const planned = memberAssignments.reduce(
                        (total, assignment) => total + Number(assignment?.planned_weight_bytes || 0),
                        0
                    );
                    const capacity = memberAssignments.reduce(
                        (total, assignment) => total + Number(assignment?.capacity_bytes || 0),
                        0
                    );
                    const available = memberRanks
                        .reduce(
                            (total, rank) => total + Number(
                                memoryNodes[rank]?.usableGiB || 0
                            ),
                            0
                        )
                        || (Number(node.memory || 0) / (1024 ** 3));
                    let memoryLabel = available > 0
                        ? `${this.clusterMemoryGiBLabel(available)} usable`
                        : 'Memory not measured';
                    if (capacity > 0) {
                        memoryLabel = `${this.formatClusterGiB(measured || planned)} / `
                            + `${this.formatClusterGiB(capacity)}`;
                    }
                    const working = this.clusterNeuralFabricFiring()
                        && Boolean(job?.live || this.clusterActivationLoading);
                    return {
                        ...node,
                        ...positions[index],
                        memoryLabel,
                        working,
                        state: working
                            ? (this.clusterNeuralFabricMode() === 'loading'
                                ? 'Loading'
                                : 'Firing')
                            : (job?.live && job.phase === 'ready'
                                ? 'Ready'
                                : (node.online ? 'Online' : 'Checking')),
                    };
                });
            },

            clusterNeuralFabricEdges() {
                const nodes = this.clusterNeuralFabricNodes();
                if (nodes.length < 2) return [];
                if (nodes.length === 2) {
                    return [
                        {
                            key: '0-1',
                            path: `M ${nodes[0].x} ${nodes[0].y} `
                                + `C 38 21, 62 21, ${nodes[1].x} ${nodes[1].y}`,
                            reverse: false,
                        },
                        {
                            key: '1-0',
                            path: `M ${nodes[1].x} ${nodes[1].y} `
                                + `C 62 79, 38 79, ${nodes[0].x} ${nodes[0].y}`,
                            reverse: true,
                        },
                    ];
                }
                return nodes.map((node, index) => {
                    const next = nodes[(index + 1) % nodes.length];
                    return {
                        key: `${node.rank}-${next.rank}`,
                        path: `M ${node.x} ${node.y} L ${next.x} ${next.y}`,
                        reverse: index % 2 === 1,
                    };
                });
            },

            clusterNeuralFabricRingPath() {
                return this.clusterNeuralFabricEdges()
                    .map(edge => edge.path)
                    .join(' ');
            },

            clusterNeuralFabricProfiles() {
                return (this.clusterNeuralFabricJob()?.performance_profiles || [])
                    .filter(profile => profile && typeof profile === 'object');
            },

            clusterNeuralFabricLatencySeconds() {
                const values = this.clusterNeuralFabricProfiles()
                    .map(profile => Number(profile.collective_latency_seconds))
                    .filter(value => Number.isFinite(value) && value > 0);
                return values.length ? Math.max(...values) : null;
            },

            clusterNeuralFabricBandwidthBytes() {
                const values = this.clusterNeuralFabricProfiles()
                    .map(profile => Number(
                        profile.collective_bandwidth_bytes_per_second
                    ))
                    .filter(value => Number.isFinite(value) && value > 0);
                return values.length ? Math.min(...values) : null;
            },

            clusterNeuralFabricLinkCapacityGbps() {
                const values = this.clusterQuickNodes()
                    .slice(1)
                    .map(node => Number(node.link_speed_gbps))
                    .filter(value => Number.isFinite(value) && value > 0);
                if (values.length) return Math.min(...values);
                const labels = [
                    this.clusterLinkStatus?.link_label,
                    ...(
                        this.clusterStatus?.transport?.thunderbolt?.ports || []
                    ).filter(port => port.peer_connected).map(port => port.speed),
                ];
                const parsed = labels.map(label => {
                    const match = String(label || '').match(
                        /(\d+(?:\.\d+)?)\s*Gb\/s/i
                    );
                    return match ? Number(match[1]) : null;
                }).filter(value => Number.isFinite(value) && value > 0);
                return parsed.length ? Math.min(...parsed) : null;
            },

            formatClusterLatency(seconds) {
                const value = Number(seconds);
                if (!Number.isFinite(value) || value <= 0) return '—';
                if (value < 0.001) return `${Math.round(value * 1_000_000)} µs`;
                if (value < 1) return `${(value * 1000).toFixed(
                    value < 0.01 ? 2 : 1
                )} ms`;
                return `${value.toFixed(2)} s`;
            },

            clusterNeuralFabricMetrics() {
                const job = this.clusterNeuralFabricJob();
                const latency = this.clusterNeuralFabricLatencySeconds();
                const bandwidth = this.clusterNeuralFabricBandwidthBytes();
                const linkCapacity = this.clusterNeuralFabricLinkCapacityGbps();
                const lastRequest = job?.metrics?.last_request;
                const requestActive = Number(
                    job?.metrics?.active_requests || 0
                ) > 0;
                const completionTokens = Math.max(
                    0,
                    Number(lastRequest?.completion_tokens || 0),
                );
                const hasFirstToken = (
                    lastRequest?.ttft_seconds !== null
                    && lastRequest?.ttft_seconds !== undefined
                    && completionTokens > 0
                );
                const prefillProgress = lastRequest?.prefill_progress;
                const promptTokens = Math.max(
                    0,
                    Number(lastRequest?.prompt_tokens || 0),
                );
                const cachedTokens = Math.min(
                    promptTokens,
                    Math.max(0, Number(lastRequest?.cached_tokens || 0)),
                );
                const uncachedTokens = Math.max(0, promptTokens - cachedTokens);
                const prefillTotal = Math.max(
                    0,
                    Number(prefillProgress?.total ?? uncachedTokens),
                );
                const prefillProcessed = Math.min(
                    prefillTotal,
                    Math.max(0, Number(prefillProgress?.processed || 0)),
                );
                const prefillActive = (
                    requestActive
                    && !hasFirstToken
                    && Boolean(prefillProgress?.active ?? true)
                );
                const livePrefillAverage = Number(
                    prefillProgress?.average_speed
                    || lastRequest?.prefill_tps
                    || prefillProgress?.speed
                    || 0
                );
                const livePrefillRecent = Number(
                    prefillProgress?.speed || livePrefillAverage || 0
                );
                const rawPrefillEta = prefillProgress?.eta;
                const prefillEta = rawPrefillEta === null
                    || rawPrefillEta === undefined
                    ? null
                    : Number(rawPrefillEta);
                const prefillDetail = [];
                if (livePrefillAverage > 0) {
                    prefillDetail.push(
                        `${this.formatClusterRate(livePrefillAverage)} avg`
                    );
                }
                if (
                    livePrefillRecent > 0
                    && (
                        livePrefillAverage <= 0
                        || Math.abs(livePrefillRecent - livePrefillAverage)
                            / livePrefillAverage > 0.05
                    )
                ) {
                    prefillDetail.push(
                        `${this.formatClusterRate(livePrefillRecent)} now`
                    );
                }
                if (Number.isFinite(prefillEta) && prefillEta >= 0) {
                    prefillDetail.push(`${this.formatDurationShort(prefillEta)} left`);
                }
                if (cachedTokens > 0) {
                    prefillDetail.push(`${this.formatClusterCount(cachedTokens)} cached`);
                }
                const resident = job
                    ? this.clusterRuntimeResidentWeights(job)
                    : 0;
                const capacity = job
                    ? this.clusterRuntimeCapacity(job)
                    : this.clusterCombinedUsableMemoryGiB() * (1024 ** 3);
                return [
                    {
                        key: 'latency',
                        label: 'Ring latency',
                        value: this.formatClusterLatency(latency),
                        detail: latency
                            ? 'Latest startup probe · slowest hop'
                            : 'Measured when the cluster starts',
                        tone: 'violet',
                    },
                    {
                        key: 'link',
                        label: bandwidth ? 'Collective throughput' : 'Link capacity',
                        value: bandwidth
                            ? this.formatClusterBandwidth(bandwidth)
                            : (linkCapacity ? `${linkCapacity} Gb/s` : '—'),
                        detail: bandwidth
                            ? '1 MiB all-reduce · startup probe · slowest rank'
                            : (linkCapacity
                                ? 'Negotiated speed · not measured throughput'
                                : 'Waiting for link measurement'),
                        tone: 'cyan',
                    },
                    {
                        key: 'prefill',
                        label: 'Prompt prefill',
                        value: prefillActive
                            ? (
                                prefillTotal > 0
                                    ? `${this.formatClusterCount(prefillProcessed)} / `
                                        + `${this.formatClusterCount(prefillTotal)}`
                                    : 'Preparing…'
                            )
                            : this.formatClusterRate(lastRequest?.prefill_tps),
                        detail: prefillActive
                            ? (prefillDetail.join(' · ') || 'Preparing prompt…')
                            : (
                                lastRequest
                                    ? `${this.formatClusterCount(uncachedTokens)} new`
                                        + (cachedTokens > 0
                                            ? ` · ${this.formatClusterCount(cachedTokens)} cached`
                                            : '')
                                    : 'No request yet'
                            ),
                        progress: prefillActive && prefillTotal > 0
                            ? prefillProcessed / prefillTotal
                            : null,
                        tone: 'blue',
                    },
                    {
                        key: 'decode',
                        label: 'Token decode',
                        value: !hasFirstToken
                            ? '—'
                            : (
                                completionTokens > 1
                                    ? this.formatClusterRate(lastRequest?.decode_tps)
                                    : 'Measuring…'
                            ),
                        detail: prefillActive
                            ? 'Starts after prefill'
                            : (
                                requestActive && hasFirstToken
                                    ? `${this.formatClusterCount(completionTokens)} generated · live`
                                    : (
                                        lastRequest
                                            ? `${this.formatClusterCount(completionTokens)} generated`
                                            : 'No request yet'
                                    )
                            ),
                        progress: null,
                        tone: 'green',
                    },
                    {
                        key: 'memory',
                        label: job ? 'Resident memory' : 'Model memory',
                        value: job
                            ? `${this.formatClusterGiB(resident)} / `
                                + `${this.formatClusterGiB(capacity)}`
                            : `${this.formatClusterGiB(capacity)} ready`,
                        detail: job
                            ? 'Weights resident / cluster capacity'
                            : 'Usable across detected accelerators',
                        tone: 'amber',
                    },
                ];
            },

            clusterNeuralFabricAriaLabel() {
                const metrics = this.clusterNeuralFabricMetrics();
                const latency = metrics.find(metric => metric.key === 'latency');
                const link = metrics.find(metric => metric.key === 'link');
                return `Neural fabric with ${this.clusterLogicalNodes().length} visual groups `
                    + `and ${this.clusterQuickNodes().length} execution ranks. `
                    + `${this.clusterNeuralFabricLabel()}. `
                    + `Ring latency ${latency?.value || 'not measured'}. `
                    + `${link?.label || 'Link'} ${link?.value || 'not measured'}.`;
            },

            clusterMemoryNodes() {
                const quickNodes = this.clusterQuickNodes();
                return (this.clusterPlanNodes || [])
                    .slice(0, quickNodes.length)
                    .map((budget, index) => {
                        const quick = quickNodes[index] || {};
                        const capacityGiB = Math.max(
                            0,
                            Number(budget.capacity_gib || 0),
                        );
                        const reserveGiB = Math.max(
                            0,
                            Number(budget.reserve_gib || 0),
                        );
                        const measuredAutomaticReserve = Number(
                            budget.automatic_reserve_gib
                        );
                        const automaticReserveGiB = (
                            Number.isFinite(measuredAutomaticReserve)
                            && measuredAutomaticReserve >= 0
                        )
                            ? Math.min(capacityGiB, measuredAutomaticReserve)
                            : Math.min(capacityGiB, reserveGiB);
                        return {
                            budget,
                            nodeId: budget.node_id || `device-${index}`,
                            name: quick.name || budget.node_id || `Device ${index + 1}`,
                            chip: quick.chip || '',
                            accelerator: quick.accelerator || 'metal',
                            acceleratorVendor: quick.acceleratorVendor || 'apple',
                            memoryKind: quick.memoryKind || 'unified',
                            fabricKind: quick.fabricKind || '',
                            fabricGroup: quick.fabricGroup || '',
                            physicalMemory: Number(quick.physicalMemory || 0),
                            capacityGiB,
                            reserveGiB,
                            usableGiB: Math.max(0, capacityGiB - reserveGiB),
                            automaticReserveGiB,
                            minGiB: Math.min(capacityGiB, 4),
                            physicalGiB: Math.max(
                                0,
                                Number(quick.physicalMemory || 0) / (1024 ** 3)
                            ),
                        };
                    });
            },

            clusterMemoryAllowanceNodes() {
                return this.clusterMemoryNodes().filter(
                    node => node.capacityGiB > 0
                );
            },

            clusterMemoryAllowanceCustomized() {
                return this.clusterMemoryAllowanceNodes().some(
                    node => this.clusterManualMemoryAllowanceGiB(node.nodeId) !== null
                );
            },

            clusterSetMemoryAllowance(node, requestedGiB) {
                if (!node?.budget) return;
                const capacity = Math.max(0, Number(node.capacityGiB || 0));
                const minimum = Math.min(capacity, Math.max(0, Number(node.minGiB || 0)));
                const allowed = Math.min(
                    capacity,
                    Math.max(minimum, Number(requestedGiB || 0))
                );
                node.budget.reserve_gib = Number((capacity - allowed).toFixed(2));
                this.clusterMemoryAllowancesGiB = {
                    ...(this.clusterMemoryAllowancesGiB || {}),
                    [node.nodeId]: Number(allowed.toFixed(2)),
                };
                // The user sets a safety ceiling; layer placement remains an
                // automatic planner decision within those ceilings.
                this.clusterMemoryLimitsManual = true;
                this.saveClusterMemoryAllowances();
                this.clusterWeightTargetsGiB = {};
                this.clusterCatalogue = null;
                this.invalidateClusterPlan();
            },

            async clusterMemoryAllowanceChanged() {
                this.clusterCatalogue = null;
                await this.loadClusterCatalogue();
                await this.previewClusterWeightBalance();
            },

            clusterResetMemoryAllowances() {
                const nodes = this.clusterMemoryAllowanceNodes();
                this.clusterMemoryAllowancesGiB = {};
                nodes.forEach(node => {
                    node.budget.reserve_gib = Number(
                        node.automaticReserveGiB.toFixed(2)
                    );
                });
                this.clusterMemoryLimitsManual = false;
                this.saveClusterMemoryAllowances();
                this.clusterWeightTargetsGiB = {};
                this.invalidateClusterPlan();
                this.clusterMemoryAllowanceChanged();
            },

            clusterMemoryGiBLabel(value) {
                const gib = Math.max(0, Number(value || 0));
                return `${Number.isInteger(gib) ? gib : gib.toFixed(1)} GiB`;
            },

            clusterCombinedUsableMemoryGiB() {
                return this.clusterMemoryNodes().reduce(
                    (total, node) => total + node.usableGiB,
                    0
                );
            },

            clusterCombinedPhysicalMemoryGiB() {
                return this.clusterMemoryNodes().reduce(
                    (total, node) => total + node.physicalGiB,
                    0
                );
            },

            clusterCombinedMemoryLabel() {
                const physical = this.clusterCombinedPhysicalMemoryGiB();
                const usable = this.clusterCombinedUsableMemoryGiB();
                return `${this.clusterMemoryGiBLabel(usable)} model-usable of `
                    + `${this.clusterMemoryGiBLabel(physical)} installed`;
            },

            clusterCombinedMemoryBreakdown() {
                return this.clusterMemoryNodes()
                    .map(node => `${node.name} ${this.clusterMemoryGiBLabel(node.usableGiB)}`)
                    .join(' + ');
            },

            clusterPairTitle() {
                const nodes = this.clusterQuickNodes();
                if (nodes.length === 1) return `${nodes[0].name} · finding workers`;
                const logical = this.clusterLogicalNodes();
                const mixed = new Set(nodes.map(node => node.accelerator)).size > 1;
                if (nodes.length === 2 && !mixed) {
                    return `${nodes[0].name} + ${nodes[1].name}`;
                }
                return `${logical.length} visual group${logical.length === 1 ? '' : 's'} · `
                    + `${nodes.length} execution rank${nodes.length === 1 ? '' : 's'}`;
            },

            clusterDeviceCountLabel() {
                const devices = this.clusterQuickNodes().length;
                const units = this.clusterLogicalNodes().length;
                return units === devices
                    ? `${devices} device${devices === 1 ? '' : 's'}`
                    : `${units} visual groups · ${devices} execution ranks`;
            },

            clusterPeerDisplayName() {
                const hostname = this.clusterPeerProbe?.status?.node?.hostname;
                if (hostname) return hostname;
                const ssh = this.clusterPeerSsh.trim();
                const discovered = (this.clusterDiscoveredPeers || []).find(
                    peer => peer.ssh === ssh
                );
                return discovered?.name || (ssh ? ssh.replace(/\.local$/i, '') : 'Connected Mac');
            },

            clusterPeerConnected() {
                return Boolean(
                    this.clusterStatus?.transport?.thunderbolt?.peer_connected
                    || this.clusterFabric?.link?.ok
                    || this.clusterPeerProbe?.runtime_compatible
                );
            },

            clusterEffectiveTransportState() {
                if (
                    !this.clusterIpsOverridden
                    && this.clusterFabric?.backend === 'jaccl'
                    && this.clusterFabric?.rdma?.ok
                ) {
                    return 'rdma_ready';
                }
                if (this.clusterPeerConnected()) return 'peer_linked_config_pending';
                return this.clusterStatus?.transport?.state || 'unknown';
            },

            clusterConnectionLabel() {
                if (this.clusterIpsOverridden) return 'Manual TCP route';
                const physical = (
                    this.clusterStatus?.transport?.thunderbolt?.ports || []
                ).find(port => port.peer_connected);
                if (physical?.speed) return physical.speed;
                if (
                    this.clusterFabric?.backend === 'jaccl'
                    && this.clusterFabric?.rdma?.ok
                ) return 'Thunderbolt RDMA';
                return this.clusterPeerConnected() ? 'Direct link' : 'Not connected';
            },

            clusterFriendlyConnectionLabel() {
                if (this.clusterFabricLoading || this.clusterLinkStatusLoading) {
                    return 'Checking Thunderbolt…';
                }
                if (this.clusterIpsOverridden) {
                    return 'TCP ring · manual addresses';
                }
                const classified = String(this.clusterLinkStatus?.link_label || '').trim();
                if (classified) return classified.replace(' at ', ' · ');
                const ssh = this.clusterPeerSsh.trim();
                const discovered = (this.clusterDiscoveredPeers || []).find(
                    peer => peer.ssh === ssh
                );
                const speed = Number(discovered?.link_speed_gbps || 0);
                if (speed >= 80) return `Thunderbolt 5 · ${speed} Gb/s`;
                if (speed >= 40) return `Thunderbolt 4 · ${speed} Gb/s`;
                if (
                    this.clusterFabric?.backend === 'jaccl'
                    && this.clusterFabric?.rdma?.ok
                ) return 'Thunderbolt RDMA';
                const physical = (
                    this.clusterStatus?.transport?.thunderbolt?.ports || []
                ).find(port => port.peer_connected);
                if (physical?.speed) return `Thunderbolt · ${physical.speed}`;
                return this.clusterPeerConnected()
                    ? 'Direct connection'
                    : 'Connecting automatically…';
            },

            clusterVisibleWarnings() {
                const warnings = this.clusterStatus?.warnings || [];
                if (this.clusterEffectiveTransportState() !== 'rdma_ready') {
                    return warnings;
                }
                return warnings.filter(
                    warning => !/no Thunderbolt peer is connected/i.test(warning)
                );
            },

            clusterTopologySummary() {
                const count = this.clusterQuickNodes().length;
                if (count === 1) {
                    return `${this.clusterStatus?.node?.hostname || 'This Mac'} · coordinator`;
                }
                const link = this.clusterIpsOverridden
                    ? 'TCP ring · manual addresses'
                    : (this.clusterFabric?.backend === 'jaccl'
                        ? 'Thunderbolt RDMA'
                        : (this.clusterTransports?.transports?.[0]
                            ? this.clusterTransportLabel(this.clusterTransports.transports[0])
                            : 'direct link'));
                return `${count} devices · ${this.clusterStatus?.node?.hostname || 'This device'} coordinates · ${count - 1} workers · ${link}`;
            },

            clusterCatalogueModels() {
                return this.clusterAllModels()
                    .filter(model => model?.model_path)
                    .map(model => ({
                        id: model.id || this.clusterModelDisplayName(model),
                        model_path: model.model_path,
                        model_source: model.model_source || '127.0.0.1',
                        model_source_python:
                            model.model_source_python
                            || model.python_executable
                            || this.clusterPythonExecutableForSsh(
                                model.model_source || '127.0.0.1'
                            )
                            || undefined,
                        source_node_id: model.source_node_id || '',
                        model_context_length:
                            model.model_context_length || null,
                    }));
            },

            clusterCatalogueInputsReady() {
                const peers = this.clusterWorkerPeers();
                if (!peers.length) return false;
                if (
                    this.clusterPeerProbeLoading
                    || this.clusterBudgetsLoading
                    || this.clusterFabricLoading
                ) return false;
                if ((this.clusterPlanNodes || []).length !== peers.length + 1) {
                    return false;
                }
                const probes = this.clusterPeerProbes || {};
                if (peers.some(peer => {
                    const probe = probes[String(peer?.ssh || '').trim()];
                    return !probe?.status?.node || probe.cached === true;
                })) {
                    return false;
                }
                return (this.clusterPlanNodes || []).every(
                    node => Number(node.capacity_bytes || 0) > 0
                );
            },

            clusterCatalogueRequestKey(models = null) {
                return JSON.stringify({
                    nodes: this.clusterNodePayloads(),
                    models: models || this.clusterCatalogueModels(),
                    execution_profile: this.clusterExecutionProfile,
                });
            },

            // Answer "what can these Macs actually run" before anything is
            // staged. The server plans each model with the real planner, so a
            // model listed as fitting is one that will load. A catalogue made
            // from placeholder peer memory is not a verdict: wait for every
            // probe and discard any response whose input snapshot went stale.
            async loadClusterCatalogue() {
                if (this.clusterCatalogueLoading) return;
                if (!this.clusterCatalogueInputsReady()) {
                    this.clusterCatalogue = null;
                    this.clusterCatalogueError = '';
                    return;
                }
                if (!this.clusterModelInventory) {
                    await this.loadClusterModelInventory();
                }
                if (!this.clusterCatalogueInputsReady()) return;
                const models = this.clusterCatalogueModels();
                if (!models.length) {
                    this.clusterCatalogueError = 'No downloaded MLX models were found.';
                    return;
                }
                const requestKey = this.clusterCatalogueRequestKey(models);
                this.clusterCatalogueLoading = true;
                this.clusterCatalogueError = '';
                this.clusterCatalogue = null;
                try {
                    const nodes = this.clusterNodePayloads();
                    const response = await fetch('/admin/api/cluster/catalogue', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            nodes,
                            models,
                            execution_profile: this.clusterExecutionProfile,
                        }),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    const body = await response.json();
                    if (
                        !this.clusterCatalogueInputsReady()
                        || requestKey !== this.clusterCatalogueRequestKey()
                    ) return;
                    if (!response.ok) {
                        this.clusterCatalogueError = this.clusterErrorMessage(
                            body.detail,
                            'Could not read that folder.',
                        );
                        return;
                    }
                    this.clusterCatalogue = body;
                    const selected = this.clusterSelectedModel();
                    if (selected) {
                        const fit = this.clusterCatalogueFit(selected.model_path);
                        const tensorSize = this.clusterFitTensorParallelSize(fit);
                        if (
                            fit?.fits === true
                            && this.clusterTensorParallelOptions().includes(tensorSize)
                        ) {
                            this.clusterPlanTensorParallelSize = tensorSize;
                        }
                        this.clusterTargetContextTokens =
                            this.clusterContextMode === 'auto'
                                ? this.clusterContextMaximumTokens()
                                : this.clusterModelTargetContext(
                                    selected,
                                    this.clusterTargetContextTokens,
                                );
                    }
                } catch (error) {
                    if (
                        this.clusterCatalogueInputsReady()
                        && requestKey === this.clusterCatalogueRequestKey()
                    ) {
                        this.clusterCatalogueError = String(error);
                    }
                } finally {
                    this.clusterCatalogueLoading = false;
                    if (
                        this.clusterCatalogueInputsReady()
                        && requestKey !== this.clusterCatalogueRequestKey()
                    ) {
                        await this.loadClusterCatalogue();
                    }
                }
            },

            clusterCatalogueSize(bytes) {
                return `${(Number(bytes) / (1024 ** 3)).toFixed(1)} GiB`;
            },

            // Every block of commands the cluster page shows copies through
            // here, so a blocking error always comes with its fix attached and
            // the affordance behaves the same wherever it appears.
            copyClusterCommands(key, commands) {
                const lines = (commands || []).filter(Boolean);
                if (!lines.length) return;
                this.copyToClipboard(lines.join('\n'));
                this.clusterCommandsCopied = { ...this.clusterCommandsCopied, [key]: true };
                setTimeout(() => {
                    this.clusterCommandsCopied = { ...this.clusterCommandsCopied, [key]: false };
                }, 2000);
            },

            // Where each Mac answers, read from both ends. An address is true
            // until macOS renumbers the port behind it, so it is discovered on
            // demand and never remembered between sessions.
            async loadClusterFabric() {
                if (this.clusterFabricLoading) return;
                const hosts = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ].filter(Boolean);
                if (hosts.length < 2) {
                    this.clusterFabricError = 'Add a worker before reading its addresses.';
                    return;
                }
                this.clusterFabricLoading = true;
                this.clusterFabricError = '';
                try {
                    const query = new URLSearchParams({ hosts: hosts.join(',') });
                    const response = await fetch(`/admin/api/cluster/fabric?${query}`);
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(
                            response, 'Could not read the addresses between these workers'
                        ));
                    }
                    this.adoptClusterFabric(await response.json());
                } catch (error) {
                    this.clusterFabricError = error?.message
                        || 'Could not read the addresses between these workers';
                } finally {
                    this.clusterFabricLoading = false;
                }
            },

            // The discovered addresses win unless someone has typed over them.
            //
            // Only a reading taken in the order this page posts is adopted: the
            // RDMA matrix is indexed by rank, so entry [i][j] read against
            // another ordering is the path to a different Mac. Rank placement
            // is free to order the hosts its own way, and mapping one onto the
            // other by position is how a rank ends up dialling itself.
            adoptClusterFabric(fabric) {
                if (!fabric) return;
                const expected = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ];
                const actual = (fabric.hosts || []).map(item => item.host);
                if (actual.length !== expected.length
                    || actual.some((host, index) => host !== expected[index])) {
                    this.loadClusterFabric();
                    return;
                }
                this.clusterFabric = fabric;
                if (!fabric.ok) {
                    // No shared address is a blocking error whose fix is a
                    // command needing administrator rights. Link status is what
                    // emits those, so bring it up rather than leaving the page
                    // stating a problem it will not help with.
                    this.loadClusterLinkStatus();
                }
                if (this.clusterIpsOverridden) return;
                const [local, peer] = fabric.hosts;
                if (local.ips?.length) {
                    this.clusterLocalIp = local.ips[0];
                    if (!this.clusterJoinControllerIp.trim()) {
                        this.clusterJoinControllerIp = this.clusterLocalIp;
                    }
                }
                if (peer.ips?.length) this.clusterPeerIp = peer.ips[0];
                this.invalidateClusterPlan();
            },

            useDiscoveredClusterAddresses() {
                this.clusterIpsOverridden = false;
                if (this.clusterFabric) this.adoptClusterFabric(this.clusterFabric);
                else this.loadClusterFabric();
            },

            // Backend is a consequence of the cable, never a preference: the
            // fabric reader is the only thing that may name one, and it falls
            // back to the TCP ring out loud rather than by accident.
            get clusterBackend() {
                // A typed address is not enough evidence to construct a JACCL
                // device matrix. Honour the override, but keep it on the TCP
                // ring until the server has verified that exact path as RDMA.
                if (this.clusterIpsOverridden) return 'ring';
                return this.clusterFabric?.backend || 'ring';
            },

            async loadClusterTransports() {
                if (this.clusterTransportsLoading) return;
                const hosts = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ].filter(Boolean);
                if (hosts.length < 2) {
                    this.clusterTransportsError = 'Add a peer before detecting the link';
                    return;
                }
                this.clusterTransportsLoading = true;
                this.clusterTransportsError = '';
                try {
                    const query = new URLSearchParams({ hosts: hosts.join(',') });
                    const response = await fetch(
                        `/admin/api/cluster/transports?${query}`
                    );
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(
                            await this.clusterResponseError(response, 'Link detection failed')
                        );
                    }
                    this.clusterTransports = await response.json();
                    await this.loadClusterLinkStatus();
                    await this.loadClusterFabric();
                } catch (error) {
                    this.clusterTransportsError = error?.message || 'Link detection failed';
                } finally {
                    this.clusterTransportsLoading = false;
                }
            },

            clusterTransportLabel(transport) {
                const version = transport.tb_version || transport.kind;
                const speed = transport.link_speed_gbps
                    ? ` · ${transport.link_speed_gbps} Gb/s`
                    : '';
                return `${version}${speed}`;
            },

            // Ask the server to work out the transport, parallelism split,
            // backend and preflight result from the peers we already know.
            // This remains useful on its own for advanced/manual planning;
            // startCluster() composes it with activation behind one click.
            async autoconfigureCluster() {
                if (this.clusterAutoconfigureLoading) return null;
                const requestRevision = Number(this._clusterPlanRevision || 0);
                const measuredNodes =
                    this.clusterAutoconfigure?.activation?.nodes || [];
                const performanceByNode = new Map(
                    measuredNodes
                        .filter(node => node?.node_id && node?.performance)
                        .map(node => [node.node_id, node.performance])
                );
                this.clusterAutoconfigureLoading = true;
                this.clusterAutoconfigureError = '';
                this.clusterAutoconfigure = null;
                try {
                    const nodes = this.clusterNodePayloads().map(node => ({
                        ...node,
                        ...(performanceByNode.has(node.node_id)
                            ? { performance: performanceByNode.get(node.node_id) }
                            : {}),
                    }));
                    const hosts = this.clusterClusterHostsPayload();
                    const body = {
                        nodes,
                        hosts,
                        // `split_gib` used to be posted here and silently
                        // dropped — ClusterAutoconfigureRequest never declared
                        // it. The split now travels where the planner reads
                        // it, as `max_weight_bytes` on the node itself.
                        execution_profile: this.clusterExecutionProfile,
                        prefer: this.clusterAutoconfigurePrefer,
                        strategy: this.clusterStrategy,
                        // Hand-entered addresses are an explicit routing
                        // decision. Rediscovery here would replace them just
                        // before activation and could put the collective back
                        // on an unrelated mDNS/default route.
                        detect_transports:
                            hosts.length > 0 && !this.clusterIpsOverridden,
                        auto_tune: this.clusterAutoTune,
                        // Measure before staging so the signed layer placement
                        // is the fast one. A post-staging re-plan can only report
                        // a better split because the needed shards may be absent.
                        measure_performance: this.clusterAutoTune,
                        sampling_rank_only: this.clusterSamplingRankOnly,
                        async_overlap: this.clusterAsyncOverlap,
                        cache_affinity: this.clusterCacheAffinity,
                        max_kv_size: this.clusterMaxKvSize === ''
                            ? null
                            : Number(this.clusterMaxKvSize),
                        target_context_tokens: Number(
                            this.clusterTargetContextTokens || 8192
                        ),
                        ring_connections_per_ip: Number(
                            this.clusterRingConnectionsPerIp
                        ),
                    };
                    if (this.clusterPlanMode === 'model') {
                        body.model_path = this.clusterPlanModelPath.trim();
                        body.model_source =
                            this.clusterPlanModelSource || '127.0.0.1';
                        if (this.clusterPlanModelSourcePython) {
                            body.model_source_python =
                                this.clusterPlanModelSourcePython;
                        }
                    } else {
                        body.model_size_bytes = Math.round(
                            Number(this.clusterPlanModelSizeGiB) * (1024 ** 3)
                        );
                        body.layer_count = Number(this.clusterPlanLayerCount);
                    }
                    const response = await fetch('/admin/api/cluster/autoconfigure', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(
                            await this.clusterResponseError(response, 'Autoconfigure failed')
                        );
                    }
                    const result = await response.json();
                    // A memory, model or context change invalidates an
                    // in-flight proposal. Never let its late response restore
                    // the old refusal (or, worse, make the old activation
                    // launchable under the new labels).
                    if (
                        requestRevision
                        !== Number(this._clusterPlanRevision || 0)
                    ) {
                        return null;
                    }
                    this.clusterAutoconfigure = result;
                    // Remember the last proposal that was ready to activate, so
                    // the next visit is genuinely one click.
                    if (result.ready_to_activate) {
                        try {
                            window.localStorage.setItem(
                                'omlx.cluster.lastGoodConfig',
                                JSON.stringify({
                                    saved_at: new Date().toISOString(),
                                    summary: result.summary,
                                    backend: result.backend,
                                    tensor_parallel_size: result.tensor_parallel_size,
                                    activation: result.activation,
                                })
                            );
                            this.clusterLastGoodConfig = result.activation;
                        } catch (error) {
                            // Private browsing or a full quota: remembering is a
                            // convenience, never a requirement.
                        }
                    }
                    // Adopt the server's choices so the manual controls agree
                    // with what the proposal says. The backend rides in on the
                    // fabric rather than on its own, so the addresses it was
                    // decided from are the addresses activation posts.
                    this.adoptClusterFabric(result.fabric);
                    this.clusterPlanTensorParallelSize = result.tensor_parallel_size;
                    this.invalidateClusterPlan();
                    this.clusterPlan = result.plan || null;
                    if (this.clusterPlan) {
                        this._clusterPlanSignature =
                            this.clusterCurrentPlanSignature();
                    }
                    return result;
                } catch (error) {
                    if (
                        requestRevision
                        !== Number(this._clusterPlanRevision || 0)
                    ) {
                        return null;
                    }
                    this.clusterAutoconfigureError = error?.message || 'Autoconfigure failed';
                    return null;
                } finally {
                    this.clusterAutoconfigureLoading = false;
                }
            },

            // The normal product path. Autoconfigure is the safety boundary:
            // only its launch-ready response can reach /deployments, and the
            // activation object is posted byte-for-byte as the server shaped
            // it. A blocker therefore stops before any rank process starts.
            async startCluster() {
                if (this.clusterLinkSetupLoading
                    || this.clusterAutoconfigureLoading
                    || this.clusterActivationLoading) {
                    return;
                }
                this.clusterError = '';
                this.clusterActivationResult = null;
                if (this.clusterPlanMode !== 'model'
                    || !this.clusterPlanModelPath.trim()) {
                    this.clusterAutoconfigureError =
                        'Choose a downloaded model before starting the cluster.';
                    return;
                }
                if (!this.clusterPeerSsh.trim()) {
                    this.clusterAutoconfigureError =
                        'Choose a worker before starting the cluster.';
                    return;
                }
                if (!await this.prepareCudaSupernodesForActivation()) return;
                const linkStatus = await this.loadClusterLinkStatus();
                if (linkStatus?.setup_available) {
                    const ready = await this.prepareClusterLink();
                    if (!ready) return;
                }
                if (!this.clusterLocalIp.trim() || !this.clusterPeerIp.trim()) {
                    await this.loadClusterFabric();
                }
                if (!this.clusterLocalIp.trim() || !this.clusterPeerIp.trim()) {
                    this.clusterAutoconfigureError =
                        this.clusterFabricError
                        || 'The workers do not have a shared cluster address yet.';
                    return;
                }
                let proposal = await this.autoconfigureCluster();
                if (!proposal) return;
                if (proposal.fabric_ready === false) {
                    this.clusterAutoconfigureError = proposal.fabric_blocker
                        || 'The workers do not have a verified cluster route yet.';
                    return;
                }
                if (proposal.staging && !proposal.staging.ready) {
                    const staged = await this.stageClusterModel(proposal);
                    if (!staged) return;
                    proposal = await this.autoconfigureCluster();
                    if (!proposal) return;
                }
                if (!proposal.ready_to_activate) {
                    this.clusterAutoconfigureError = proposal.staging?.error
                        || (proposal.staging && !proposal.staging.ready
                            ? 'The model files could not be prepared on every worker.'
                            : proposal.preflight)
                        || 'Resolve the setup checks below, then try Start Cluster again.';
                    return;
                }
                if (!proposal.activation || typeof proposal.activation !== 'object') {
                    this.clusterAutoconfigureError =
                        'Automatic setup did not return a safe activation request.';
                    return;
                }
                await this.activateClusterProposal(proposal.activation);
            },

            async activateClusterProposal(activation) {
                this.clusterActivationLoading = true;
                this.clusterActivationResult = null;
                this.clusterPlanChanges = null;
                this.clusterError = '';
                this.startClusterActivationProgress();
                try {
                    const response = await fetch('/admin/api/cluster/deployments', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(activation),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(
                            await this.clusterResponseError(response, 'Activation failed')
                        );
                    }
                    this.clusterActivationResult = await response.json();
                    this.clusterPlanChanges =
                        this.clusterActivationResult.plan_changes || null;
                    if (this.clusterActivationResult.plan) {
                        this.clusterPlan = this.clusterActivationResult.plan;
                    }
                    this.clusterActivationProgress = 'Ready · canary passed';
                    await this.loadClusterRuntime();
                    await this.loadClusterDeployments();
                } catch (error) {
                    this.clusterError = error?.message || 'Activation failed';
                } finally {
                    this.stopClusterActivationProgress();
                    this.clusterActivationProgress = '';
                    this.clusterActivationLoading = false;
                }
            },

            clusterCurrentPlanSignature() {
                return JSON.stringify({
                    mode: this.clusterPlanMode,
                    model: this.clusterPlanMode === 'model'
                        ? {
                            path: this.clusterPlanModelPath.trim(),
                            source: this.clusterPlanModelSource || '127.0.0.1',
                        }
                        : {
                            size_gib: Number(this.clusterPlanModelSizeGiB),
                            layer_count: Number(this.clusterPlanLayerCount),
                        },
                    // Built from the payload activation actually posts, so a
                    // control that changes the plan cannot leave a stale one
                    // approved. Clicking Workstation used to be invisible
                    // here: the plan stayed "fresh" while the role it was
                    // built from no longer matched the one being sent.
                    nodes: this.clusterNodePayloads(),
                    execution_profile: this.clusterExecutionProfile,
                    tensor_parallel_size: Number(this.clusterPlanTensorParallelSize),
                    target_context_tokens: Number(
                        this.clusterTargetContextTokens || 8192
                    ),
                });
            },

            addClusterPlanNode() {
                this._clusterNodeKey += 1;
                this.clusterPlanNodes.push({
                    key: this._clusterNodeKey,
                    node_id: `node-${this.clusterPlanNodes.length + 1}`,
                    capacity_gib: 64,
                    reserve_gib: 8,
                    // Named, not left undefined: the payload sends 'headless'
                    // either way, and a node with no role leaves both buttons
                    // unlit while the server plans it as one of them.
                    role: 'headless',
                });
                this.normalizeClusterTensorParallelSize();
                this.invalidateClusterPlan();
            },

            removeClusterPlanNode(index) {
                if (this.clusterPlanNodes.length <= 1) return;
                this.clusterPlanNodes.splice(index, 1);
                this.normalizeClusterTensorParallelSize();
                this.invalidateClusterPlan();
            },

            useLocalClusterNode(index) {
                if (!this.clusterStatus || !this.clusterPlanNodes[index]) return;
                const gib = 1024 ** 3;
                const exactCapacity = Number(
                    this.clusterStatus.node.admission_ceiling_bytes
                    || this.clusterStatus.node.recommended_working_set_bytes
                    || 0
                );
                this.clusterPlanNodes[index].node_id = this.clusterNodeId(
                    this.clusterStatus.node.hostname,
                    'this-mac',
                );
                this.clusterPlanNodes[index].capacity_gib = Number(
                    (exactCapacity / gib).toFixed(2)
                );
                this.clusterPlanNodes[index].capacity_bytes = exactCapacity;
                this.invalidateClusterPlan();
            },

            // Every endpoint that plans gets the *same* node objects. /plan and
            // /deployments used to build their own: activation's copy carried
            // only node_id, capacity and reserve, so `role` fell back to
            // headless and `max_weight_bytes` to "planner, balance freely" —
            // the Workstation button and the split slider reached the preview
            // and nothing that launched. Building the payload once is what
            // makes that class of drift impossible rather than merely fixed.
            // `validate` is off by default because this is also what the
            // staleness signature is built from, and that runs inside an Alpine
            // binding on every render — a throw there would take the panel down
            // rather than disable a button.
            clusterNodePayloads({ validate = false } = {}) {
                const gib = 1024 ** 3;
                return this.clusterPlanNodes.map((node, index) => {
                    const nodeId = String(node.node_id || '').trim();
                    const capacityGiB = Number(node.capacity_gib);
                    const reserveGiB = Number(node.reserve_gib);
                    const measuredCapacityBytes = Number(node.capacity_bytes || 0);
                    const capacityBytes = Number.isFinite(measuredCapacityBytes)
                        && measuredCapacityBytes > 0
                        ? Math.round(measuredCapacityBytes)
                        : Math.round(capacityGiB * gib);
                    const manualAllowance = (
                        typeof this.clusterManualMemoryAllowanceGiB === 'function'
                    )
                        ? this.clusterManualMemoryAllowanceGiB(nodeId)
                        : null;
                    if (validate) {
                        if (!nodeId) throw new Error(`Rank ${index} needs a node name`);
                        if (!Number.isFinite(capacityGiB) || capacityGiB <= 0) {
                            throw new Error(`Rank ${index} needs a positive memory budget`);
                        }
                        if (!Number.isFinite(reserveGiB) || reserveGiB < 0 || reserveGiB >= capacityGiB) {
                            throw new Error(`Rank ${index} reserve must be smaller than its budget`);
                        }
                    }
                    const measuredAutomaticReserveBytes = Number(
                        node.automatic_reserve_bytes || 0
                    );
                    const reserveBytes = manualAllowance !== null
                        ? Math.max(
                            0,
                            capacityBytes - Math.round(
                                Math.min(
                                    capacityBytes / gib,
                                    Number(manualAllowance),
                                ) * gib
                            ),
                        )
                        : (
                            Number.isFinite(measuredAutomaticReserveBytes)
                            && measuredAutomaticReserveBytes > 0
                                ? Math.round(measuredAutomaticReserveBytes)
                                : Math.round(reserveGiB * gib)
                        );
                    const payload = {
                        node_id: nodeId,
                        // Do not round a measured ceiling through the display's
                        // GiB value. A 0.3 GiB drift here was enough for the GUI
                        // to approve a stage the rank correctly refused.
                        capacity_bytes: capacityBytes,
                        reserve_bytes: reserveBytes,
                        manual_memory_limit: manualAllowance !== null,
                        role: node.role || 'headless',
                        memory_guard_tier:
                            this.globalSettings?.memory?.memory_guard_tier || 'balanced',
                    };
                    if (node.accelerator) payload.accelerator = node.accelerator;
                    if (node.fabric_kind) payload.fabric_kind = node.fabric_kind;
                    if (node.fabric_group_id) {
                        payload.fabric_group_id = node.fabric_group_id;
                    }
                    if (node.fabric_verified) payload.fabric_verified = true;
                    const target = Number(
                        (this.clusterWeightTargetsGiB || {})[nodeId]
                    );
                    if (Number.isFinite(target) && target > 0) {
                        // Measured memory and a Workstation role may reduce the
                        // safe ceiling after the slider was moved. Keep the
                        // preference inside the budget activation will use.
                        const usableGiB = Math.max(0, capacityGiB - reserveGiB);
                        payload.target_weight_bytes = Math.round(
                            Math.min(target, usableGiB) * gib
                        );
                    } else if (
                        index === 0
                        && this.clusterSplitGiB !== null
                        && this.clusterPlanNodes.length === 2
                    ) {
                        payload.max_weight_bytes = Math.round(
                            Number(this.clusterSplitGiB) * gib
                        );
                    }
                    return payload;
                });
            },

            // Give the simple page an automatic, read-only split preview.
            // This only inspects model metadata and invokes the planner; ranks
            // are launched exclusively by the explicit Start action.
            async previewClusterWeightBalance() {
                if (
                    this.clusterPlanLoading
                    || this.clusterAutoconfigureLoading
                    || this.clusterActivationLoading
                    || this.clusterWeightPlan()
                    || this.clusterPlanNodes.length < 2
                    || (this.clusterPlanMode === 'model'
                        && !this.clusterPlanModelPath.trim())
                ) {
                    return;
                }
                const key = this.clusterCurrentPlanSignature();
                if (key === this._clusterAutoPlanKey) return;
                this._clusterAutoPlanKey = key;
                await this.runClusterPlan();
            },

            async runClusterPlan() {
                if (this.clusterPlanLoading) return;
                const requestRevision = Number(this._clusterPlanRevision || 0);
                this.clusterPlanLoading = true;
                // Keep the prior split visible while a slider re-plans. The
                // signature guard still prevents activating that stale plan.
                this.clusterPlanError = '';
                this.clusterActivationResult = null;
                this.clusterPlanChanges = null;
                try {
                    const gib = 1024 ** 3;
                    const nodes = this.clusterNodePayloads({ validate: true });

                    const request = {
                        nodes,
                        execution_profile: this.clusterExecutionProfile,
                        // Sent because the planner branches on it: without it
                        // /plan defaulted to 1 and previewed an unequal
                        // pipeline while activation, carrying the size the
                        // one-click setup chose, deployed a hybrid plan.
                        tensor_parallel_size: Number(this.clusterPlanTensorParallelSize),
                        target_context_tokens: Number(
                            this.clusterTargetContextTokens || 8192
                        ),
                    };
                    if (this.clusterPlanMode === 'model') {
                        const path = this.clusterPlanModelPath.trim();
                        if (!path) throw new Error('Enter a downloaded model directory');
                        request.model_path = path;
                        request.model_source =
                            this.clusterPlanModelSource || '127.0.0.1';
                        if (this.clusterPlanModelSourcePython) {
                            request.model_source_python =
                                this.clusterPlanModelSourcePython;
                        }
                    } else {
                        const modelSizeGiB = Number(this.clusterPlanModelSizeGiB);
                        const layerCount = Number(this.clusterPlanLayerCount);
                        if (!Number.isFinite(modelSizeGiB) || modelSizeGiB <= 0) {
                            throw new Error('Enter a positive model weight size');
                        }
                        if (!Number.isInteger(layerCount) || layerCount <= 0) {
                            throw new Error('Enter a positive whole-number layer count');
                        }
                        request.model_size_bytes = Math.round(modelSizeGiB * gib);
                        request.layer_count = layerCount;
                    }

                    const response = await fetch('/admin/api/cluster/plan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(request),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Could not build shard plan'));
                    }
                    const result = await response.json();
                    if (
                        requestRevision
                        !== Number(this._clusterPlanRevision || 0)
                    ) {
                        return;
                    }
                    this.clusterPlan = result;
                    this._clusterPlanSignature = this.clusterCurrentPlanSignature();
                } catch (error) {
                    if (
                        requestRevision
                        !== Number(this._clusterPlanRevision || 0)
                    ) {
                        return;
                    }
                    this.clusterPlanError = error?.message || 'Could not build shard plan';
                } finally {
                    this.clusterPlanLoading = false;
                }
            },

            async loadClusterNodeRoles() {
                if (this.clusterNodeRoles.length) return;
                try {
                    const response = await fetch('/admin/api/cluster/node-roles');
                    if (response.ok) {
                        this.clusterNodeRoles = (await response.json()).roles || [];
                    }
                } catch (error) { /* tooltips are optional; planning is not */ }
            },

            clusterRoleDetail(key) {
                return (this.clusterNodeRoles.find(r => r.key === key) || {}).detail || '';
            },

            clusterRoleSummary(key) {
                return (this.clusterNodeRoles.find(r => r.key === key) || {}).summary || '';
            },

            // Ask each Mac what it can actually offer. Installed RAM overstates
            // a MacBook by ~20 GiB because the GPU cannot address it all, and a
            // plan built on that number is refused at load.
            async measureClusterBudgets() {
                if (this.clusterBudgetsLoading) return;
                const hosts = this.clusterBudgetHostsPayload();
                if (!hosts.length) {
                    this.clusterBudgetsError = 'Add a worker first.';
                    return;
                }
                this.clusterBudgetsLoading = true;
                this.clusterBudgetsError = '';
                try {
                    const roles = {};
                    this.clusterPlanNodes.forEach(node => {
                        roles[String(node.node_id || '').trim()] = node.role || 'headless';
                    });
                    const response = await fetch('/admin/api/cluster/node-budgets', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ hosts, roles }),
                    });
                    const body = await response.json();
                    if (!response.ok) {
                        this.clusterBudgetsError = this.clusterErrorMessage(
                            body.detail,
                            'Could not measure the workers.',
                        );
                        return;
                    }
                    const gib = 1024 ** 3;
                    let changed = false;
                    (body.nodes || []).forEach(measured => {
                        const node = this.clusterPlanNodes.find(
                            n => String(n.node_id || '').trim() === measured.node_id
                        );
                        if (!node || !measured.capacity_bytes) return;
                        const capacityGiB = Number(
                            (measured.capacity_bytes / gib).toFixed(2)
                        );
                        const automaticReserveGiB = Number(
                            (measured.reserve_bytes / gib).toFixed(2)
                        );
                        const manualAllowedGiB =
                            this.clusterManualMemoryAllowanceGiB(measured.node_id);
                        const reserveGiB = manualAllowedGiB === null
                            ? automaticReserveGiB
                            : Math.max(
                                0,
                                capacityGiB - Math.min(capacityGiB, manualAllowedGiB)
                            );
                        // The admission ceiling is derived from live memory
                        // pressure and wobbles by a few hundred MiB between
                        // polls. Re-planning the whole catalogue for that wobble
                        // emptied the model list and reset the topology controls
                        // every ten seconds. Only a drift that could change a
                        // plan invalidates anything.
                        const drift = (a, b) =>
                            Math.abs(Number(a || 0) - b) > 0.5;
                        if (
                            !drift(node.capacity_gib, capacityGiB)
                            && !drift(node.reserve_gib, reserveGiB)
                            && !drift(node.automatic_reserve_gib, automaticReserveGiB)
                        ) {
                            node.measured = measured.summary;
                            return;
                        }
                        changed = true;
                        node.capacity_gib = capacityGiB;
                        node.reserve_gib = reserveGiB;
                        node.automatic_reserve_gib = automaticReserveGiB;
                        node.capacity_bytes = Number(measured.capacity_bytes);
                        node.automatic_reserve_bytes = Number(
                            measured.reserve_bytes
                        );
                        node.measured = measured.summary;
                    });
                    if (changed) {
                        this.clusterCatalogue = null;
                        this.invalidateClusterPlan();
                    }
                    // Budgets are now known for this topology. The recurring
                    // discovery poll must not re-measure them: the peer's live
                    // admission ceiling wobbles by more than the drift guard on
                    // a busy Mac, and re-measuring every 10 s emptied the model
                    // list and reset the topology controls (visible jitter).
                    // A peer or node change resets this flag (see
                    // invalidateClusterPeer / the join-status merge); an
                    // explicit peer selection re-measures directly.
                    this._clusterBudgetsMeasured = true;
                } catch (error) {
                    this.clusterBudgetsError = String(error);
                } finally {
                    this.clusterBudgetsLoading = false;
                }
            },

            clusterBudgetHostsPayload() {
                const nodes = this.clusterPlanNodes || [];
                const sshHosts = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ];
                if (nodes.length < 2 || sshHosts.length !== nodes.length) return [];
                return nodes.map((node, rank) => ({
                    node_id: String(node.node_id || '').trim(),
                    ssh: sshHosts[rank],
                    ...(node.python_executable
                        ? { python_executable: node.python_executable }
                        : {}),
                }));
            },

            clusterClusterHostsPayload() {
                const nodes = this.clusterPlanNodes;
                const sshHosts = [
                    '127.0.0.1',
                    ...this.clusterWorkerPeers().map(peer => peer.ssh),
                ];
                if (nodes.length < 2 || sshHosts.length !== nodes.length) return [];
                const fabricHosts = this.clusterFabric?.hosts || [];
                const useDiscoveredFabric = !this.clusterIpsOverridden;
                return nodes.map((node, rank) => {
                    const ssh = sshHosts[rank];
                    const fabric = fabricHosts.find(item => item.host === ssh);
                    const fallbackIp = rank === 0
                        ? this.clusterLocalIp.trim()
                        : (rank === 1 ? this.clusterPeerIp.trim() : '');
                    return {
                        node_id: String(node.node_id || '').trim(),
                        ssh,
                        ips: useDiscoveredFabric && fabric?.ips?.length
                            ? [...fabric.ips]
                            : (fallbackIp ? [fallbackIp] : []),
                        rdma: useDiscoveredFabric ? (fabric?.rdma || []) : [],
                        ...(node.python_executable
                            ? { python_executable: node.python_executable }
                            : {}),
                    };
                });
            },

            clusterPinnedAssignment() {
                return (this.clusterWeightPlan()?.assignments || [])
                    .find(item => item.rank === 0) || null;
            },

            clusterPinnedWeightGiB() {
                const item = this.clusterPinnedAssignment();
                if (!item) return 0;
                return ((item.layer_weight_bytes || 0) + (item.fixed_weight_bytes || 0))
                    / (1024 ** 3);
            },

            clusterSplitAvailable() {
                const plan = this.clusterWeightPlan();
                const selected = this.clusterSelectedModel();
                const fit = selected
                    ? this.clusterCatalogueFit(selected.model_path)
                    : null;
                return Boolean(
                    plan
                    && Number(plan.tensor_parallel_size || 1) === 1
                    && (plan.assignments || []).length >= 2
                    // The generic memory planner can divide any layer list,
                    // but mlx-lm cannot execute a pipeline for every model
                    // architecture. Never show an adjustable multi-Mac split
                    // that activation will correctly refuse.
                    && fit?.splittable !== false
                );
            },

            clusterWeightPlan() {
                return this.clusterPlan || this.clusterAutoconfigure?.plan || null;
            },

            clusterWeightNodes() {
                const plan = this.clusterWeightPlan();
                const budgets = new Map(
                    (this.clusterPlanNodes || []).map(
                        node => [String(node.node_id || '').trim(), node]
                    )
                );
                return [...(plan?.assignments || [])]
                    .sort((left, right) => left.rank - right.rank)
                    .map((assignment, index) => {
                        const budget = budgets.get(assignment.node_id)
                            || this.clusterPlanNodes[index]
                            || {};
                        const actualGiB = (
                            Number(assignment.layer_weight_bytes || 0)
                            + Number(assignment.fixed_weight_bytes || 0)
                        ) / (1024 ** 3);
                        return {
                            ...assignment,
                            budget,
                            actualGiB,
                            usableGiB: Math.max(
                                0,
                                Number(budget.capacity_gib || 0)
                                - Number(budget.reserve_gib || 0)
                            ),
                        };
                    });
            },

            clusterWeightTargetGiB(node) {
                const selected = Number(
                    this.clusterWeightTargetsGiB[node?.node_id]
                );
                return Number.isFinite(selected) && selected > 0
                    ? selected
                    : Number(node?.actualGiB || 0);
            },

            clusterWeightTargetsCustomized() {
                return Object.values(this.clusterWeightTargetsGiB || {})
                    .some(value => Number.isFinite(Number(value)) && Number(value) > 0);
            },

            clusterWeightMinimumGiB() {
                const model = this.clusterWeightPlan()?.model || {};
                const fixed = Number(model.fixed_weight_bytes || 0) / (1024 ** 3);
                const layers = (model.layer_weight_bytes || [])
                    .map(value => Number(value) / (1024 ** 3))
                    .filter(value => Number.isFinite(value) && value > 0);
                return fixed + (layers.length ? Math.min(...layers) : 0.25);
            },

            clusterWeightTotalGiB() {
                return this.clusterWeightNodes().reduce(
                    (total, node) => total + Number(node.actualGiB || 0),
                    0
                );
            },

            clusterWeightBounds(node) {
                const nodes = this.clusterWeightNodes();
                const total = this.clusterWeightTotalGiB();
                const minimum = this.clusterWeightMinimumGiB();
                const others = nodes.filter(item => item.node_id !== node?.node_id);
                const othersMinimum = others.length * minimum;
                const othersMaximum = others.reduce(
                    (sum, item) => sum + item.usableGiB,
                    0
                );
                const min = Math.max(
                    minimum,
                    total - othersMaximum
                );
                const max = Math.min(
                    Number(node?.usableGiB || total),
                    total - othersMinimum
                );
                return {
                    min: Number(Math.max(0.25, min).toFixed(2)),
                    max: Number(Math.max(min, max).toFixed(2)),
                    step: 0.5,
                };
            },

            clusterSplitBounds() {
                const first = this.clusterWeightNodes()[0] || {};
                return this.clusterWeightBounds(first);
            },

            clusterSetWeightTarget(node, requested) {
                const nodes = this.clusterWeightNodes();
                if (nodes.length < 2 || !node) return;
                const targets = Object.fromEntries(
                    nodes.map(item => [
                        item.node_id,
                        this.clusterWeightTargetGiB(item),
                    ])
                );
                const bounds = this.clusterWeightBounds(node);
                const previous = targets[node.node_id];
                const desired = Math.min(
                    bounds.max,
                    Math.max(bounds.min, Number(requested))
                );
                const delta = desired - previous;
                targets[node.node_id] = desired;
                const others = nodes.filter(item => item.node_id !== node.node_id);
                const capacity = others.map(item => {
                    const current = targets[item.node_id];
                    const itemBounds = this.clusterWeightBounds(item);
                    return {
                        item,
                        current,
                        room: delta > 0
                            ? Math.max(0, current - itemBounds.min)
                            : Math.max(0, itemBounds.max - current),
                    };
                });
                const available = capacity.reduce(
                    (sum, entry) => sum + entry.room,
                    0
                );
                if (available > 0 && delta !== 0) {
                    capacity.forEach(entry => {
                        const share = Math.min(
                            entry.room,
                            Math.abs(delta) * (entry.room / available)
                        );
                        targets[entry.item.node_id] = entry.current
                            + (delta > 0 ? -share : share);
                    });
                }
                this.clusterWeightTargetsGiB = Object.fromEntries(
                    Object.entries(targets).map(([key, value]) => [
                        key,
                        Number(Number(value).toFixed(2)),
                    ])
                );
            },

            clusterPlanModelTotalGiB() {
                const gib = 1024 ** 3;
                if (this.clusterWeightPlan()?.model) {
                    const model = this.clusterWeightPlan().model;
                    const layers = (model.layer_weight_bytes || []).reduce((a, b) => a + b, 0);
                    return (layers + (model.fixed_weight_bytes || 0)) / gib;
                }
                return Number(this.clusterPlanModelSizeGiB) || 0;
            },

            // Replan as the slider moves. Coalesced so a drag issues one
            // request per settled position rather than one per pixel.
            clusterSplitChanged() {
                if (this._clusterSplitTimer) clearTimeout(this._clusterSplitTimer);
                this._clusterSplitTimer = setTimeout(() => {
                    this.runClusterPlan();
                }, 250);
            },

            clusterResetSplit() {
                this.clusterSplitGiB = null;
                this.clusterWeightTargetsGiB = {};
                this.clusterSplitChanged();
            },

            clusterWeightColor(index) {
                return [
                    '#818cf8',
                    '#14b8a6',
                    '#38bdf8',
                    '#a78bfa',
                    '#f59e0b',
                    '#34d399',
                ][Number(index) % 6];
            },

            clusterGiB(bytes) {
                return `${(Number(bytes || 0) / (1024 ** 3)).toFixed(1)} GiB`;
            },

            clusterTokens(count) {
                const value = Number(count || 0);
                if (!value) return 'unknown';
                if (value >= 1024 && value % 1024 === 0) {
                    return `${(value / 1024).toLocaleString()}k`;
                }
                if (value >= 1000) return `${Math.round(value / 1000).toLocaleString()}k`;
                return value.toLocaleString();
            },

            // What share of the model each node carries, for the bar.
            clusterSplitPercent(assignment) {
                const total = this.clusterWeightTotalGiB() * (1024 ** 3);
                if (!total) return 0;
                const held = (assignment.layer_weight_bytes || 0)
                    + (assignment.fixed_weight_bytes || 0);
                return Math.max(2, Math.round((held / total) * 100));
            },

            // What the server actually held back on this node, read off the
            // plan rather than recomputed here. A role's reserve rule lives on
            // one side only — restating it in JavaScript is how the number in
            // the card and the number in the plan drift apart.
            clusterPlannedReserveGiB(nodeId) {
                const item = (this.clusterPlan?.assignments || []).find(
                    entry => entry.node_id === String(nodeId || '').trim()
                );
                if (!item) return null;
                return Number(item.reserve_bytes || 0) / (1024 ** 3);
            },

            clusterPlanAssignmentsByLayer() {
                if (!this.clusterWeightPlan()?.assignments) return [];
                return [...this.clusterWeightPlan().assignments].sort(
                    (left, right) => left.start_layer - right.start_layer
                );
            },

            // The four things that must be true, in order, before a cluster can
            // serve. Each returns 'done' | 'active' | 'todo' so the stepper can
            // show where the user actually is rather than a wall of controls.
            get clusterSteps() {
                const paired = Boolean(
                    this.clusterPeerProbe && this.clusterPeerProbe.runtime_compatible
                );
                const linked = Boolean(
                    this.clusterTransports || this.clusterAutoconfigure
                );
                const planned = Boolean(
                    this.clusterAutoconfigure && this.clusterAutoconfigure.ready_to_activate
                );
                const liveJobs = (this.clusterStatus?.runtime_jobs?.jobs || [])
                    .filter(job => job.live);
                const running = liveJobs.length > 0;
                const configured = (this.clusterDeployments || []).length;

                const state = (done, isActive) =>
                    done ? 'done' : (isActive ? 'active' : 'todo');

                return [
                    {
                        key: 'pair',
                        title: 'Connect the workers',
                        hint: paired
                            ? this.clusterPeerProbe.status.node.hostname
                            : 'Find a peer and exchange keys',
                        state: state(paired, true),
                    },
                    {
                        key: 'link',
                        title: 'Check the link',
                        hint: this.clusterAutoconfigure
                            ? this.clusterAutoconfigure.link
                            : (linked ? 'Detected' : 'Thunderbolt, Ethernet or RDMA'),
                        state: state(linked, paired),
                    },
                    {
                        key: 'plan',
                        title: 'Split the model',
                        hint: this.clusterAutoconfigure
                            ? this.clusterAutoconfigure.summary
                            : 'Choose a model and set up automatically',
                        state: state(planned, linked),
                    },
                    {
                        key: 'run',
                        title: 'Activate',
                        hint: running
                            ? `${liveJobs.length} running`
                            : (configured
                                ? `${configured} configured · stopped`
                                : 'Start serving across the cluster'),
                        state: state(running, planned),
                    },
                ];
            },

            get clusterCurrentStep() {
                const active = this.clusterSteps.find(step => step.state === 'active');
                return active ? active.key : 'run';
            },

            // Every reason a peer would refuse to serve this model, each with
            // the command that fixes it where one exists. Empty until a setup
            // run has actually asked the peers.
            get clusterPreflightBlockers() {
                return this.clusterAutoconfigure?.preflight_issues || [];
            },

            // One identity per issue, shared by the list key and the copy
            // affordance so two panels cannot disagree about which is which.
            clusterIssueKey(issue) {
                return `${issue.node_id}/${issue.kind}/${issue.detail}`;
            },

            // A plan built before the last change to a control that changes it.
            // Activation refuses these server-side too; saying so here is what
            // stops the refusal being a mystery.
            clusterPlanIsStale() {
                return Boolean(this.clusterPlan)
                    && this._clusterPlanSignature !== this.clusterCurrentPlanSignature();
            },

            clusterActivationReady() {
                const maxKvSize = this.clusterMaxKvSize === ''
                    ? null
                    : Number(this.clusterMaxKvSize);
                const connections = Number(this.clusterRingConnectionsPerIp);
                const hosts = this.clusterClusterHostsPayload();
                if (
                    this.clusterActivationLoading
                    || this.clusterPlanMode !== 'model'
                    || !this.clusterPlan
                    || this.clusterPlanNodes.length < 2
                    || hosts.length !== this.clusterPlanNodes.length
                    || hosts.some(host => !host.ips?.length)
                    || !this.clusterPlanModelPath.trim()
                    || this._clusterPlanSignature !== this.clusterCurrentPlanSignature()
                    // A peer that cannot import what this model needs dies mid
                    // load, after every other rank has paid for its weights.
                    || this.clusterPreflightBlockers.length > 0
                    || (maxKvSize !== null && (
                        !Number.isInteger(maxKvSize) || maxKvSize <= 0
                    ))
                    || (
                        this.clusterBackend === 'ring'
                        && (
                            !Number.isInteger(connections)
                            || connections < 1
                            || connections > 32
                        )
                    )
                ) {
                    return false;
                }
                // A non-ring backend is only ever named by the fabric reader,
                // and only when it derived a full RDMA matrix — so this asserts
                // what activation is about to post rather than a device name
                // someone remembered.
                if (this.clusterBackend !== 'ring') {
                    return Boolean(this.clusterFabric?.rdma?.ok);
                }
                return true;
            },

            async activateClusterDeployment() {
                if (!this.clusterActivationReady()) return;
                if (!await this.prepareCudaSupernodesForActivation({
                    rebuildPlan: true,
                })) return;
                this.clusterActivationLoading = true;
                this.clusterActivationResult = null;
                this.clusterPlanChanges = null;
                this.clusterError = '';
                this.startClusterActivationProgress();
                try {
                    const nodes = this.clusterNodePayloads();
                    const hosts = this.clusterClusterHostsPayload();
                    const response = await fetch('/admin/api/cluster/deployments', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_path: this.clusterPlanModelPath.trim(),
                            model_source:
                                this.clusterPlanModelSource || '127.0.0.1',
                            model_source_python:
                                this.clusterPlanModelSourcePython || undefined,
                            backend: this.clusterBackend,
                            nodes,
                            hosts,
                            preflight: true,
                            execution_profile: this.clusterExecutionProfile,
                            auto_tune: this.clusterAutoTune,
                            sampling_rank_only: this.clusterSamplingRankOnly,
                            async_overlap: this.clusterAsyncOverlap,
                            cache_affinity: this.clusterCacheAffinity,
                            max_kv_size: this.clusterMaxKvSize === ''
                                ? null
                                : Number(this.clusterMaxKvSize),
                            ring_connections_per_ip: this.clusterBackend === 'ring'
                                ? Number(this.clusterRingConnectionsPerIp)
                                : null,
                            tensor_parallel_size: Number(this.clusterPlanTensorParallelSize),
                            target_context_tokens: Number(
                                this.clusterTargetContextTokens || 8192
                            ),
                            // The plan on screen, named. The server refuses to
                            // activate anything else, so a payload that has
                            // drifted from the preview fails loudly here
                            // instead of quietly launching a different split.
                            approved_placement: this.clusterPlan?.placement_signature || '',
                        }),
                    });
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Activation failed'));
                    }
                    this.clusterActivationResult = await response.json();
                    // The server may report a faster candidate, but it keeps
                    // the signed/staged placement if that candidate moves
                    // layers. Surface the measurement without claiming it ran.
                    this.clusterPlanChanges = this.clusterActivationResult.plan_changes || null;
                    if (this.clusterActivationResult.plan) {
                        this.clusterPlan = this.clusterActivationResult.plan;
                    }
                    this.clusterActivationProgress = 'Ready · canary passed';
                    await this.loadClusterRuntime();
                    await this.loadClusterDeployments();
                } catch (error) {
                    this.clusterError = error?.message || 'Activation failed';
                } finally {
                    this.stopClusterActivationProgress();
                    this.clusterActivationProgress = '';
                    this.clusterActivationLoading = false;
                }
            },

            async deactivateClusterDeployment(deploymentId) {
                if (this.clusterDeactivatingId) return;
                this.clusterDeactivatingId = deploymentId;
                this.clusterError = '';
                try {
                    const response = await fetch(
                        `/admin/api/cluster/deployments/${encodeURIComponent(deploymentId)}`,
                        { method: 'DELETE' },
                    );
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(await this.clusterResponseError(response, 'Deactivation failed'));
                    }
                    await this.loadClusterDeployments();
                } catch (error) {
                    this.clusterError = error?.message || 'Deactivation failed';
                } finally {
                    this.clusterDeactivatingId = '';
                }
            },

            formatClusterGiB(bytes) {
                const value = Number(bytes);
                if (!Number.isFinite(value)) return '—';
                return `${(value / (1024 ** 3)).toFixed(2)} GiB`;
            },

            formatClusterSeconds(seconds) {
                const value = Number(seconds);
                if (!Number.isFinite(value)) return '—';
                return value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(2)} s`;
            },

            formatClusterRate(rate) {
                const value = Number(rate);
                if (!Number.isFinite(value) || value < 0) return '—';
                if (value >= 1000) return `${(value / 1000).toFixed(1)}k tok/s`;
                return `${value.toFixed(value >= 100 ? 0 : 1)} tok/s`;
            },

            formatClusterCount(count) {
                const value = Number(count);
                if (!Number.isFinite(value) || value < 0) return '—';
                return Math.round(value).toLocaleString();
            },

            formatClusterPercent(value) {
                const numeric = Number(value);
                if (!Number.isFinite(numeric) || numeric < 0) return '—';
                return `${Math.round(numeric * 100)}%`;
            },

            formatClusterBandwidth(bytesPerSecond) {
                const value = Number(bytesPerSecond);
                if (!Number.isFinite(value) || value <= 0) return '—';
                return `${(value / (1024 ** 3)).toFixed(2)} GiB/s`;
            },

            clusterRuntimeProfile(job) {
                return (job?.performance_profiles || []).find(
                    profile => profile.rank === job.rank
                ) || null;
            },

            clusterRuntimePrimaryJob() {
                const jobs = this.clusterStatus?.runtime_jobs?.jobs || [];
                return jobs.find(job => job.live)
                    || [...jobs].sort(
                        (left, right) => String(right.updated_at || '')
                            .localeCompare(String(left.updated_at || ''))
                    )[0]
                    || null;
            },

            clusterRuntimePhaseLabel(job) {
                if (job?.live && job.phase === 'ready') return 'Cluster ready';
                if (job?.live) return 'Loading cluster';
                if (['peer_lost', 'launcher_lost', 'failed'].includes(job?.phase)) {
                    return 'Cluster stopped';
                }
                return 'Previous cluster run';
            },

            clusterRuntimeAssignments(job) {
                return [...(job?.assignments || [])].sort(
                    (left, right) => left.start_layer - right.start_layer
                );
            },

            clusterRuntimeRankStatus(job, assignment) {
                return (job?.ranks || []).find(
                    rank => Number(rank.rank) === Number(assignment?.rank)
                ) || null;
            },

            clusterRuntimeResidentWeights(job) {
                const ranks = job?.ranks || [];
                if (ranks.length) {
                    return ranks.reduce(
                        (total, rank) =>
                            total + Number(rank.measured_weight_bytes || 0),
                        0,
                    );
                }
                return Number(job?.measured_weight_bytes || 0);
            },

            clusterRuntimeCapacity(job) {
                return (job?.assignments || []).reduce(
                    (total, assignment) =>
                        total + Number(assignment.capacity_bytes || 0),
                    0,
                );
            },

            clusterPublicEndpoint() {
                return `${window.location.origin}/v1`;
            },

            clusterLiveLauncher() {
                const job = this.clusterNeuralFabricJob();
                if (!job) return null;
                return (this.clusterStatus?.runtime_jobs?.launchers || []).find(
                    launcher => launcher.deployment_id === job.deployment_id
                ) || null;
            },

            clusterLiveModelId() {
                return this.clusterLiveLauncher()?.model_id
                    || this.clusterActivationResult?.api?.model
                    || this.clusterLiveModelName();
            },

            clusterLiveModelName() {
                const job = this.clusterNeuralFabricJob();
                const deployment = (this.clusterDeployments || []).find(
                    item => item.deployment_id === job?.deployment_id
                ) || this.clusterPrimaryDeployment();
                const modelPath = deployment?.model || this.clusterPlanModelPath;
                const model = this.clusterAllModels().find(
                    item => item.model_path === modelPath
                ) || this.clusterSelectedModel();
                if (model) return this.clusterModelDisplayName(model);
                const parts = String(modelPath || job?.deployment_id || 'Cluster model')
                    .split('/');
                return parts[parts.length - 1] || 'Cluster model';
            },

            clusterRuntimeLayerCount(job) {
                const assignments = job?.assignments || [];
                return assignments.reduce(
                    (maximum, assignment) => Math.max(maximum, assignment.end_layer || 0),
                    0,
                );
            },

            clusterStateLabel(state) {
                return {
                    unavailable: 'Unavailable',
                    disabled: 'RDMA disabled',
                    enabled_no_peer: 'Ready · no peer',
                    peer_linked_config_pending: 'Peer linked · setup pending',
                    rdma_ready: 'Thunderbolt RDMA ready',
                }[state] || String(state || 'Unknown').replaceAll('_', ' ');
            },

            clusterTransportTone(state) {
                if (state === 'rdma_ready') {
                    return 'bg-green-50 text-green-700 border border-green-200';
                }
                if (state === 'peer_linked_config_pending') {
                    return 'bg-blue-50 text-blue-700 border border-blue-200';
                }
                if (state === 'enabled_no_peer') {
                    return 'bg-amber-50 text-amber-700 border border-amber-200';
                }
                return 'bg-neutral-100 text-neutral-600 border border-neutral-200';
            },

            clusterTransportIconTone(state) {
                if (state === 'rdma_ready') return 'bg-green-50 text-green-700';
                if (state === 'peer_linked_config_pending') return 'bg-blue-50 text-blue-700';
                if (state === 'enabled_no_peer') return 'bg-amber-50 text-amber-700';
                return 'bg-neutral-100 text-neutral-500';
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
                            network: { ...this.globalSettings.network, ...data.network },
                            auth: { ...this.globalSettings.auth, ...data.auth },
                            claude_code: { ...this.globalSettings.claude_code, ...data.claude_code },
                            integrations: { ...this.globalSettings.integrations, ...data.integrations },
                            idle_timeout: { ...this.globalSettings.idle_timeout, ...data.idle_timeout },
                            system: { ...this.globalSettings.system, ...data.system },
                        };
                        this.globalSettings.ui = data.ui || { language: 'en' };
                        if (
                            !this.globalSettings.server.distributed_inference_active
                            && this.mainTab === 'cluster'
                        ) {
                            this.mainTab = 'status';
                            this.syncTabStateToUrl();
                        }

                        // Sync idle timeout select value
                        this.idleTimeoutValue = this.globalSettings.idle_timeout?.idle_timeout_seconds != null
                            ? String(this.globalSettings.idle_timeout.idle_timeout_seconds)
                            : '';

                        // Normalize memory guard tier to one of the known values.
                        const validTiers = ['safe', 'balanced', 'aggressive', 'custom'];
                        if (!validTiers.includes(this.globalSettings.memory.memory_guard_tier)) {
                            this.globalSettings.memory.memory_guard_tier = 'balanced';
                        }

                        // Calculate cache percent from stored value (based on total capacity)
                        this.cachePercent = this.parseCacheToPercent(
                            this.globalSettings.cache.ssd_cache_max_size,
                            this.globalSettings.system.ssd_total_bytes
                        );
                        // Sync the cache string value from percent
                        this.updateCacheFromSlider();

                        // Calculate hot cache percent from stored value
                        this.globalSettings.cache.hot_cache_max_size = this.normalizeHotCacheMaxSize(
                            this.globalSettings.cache.hot_cache_max_size
                        );
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
                if (!s.server.max_audio_upload_size) errors.push('Maximum Audio Upload Size');
                if (!s.model.model_dirs || !s.model.model_dirs.some(d => d.trim())) errors.push('Model Directory');
                if (!s.scheduler.max_concurrent_requests) errors.push('Max Concurrent Requests');
                if (!s.scheduler.embedding_batch_size) errors.push('Embedding Batch Size');
                if (!s.cache.ssd_cache_max_size) errors.push('Max Cache Size');
                if (!s.sampling.max_context_window) errors.push('Max Context Window');
                if (!s.sampling.max_tokens) errors.push('Max Tokens');
                if (s.cache.gdn_snapshot_storage === 'ssd_sidecar' && s.cache.hot_cache_only) {
                    errors.push('GDN SSD sidecar requires Hot Cache Only to be disabled');
                }

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
                            sse_keepalive_mode: this.globalSettings.server.sse_keepalive_mode,
                            burst_decode_mode: this.globalSettings.server.burst_decode_mode,
                            preserve_mid_system_cache: this.globalSettings.server.preserve_mid_system_cache,
                            distributed_inference_enabled: this.globalSettings.server.distributed_inference_enabled,
                            max_audio_upload_size: this.globalSettings.server.max_audio_upload_size,
                            model_dirs: this.globalSettings.model.model_dirs.filter(d => d.trim()),
                            model_fallback: this.globalSettings.model.model_fallback,
                            hide_helper_models: this.globalSettings.model.hide_helper_models,
                            memory_prefill_memory_guard: this.globalSettings.memory.prefill_memory_guard,
                            memory_guard_tier: this.globalSettings.memory.memory_guard_tier,
                            memory_guard_custom_ceiling_gb: this.globalSettings.memory.memory_guard_custom_ceiling_gb,
                            max_concurrent_requests: this.globalSettings.scheduler.max_concurrent_requests,
                            embedding_batch_size: this.globalSettings.scheduler.embedding_batch_size,
                            chunked_prefill: this.globalSettings.scheduler.chunked_prefill,
                            prefill_priority: this.globalSettings.scheduler.prefill_priority,
                            decode_fairness: this.globalSettings.scheduler.decode_fairness,
                            cache_enabled: this.globalSettings.cache.enabled,
                            ssd_cache_dir: this.globalSettings.cache.ssd_cache_dir,
                            ssd_cache_max_size: this.globalSettings.cache.ssd_cache_max_size,
                            hot_cache_max_size: this.normalizeHotCacheMaxSize(
                                this.globalSettings.cache.hot_cache_max_size
                            ),
                            initial_cache_blocks: this.globalSettings.cache.initial_cache_blocks,
                            hot_cache_only: this.globalSettings.cache.hot_cache_only,
                            hot_cache_write_through: this.globalSettings.cache.hot_cache_write_through,
                            ane_compile_cache: this.globalSettings.cache.ane_compile_cache,
                            gdn_snapshot_storage: this.globalSettings.cache.gdn_snapshot_storage,
                            gdn_ssd_pending_max_size: this.globalSettings.cache.gdn_ssd_pending_max_size,
                            gdn_sidecar_precision: this.globalSettings.cache.gdn_sidecar_precision,
                            sampling_max_context_window: this.globalSettings.sampling.max_context_window,
                            sampling_max_context_window_policy: this.globalSettings.sampling.max_context_window_policy || null,
                            sampling_max_tokens: this.globalSettings.sampling.max_tokens,
                            sampling_temperature: this.globalSettings.sampling.temperature,
                            sampling_top_p: this.globalSettings.sampling.top_p,
                            sampling_top_k: this.globalSettings.sampling.top_k,
                            sampling_repetition_penalty: this.globalSettings.sampling.repetition_penalty,
                            mcp_config: this.globalSettings.mcp.config_path,
                            mcp_expose_tools: this.globalSettings.mcp.expose_tools,
                            hf_cache_enabled: this.globalSettings.huggingface.hf_cache_enabled,
                            network_http_proxy: this.globalSettings.network.http_proxy,
                            network_https_proxy: this.globalSettings.network.https_proxy,
                            network_no_proxy: this.globalSettings.network.no_proxy,
                            network_ca_bundle: this.globalSettings.network.ca_bundle,
                            ...(this.globalSettings.auth.api_key ? { api_key: this.globalSettings.auth.api_key } : {}),
                            skip_api_key_verification: this.globalSettings.auth.skip_api_key_verification,
                            idle_timeout_seconds: this.globalSettings.idle_timeout?.idle_timeout_seconds ?? null,
                        }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        this.saveSuccess = true;
                        this.saveMessage = data.message || 'Settings saved successfully';
                        // Refresh stats and model list (cache changes unload models)
                        await this.loadStats();
                        await this.loadModels();
                        setTimeout(() => { this.saveSuccess = false; }, 5000);
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await response.json();
                        this.saveError = Array.isArray(data.detail) ? data.detail.map(e => (e && typeof e === 'object') ? (e.msg || JSON.stringify(e)) : String(e)).join(', ') : (data.detail || window.t('js.error.save_settings_failed'));
                        // Reload settings to revert to server values
                        await this.loadGlobalSettings();
                    }
                } catch (err) {
                    console.error('Failed to save global settings:', err);
                    this.saveError = window.t('js.error.save_settings_failed');
                    // Reload settings to revert to server values
                    await this.loadGlobalSettings();
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
                        if (field === 'is_default') {
                            if (value === true) {
                                // Exactly this model is default now; every other
                                // model's flag clears.
                                this.models.forEach(m => { m.is_default = (m.id === modelId); });
                            } else {
                                const model = this.models.find(m => m.id === modelId);
                                if (model) model.is_default = false;
                            }
                        } else if (field === 'is_pinned') {
                            const model = this.models.find(m => m.id === modelId);
                            if (model) model.pinned = value;
                        } else if (field === 'is_hidden') {
                            const model = this.models.find(m => m.id === modelId);
                            if (model) model.is_hidden = value;
                        } else if (field === 'is_favorite') {
                            const model = this.models.find(m => m.id === modelId);
                            if (model) model.is_favorite = value;
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
                        const data = await response.json();
                        const model = this.models.find(m => m.id === modelId);
                        if (model && data.status === 'ok') model.loaded = false;
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

            // ===== Profiles / Templates =====
            formValuesForProfile() {
                const ms = this.modelSettings;
                const out = {};
                const isDiffusion = !!ms.is_diffusion_model;

                for (const k of this.profileFields.universal.concat(this.profileFields.model_specific)) {
                    if (k === 'chat_template_kwargs' || k === 'forced_ct_kwargs') continue;  // handle below
                    if (isDiffusion && this.isDiffusionUnsupportedProfileField(k)) continue;
                    if (k === 'thinking_budget_enabled') {
                        if (ms.enableThinkingBudget) out.thinking_budget_enabled = true;
                        continue;
                    }
                    if (k === 'thinking_budget_tokens') {
                        if (ms.enableThinkingBudget && ms.thinking_budget_tokens) {
                            out.thinking_budget_tokens = Number(ms.thinking_budget_tokens);
                        }
                        continue;
                    }
                    if (k === 'index_cache_freq') {
                        if (ms.enableIndexCache) out.index_cache_freq = ms.index_cache_freq || 4;
                        continue;
                    }
                    if (k === 'max_tool_result_tokens') {
                        if (ms.enableToolResultLimit && ms.max_tool_result_tokens) {
                            out.max_tool_result_tokens = Number(ms.max_tool_result_tokens);
                        }
                        continue;
                    }
                    if (k === 'guided_grammar_enabled') {
                        out.guided_grammar_enabled = !!ms.guided_grammar_enabled;
                        continue;
                    }
                    if (k === 'guided_grammar') {
                        const g = ms.guided_grammar_enabled ? (ms.guided_grammar || '').trim() : '';
                        if (g) out.guided_grammar = g;
                        continue;
                    }
                    // Standard field: omit unset values entirely — the server
                    // treats absent universal keys as "reset to default" when
                    // the profile is applied (snapshot semantics).
                    let v = ms[k];
                    if (v === undefined || v === null || v === '') continue;
                    if (typeof v === 'string' && !isNaN(Number(v))) v = Number(v);
                    out[k] = v;
                }

                // Build chat_template_kwargs and forced_ct_kwargs from ctKwargEntries
                const ctk = {};
                const forced = [];
                for (const e of (ms.ctKwargEntries || [])) {
                    if (e.type === 'enable_thinking') {
                        if (isDiffusion) continue;
                        ctk.enable_thinking = e.value === 'true';
                        if (e.force) forced.push('enable_thinking');
                    } else if (e.type === 'reasoning_effort') {
                        if (isDiffusion) continue;
                        const rawEffort = e.custom ? e.customValue : e.value;
                        const effort = this.coerceKwargValue(rawEffort);
                        if (String(effort).trim() !== '') {
                            ctk.reasoning_effort = effort;
                            if (e.force) forced.push('reasoning_effort');
                        }
                    } else if (e.type === 'custom' && e.key && e.key.trim()) {
                        if (isDiffusion && this.isDiffusionUnsupportedCtKwarg(e.key.trim())) {
                            continue;
                        }
                        ctk[e.key.trim()] = this.coerceKwargValue(e.value);
                        if (e.force) forced.push(e.key.trim());
                    }
                }
                if (Object.keys(ctk).length > 0) out.chat_template_kwargs = ctk;
                if (forced.length > 0) out.forced_ct_kwargs = forced;

                return out;
            },
            formValuesForTemplate() {
                const full = this.formValuesForProfile();
                const out = {};
                for (const k of this.profileFields.universal) {
                    if (k in full) out[k] = full[k];
                }
                return out;
            },
            computeDrift() {
                if (!this.activeProfileName) { this.profilesDrift = false; return; }
                const active = this.profiles.find(p => p.name === this.activeProfileName);
                if (!active) { this.profilesDrift = false; return; }
                const form = this.formValuesForProfile();
                for (const [k, v] of Object.entries(active.settings || {})) {
                    if (JSON.stringify(form[k]) !== JSON.stringify(v)) {
                        this.profilesDrift = true;
                        return;
                    }
                }
                this.profilesDrift = false;
            },
            matchedPreset(settings) {
                // Return the preset whose universal-field settings match the current model
                // settings exactly, otherwise null. Used by the models list to show "which
                // preset was applied" as a pill without any server-side tracking.
                if (!settings || !this.presets || this.presets.length === 0) return null;
                const universal = this.profileFields.universal || [];
                if (universal.length === 0) return null;
                const canonical = v => {
                    if (v === undefined || v === null || v === false) return null;
                    if (typeof v === 'object') {
                        return JSON.stringify(v, Object.keys(v).sort());
                    }
                    return v;
                };
                for (const p of this.presets) {
                    const ps = p.settings || {};
                    let ok = true;
                    for (const k of universal) {
                        if (canonical(ps[k]) !== canonical(settings[k])) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) return p;
                }
                return null;
            },
            profileTooltip(profile) {
                const lines = [];
                if (profile?.expose_as_model && profile.model_id) {
                    lines.push(profile.model_id);
                }
                const description = (profile?.description || '').trim();
                if (description) lines.push(description);
                return lines.join('\n');
            },
            async loadProfilesForModel(modelId) {
                this.profiles = [];
                try {
                    const r = await fetch(`/admin/api/models/${encodeURIComponent(modelId)}/profiles`);
                    if (r.ok) {
                        const data = await r.json();
                        this.profiles = data.profiles || [];
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (e) {
                    console.error('Failed to load profiles:', e);
                }
            },
            async loadTemplates() {
                try {
                    const r = await fetch('/admin/api/profile-templates');
                    if (r.ok) {
                        const data = await r.json();
                        this.templates = data.templates || [];
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (e) {
                    console.error('Failed to load templates:', e);
                }
            },
            async loadProfileFields() {
                try {
                    const r = await fetch('/admin/api/profile-fields');
                    if (r.ok) {
                        const data = await r.json();
                        this.profileFields = {
                            universal: data.universal || [],
                            model_specific: data.model_specific || [],
                        };
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (e) {
                    console.error('Failed to load profile field definitions:', e);
                }
            },

            async loadPresets() {
                // Use localStorage cache if present, otherwise fall back to the bundled file.
                const cached = localStorage.getItem('omlx_preset_cache');
                if (cached) {
                    try {
                        const parsed = JSON.parse(cached);
                        this.presets = parsed.presets || [];
                        return;
                    } catch (e) { /* corrupted, fall through */ }
                }
                try {
                    const r = await fetch('/admin/static/omlx_preset.json');
                    if (r.ok) {
                        const data = await r.json();
                        this.presets = data.presets || [];
                    }
                } catch (e) {
                    console.error('Failed to load bundled presets:', e);
                }
            },

            async refreshPresets() {
                if (this.refreshingPresets) return;
                this.refreshingPresets = true;
                try {
                    const r = await fetch('/admin/api/presets/refresh', { method: 'POST' });
                    if (r.ok) {
                        const data = await r.json();
                        this.presets = data.presets || [];
                        localStorage.setItem('omlx_preset_cache', JSON.stringify(data));
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (e) {
                    console.error('Preset refresh failed:', e);
                } finally {
                    this.refreshingPresets = false;
                }
            },

            // Floating tooltip shared by preset/profile pills (position:fixed escapes
            // the scroll container's overflow clipping, unlike absolute+group-hover).
            tip: { visible: false, text: '', x: 0, y: 0 },

            showTip(el, text) {
                if (!text) return;
                const rect = el.getBoundingClientRect();
                this.tip = {
                    visible: true,
                    text: text,
                    x: rect.left + rect.width / 2,
                    y: rect.bottom + 6,
                };
            },
            hideTip() {
                this.tip.visible = false;
            },

            isDiffusionModel(model) {
                const modelType = String(model?.config_model_type || '')
                    .toLowerCase()
                    .replace(/-/g, '_');
                return DIFFUSION_CONFIG_MODEL_TYPES.has(modelType);
            },

            isQwen35AnePrefillModel(model) {
                const modelType = String(model?.config_model_type || '')
                    .toLowerCase()
                    .replace(/-/g, '_');
                return QWEN35_ANE_CONFIG_PREFIXES.some(prefix => modelType.startsWith(prefix));
            },

            isDiffusionUnsupportedProfileField(field) {
                return DIFFUSION_UNSUPPORTED_PROFILE_FIELDS.has(field);
            },

            isDiffusionUnsupportedCtKwarg(key) {
                return DIFFUSION_UNSUPPORTED_CT_KWARGS.has(key);
            },

            draftModelSearchText(model) {
                return [
                    model?.id,
                    model?.name,
                    model?.model_path,
                    model?.source_repo_id,
                    model?.config_model_type,
                ].filter(Boolean).join(' ').toLowerCase();
            },

            isDraftModelBaseCandidate(model) {
                if (!model || model.virtual) return false;
                if (model.id === this.selectedModel?.id) return false;
                return model.model_type === 'llm' || model.model_type === 'vlm' || !model.model_type;
            },

            isDflashDraftModel(model) {
                const configType = String(model?.config_model_type || '').toLowerCase();
                if (DFLASH_DRAFTER_CONFIG_MODEL_TYPES.has(configType)) {
                    return true;
                }
                // DFlash 2 checkpoints are versioned ("-DFlash2"), so allow an
                // optional numeric suffix after the "dflash" token.
                return /(^|[-_/\s])dflash[0-9]*($|[-_/\s])/i.test(this.draftModelSearchText(model));
            },

            isVlmMtpDraftModel(model) {
                const configType = String(model?.config_model_type || '').toLowerCase();
                if (configType) {
                    return VLM_MTP_DRAFTER_CONFIG_MODEL_TYPES.has(configType);
                }
                return /assistant|(^|[-_/\s])mtp($|[-_/\s])/i.test(this.draftModelSearchText(model));
            },

            isSpecPrefillDraftModel(model) {
                return !this.isDflashDraftModel(model)
                    && !this.isVlmMtpDraftModel(model);
            },

            draftModelCandidates(filterFn, { fallbackToBase = true } = {}) {
                const base = (this.models || []).filter((model) => (
                    this.isDraftModelBaseCandidate(model)
                ));
                const filtered = base.filter(filterFn);
                return (filtered.length > 0 || !fallbackToBase) ? filtered : base;
            },

            specPrefillDraftModelCandidates() {
                return this.draftModelCandidates((model) => this.isSpecPrefillDraftModel(model));
            },

            dflashDraftModelCandidates() {
                return this.draftModelCandidates((model) => this.isDflashDraftModel(model));
            },

            vlmMtpDraftModelCandidates() {
                return this.draftModelCandidates(
                    (model) => this.isVlmMtpDraftModel(model),
                    { fallbackToBase: false },
                );
            },

            // Settings that materialize as per-request logits processors,
            // which the VLM MTP decode path cannot apply (#2399). Mirrors
            // vlm_mtp_processor_conflicts() in model_settings.py; neutral
            // values (repetition 1.0, presence 0.0) do not conflict.
            // Thinking budget is exempt: it is applied on the vlm_mtp path
            // at verify time (MTPProcessingSampler).
            vlmMtpProcessorConflict() {
                const ms = this.modelSettings;
                if (!ms) return false;
                const num = (v) => (v === null || v === undefined || v === '' ? null : Number(v));
                const rep = num(ms.repetition_penalty);
                const pres = num(ms.presence_penalty);
                return (rep !== null && rep !== 1.0)
                    || (pres !== null && pres !== 0.0)
                    || !!ms.guided_grammar_enabled;
            },

            // Coerce a raw kwarg string from the panel into its JSON type:
            // 'true'/'false' -> boolean, finite numeric strings -> number.
            coerceKwargValue(v) {
                if (v === 'true') return true;
                if (v === 'false') return false;
                if (String(v).trim() !== '') {
                    const numeric = Number(v);
                    if (Number.isFinite(numeric)) return numeric;
                }
                return v;
            },

            buildCtKwargEntries(chatTemplateKwargs, forcedCtKwargs, isDiffusion = false) {
                const ctk = chatTemplateKwargs || {};
                const forced = new Set(forcedCtKwargs || []);
                const entries = [];
                for (const [key, value] of Object.entries(ctk)) {
                    if (isDiffusion && this.isDiffusionUnsupportedCtKwarg(key)) {
                        continue;
                    }
                    if (key === 'enable_thinking') {
                        entries.push({
                            type: 'enable_thinking',
                            value: String(value),
                            force: forced.has('enable_thinking'),
                        });
                    } else if (key === 'reasoning_effort') {
                        const isPreset = typeof value === 'string'
                            && REASONING_EFFORT_PRESETS.has(value);
                        entries.push({
                            type: 'reasoning_effort',
                            value: isPreset ? value : 'low',
                            custom: !isPreset,
                            customValue: isPreset ? '' : String(value),
                            force: forced.has('reasoning_effort'),
                        });
                    } else {
                        entries.push({
                            type: 'custom',
                            key,
                            value: String(value),
                            force: forced.has(key),
                        });
                    }
                }
                return entries;
            },

            buildModelSettingsState(model, settings) {
                const s = settings || {};
                const isDiffusion = this.isDiffusionModel(model);
                const ctKwargEntries = this.buildCtKwargEntries(
                    s.chat_template_kwargs,
                    s.forced_ct_kwargs,
                    isDiffusion,
                );
                const isOcr = OCR_CONFIG_MODEL_TYPES.has(model?.config_model_type || '');
                return {
                    model_alias: s.model_alias || '',
                    model_type_override: s.model_type_override || '',
                    max_context_window: s.max_context_window || null,
                    max_tokens: s.max_tokens || null,
                    temperature: isOcr ? 0.0 : (s.temperature ?? null),
                    top_p: s.top_p ?? null,
                    top_k: s.top_k ?? null,
                    repetition_penalty: s.repetition_penalty ?? null,
                    min_p: s.min_p ?? null,
                    presence_penalty: s.presence_penalty ?? null,
                    force_sampling: s.force_sampling || false,
                    enable_thinking: s.enable_thinking ?? null,
                    thinking_default: model?.thinking_default ?? null,
                    qwen4_ple_ssd_offload: model?.qwen4_ple_ssd_offload_forced === true
                        || s.qwen4_ple_ssd_offload === true,
                    qwen4_ple_ssd_offload_supported:
                        model?.qwen4_ple_ssd_offload_supported === true,
                    qwen4_ple_ssd_offload_forced:
                        model?.qwen4_ple_ssd_offload_forced === true,
                    enableThinkingBudget: !!(s.thinking_budget_tokens),
                    thinking_budget_tokens: s.thinking_budget_tokens || null,
                    guided_grammar_enabled: s.guided_grammar_enabled || false,
                    guided_grammar: s.guided_grammar || '',
                    enableToolResultLimit: !!(s.max_tool_result_tokens),
                    max_tool_result_tokens: s.max_tool_result_tokens || null,
                    reasoning_parser: s.reasoning_parser || '',
                    ttl_seconds: s.ttl_seconds ?? null,
                    enableIndexCache: !!(s.index_cache_freq),
                    index_cache_freq: s.index_cache_freq || null,
                    turboquant_kv_enabled: s.turboquant_kv_enabled || false,
                    turboquant_kv_bits: s.turboquant_kv_bits || 4,
                    qwen35_ane_prefill_enabled: s.qwen35_ane_prefill_enabled || false,
                    qwen35_ane_prefill_sequence_length: s.qwen35_ane_prefill_sequence_length || 2048,
                    qwen35_ane_prefill_tail_padding_min_tokens: s.qwen35_ane_prefill_tail_padding_min_tokens ?? 0,
                    qwen35_ane_prefill_fraction: s.qwen35_ane_prefill_fraction ?? 0.53,
                    qwen35_ane_prefill_fused_down: s.qwen35_ane_prefill_fused_down || false,
                    qwen35_ane_prefill_max_layers: s.qwen35_ane_prefill_max_layers || 64,
                    qwen35_ane_prefill_dual_ane: s.qwen35_ane_prefill_dual_ane !== false,
                    qwen35_ane_prefill_gdn: s.qwen35_ane_prefill_gdn !== false,
                    qwen35_ane_prefill_gdn_fraction: s.qwen35_ane_prefill_gdn_fraction ?? 0.5,
                    qwen35_ane_prefill_gdn_max_layers: s.qwen35_ane_prefill_gdn_max_layers ?? 48,
                    qwen35_ane_prefill_cpu_enabled: s.qwen35_ane_prefill_cpu_enabled || false,
                    qwen35_ane_prefill_cpu_fraction: s.qwen35_ane_prefill_cpu_fraction ?? 0.135,
                    qwen35_ane_prefill_cpu_down_fraction: s.qwen35_ane_prefill_cpu_down_fraction ?? 0,
                    qwen35_ane_prefill_cpu_gdn_fraction: s.qwen35_ane_prefill_cpu_gdn_fraction ?? 0,
                    qwen35_ane_prefill_cpu_threads: s.qwen35_ane_prefill_cpu_threads ?? 8,
                    qwen35_ane_prefill_cpu_shared_resource: s.qwen35_ane_prefill_cpu_shared_resource !== false,
                    specprefill_enabled: s.specprefill_enabled || false,
                    specprefill_draft_model: s.specprefill_draft_model || '',
                    specprefill_keep_pct: s.specprefill_keep_pct ? String(s.specprefill_keep_pct) : '0.2',
                    specprefill_threshold: s.specprefill_threshold || null,
                    dflash_enabled: s.dflash_enabled || false,
                    dflash_draft_model: s.dflash_draft_model || '',
                    dflash_draft_quant_enabled: s.dflash_draft_quant_enabled || false,
                    dflash_draft_quant_weight_bits: s.dflash_draft_quant_weight_bits || 4,
                    dflash_draft_quant_activation_bits: s.dflash_draft_quant_activation_bits || 16,
                    dflash_draft_quant_group_size: s.dflash_draft_quant_group_size || 64,
                    dflash_max_ctx: s.dflash_max_ctx ?? null,
                    dflash_in_memory_cache: s.dflash_in_memory_cache !== false,
                    dflash_in_memory_cache_max_entries: s.dflash_in_memory_cache_max_entries || 4,
                    dflash_in_memory_cache_max_gib: s.dflash_in_memory_cache_max_bytes
                        ? Math.round(s.dflash_in_memory_cache_max_bytes / (1024 ** 3))
                        : 8,
                    dflash_ssd_cache: s.dflash_ssd_cache || false,
                    dflash_ssd_cache_max_gib: s.dflash_ssd_cache_max_bytes
                        ? Math.round(s.dflash_ssd_cache_max_bytes / (1024 ** 3))
                        : 20,
                    dflash_draft_window_size: s.dflash_draft_window_size ?? null,
                    dflash_draft_sink_size: s.dflash_draft_sink_size ?? 0,
                    dflash_block_size: s.dflash_block_size ?? null,
                    dflash_verify_mode: s.dflash_verify_mode || 'adaptive',
                    dflash_compatible: model?.dflash_compatible !== false,
                    dflash_compatibility_reason: model?.dflash_compatibility_reason || '',
                    dflash_ssd_cache_available: !!model?.dflash_ssd_cache_available,
                    mtp_enabled: s.mtp_enabled || false,
                    mtp_compatible: model?.mtp_compatible === true,
                    mtp_compatibility_reason: model?.mtp_compatibility_reason || '',
                    is_paroquant: model?.is_paroquant === true,
                    paroquant_reason: model?.paroquant_reason || '',
                    vlm_mtp_enabled: s.vlm_mtp_enabled || false,
                    vlm_mtp_draft_model: s.vlm_mtp_draft_model || '',
                    vlm_mtp_draft_block_size: s.vlm_mtp_draft_block_size ?? null,
                    ctKwargEntries,
                    is_diffusion_model: isDiffusion,
                    trust_remote_code: s.trust_remote_code || false,
                };
            },

            _resetPresetApplicableFields() {
                // Reset all fields a preset can touch so switching presets does not leave
                // stale values. Intentionally does NOT touch model_alias / model_type_override
                // / is_pinned / is_default / turboquant_* / dflash_* / specprefill_* / index_cache_*.
                const ms = this.modelSettings;
                ms.temperature = null;
                ms.top_p = null;
                ms.top_k = null;
                ms.min_p = null;
                ms.repetition_penalty = null;
                ms.presence_penalty = null;
                ms.force_sampling = false;
                ms.max_context_window = null;
                ms.max_tokens = null;
                ms.reasoning_parser = null;
                ms.guided_grammar_enabled = false;
                ms.guided_grammar = '';
                ms.ttl_seconds = null;
                ms.enable_thinking = null;
                ms.enableThinkingBudget = false;
                ms.thinking_budget_tokens = null;
                ms.enableToolResultLimit = false;
                ms.max_tool_result_tokens = null;
                ms.ctKwargEntries = [];
            },

            applyPresetToForm(preset) {
                // Reset first so previous preset's fields (e.g. presence_penalty) do not stick.
                this._resetPresetApplicableFields();
                const s = preset.settings || {};
                const ms = this.modelSettings;
                const isDiffusion = !!ms.is_diffusion_model;
                for (const k of Object.keys(s)) {
                    if (isDiffusion
                        && k !== 'chat_template_kwargs'
                        && k !== 'forced_ct_kwargs'
                        && this.isDiffusionUnsupportedProfileField(k)) {
                        continue;
                    }
                    if (k === 'thinking_budget_enabled') {
                        ms.enableThinkingBudget = !!s[k];
                    } else if (k === 'max_tool_result_tokens') {
                        ms.enableToolResultLimit = s[k] != null;
                        ms.max_tool_result_tokens = s[k] ?? null;
                    } else if (k === 'guided_grammar_enabled') {
                        ms.guided_grammar_enabled = !!s[k];
                    } else if (k === 'guided_grammar') {
                        ms.guided_grammar = s[k] || '';
                    } else if (k === 'chat_template_kwargs' || k === 'forced_ct_kwargs') {
                        ms.ctKwargEntries = this.buildCtKwargEntries(
                            s.chat_template_kwargs,
                            s.forced_ct_kwargs,
                            isDiffusion,
                        );
                    } else {
                        ms[k] = s[k];
                    }
                }
                this.activeProfileName = null;
                this.profilesDrift = false;
            },

            setScope(scope) {
                this.profileScope = scope;
                try { localStorage.setItem('omlx_profile_scope', scope); } catch (e) {}
            },

            isValidProfileName(name) {
                // Mirror of the backend rule (validate_profile_name) and the
                // Mac app's isValidSlug. api_name is the exposed model ID
                // suffix (<model>:<api_name>), so it must be a clean slug.
                return /^[a-z0-9][a-z0-9_-]{0,31}$/.test((name || '').trim());
            },
            slugifyProfileApiName(value) {
                let slug = (value || '')
                    .normalize('NFKD')
                    .replace(/[\u0300-\u036f]/g, '')
                    .toLowerCase()
                    .replace(/[^a-z0-9_-]+/g, '-')
                    .replace(/-+/g, '-')
                    .replace(/^[-_]+|[-_]+$/g, '')
                    .slice(0, 32)
                    .replace(/[-_]+$/g, '');
                return slug || 'profile';
            },
            async createProfile() {
                if (!this.selectedModel) return;
                this.profileError = '';
                const displayName = (this.newProfile.display_name || '').trim();
                if (!displayName) {
                    this.profileError = 'Name required';
                    return;
                }
                const apiName = (this.newProfile.api_name || this.slugifyProfileApiName(displayName)).trim();
                if (!this.isValidProfileName(apiName)) {
                    this.profileError = window.t('modal.model_settings.profiles.invalid_name');
                    return;
                }
                const autoId = 'p-' + Date.now().toString(36) + '-' +
                               Math.random().toString(36).slice(2, 6);
                const body = {
                    name: autoId,
                    display_name: displayName,
                    api_name: apiName,
                    description: (this.newProfile.description || '').trim() || null,
                    settings: this.formValuesForProfile(),
                    also_save_as_template: false,
                };
                try {
                    const r = await fetch(
                        `/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/profiles`,
                        { method: 'POST', headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify(body) }
                    );
                    if (r.ok) {
                        await this.loadProfilesForModel(this.selectedModel.id);
                        if (body.also_save_as_template) await this.loadTemplates();
                        this.showNewProfileForm = false;
                        this.newProfile = { display_name: '', api_name: '', api_name_touched: false, description: '', also_as_template: false };
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await r.json().catch(() => ({}));
                        this.profileError = data.detail || 'Failed to save profile';
                    }
                } catch (e) {
                    this.profileError = String(e);
                }
            },
            async applyProfileToForm(profile) {
                const seq = ++this._applySeq;
                this.profileError = '';
                try {
                    const r = await fetch(
                        `/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/profiles/${encodeURIComponent(profile.name)}/apply`,
                        { method: 'POST' }
                    );
                    if (seq !== this._applySeq) return;  // superseded by a newer click
                    if (r.ok) {
                        const data = await r.json();
                        const activeName = data.settings?.active_profile_name || profile.name;
                        const settings = {
                            ...(data.settings || {}),
                            active_profile_name: activeName,
                        };
                        this.modelSettings = this.buildModelSettingsState(
                            this.selectedModel,
                            settings,
                        );
                        if (this.selectedModel) {
                            this.selectedModel.settings = { ...settings };
                        }
                        this.activeProfileName = activeName;
                        this.profilesDrift = false;
                        // Update the models list so the profile badge reflects the change
                        const m = this.models.find(m => m.id === this.selectedModel.id);
                        if (m) m.settings = { ...settings };
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await r.json().catch(() => ({}));
                        this.profileError = data.detail || 'Failed to apply profile';
                    }
                } catch (e) {
                    this.profileError = String(e);
                    console.error('Failed to apply profile:', e);
                }
            },
            async applyTemplateToForm(template) {
                // Check if a profile with this template's name already exists
                const existingProfile = this.profiles.find(p => p.name === template.name);

                if (existingProfile) {
                    // Global templates are the source of truth in this scope.
                    const updatedProfile = await this.updateProfile(existingProfile.name, {
                        settings: template.settings,
                        source_template: template.name,
                    });
                    if (updatedProfile) {
                        await this.applyProfileToForm(updatedProfile);
                    }
                } else {
                    // Create a new profile from the template
                    const body = {
                        name: template.name,
                        display_name: template.display_name,
                        api_name: this.slugifyProfileApiName(template.display_name || template.name),
                        description: template.description || null,
                        settings: template.settings,
                        source_template: template.name,
                    };
                    
                    try {
                        const r = await fetch(
                            `/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/profiles`,
                            { method: 'POST', headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify(body) }
                        );
                        if (r.ok) {
                            // Reload profiles first to include the new one
                            await this.loadProfilesForModel(this.selectedModel.id);
                            // Find the newly created profile in the refreshed list
                            const newProfile = this.profiles.find(p => p.name === template.name);
                            if (newProfile) {
                                await this.applyProfileToForm(newProfile);
                            }
                        }
                    } catch (e) {
                        console.error('Failed to create profile from template:', e);
                    }
                }
            },
            async deleteProfile(name) {
                if (!this.selectedModel) return;
                try {
                    const r = await fetch(
                        `/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/profiles/${encodeURIComponent(name)}`,
                        { method: 'DELETE' }
                    );
                    if (r.ok) {
                        if (this.activeProfileName === name) this.activeProfileName = null;
                        await this.loadProfilesForModel(this.selectedModel.id);
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (e) {
                    console.error('Delete profile failed:', e);
                } finally {
                    this.profileDeleteConfirm = null;
                }
            },
            updateProfileFromEdit(p) {
                // Edit-dialog save. Internal profile name stays stable; api_name
                // is the API-visible suffix used by exposed model IDs.
                this.profileError = '';
                const displayName = (p._editDisplayName ?? p.display_name ?? p.name).trim();
                const apiName = (p._editApiName ?? p.api_name ?? p.name).trim();
                const description = (p._editDescription ?? p.description ?? '').trim();
                const exposeAsModel = !!(p._editExposeAsModel ?? p.expose_as_model);
                if (!displayName) {
                    this.profileError = 'Name required';
                    return;
                }
                if (!this.isValidProfileName(apiName)) {
                    this.profileError = window.t('modal.model_settings.profiles.invalid_name');
                    return;
                }
                const patch = {
                    display_name: displayName,
                    api_name: apiName,
                    description: description,
                    expose_as_model: exposeAsModel,
                };
                return this.updateProfile(p.name, patch);
            },
            updateProfileSettingsFromForm(p) {
                return this.updateProfile(p.name, {
                    settings: this.formValuesForProfile(),
                });
            },
            async updateProfile(name, patch) {
                // patch: { new_name?, display_name?, api_name?, description?, expose_as_model?, settings?, also_save_as_template? }
                if (!this.selectedModel) return;
                this.profileError = '';
                try {
                    const r = await fetch(
                        `/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/profiles/${encodeURIComponent(name)}`,
                        { method: 'PUT', headers: {'Content-Type':'application/json'},
                          body: JSON.stringify(patch) }
                    );
                    if (r.ok) {
                        const data = await r.json();
                        if (this.activeProfileName === name && patch.new_name) {
                            this.activeProfileName = patch.new_name;
                        }
                        await this.loadProfilesForModel(this.selectedModel.id);
                        if (patch.also_save_as_template) await this.loadTemplates();
                        this.editingProfile = null;
                        return data.profile;
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await r.json().catch(() => ({}));
                        this.profileError = data.detail || 'Failed to update profile';
                    }
                } catch (e) {
                    this.profileError = String(e);
                }
            },
            async createTemplate() {
                this.profileError = '';
                const displayName = this.newTemplate.display_name.trim();
                if (!displayName) {
                    this.profileError = 'Name required';
                    return;
                }
                const autoId = 't-' + Date.now().toString(36) + '-' +
                               Math.random().toString(36).slice(2, 6);
                const body = {
                    name: autoId,
                    display_name: displayName,
                    description: this.newTemplate.description.trim() || null,
                    // Only universal fields — server will filter again defensively.
                    settings: this.formValuesForTemplate(),
                };
                try {
                    const r = await fetch('/admin/api/profile-templates', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(body),
                    });
                    if (r.ok) {
                        await this.loadTemplates();
                        this.showNewTemplateForm = false;
                        this.newTemplate = { name: '', display_name: '', description: '' };
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await r.json().catch(() => ({}));
                        this.profileError = data.detail || 'Failed to save template';
                    }
                } catch (e) {
                    this.profileError = String(e);
                }
            },
            async updateTemplate(name, patch) {
                this.profileError = '';
                try {
                    const r = await fetch(
                        `/admin/api/profile-templates/${encodeURIComponent(name)}`,
                        { method: 'PUT', headers: {'Content-Type':'application/json'},
                          body: JSON.stringify(patch) }
                    );
                    if (r.ok) {
                        await this.loadTemplates();
                        this.editingTemplate = null;
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    } else {
                        const data = await r.json().catch(() => ({}));
                        this.profileError = data.detail || 'Failed to update template';
                    }
                } catch (e) {
                    this.profileError = String(e);
                }
            },
            async deleteTemplate(name) {
                try {
                    const r = await fetch(
                        `/admin/api/profile-templates/${encodeURIComponent(name)}`,
                        { method: 'DELETE' }
                    );
                    if (r.ok) {
                        await this.loadTemplates();
                    } else if (r.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (e) {
                    console.error('Delete template failed:', e);
                } finally {
                    this.templateDeleteConfirm = null;
                }
            },

            aneTuningForSelectedModel() {
                return !!this.selectedModel
                    && this.aneTuning.modelId === this.selectedModel.id;
            },

            aneTuningProgressPercent() {
                const current = Number(this.aneTuning.status?.current || 0);
                const total = Number(
                    this.aneTuning.status?.total || this.aneTuning.total || 0
                );
                if (total <= 0) return 0;
                return Math.max(0, Math.min(100, current / total * 100));
            },

            aneTuningRecommendationText() {
                const recommendation = this.aneTuning.status?.recommendation;
                if (!recommendation) return '';
                const measured = recommendation.processing_tps !== null
                    && recommendation.processing_tps !== undefined;
                const speed = Number(recommendation.processing_tps || 0).toFixed(1);
                const speedup = Number(recommendation.speedup_percent || 0);
                const speedupText = `${speedup >= 0 ? '+' : ''}${speedup.toFixed(1)}%`;
                const speedSuffix = measured
                    ? ` · ${speed} prompt tok/s · ${speedupText}`
                    : '';
                if (!recommendation.enabled) {
                    return `GPU only${speedSuffix}`;
                }
                const parts = [
                    `${recommendation.fused_down ? 'Fused MLP per ANE' : 'MLP'} ${Math.round(Number(recommendation.mlp_fraction) * 100)}%`,
                ];
                if (recommendation.gdn_enabled) {
                    parts.push(
                        `GDN ${Math.round(Number(recommendation.gdn_fraction) * 100)}%`
                    );
                } else {
                    parts.push('GDN off');
                }
                if (recommendation.cpu_enabled) {
                    parts.push(
                        `CPU gate ${Math.round(Number(recommendation.cpu_fraction || 0) * 100)}%`,
                        `CPU down ${Math.round(Number(recommendation.cpu_down_fraction || 0) * 100)}%`,
                        `CPU GDN ${Math.round(Number(recommendation.cpu_gdn_fraction || 0) * 100)}%`,
                    );
                }
                if (Number(recommendation.tail_padding_min_tokens || 0) > 0) {
                    parts.push(
                        `Pad tails ≥${Number(recommendation.tail_padding_min_tokens)}`
                    );
                }
                return `${parts.join(' · ')}${speedSuffix}`;
            },

            aneTuningResultText(result) {
                if (result?.processing_tps === null
                    || result?.processing_tps === undefined) {
                    if (result?.latency_ms !== null
                        && result?.latency_ms !== undefined) {
                        return `${Number(result.latency_ms).toFixed(2)} ms`;
                    }
                    // Keep unfinished rows visible but leave their result cell
                    // blank, including the candidate that stopped the run.
                    return '';
                }
                const speed = Number(result.processing_tps).toFixed(1);
                if (result.speedup_percent === null
                    || result.speedup_percent === undefined) {
                    return speed;
                }
                const speedup = Number(result.speedup_percent);
                return `${speed} (${speedup >= 0 ? '+' : ''}${speedup.toFixed(1)}%)`;
            },

            _scheduleANETuningPoll() {
                if (this._aneTuningPollTimer) {
                    clearTimeout(this._aneTuningPollTimer);
                }
                if (!this.aneTuning.running) return;
                this._aneTuningPollTimer = setTimeout(
                    () => this.pollANETuning(),
                    1000,
                );
            },

            async startANETuning() {
                if (!this.selectedModel || this.aneTuning.running) return;
                const modelId = this.selectedModel.id;
                if (this._aneTuningPollTimer) {
                    clearTimeout(this._aneTuningPollTimer);
                    this._aneTuningPollTimer = null;
                }
                this.aneTuning = {
                    tuningId: null,
                    modelId,
                    running: true,
                    cancelling: false,
                    applying: false,
                    applied: false,
                    total: 0,
                    status: null,
                    error: '',
                };
                try {
                    const response = await fetch('/admin/api/bench/ane-tune/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: modelId,
                            sequence_length: parseInt(
                                this.modelSettings.qwen35_ane_prefill_sequence_length
                            ) || 2048,
                            repeats: 2,
                            allow_cpu: this.aneTuningOverrides.allowCpu,
                            allow_cpu_gate: this.aneTuningOverrides.allowCpu
                                && this.aneTuningOverrides.allowCpuGate,
                            allow_cpu_down: this.aneTuningOverrides.allowCpu
                                && this.aneTuningOverrides.allowCpuDown,
                            allow_ane_gdn: this.aneTuningOverrides.allowAneGdn,
                            allow_cpu_gdn: this.aneTuningOverrides.allowCpu
                                && this.aneTuningOverrides.allowAneGdn
                                && this.aneTuningOverrides.allowCpuGdn,
                            allow_cpu_shared_resource: this.aneTuningOverrides.allowCpu
                                && this.aneTuningOverrides.allowCpuSharedResource,
                        }),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to start ANE tuning.');
                    }
                    this.aneTuning.tuningId = data.tuning_id;
                    this.aneTuning.total = Number(data.total || 0);
                    await this.pollANETuning();
                } catch (error) {
                    this.aneTuning.running = false;
                    this.aneTuning.error = error.message || String(error);
                }
            },

            async pollANETuning() {
                const tuningId = this.aneTuning.tuningId;
                if (!tuningId || !this.aneTuning.running) return;
                try {
                    const response = await fetch(
                        `/admin/api/bench/ane-tune/${encodeURIComponent(tuningId)}/results`
                    );
                    const data = await response.json().catch(() => ({}));
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to read ANE tuning progress.');
                    }
                    if (this.aneTuning.tuningId !== tuningId) return;
                    if (!data.termination_reason && data.status === 'error') {
                        data.termination_reason = data.error || data.message || 'ANE tuning failed.';
                    }
                    this.aneTuning.status = data;
                    this.aneTuning.total = Number(data.total || this.aneTuning.total || 0);
                    this.aneTuning.running = data.status === 'running';
                    this.aneTuning.cancelling = false;
                    if (data.status === 'error' || data.status === 'cancelled') {
                        // Early termination belongs beside the partial result
                        // matrix. Keep this field for request/transport errors.
                        this.aneTuning.error = '';
                    }
                    this._scheduleANETuningPoll();
                } catch (error) {
                    this.aneTuning.running = false;
                    this.aneTuning.cancelling = false;
                    this.aneTuning.error = error.message || String(error);
                }
            },

            async cancelANETuning() {
                const tuningId = this.aneTuning.tuningId;
                if (!tuningId || !this.aneTuning.running || this.aneTuning.cancelling) return;
                this.aneTuning.cancelling = true;
                try {
                    const response = await fetch(
                        `/admin/api/bench/ane-tune/${encodeURIComponent(tuningId)}/cancel`,
                        { method: 'POST' },
                    );
                    const data = await response.json().catch(() => ({}));
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to cancel ANE tuning.');
                    }
                    await this.pollANETuning();
                } catch (error) {
                    this.aneTuning.cancelling = false;
                    this.aneTuning.error = error.message || String(error);
                }
            },

            async applyANETuningRecommendation() {
                if (!this.selectedModel || !this.aneTuningForSelectedModel()) return;
                const recommendation = this.aneTuning.status?.recommendation;
                if (!recommendation || this.aneTuning.applying) return;
                const patch = {
                    qwen35_ane_prefill_enabled: !!recommendation.enabled,
                    qwen35_ane_prefill_sequence_length: Number(recommendation.sequence_length),
                    qwen35_ane_prefill_tail_padding_min_tokens: Number(
                        recommendation.tail_padding_min_tokens || 0
                    ),
                };
                if (recommendation.enabled) {
                    patch.qwen35_ane_prefill_fraction = Number(recommendation.mlp_fraction);
                    patch.qwen35_ane_prefill_fused_down = !!recommendation.fused_down;
                    patch.qwen35_ane_prefill_gdn = !!recommendation.gdn_enabled;
                    if (recommendation.gdn_enabled) {
                        patch.qwen35_ane_prefill_gdn_fraction = Number(
                            recommendation.gdn_fraction
                        );
                    }
                    patch.qwen35_ane_prefill_cpu_enabled = !!recommendation.cpu_enabled;
                    patch.qwen35_ane_prefill_cpu_fraction = Number(
                        recommendation.cpu_fraction || 0
                    );
                    patch.qwen35_ane_prefill_cpu_down_fraction = Number(
                        recommendation.cpu_down_fraction || 0
                    );
                    patch.qwen35_ane_prefill_cpu_gdn_fraction = Number(
                        recommendation.cpu_gdn_fraction || 0
                    );
                    if (recommendation.cpu_threads !== null
                        && recommendation.cpu_threads !== undefined) {
                        patch.qwen35_ane_prefill_cpu_threads = Number(
                            recommendation.cpu_threads
                        );
                    }
                    if (recommendation.cpu_shared_resource !== null
                        && recommendation.cpu_shared_resource !== undefined) {
                        patch.qwen35_ane_prefill_cpu_shared_resource =
                            !!recommendation.cpu_shared_resource;
                    }
                }

                this.aneTuning.applying = true;
                this.aneTuning.error = '';
                try {
                    const response = await fetch(
                        `/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/settings`,
                        {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(patch),
                        },
                    );
                    const data = await response.json().catch(() => ({}));
                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to apply ANE tuning result.');
                    }
                    Object.assign(this.modelSettings, patch);
                    const model = this.models.find(item => item.id === this.selectedModel.id);
                    if (model && data.settings) {
                        model.settings = { ...data.settings };
                    }
                    this.aneTuning.applied = true;
                } catch (error) {
                    this.aneTuning.error = error.message || String(error);
                } finally {
                    this.aneTuning.applying = false;
                }
            },

            async openModelSettings(model) {
                this.profileError = '';
                this.showNewProfileForm = false;
                this.showNewTemplateForm = false;
                this.editingProfile = null;
                this.editingTemplate = null;
                this.profileDeleteConfirm = null;
                this.templateDeleteConfirm = null;
                const isDiffusion = this.isDiffusionModel(model);
                this.activeProfileName = isDiffusion
                    ? null
                    : ((model.settings && model.settings.active_profile_name) || null);
                if (isDiffusion) {
                    this.profiles = [];
                    this.templates = [];
                    this.profilesDrift = false;
                } else {
                    try {
                        const saved = localStorage.getItem('omlx_profile_scope');
                        if (saved === 'preset' || saved === 'global' || saved === 'model') {
                            this.profileScope = saved;
                        }
                    } catch (e) {}
                    await Promise.all([
                        this.loadProfilesForModel(model.id),
                        this.loadTemplates(),
                    ]);
                    if (this.reasoningParsers.length === 0) {
                        try {
                            const resp = await fetch('/admin/api/grammar/parsers');
                            if (resp.ok) this.reasoningParsers = await resp.json();
                            else if (resp.status === 401) window.location.href = '/admin';
                        } catch (_) { /* network error */ }
                    }
                }
                this.selectedModel = model;
                this.modelSettings = this.buildModelSettingsState(
                    model,
                    model.settings || {},
                );
                if (isDiffusion) {
                    this.profilesDrift = false;
                } else {
                    this.computeDrift();
                }
                this.showModelSettingsModal = true;
            },

            async importMtplxSidecar() {
                if (!this.selectedModel || this.importingMtplx) return;
                this.importingMtplx = true;
                try {
                    const response = await fetch(`/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/import-mtplx`, {
                        method: 'POST',
                    });
                    const data = await response.json().catch(() => ({}));
                    if (!response.ok) {
                        alert(data.detail || window.t('js.error.mtplx_import_failed'));
                        return;
                    }
                    if (data.message) alert(data.message);
                    // Refresh so mtp_compatible flips and the toggle unlocks.
                    await this.loadModels();
                    const model = this.models.find(m => m.id === this.selectedModel.id);
                    if (model) await this.openModelSettings(model);
                } catch (e) {
                    alert(window.t('js.error.mtplx_import_failed'));
                } finally {
                    this.importingMtplx = false;
                }
            },

            validateQwenAneSettings() {
                if (!this.modelSettings.qwen35_ane_prefill_enabled) return null;

                const integer = (value, label, minimum) => {
                    if (value === '' || value === null || value === undefined) {
                        return `${label} is required.`;
                    }
                    const number = Number(value);
                    if (!Number.isInteger(number)) return `${label} must be an integer.`;
                    if (number < minimum) return `${label} must be at least ${minimum}.`;
                    return null;
                };
                const fraction = (value, label, minimum, maximum) => {
                    if (value === '' || value === null || value === undefined) {
                        return `${label} is required.`;
                    }
                    const number = Number(value);
                    if (!Number.isFinite(number)) return `${label} must be a number.`;
                    if (number < minimum || number > maximum) {
                        return `${label} must be between ${minimum} and ${maximum}.`;
                    }
                    return null;
                };

                const sequenceLength = Number(this.modelSettings.qwen35_ane_prefill_sequence_length);
                let error = integer(sequenceLength, 'ANE prompt block', 1024);
                if (!error && sequenceLength % 64 !== 0) {
                    error = 'ANE prompt block must be a multiple of 64.';
                }
                if (error) return error;
                error = integer(
                    this.modelSettings.qwen35_ane_prefill_tail_padding_min_tokens,
                    'ANE tail padding threshold',
                    0,
                );
                if (error) return error;
                if (Number(this.modelSettings.qwen35_ane_prefill_tail_padding_min_tokens) >= sequenceLength) {
                    return 'ANE tail padding threshold must be less than the prompt block.';
                }
                error = fraction(this.modelSettings.qwen35_ane_prefill_fraction, 'MLP ANE fraction', 0.05, 0.90);
                if (error) return error;
                error = integer(this.modelSettings.qwen35_ane_prefill_max_layers, 'ANE MLP layer limit', 1);
                if (error) return error;

                if (this.modelSettings.qwen35_ane_prefill_cpu_enabled) {
                    error = fraction(this.modelSettings.qwen35_ane_prefill_cpu_fraction, 'CPU MLP fraction', 0, 0.25);
                    if (error) return error;
                    error = fraction(this.modelSettings.qwen35_ane_prefill_cpu_down_fraction, 'CPU MLP down fraction', 0, 0.50);
                    if (error) return error;
                    error = fraction(this.modelSettings.qwen35_ane_prefill_cpu_gdn_fraction, 'CPU GDN fraction', 0, 0.50);
                    if (error) return error;
                    error = integer(this.modelSettings.qwen35_ane_prefill_cpu_threads, 'CPU worker count', 0);
                    if (error) return error;
                    if (Number(this.modelSettings.qwen35_ane_prefill_cpu_threads) > 64) {
                        return 'CPU worker count must be between 0 and 64.';
                    }
                    if (Number(this.modelSettings.qwen35_ane_prefill_fraction)
                        + Number(this.modelSettings.qwen35_ane_prefill_cpu_fraction) >= 1) {
                        return 'MLP ANE and CPU fractions must total less than 1.0.';
                    }
                }

                if (this.modelSettings.qwen35_ane_prefill_gdn) {
                    error = fraction(this.modelSettings.qwen35_ane_prefill_gdn_fraction, 'GDN ANE fraction', 0.05, 0.90);
                    if (error) return error;
                    if (this.modelSettings.qwen35_ane_prefill_cpu_enabled
                        && Number(this.modelSettings.qwen35_ane_prefill_gdn_fraction)
                        + Number(this.modelSettings.qwen35_ane_prefill_cpu_gdn_fraction) >= 1) {
                        return 'GDN ANE and CPU fractions must total less than 1.0.';
                    }
                    error = integer(this.modelSettings.qwen35_ane_prefill_gdn_max_layers, 'ANE GDN layer limit', 0);
                    if (error) return error;
                }
                return null;
            },

            async saveModelSettings() {
                if (!this.selectedModel) return;

                const qwenAneValidationError = this.validateQwenAneSettings();
                if (qwenAneValidationError) {
                    alert(qwenAneValidationError);
                    return;
                }

                this.savingModelSettings = true;
                try {
                    const response = await fetch(`/admin/api/models/${encodeURIComponent(this.selectedModel.id)}/settings`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify((() => {
                            const isDiffusion = !!this.modelSettings.is_diffusion_model;
                            // Build chat_template_kwargs and forced_ct_kwargs from ctKwargEntries
                            const chatTemplateKwargs = {};
                            const forcedCtKwargs = [];
                            for (const entry of this.modelSettings.ctKwargEntries) {
                                if (entry.type === 'enable_thinking') {
                                    if (isDiffusion) continue;
                                    chatTemplateKwargs.enable_thinking = entry.value === 'true';
                                    if (entry.force) forcedCtKwargs.push('enable_thinking');
                                } else if (entry.type === 'reasoning_effort') {
                                    if (isDiffusion) continue;
                                    const rawEffort = entry.custom
                                        ? entry.customValue
                                        : entry.value;
                                    const effort = this.coerceKwargValue(rawEffort);
                                    if (String(effort).trim() !== '') {
                                        chatTemplateKwargs.reasoning_effort = effort;
                                        if (entry.force) forcedCtKwargs.push('reasoning_effort');
                                    }
                                } else if (entry.type === 'custom' && entry.key && entry.key.trim()) {
                                    const val = this.coerceKwargValue(entry.value);
                                    const key = entry.key.trim();
                                    if (isDiffusion && this.isDiffusionUnsupportedCtKwarg(key)) {
                                        continue;
                                    }
                                    chatTemplateKwargs[key] = val;
                                    if (entry.force) forcedCtKwargs.push(key);
                                }
                            }
                            const payload = {
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
                                reasoning_parser: this.modelSettings.reasoning_parser || null,
                                ttl_seconds: this.modelSettings.ttl_seconds || null,
                                index_cache_freq: this.modelSettings.enableIndexCache
                                    ? (this.modelSettings.index_cache_freq || 4)
                                    : 0,
                                enable_thinking: this.modelSettings.enable_thinking,
                                qwen4_ple_ssd_offload:
                                    !!this.modelSettings.qwen4_ple_ssd_offload,
                                thinking_budget_enabled: this.modelSettings.enableThinkingBudget,
                                thinking_budget_tokens: this.modelSettings.enableThinkingBudget
                                    ? (this.modelSettings.thinking_budget_tokens || null)
                                    : 0,
                                guided_grammar_enabled: this.modelSettings.guided_grammar_enabled,
                                guided_grammar: this.modelSettings.guided_grammar_enabled
                                    ? (this.modelSettings.guided_grammar || null)
                                    : null,
                                max_tool_result_tokens: this.modelSettings.enableToolResultLimit
                                    ? (this.modelSettings.max_tool_result_tokens || null)
                                    : 0,
                                chat_template_kwargs: Object.keys(chatTemplateKwargs).length > 0
                                    ? chatTemplateKwargs : null,
                                forced_ct_kwargs: forcedCtKwargs.length > 0
                                    ? forcedCtKwargs : null,
                                turboquant_kv_enabled: this.modelSettings.turboquant_kv_enabled,
                                turboquant_kv_bits: this.modelSettings.turboquant_kv_enabled
                                    ? (parseFloat(this.modelSettings.turboquant_kv_bits) || 4)
                                    : 4,
                                qwen35_ane_prefill_enabled: !!this.modelSettings.qwen35_ane_prefill_enabled,
                                // Validation only runs when the feature is enabled, so a
                                // blank numeric input must fall back to the server default
                                // instead of coercing to 0 and failing an unrelated save.
                                qwen35_ane_prefill_sequence_length: Number(this.modelSettings.qwen35_ane_prefill_sequence_length) || 2048,
                                qwen35_ane_prefill_tail_padding_min_tokens: Number.isFinite(Number(this.modelSettings.qwen35_ane_prefill_tail_padding_min_tokens))
                                    ? Number(this.modelSettings.qwen35_ane_prefill_tail_padding_min_tokens)
                                    : 0,
                                qwen35_ane_prefill_fraction: Number(this.modelSettings.qwen35_ane_prefill_fraction) || 0.53,
                                qwen35_ane_prefill_max_layers: Number(this.modelSettings.qwen35_ane_prefill_max_layers) || 64,
                                qwen35_ane_prefill_dual_ane: !!this.modelSettings.qwen35_ane_prefill_dual_ane,
                                qwen35_ane_prefill_gdn: !!this.modelSettings.qwen35_ane_prefill_gdn,
                                qwen35_ane_prefill_gdn_fraction: Number(this.modelSettings.qwen35_ane_prefill_gdn_fraction) || 0.5,
                                qwen35_ane_prefill_gdn_max_layers: Number.isFinite(Number(this.modelSettings.qwen35_ane_prefill_gdn_max_layers))
                                    ? Number(this.modelSettings.qwen35_ane_prefill_gdn_max_layers)
                                    : 48,
                                qwen35_ane_prefill_cpu_enabled: !!this.modelSettings.qwen35_ane_prefill_cpu_enabled,
                                qwen35_ane_prefill_cpu_fraction: Number(this.modelSettings.qwen35_ane_prefill_cpu_fraction),
                                qwen35_ane_prefill_cpu_down_fraction: Number.isFinite(Number(this.modelSettings.qwen35_ane_prefill_cpu_down_fraction))
                                    ? Number(this.modelSettings.qwen35_ane_prefill_cpu_down_fraction)
                                    : 0,
                                qwen35_ane_prefill_cpu_gdn_fraction: Number.isFinite(Number(this.modelSettings.qwen35_ane_prefill_cpu_gdn_fraction))
                                    ? Number(this.modelSettings.qwen35_ane_prefill_cpu_gdn_fraction)
                                    : 0,
                                qwen35_ane_prefill_cpu_threads: Number.isFinite(Number(this.modelSettings.qwen35_ane_prefill_cpu_threads))
                                    ? Number(this.modelSettings.qwen35_ane_prefill_cpu_threads)
                                    : 8,
                                qwen35_ane_prefill_cpu_shared_resource: !!this.modelSettings.qwen35_ane_prefill_cpu_shared_resource,
                                specprefill_enabled: this.modelSettings.specprefill_enabled,
                                specprefill_draft_model: this.modelSettings.specprefill_draft_model || null,
                                specprefill_keep_pct: this.modelSettings.specprefill_enabled
                                    ? parseFloat(this.modelSettings.specprefill_keep_pct) || 0.2
                                    : null,
                                specprefill_threshold: this.modelSettings.specprefill_enabled
                                    ? (this.modelSettings.specprefill_threshold || null)
                                    : null,
                                dflash_enabled: this.modelSettings.dflash_enabled,
                                dflash_draft_model: this.modelSettings.dflash_draft_model || null,
                                dflash_draft_quant_enabled: this.modelSettings.dflash_enabled && !!this.modelSettings.dflash_draft_quant_enabled,
                                dflash_draft_quant_weight_bits: this.modelSettings.dflash_enabled && this.modelSettings.dflash_draft_quant_enabled
                                    ? parseInt(this.modelSettings.dflash_draft_quant_weight_bits)
                                    : null,
                                dflash_draft_quant_activation_bits: this.modelSettings.dflash_enabled && this.modelSettings.dflash_draft_quant_enabled
                                    ? parseInt(this.modelSettings.dflash_draft_quant_activation_bits)
                                    : null,
                                dflash_draft_quant_group_size: this.modelSettings.dflash_enabled && this.modelSettings.dflash_draft_quant_enabled
                                    ? parseInt(this.modelSettings.dflash_draft_quant_group_size)
                                    : null,
                                dflash_max_ctx: this.modelSettings.dflash_enabled && this.modelSettings.dflash_max_ctx
                                    ? parseInt(this.modelSettings.dflash_max_ctx)
                                    : null,
                                dflash_in_memory_cache: this.modelSettings.dflash_enabled
                                    ? !!this.modelSettings.dflash_in_memory_cache
                                    : true,
                                dflash_in_memory_cache_max_entries: this.modelSettings.dflash_enabled
                                    ? (parseInt(this.modelSettings.dflash_in_memory_cache_max_entries) || 4)
                                    : 4,
                                dflash_in_memory_cache_max_bytes: this.modelSettings.dflash_enabled
                                    ? Math.max(1, parseInt(this.modelSettings.dflash_in_memory_cache_max_gib) || 8) * (1024 ** 3)
                                    : 8 * (1024 ** 3),
                                dflash_ssd_cache: this.modelSettings.dflash_enabled
                                    && !!this.modelSettings.dflash_in_memory_cache
                                    && !!this.modelSettings.dflash_ssd_cache_available
                                    && !!this.modelSettings.dflash_ssd_cache,
                                dflash_ssd_cache_max_bytes: this.modelSettings.dflash_enabled
                                    ? Math.max(1, parseInt(this.modelSettings.dflash_ssd_cache_max_gib) || 20) * (1024 ** 3)
                                    : 20 * (1024 ** 3),
                                // Long-context tuning. Empty → oMLX default.
                                dflash_draft_window_size: this.modelSettings.dflash_enabled
                                    && this.modelSettings.dflash_draft_window_size
                                    ? parseInt(this.modelSettings.dflash_draft_window_size)
                                    : null,
                                dflash_draft_sink_size: this.modelSettings.dflash_enabled
                                    && this.modelSettings.dflash_draft_sink_size !== null
                                    && this.modelSettings.dflash_draft_sink_size !== undefined
                                    && this.modelSettings.dflash_draft_sink_size !== ''
                                    ? parseInt(this.modelSettings.dflash_draft_sink_size)
                                    : 0,
                                dflash_block_size: this.modelSettings.dflash_enabled
                                    && this.modelSettings.dflash_block_size
                                    ? parseInt(this.modelSettings.dflash_block_size)
                                    : null,
                                dflash_verify_mode: this.modelSettings.dflash_enabled
                                    ? (this.modelSettings.dflash_verify_mode || 'adaptive')
                                    : null,
                                mtp_enabled: !!this.modelSettings.mtp_enabled,
                                vlm_mtp_enabled: !!this.modelSettings.vlm_mtp_enabled,
                                vlm_mtp_draft_model: this.modelSettings.vlm_mtp_enabled
                                    ? (this.modelSettings.vlm_mtp_draft_model || null)
                                    : null,
                                vlm_mtp_draft_block_size: this.modelSettings.vlm_mtp_enabled
                                    && this.modelSettings.vlm_mtp_draft_block_size
                                    ? parseInt(this.modelSettings.vlm_mtp_draft_block_size)
                                    : null,
                                trust_remote_code: this.modelSettings.trust_remote_code,
                            };
                            if (isDiffusion) {
                                Object.assign(payload, {
                                    top_p: null,
                                    top_k: null,
                                    repetition_penalty: null,
                                    min_p: null,
                                    presence_penalty: null,
                                    force_sampling: false,
                                    reasoning_parser: null,
                                    index_cache_freq: 0,
                                    enable_thinking: null,
                                    thinking_budget_enabled: false,
                                    thinking_budget_tokens: 0,
                                    guided_grammar_enabled: false,
                                    guided_grammar: null,
                                    max_tool_result_tokens: 0,
                                    turboquant_kv_enabled: false,
                                    turboquant_kv_bits: 4,
                                    qwen35_ane_prefill_enabled: false,
                                    qwen35_ane_prefill_sequence_length: 2048,
                                    qwen35_ane_prefill_tail_padding_min_tokens: 0,
                                    qwen35_ane_prefill_fraction: 0.53,
                                    qwen35_ane_prefill_max_layers: 64,
                                    qwen35_ane_prefill_dual_ane: true,
                                    qwen35_ane_prefill_gdn: true,
                                    qwen35_ane_prefill_gdn_fraction: 0.5,
                                    qwen35_ane_prefill_gdn_max_layers: 48,
                                    qwen35_ane_prefill_cpu_enabled: false,
                                    qwen35_ane_prefill_cpu_fraction: 0.135,
                                    qwen35_ane_prefill_cpu_down_fraction: 0,
                                    qwen35_ane_prefill_cpu_gdn_fraction: 0,
                                    qwen35_ane_prefill_cpu_threads: 8,
                                    qwen35_ane_prefill_cpu_shared_resource: true,
                                    specprefill_enabled: false,
                                    specprefill_draft_model: null,
                                    specprefill_keep_pct: null,
                                    specprefill_threshold: null,
                                    dflash_enabled: false,
                                    dflash_draft_model: null,
                                    dflash_draft_quant_enabled: false,
                                    dflash_draft_quant_weight_bits: null,
                                    dflash_draft_quant_activation_bits: null,
                                    dflash_draft_quant_group_size: null,
                                    dflash_max_ctx: null,
                                    dflash_in_memory_cache: true,
                                    dflash_in_memory_cache_max_entries: 4,
                                    dflash_in_memory_cache_max_bytes: 8 * (1024 ** 3),
                                    dflash_ssd_cache: false,
                                    dflash_ssd_cache_max_bytes: 20 * (1024 ** 3),
                                    dflash_draft_window_size: null,
                                    dflash_draft_sink_size: null,
                                    dflash_block_size: null,
                                    dflash_verify_mode: null,
                                    mtp_enabled: false,
                                    vlm_mtp_enabled: false,
                                    vlm_mtp_draft_model: null,
                                    vlm_mtp_draft_block_size: null,
                                });
                            }
                            return payload;
                        })()),
                    });

                    if (response.ok) {
                        // Refresh the model list to update badges
                        await this.loadModels();
                        const data = await response.json();
                        this.showModelSettingsModal = false;
                        if (data.requires_reload) {
                            if (data.auto_reloaded) {
                                alert(window.t('js.info.model_settings_auto_reloaded'));
                            } else if (data.auto_unloaded) {
                                alert(window.t('js.info.model_settings_auto_unloaded'));
                            } else {
                                alert(window.t('js.info.model_type_reload_required'));
                            }
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
                        this.modelSettings.max_tokens = null;
                        this.modelSettings.min_p = null;
                        this.modelSettings.presence_penalty = null;
                        this.modelSettings.force_sampling = false;
                        this.modelSettings.reasoning_parser = null;
                        this.modelSettings.guided_grammar_enabled = false;
                        this.modelSettings.guided_grammar = '';
                        this.modelSettings.ttl_seconds = null;
                        this.modelSettings.enableIndexCache = false;
                        this.modelSettings.index_cache_freq = 0;
                        this.modelSettings.enable_thinking = false;
                        this.modelSettings.enableThinkingBudget = false;
                        this.modelSettings.thinking_budget_tokens = 0;
                        this.modelSettings.enableToolResultLimit = false;
                        this.modelSettings.max_tool_result_tokens = 0;
                        this.modelSettings.ctKwargEntries = [];
                        this.modelSettings.turboquant_kv_enabled = false;
                        this.modelSettings.turboquant_kv_bits = 4;
                        this.modelSettings.qwen35_ane_prefill_enabled = false;
                        this.modelSettings.qwen35_ane_prefill_sequence_length = 2048;
                        this.modelSettings.qwen35_ane_prefill_tail_padding_min_tokens = 0;
                        this.modelSettings.qwen35_ane_prefill_fraction = 0.53;
                        this.modelSettings.qwen35_ane_prefill_max_layers = 64;
                        this.modelSettings.qwen35_ane_prefill_dual_ane = true;
                        this.modelSettings.qwen35_ane_prefill_gdn = true;
                        this.modelSettings.qwen35_ane_prefill_gdn_fraction = 0.5;
                        this.modelSettings.qwen35_ane_prefill_gdn_max_layers = 48;
                        this.modelSettings.qwen35_ane_prefill_cpu_enabled = false;
                        this.modelSettings.qwen35_ane_prefill_cpu_fraction = 0.135;
                        this.modelSettings.qwen35_ane_prefill_cpu_down_fraction = 0;
                        this.modelSettings.qwen35_ane_prefill_cpu_gdn_fraction = 0;
                        this.modelSettings.qwen35_ane_prefill_cpu_threads = 8;
                        this.modelSettings.qwen35_ane_prefill_cpu_shared_resource = true;
                        this.modelSettings.specprefill_enabled = false;
                        this.modelSettings.specprefill_draft_model = null;
                        this.modelSettings.specprefill_keep_pct = 0.2;
                        this.modelSettings.specprefill_threshold = null;
                        this.modelSettings.dflash_enabled = false;
                        this.modelSettings.dflash_draft_model = null;
                        this.modelSettings.dflash_draft_quant_enabled = false;
                        this.modelSettings.dflash_draft_quant_weight_bits = null;
                        this.modelSettings.dflash_draft_quant_activation_bits = null;
                        this.modelSettings.dflash_draft_quant_group_size = null;
                        this.modelSettings.dflash_max_ctx = null;
                        this.modelSettings.dflash_in_memory_cache = true;
                        this.modelSettings.dflash_in_memory_cache_max_entries = 4;
                        this.modelSettings.dflash_in_memory_cache_max_gib = 8;
                        this.modelSettings.dflash_ssd_cache = false;
                        this.modelSettings.dflash_ssd_cache_max_gib = 20;
                        this.modelSettings.dflash_draft_window_size = null;
                        this.modelSettings.dflash_draft_sink_size = 0;
                        this.modelSettings.dflash_block_size = null;
                        this.modelSettings.dflash_verify_mode = 'adaptive';
                        this.modelSettings.mtp_enabled = false;
                        this.modelSettings.trust_remote_code = false;
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
                this.activeProfileName = null;
                this.profilesDrift = false;
            },

            // Status tab functions
            // Normalizes a host string for safe URL embedding:
            //  - unwraps existing IPv6 brackets so we can re-bracket consistently
            //  - maps unspecified bind addresses (0.0.0.0, ::) to a placeholder
            //    since they are not routable from a client
            //  - maps `localhost` to 127.0.0.1 for consistency with other URLs
            //  - bracket-wraps IPv6 addresses per RFC 3986 (`http://[::1]:8000/v1`)
            formatDisplayHost(host) {
                const value = (host || '').trim();
                if (!value) return '127.0.0.1';

                const unwrapped = value.startsWith('[') && value.endsWith(']')
                    ? value.slice(1, -1)
                    : value;

                if (unwrapped === '0.0.0.0' || unwrapped === '::') return 'your-ip-address';
                if (unwrapped === 'localhost') return '127.0.0.1';
                if (unwrapped.includes(':')) return `[${unwrapped}]`;
                return unwrapped;
            },

            get displayHost() {
                const host = this.selectedAlias || this.stats.host || '127.0.0.1';
                return this.formatDisplayHost(host);
            },

            get ttlPlaceholder() {
                if (this.selectedModel?.pinned) return window.t('modal.model_settings.ttl_pinned');
                const globalTtl = this.globalSettings.idle_timeout?.idle_timeout_seconds;
                if (globalTtl) {
                    return window.t('modal.model_settings.ttl_global_fallback').replace('{seconds}', globalTtl);
                }
                return window.t('modal.model_settings.ttl_no_ttl');
            },

            async loadServerInfo() {
                try {
                    const response = await fetch('/admin/api/server-info');
                    if (response.ok) {
                        const data = await response.json();
                        const aliases = Array.isArray(data.aliases) ? data.aliases : [];
                        this.serverAliases = aliases;
                        // Preserve user selection across reloads if still valid;
                        // otherwise default to the first alias when available.
                        if (this.selectedAlias && !aliases.includes(this.selectedAlias)) {
                            this.selectedAlias = '';
                        }
                        if (!this.selectedAlias && aliases.length > 0) {
                            this.selectedAlias = aliases[0];
                        }
                    } else if (response.status === 401) {
                        window.location.href = '/admin';
                    }
                } catch (err) {
                    console.error('Failed to load server info:', err);
                }
            },

            async restartServerStart() {
                if (this.restartServer.status === 'restarting'
                    || this.restartServer.status === 'waiting') {
                    return;
                }
                if (!window.confirm(window.t('settings.server.restart_confirm'))) {
                    return;
                }

                this.restartServer = {
                    status: 'restarting',
                    message: window.t('settings.server.restart_status_sending'),
                };

                let response;
                try {
                    response = await fetch('/admin/api/server/restart', { method: 'POST' });
                } catch (err) {
                    // Network errors mid-restart are expected if the server
                    // dies before sending the 202; fall through to polling.
                    this.restartServer = {
                        status: 'waiting',
                        message: window.t('settings.server.restart_status_waiting'),
                    };
                    this._restartServerPoll();
                    return;
                }

                if (response.status === 503) {
                    let msg = window.t('settings.server.restart_status_unavailable');
                    try {
                        const data = await response.json();
                        if (data && data.detail) msg = data.detail;
                    } catch (e) { /* ignore */ }
                    this.restartServer = { status: 'unsupported', message: msg };
                    return;
                }

                if (response.status === 401) {
                    window.location.href = '/admin';
                    return;
                }

                if (response.status !== 202) {
                    this.restartServer = {
                        status: 'error',
                        message: window.t('settings.server.restart_status_unexpected')
                            .replace('{status}', String(response.status)),
                    };
                    return;
                }

                this.restartServer = {
                    status: 'waiting',
                    message: window.t('settings.server.restart_status_waiting'),
                };
                this._restartServerPoll();
            },

            _restartServerPoll() {
                const deadline = Date.now() + 60000;  // 60s max wait
                let sawDownAt = 0;
                const tick = async () => {
                    if (Date.now() > deadline) {
                        this.restartServer = {
                            status: 'error',
                            message: window.t('settings.server.restart_status_timeout'),
                        };
                        return;
                    }
                    let alive = false;
                    try {
                        const r = await fetch('/health', { cache: 'no-store' });
                        alive = r.ok;
                    } catch (e) {
                        alive = false;
                    }
                    if (!alive) {
                        // First time we see it down — record it. We require a
                        // down-then-up transition before declaring success, so
                        // a fast supervisor that hasn't killed the old process
                        // yet doesn't trick us into "instant success".
                        if (!sawDownAt) sawDownAt = Date.now();
                        setTimeout(tick, 1000);
                        return;
                    }
                    // Alive again. If we never observed the down state, the
                    // restart hasn't actually fired yet — keep polling.
                    if (!sawDownAt) {
                        setTimeout(tick, 1000);
                        return;
                    }
                    this.restartServer = {
                        status: 'idle',
                        message: window.t('settings.server.restart_status_back'),
                    };
                    // Small delay so the user sees the success state, then
                    // reload to ensure all caches/sessions re-sync.
                    setTimeout(() => window.location.reload(), 500);
                };
                tick();
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
                // Deny LSP: its schema joins the tools array mid-session and
                // re-prefills the whole conversation on a caching server (#2349).
                parts.push('claude --disallowedTools LSP');
                return parts.join(' ');
            },

            async saveClaudeCodeSettings() {
                try {
                    const response = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
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

            _launchCmd(tool) {
                const raw = this.stats.cli_prefix || 'omlx';
                const cli = raw === 'omlx' ? raw : this.shellQuote(raw);
                return `${cli} launch ${tool}`;
            },

            get claudeCommand() {
                return this._launchCmd('claude');
            },

            get codexCommand() {
                return this._launchCmd('codex');
            },

            get codexAppCommand() {
                return this._launchCmd('codex_app');
            },

            get copilotCommand() {
                return this._launchCmd('copilot');
            },

            get opencodeCommand() {
                return this._launchCmd('opencode');
            },

            get openclawCommand() {
                const profile = this.globalSettings.integrations.openclaw_tools_profile || 'coding';
                return `${this._launchCmd('openclaw')} --tools-profile ${profile}`;
            },

            get hermesCommand() {
                return this._launchCmd('hermes');
            },

            get piCommand() {
                return this._launchCmd('pi');
            },

            get markitdownOcrModels() {
                return (this.models || []).filter((model) => {
                    const configType = String(model.config_model_type || '').toLowerCase();
                    return configType.includes('ocr');
                });
            },

            async saveIntegrationSettings() {
                try {
                    const response = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            integrations_copilot_model: this.globalSettings.integrations.copilot_model,
                            integrations_codex_model: this.globalSettings.integrations.codex_model,
                            integrations_opencode_model: this.globalSettings.integrations.opencode_model,
                            integrations_openclaw_model: this.globalSettings.integrations.openclaw_model,
                            integrations_hermes_model: this.globalSettings.integrations.hermes_model,
                            integrations_pi_model: this.globalSettings.integrations.pi_model,
                            integrations_openclaw_tools_profile: this.globalSettings.integrations.openclaw_tools_profile,
                            markitdown_enabled: this.globalSettings.integrations.markitdown_enabled,
                            markitdown_expose_model: this.globalSettings.integrations.markitdown_expose_model,
                            markitdown_max_file_size_mb: this.globalSettings.integrations.markitdown_max_file_size_mb,
                            markitdown_max_files_per_request: this.globalSettings.integrations.markitdown_max_files_per_request,
                            markitdown_pdf_processing_engine: this.globalSettings.integrations.markitdown_pdf_processing_engine,
                            web_search_provider: this.globalSettings.integrations.web_search_provider,
                            web_search_brave_api_key: this.globalSettings.integrations.web_search_brave_api_key,
                            web_search_searxng_url: this.globalSettings.integrations.web_search_searxng_url,
                            web_search_ddgs_backends: this.globalSettings.integrations.web_search_ddgs_backends,
                            web_search_max_results: this.globalSettings.integrations.web_search_max_results,
                            web_search_content_mode: this.globalSettings.integrations.web_search_content_mode,
                            web_search_content_truncate: this.globalSettings.integrations.web_search_content_truncate,
                            web_search_content_max_chars: this.globalSettings.integrations.web_search_content_max_chars,
                        }),
                    });
                    if (!response.ok) {
                        console.error('Failed to save integration settings');
                    }
                } catch (err) {
                    console.error('Failed to save integration settings:', err);
                }
            },

            ddgsBackendChecked(name) {
                return (this.globalSettings.integrations.web_search_ddgs_backends || '')
                    .split(',').map(s => s.trim()).filter(Boolean).includes(name);
            },

            toggleDdgsBackend(name) {
                const current = (this.globalSettings.integrations.web_search_ddgs_backends || '')
                    .split(',').map(s => s.trim()).filter(Boolean);
                const next = current.includes(name)
                    ? current.filter(b => b !== name)
                    : [...current, name];
                this.globalSettings.integrations.web_search_ddgs_backends = next.join(',');
                this.saveIntegrationSettings();
            },

            async testWebSearch() {
                if (this.webSearchTest.running) return;
                this.webSearchTest = { running: true, ok: null, message: '' };
                try {
                    const response = await fetch('/admin/api/web-search/test', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            provider: this.globalSettings.integrations.web_search_provider,
                            brave_api_key: this.globalSettings.integrations.web_search_brave_api_key || '',
                            searxng_url: this.globalSettings.integrations.web_search_searxng_url || '',
                            ddgs_backends: this.globalSettings.integrations.web_search_ddgs_backends || '',
                            max_results: this.globalSettings.integrations.web_search_max_results,
                        }),
                    });
                    const payload = await response.json();
                    if (payload.ok) {
                        this.webSearchTest = {
                            running: false,
                            ok: true,
                            message: window.t('settings.integrations.websearch.test_success')
                                .replace('{count}', (payload.results || []).length),
                        };
                    } else {
                        const message = payload.error?.message
                            || window.t('settings.integrations.websearch.test_failed');
                        this.webSearchTest = { running: false, ok: false, message };
                    }
                } catch (err) {
                    this.webSearchTest = {
                        running: false,
                        ok: false,
                        message: window.t('settings.integrations.websearch.test_failed'),
                    };
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

            async loadStats(includeAlltime = true) {
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

                    if (!includeAlltime) {
                        return;
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

            async clearSsdCache() {
                try {
                    const resp = await fetch('/admin/api/ssd-cache/clear', { method: 'POST' });
                    if (!resp.ok) console.error('SSD cache clear failed:', resp.status);
                    this.showClearSsdCacheConfirm = false;
                    await this.loadStats();
                } catch (err) {
                    console.error('Failed to clear SSD cache:', err);
                    this.showClearSsdCacheConfirm = false;
                }
            },

            async clearHotCache() {
                try {
                    const resp = await fetch('/admin/api/hot-cache/clear', { method: 'POST' });
                    if (!resp.ok) console.error('Hot cache clear failed:', resp.status);
                    this.showClearHotCacheConfirm = false;
                    await this.loadStats();
                } catch (err) {
                    console.error('Failed to clear hot cache:', err);
                    this.showClearHotCacheConfirm = false;
                }
            },

            startStatsRefresh() {
                this.stopStatsRefresh();
                this.loadStats();
                this._statsRefreshTimer = setInterval(() => {
                    this.loadStats(false);
                }, 500);
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

            cacheObsCumulative(stats, selectedModel) {
                const entries = stats.runtime_cache?.models || [];
                if (entries.length === 0) return {};

                if (selectedModel) {
                    const entry = entries.find(m => m.id === selectedModel);
                    return entry?.cache_rates?.cumulative || {};
                }

                const sumKeys = ['prefix_hits', 'prefix_misses', 'evictions', 'ssd_hot_hits', 'ssd_disk_loads', 'ssd_saves', 'hot_cache_evictions', 'hot_cache_promotions'];
                let agg = {};

                for (const m of entries) {
                    const c = m.cache_rates?.cumulative;
                    if (!c || Object.keys(c).length === 0) continue;
                    for (const k of sumKeys) {
                        agg[k] = (agg[k] || 0) + (c[k] || 0);
                    }
                }

                const ph = agg.prefix_hits || 0;
                const pm = agg.prefix_misses || 0;
                const sh = agg.ssd_hot_hits || 0;
                const sd = agg.ssd_disk_loads || 0;
                agg.prefix_hit_rate = (ph + pm) > 0 ? ph / (ph + pm) : 0;
                agg.ssd_hot_rate = (sh + sd) > 0 ? sh / (sh + sd) : 0;

                return agg;
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

            formatByteCount(bytes) {
                if (bytes == null || !Number.isFinite(bytes)) return '';
                if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
                if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
                if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
                return Math.max(0, Math.round(bytes)) + ' B';
            },

            formatTokenCount(n) {
                if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
                if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
                return String(n);
            },

            formatDFlashSessionStats(totals) {
                if (!totals || totals.requests <= 1) return '';

                const parts = [];
                if (totals.speculative_requests > 0) {
                    parts.push(
                        Math.round((totals.acceptance_ratio || 0) * 100) + '% ' +
                        window.t('status.active_models.dflash_draft_share'),
                        (totals.accepted_draft_tokens_per_cycle || 0).toFixed(2) + ' ' +
                        window.t('status.active_models.dflash_accepted_draft_per_cycle'),
                        (totals.tokens_per_cycle || 0).toFixed(2) + ' ' +
                        window.t('status.active_models.dflash_output_per_cycle'),
                        totals.speculative_requests + ' ' +
                        window.t('status.active_models.dflash_speculative_requests'),
                    );
                }
                if (totals.fallback_requests > 0) {
                    parts.push(
                        totals.fallback_requests + ' ' +
                        window.t('status.active_models.dflash_fallback_requests'),
                    );
                }
                return window.t('status.active_models.dflash_session') + ': ' + parts.join(' · ');
            },

            formatDurationShort(seconds) {
                if (seconds == null || !Number.isFinite(seconds)) return '—';
                if (seconds < 1) return seconds.toFixed(1) + 's';
                if (seconds < 60) return Math.round(seconds) + 's';
                const minutes = Math.floor(seconds / 60);
                const rem = Math.round(seconds % 60);
                if (minutes < 60) return minutes + 'm ' + rem + 's';
                const hours = Math.floor(minutes / 60);
                return hours + 'h ' + (minutes % 60) + 'm';
            },

            formatActivityAge(seconds) {
                if (seconds == null || !Number.isFinite(seconds)) return '';
                return 'last token ' + this.formatDurationShort(seconds) + ' ago';
            },

            formatActivityMetadata(activity) {
                const parts = [];
                if (activity.input_count != null) parts.push(activity.input_count + ' inputs');
                if (activity.document_count != null) parts.push(activity.document_count + ' docs');
                if (activity.token_count != null) parts.push(this.formatTokenCount(activity.token_count) + ' tok');
                if (activity.text_length != null) parts.push(activity.text_length + ' chars');
                if (activity.chunk_count != null) parts.push(activity.chunk_count + ' chunks');
                if (activity.output_bytes != null) parts.push(this.formatByteCount(activity.output_bytes));
                if (activity.file_size_bytes != null && activity.file_size_bytes > 0) parts.push(this.formatByteCount(activity.file_size_bytes));
                return parts.join(' · ');
            },

            activityDotClass(seconds) {
                if (seconds == null || !Number.isFinite(seconds)) return 'bg-green-400 animate-pulse';
                if (seconds < 15) return 'bg-green-400 animate-pulse';
                if (seconds < 30) return 'bg-amber-400 animate-pulse';
                return 'bg-red-400';
            },

            get runtimeHotCachePercent() {
                const rc = this.stats.runtime_cache;
                if (!rc || !rc.hot_cache_max_bytes) return 0;
                return Math.min(100, (rc.hot_cache_size_bytes / rc.hot_cache_max_bytes) * 100);
            },

            get runtimeSsdCachePercent() {
                const rc = this.stats.runtime_cache;
                if (!rc || !rc.disk_max_bytes) return 0;
                return Math.min(100, (rc.total_size_bytes / rc.disk_max_bytes) * 100);
            },

            get activeModelsPressurePercent() {
                const mp = this.stats.active_models?.memory_pressure;
                if (!mp || !mp.hard_bytes) return 0;
                return Math.min(100, (mp.current_bytes / mp.hard_bytes) * 100);
            },

            get activeModelsSoftPercent() {
                const mp = this.stats.active_models?.memory_pressure;
                if (!mp || !mp.hard_bytes || !mp.soft_bytes) return 0;
                return Math.min(100, (mp.soft_bytes / mp.hard_bytes) * 100);
            },

            get activeModelsPressureBarColor() {
                const pct = this.activeModelsPressurePercent;
                if (pct >= 90) return '#ef4444';
                if (pct >= 80) return '#f97316';
                if (pct >= 70) return '#f59e0b';
                if (pct >= 60) return '#facc15';
                return '#22c55e';
            },

            get activeModelsPressureBarStyle() {
                return `width: ${this.activeModelsPressurePercent}%; height: 100%; display: block; background-color: ${this.activeModelsPressureBarColor};`;
            },

            get activeModelsSoftMarkerStyle() {
                return `left: ${this.activeModelsSoftPercent}%; width: 1px; background-color: rgba(64, 64, 64, 0.6);`;
            },

            activeModelsPressureLabel() {
                const mp = this.stats.active_models?.memory_pressure;
                if (!mp || !mp.enabled || !mp.hard_bytes) {
                    return window.t('status.active_models.enforcer_disabled');
                }
                return `${this.formatSizeBytes(mp.current_bytes)} / ${this.formatSizeBytes(mp.soft_bytes)} soft / ${this.formatSizeBytes(mp.hard_bytes)} hard`;
            },

            modelSizeLabel(model) {
                if (!model) return '-';
                const estimated = model.estimated_size_formatted || '-';
                if (model.is_loading) {
                    return estimated;
                }
                // actual_size is a rough phys_footprint delta captured at load
                // time and can include neighboring KV growth — mark with ~obs
                // so it doesn't read as exact.
                const actual = model.actual_size_formatted;
                if (!actual) {
                    return estimated;
                }
                if (!estimated || estimated === actual) {
                    return `~${actual} obs`;
                }
                return `~${actual} obs / ${estimated} est`;
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

            // Shared external endpoint settings (both bench tabs)
            saveExternalEndpoint() {
                localStorage.setItem('omlx_bench_external_base_url', this.externalBaseUrl.trim());
                localStorage.setItem('omlx_bench_external_api_key', this.externalApiKey);
                localStorage.setItem('omlx_bench_external_model', this.externalModel.trim());
            },

            externalConfigValid() {
                return !!(this.externalBaseUrl.trim() && this.externalModel.trim());
            },

            externalRequestBody() {
                return {
                    base_url: this.externalBaseUrl.trim(),
                    api_key: this.externalApiKey,
                    model: this.externalModel.trim(),
                };
            },

            parseAccuracyExtraBody() {
                const raw = this.accExternalExtraBody.trim();
                if (!raw) return {};

                let value;
                try {
                    value = JSON.parse(raw);
                } catch (_) {
                    throw new Error(window.t('js.error.external_extra_body_invalid_json'));
                }
                if (value === null || Array.isArray(value) || typeof value !== 'object') {
                    throw new Error(window.t('js.error.external_extra_body_object_required'));
                }

                const protectedFields = new Set([
                    'model', 'messages', 'stream', 'stream_options',
                    'max_tokens', 'temperature', 'api_key', 'authorization',
                ]);
                const blocked = Object.keys(value).filter(
                    key => protectedFields.has(key.toLowerCase())
                );
                if (blocked.length > 0) {
                    throw new Error(
                        window.t('js.error.external_extra_body_protected')
                            .replace('{fields}', blocked.sort().join(', '))
                    );
                }
                return value;
            },

            saveAccExternalMaxTokens() {
                localStorage.setItem('omlx_acc_external_max_tokens', this.accExternalMaxTokens.trim());
            },

            // Optional accuracy-only floor for per-question max_tokens.
            // Returns null when unset; throws on invalid input.
            parseAccuracyMaxTokens() {
                const raw = this.accExternalMaxTokens.trim();
                if (!raw) return null;
                const value = Number(raw);
                if (!Number.isInteger(value) || value < 1 || value > 1000000) {
                    throw new Error(window.t('js.error.external_max_tokens_invalid'));
                }
                return value;
            },

            accuracyExternalRequestBody() {
                const body = this.externalRequestBody();
                const extraBody = this.parseAccuracyExtraBody();
                if (Object.keys(extraBody).length > 0) body.extra_body = extraBody;
                const maxTokens = this.parseAccuracyMaxTokens();
                if (maxTokens !== null) body.max_tokens_override = maxTokens;
                return body;
            },

            // Benchmark functions
            async startBenchmark() {
                if (this.benchExternalEnabled) {
                    if (!this.externalConfigValid()) {
                        this.benchError = window.t('js.error.external_endpoint_required');
                        return;
                    }
                } else if (!this.benchModelId) {
                    return;
                }

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
                this.benchBatchResults = [];
                this.benchError = '';
                this.benchBenchId = null;
                this.benchUploadResults = [];
                this.benchUploadDone = null;
                this.benchUploading = false;
                this.benchUploadSkipped = null;
                this.benchUploadFlags = [];
                this.benchRunExternal = this.benchExternalEnabled
                    ? { base_url: this.externalBaseUrl.trim(), model: this.externalModel.trim() }
                    : null;

                try {
                    const response = await fetch('/admin/api/bench/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: this.benchExternalEnabled ? this.externalModel.trim() : this.benchModelId,
                            context_profile: this.benchContextProfile,
                            align_prompt_to_ane: this.benchAlignPromptToAne,
                            prompt_lengths: promptLengths,
                            generation_length: 128,
                            batch_sizes: batchSizes,
                            force_lm_engine: this.benchExternalEnabled ? false : this.benchForceLmEngine,
                            external: this.benchExternalEnabled ? this.externalRequestBody() : null,
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
                            // SSE replay-on-subscribe re-delivers every event on
                            // every reconnect (incl. page refresh), so append-only
                            // arrays must dedupe. Single rows are keyed by
                            // (pp, tg); batch rows by batch_size.
                            if (data.data.test_type === 'single') {
                                const exists = this.benchSingleResults.some(
                                    r => this.benchRequestedPp(r) === this.benchRequestedPp(data.data)
                                        && r.tg === data.data.tg
                                );
                                if (!exists) {
                                    this.benchSingleResults = [...this.benchSingleResults, data.data];
                                }
                            } else if (data.data.test_type === 'batch') {
                                const exists = this.benchBatchResults.some(
                                    r => r.batch_size === data.data.batch_size
                                );
                                if (!exists) {
                                    this.benchBatchResults = [...this.benchBatchResults, data.data];
                                }
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
                            // Dedupe on replay: upload entries are unique by context_length.
                            const exists = this.benchUploadResults.some(
                                r => r.context_length === data.data.context_length
                            );
                            if (!exists) {
                                this.benchUploadResults = [...this.benchUploadResults, data.data];
                            }
                        } else if (data.type === 'upload_done') {
                            this.benchUploadDone = data.data;
                            this.benchUploadFlags = data.data.feature_flags || [];
                            this.benchUploading = false;
                            this.benchRunning = false;
                            this.benchProgress = null;
                            es.close();
                            this.benchEventSource = null;
                        } else if (data.type === 'upload_skipped') {
                            this.benchUploadSkipped = {
                                reason: data.reason || 'external_endpoint',
                                features: data.features || [],
                            };
                            this.benchUploading = false;
                            this.benchRunning = false;
                            this.benchProgress = null;
                            es.close();
                            this.benchEventSource = null;
                            this.loadModels();
                        } else if (data.type === 'error') {
                            this.benchError = data.message;
                            this.benchRunning = false;
                            this.benchProgress = null;
                            es.close();
                            this.benchEventSource = null;
                            this.loadModels();
                        }

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

            // Context benchmark functions
            async startContextBenchmark() {
                if (!this.ctxBenchModelId || this.ctxBenchRunning) return;

                this.ctxBenchRunning = true;
                this.ctxBenchProgress = null;
                this.ctxBenchResult = null;
                this.ctxBenchError = '';
                this.ctxBenchBenchId = null;

                try {
                    const response = await fetch('/admin/api/bench/context/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_id: this.ctxBenchModelId,
                            target_tokens: this.ctxBenchTarget,
                        }),
                    });

                    if (response.status === 401) {
                        window.location.href = '/admin';
                        return;
                    }

                    if (!response.ok) {
                        const data = await response.json();
                        this.ctxBenchError = data.detail || window.t('js.error.start_context_bench_failed');
                        this.ctxBenchRunning = false;
                        return;
                    }

                    const data = await response.json();
                    this.ctxBenchBenchId = data.bench_id;
                    this.connectContextBenchSSE(data.bench_id);
                } catch (err) {
                    console.error('Failed to start context benchmark:', err);
                    this.ctxBenchError = window.t('js.error.start_context_bench_failed');
                    this.ctxBenchRunning = false;
                }
            },

            connectContextBenchSSE(benchId) {
                if (this.ctxBenchEventSource) {
                    this.ctxBenchEventSource.close();
                }

                const es = new EventSource(`/admin/api/bench/context/${benchId}/stream`);
                this.ctxBenchEventSource = es;

                es.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);

                        if (data.type === 'progress') {
                            this.ctxBenchProgress = {
                                phase: data.phase,
                                progress: data.progress,
                                message: data.message,
                            };
                        } else if (data.type === 'result') {
                            this.ctxBenchResult = data.data;
                        } else if (data.type === 'done') {
                            this.ctxBenchRunning = false;
                            this.ctxBenchProgress = null;
                            es.close();
                            this.ctxBenchEventSource = null;
                            // The applied setting changed the model row.
                            this.loadModels();
                        } else if (data.type === 'error') {
                            this.ctxBenchError = data.message;
                            this.ctxBenchRunning = false;
                            this.ctxBenchProgress = null;
                            es.close();
                            this.ctxBenchEventSource = null;
                            this.loadModels();
                        }
                    } catch (err) {
                        console.error('Failed to parse SSE event:', err);
                    }
                };

                es.onerror = () => {
                    if (this.ctxBenchRunning) {
                        this.ctxBenchError = window.t('js.error.benchmark_connection_lost');
                        this.ctxBenchRunning = false;
                        this.ctxBenchProgress = null;
                    }
                    es.close();
                    this.ctxBenchEventSource = null;
                };
            },

            async cancelContextBenchmark() {
                if (!this.ctxBenchBenchId) return;
                try {
                    await fetch(`/admin/api/bench/context/${this.ctxBenchBenchId}/cancel`, { method: 'POST' });
                } catch (err) {
                    console.error('Failed to cancel context benchmark:', err);
                }
                // SSE handler will update state when the error event arrives
            },

            async loadCtxBenchState() {
                // Attach to an in-flight context bench (page refresh, other tab).
                try {
                    const resp = await fetch('/admin/api/bench/context/active');
                    if (!resp.ok) return;
                    const data = await resp.json();
                    if (!data.running || !data.bench_id) return;
                    if (this.ctxBenchBenchId === data.bench_id && this.ctxBenchEventSource) {
                        return;
                    }
                    this.ctxBenchBenchId = data.bench_id;
                    this.ctxBenchModelId = data.model_id;
                    if (data.target_tokens) this.ctxBenchTarget = data.target_tokens;
                    this.ctxBenchRunning = true;
                    this.ctxBenchResult = null;
                    this.ctxBenchError = '';
                    this.connectContextBenchSSE(data.bench_id);
                } catch (err) {
                    console.error('Failed to load context bench state:', err);
                }
            },

            ctxBenchCappedByLabel() {
                const capped = this.ctxBenchResult?.capped_by;
                if (capped === 'target') return window.t('ctx_bench.capped.target');
                if (capped === 'native') return window.t('ctx_bench.capped.native');
                return window.t('ctx_bench.capped.memory');
            },

            // Native context length of the selected bench model (0 = unknown).
            ctxBenchNativeLimit() {
                const m = this.models.find(m => m.id === this.ctxBenchModelId);
                return (m && m.model_context_length) || 0;
            },

            // Target presets the selected model can actually reach. Unknown
            // native -> full list; native below the smallest preset -> keep
            // the smallest (the server caps the search at native anyway).
            ctxBenchTargetOptions() {
                const all = [16384, 32768, 65536, 131072, 262144, 524288];
                const native = this.ctxBenchNativeLimit();
                if (!native) return all;
                const filtered = all.filter(t => t <= native);
                return filtered.length ? filtered : [all[0]];
            },

            // Keep the selected target inside the model's reachable presets.
            ctxBenchClampTarget() {
                const options = this.ctxBenchTargetOptions();
                if (!options.includes(this.ctxBenchTarget)) {
                    this.ctxBenchTarget = options[options.length - 1];
                }
            },

            // Narrow-patch save of the global Prefill Priority setting from
            // the bench tab (mirrors the Settings row; applied live server-side).
            async saveCtxBenchPriority(value) {
                if (this.ctxBenchRunning) return;
                const prev = this.globalSettings.scheduler.prefill_priority;
                if (prev === value) return;
                this.globalSettings.scheduler.prefill_priority = value;
                try {
                    const resp = await fetch('/admin/api/global-settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prefill_priority: value }),
                    });
                    if (!resp.ok) throw new Error('HTTP ' + resp.status);
                } catch (err) {
                    console.error('Failed to save prefill priority:', err);
                    this.globalSettings.scheduler.prefill_priority = prev;
                    this.ctxBenchError = window.t('js.error.save_prefill_priority_failed');
                }
            },

            benchGetSpeedup(batchResult) {
                const baseline = this.benchFindSingle(1024);
                if (!baseline || !baseline.gen_tps || baseline.gen_tps <= 0) return null;
                if (batchResult.tg_tps === null || batchResult.tg_tps === undefined) return null;
                return batchResult.tg_tps / baseline.gen_tps;
            },

            benchRequestedPp(result) {
                return result?.requested_pp ?? result?.pp;
            },

            benchFindSingle(requestedPp) {
                return this.benchSingleResults.find(
                    r => this.benchRequestedPp(r) === requestedPp
                );
            },

            benchSingleTestLabel(result) {
                const requested = this.benchRequestedPp(result);
                const actual = result?.pp;
                if (requested !== actual) {
                    return `pp${actual} (requested pp${requested})/tg${result.tg}`;
                }
                return `pp${actual}/tg${result.tg}`;
            },

            benchBatchPromptSummary() {
                const result = this.benchBatchResults[0];
                if (!result || result.requested_pp === undefined) {
                    return window.t('bench.results.batch.subtitle');
                }
                const requested = result.requested_pp;
                const minimum = result.prompt_tokens_min ?? result.pp;
                const maximum = result.prompt_tokens_max ?? result.pp;
                const actual = minimum === maximum
                    ? `actual pp${minimum}`
                    : `actual pp${minimum}-${maximum}`;
                return `requested pp${requested} / ${actual} / tg${result.tg}`;
            },

            // Unmeasured metrics (tpot_ms/gen_tps/tg_tps, plus ttft/pp when
            // no content delta was ever observed) come through as null
            // rather than a misleading 0.0 — render them as N/A.
            benchFmtNum(value, decimals, suffix = '') {
                if (value === null || value === undefined) return 'N/A';
                return value.toFixed(decimals) + suffix;
            },

            // Per-request pp TPS, null when the aggregate itself is unmeasured.
            benchPpPerReq(batchResult) {
                const pp = batchResult.pp_tps;
                if (pp === null || pp === undefined) return null;
                return pp / batchResult.batch_size;
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
                if (this.benchRunExternal) {
                    lines.push(`Benchmark Model: ${this.benchRunExternal.model} @ ${this.benchRunExternal.base_url}`);
                    lines.push('Engine: External OpenAI-compatible endpoint');
                } else {
                    lines.push(`Benchmark Model: ${this.benchModelId}`);
                    lines.push(`Engine: ${this.benchForceLmEngine ? 'Force mlx-lm' : 'Auto'}`);
                }
                lines.push(`Context: ${this.benchContextLabel(this.benchContextProfile)}`);
                lines.push('='.repeat(80));

                // Single Request Results
                if (this.benchSingleResults.length > 0) {
                    lines.push('');
                    lines.push('Single Request Results');
                    lines.push('-'.repeat(80));
                    const hdr = [rpad('Test', 32), pad('TTFT(ms)', 10), pad('TPOT(ms)', 10), pad('pp TPS', 12), pad('tg TPS', 12), pad('E2E(s)', 10), pad('Throughput', 12), pad('Peak Mem', 10)];
                    lines.push(hdr.join('  '));
                    for (const r of this.benchSingleResults) {
                        const row = [
                            rpad(this.benchSingleTestLabel(r), 32),
                            pad(this.benchFmtNum(r.ttft_ms, 1), 10),
                            pad(this.benchFmtNum(r.tpot_ms, 2), 10),
                            pad(this.benchFmtNum(r.processing_tps, 1, ' tok/s'), 12),
                            pad(this.benchFmtNum(r.gen_tps, 1, ' tok/s'), 12),
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
                    const baseline = this.benchFindSingle(1024);
                    lines.push('');
                    lines.push(`${title}`);
                    lines.push(subtitle);
                    lines.push('-'.repeat(80));
                    const hdr = [rpad('Batch', 8), pad('tg TPS', 12), pad('Speedup', 8), pad('pp TPS', 12), pad('pp TPS/req', 12), pad('TTFT(ms)', 10), pad('E2E(s)', 10)];
                    lines.push(hdr.join('  '));
                    if (baseline) {
                        const row = [
                            rpad('1x', 8),
                            pad(this.benchFmtNum(baseline.gen_tps, 1, ' tok/s'), 12),
                            pad('1.00x', 8),
                            pad(this.benchFmtNum(baseline.processing_tps, 1, ' tok/s'), 12),
                            pad(this.benchFmtNum(baseline.processing_tps, 1, ' tok/s'), 12),
                            pad(this.benchFmtNum(baseline.ttft_ms, 1), 10),
                            pad(baseline.e2e_latency_s.toFixed(3), 10),
                        ];
                        lines.push(row.join('  '));
                    }
                    for (const r of results) {
                        const speedup = this.benchGetSpeedup(r);
                        const row = [
                            rpad(r.batch_size + 'x', 8),
                            pad(this.benchFmtNum(r.tg_tps, 1, ' tok/s'), 12),
                            pad(speedup !== null ? speedup.toFixed(2) + 'x' : 'N/A', 8),
                            pad(this.benchFmtNum(r.pp_tps, 1, ' tok/s'), 12),
                            pad(this.benchFmtNum(this.benchPpPerReq(r), 1, ' tok/s'), 12),
                            pad(this.benchFmtNum(r.avg_ttft_ms, 1), 10),
                            pad(r.e2e_latency_s.toFixed(3), 10),
                        ];
                        lines.push(row.join('  '));
                    }
                };

                buildBatchText(
                    'Continuous Batching',
                    this.benchBatchPromptSummary(),
                    this.benchBatchResults
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

            async loadBenchState() {
                // Discover an in-progress throughput run on tab/page load so
                // a second tab (or a refresh) can attach to its SSE stream
                // and replay the run's full event history.
                //
                // Three cases:
                //   1. No active run → clear any stale banner state.
                //   2. Active run, this tab is already attached → no-op.
                //   3. Active run, this tab is fresh → auto-attach.
                //   4. Active run, this tab is displaying a *different*
                //      completed bench → show banner; let the user decide
                //      whether to clobber their result view.
                try {
                    const resp = await fetch('/admin/api/bench/active');
                    if (!resp.ok) return;
                    const data = await resp.json();

                    if (!data.running || !data.bench_id) {
                        this.benchOtherActive = null;
                        return;
                    }

                    // Already attached to this bench — nothing to do.
                    if (this.benchBenchId === data.bench_id && this.benchEventSource) {
                        return;
                    }

                    // We have completed results from a DIFFERENT bench on
                    // screen — don't silently swap them out. Show a banner
                    // so the user can explicitly accept the new bench.
                    const hasStaleResults = !this.benchRunning
                        && this.benchBenchId
                        && this.benchBenchId !== data.bench_id
                        && (this.benchSingleResults.length > 0
                            || this.benchBatchResults.length > 0);
                    if (hasStaleResults) {
                        this.benchOtherActive = {
                            bench_id: data.bench_id,
                            model_id: data.model_id,
                            context_profile: data.context_profile || 'code_python',
                            force_lm_engine: !!data.force_lm_engine,
                            external: !!data.external,
                        };
                        return;
                    }

                    // Fresh slate: attach.
                    this.benchBenchId = data.bench_id;
                    this._restoreBenchRunSource(data);
                    this.benchRunning = true;
                    this.benchOtherActive = null;
                    this.connectBenchSSE(data.bench_id);
                } catch (err) {
                    console.error('Failed to load bench state:', err);
                }
            },

            // Restore the config UI from an active run discovered via
            // /api/bench/active. External model ids aren't in the local
            // dropdown, so the external flag drives which controls light up.
            _restoreBenchRunSource(data) {
                this.benchContextProfile = data.context_profile || 'code_python';
                if (data.external) {
                    this.benchExternalEnabled = true;
                    this.benchRunExternal = {
                        // base_url is intentionally not exposed by the API;
                        // fall back to this browser's stored setting.
                        base_url: this.externalBaseUrl.trim(),
                        model: data.model_id,
                    };
                } else {
                    this.benchModelId = data.model_id;
                    this.benchForceLmEngine = !!data.force_lm_engine;
                    this.benchExternalEnabled = false;
                    this.benchRunExternal = null;
                }
            },

            benchContextLabel(profile) {
                const keys = {
                    code_python: 'bench.config.context.code_python',
                    code_mixed: 'bench.config.context.code_mixed',
                    novel_ko: 'bench.config.context.novel_ko',
                    novel_en: 'bench.config.context.novel_en',
                    novel_ja: 'bench.config.context.novel_ja',
                };
                return window.t(keys[profile] || keys.code_python);
            },

            // User clicked "View live" on the banner — clear the stale
            // result display, attach to the active run. The replay-on-
            // subscribe stream re-delivers every event so the new bench
            // populates its table from the start.
            acceptOtherBench() {
                if (!this.benchOtherActive) return;
                const other = this.benchOtherActive;
                this.benchOtherActive = null;
                this.benchBenchId = other.bench_id;
                this._restoreBenchRunSource(other);
                this.benchRunning = true;
                this.benchSingleResults = [];
                this.benchBatchResults = [];
                this.benchUploadResults = [];
                this.benchUploadDone = null;
                this.benchUploadSkipped = null;
                this.benchUploadFlags = [];
                this.benchProgress = null;
                this.benchError = '';
                this.connectBenchSSE(other.bench_id);
            },

            dismissOtherBench() {
                // Hide for now; the banner reappears next loadBenchState if
                // the run is still active. Use case: user wants to keep
                // reviewing their previous result for a moment.
                this.benchOtherActive = null;
            },

            // Bench sub-tab
            setBenchTab(tab) {
                if (!DASHBOARD_BENCH_TABS.has(tab)) return;
                this.benchTab = tab;
                this.mainTab = 'bench';
                this.syncTabStateToUrl();
                if (tab === 'throughput') {
                    this.loadBenchDeviceInfo();
                    this.loadBenchState();
                }
                if (tab === 'context') {
                    this.loadCtxBenchState();
                    // The priority segment mirrors the global setting —
                    // refresh in case it changed on the Settings tab or in
                    // another window.
                    this.loadGlobalSettings();
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
                let externalRequest = null;
                if (this.accExternalEnabled) {
                    if (!this.externalConfigValid()) {
                        this.accError = window.t('js.error.external_endpoint_required');
                        return;
                    }
                    try {
                        externalRequest = this.accuracyExternalRequestBody();
                    } catch (err) {
                        this.accError = err.message;
                        return;
                    }
                } else if (!this.accModelId) {
                    return;
                }
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
                            model_id: this.accExternalEnabled ? this.externalModel.trim() : this.accModelId,
                            benchmarks: Object.fromEntries(
                                selected.map(k => [k, this.accSampleSizes[k]])
                            ),
                            batch_size: this.accBatchSize,
                            enable_thinking: this.accExternalEnabled ? false : this.accEnableThinking,
                            sampling_profile: this.accSamplingProfile,
                            external: externalRequest,
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
                                // Dedupe on replay: accuracy results are unique by
                                // (model_id, benchmark).
                                {
                                    const exists = this.accAllResults.some(
                                        r => r.model_id === data.data.model_id
                                          && r.benchmark === data.data.benchmark
                                    );
                                    if (!exists) {
                                        data.data._showCategories = false;
                                        this.accAllResults.push(data.data);
                                    }
                                }
                                break;
                            case 'upload':
                                // Community upload outcome for one suite. Idempotent
                                // on replay: keyed to the same (model_id, benchmark)
                                // as its result card. Array reassign for reactivity.
                                {
                                    const idx = this.accAllResults.findIndex(
                                        r => r.model_id === data.data.model_id
                                          && r.benchmark === data.data.benchmark
                                    );
                                    if (idx >= 0) {
                                        const updated = { ...this.accAllResults[idx], upload: data.data };
                                        this.accAllResults.splice(idx, 1, updated);
                                        this.accAllResults = [...this.accAllResults];
                                    }
                                }
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

                // Full sizes lookup
                const fullSizes = {};
                for (const grp of this.accBenchmarkGroups) {
                    for (const bl of grp.benchmarks) fullSizes[bl.key] = bl.fullSize;
                }

                // Determine column widths
                const modelWidth = Math.max(12, ...models.map(m => m.length + 2));
                const modeW = 8;
                const sampledW = 14;
                const benchWidth = Math.max(14, ...benchmarks.map(b => b.length + 2));

                let lines = [];
                lines.push('Intelligence Benchmark Comparison');
                lines.push('');

                // Header row
                let header = rpad('', benchWidth) + rpad('Mode', modeW) + rpad('Sampled', sampledW);
                for (const m of models) header += pad(m, modelWidth);
                lines.push(header);
                lines.push('-'.repeat(benchWidth + modeW + sampledW + models.length * modelWidth));

                // Data rows
                for (const b of benchmarks) {
                    // Get sample info from first available result for this benchmark
                    const sample = models.map(m => lookup[m]?.[b]).find(r => r);
                    const total = sample?.total || 0;
                    const full = fullSizes[b] || 0;
                    const isFull = total >= full;
                    const mode = isFull ? 'Full' : 'Sample';
                    const sampledStr = isFull ? String(full) : (total + '/' + full);

                    let row = rpad(b.toUpperCase(), benchWidth) + rpad(mode, modeW) + rpad(sampledStr, sampledW);
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
                    lines.push(rpad('Benchmark', 16) + pad('Accuracy', 10) + pad('Correct', 10) + pad('Total', 8) + pad('Time(s)', 10) + pad('Think', 8));
                    lines.push('-'.repeat(62));
                    for (const r of this.accAllResults.filter(r => r.model_id === m)) {
                        lines.push(
                            rpad(r.benchmark.toUpperCase(), 16) +
                            pad((r.accuracy * 100).toFixed(1) + '%', 10) +
                            pad(r.correct, 10) +
                            pad(r.total, 8) +
                            pad(r.time_s, 10) +
                            pad(r.thinking_used ? 'Yes' : 'No', 8)
                        );
                        if (r.external) {
                            lines.push(
                                `  Valid responses: ${r.valid_response_count}/${r.total}` +
                                ` (${(r.valid_response_rate * 100).toFixed(1)}%)` +
                                ` · Valid-answer accuracy: ${(r.valid_answer_accuracy * 100).toFixed(1)}%` +
                                ` · Empty: ${r.empty_content_count}` +
                                ` · Truncated: ${r.truncated_count}` +
                                ` · Timeout: ${r.timeout_count}` +
                                ` · HTTP: ${r.http_error_count}` +
                                ` · Connection: ${r.connection_error_count}` +
                                ` · Invalid: ${r.invalid_response_count}` +
                                ` · Parse: ${r.parse_error_count}`
                            );
                        }
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

            accDownloadResult(r, format) {
                const filename = `${r.model_id}_${r.benchmark}.${format}`;
                let content, mime;
                const qr = r.question_results || [];

                if (format === 'json') {
                    const exportData = {
                        model_id: r.model_id,
                        benchmark: r.benchmark,
                        accuracy: r.accuracy,
                        correct: r.correct,
                        total: r.total,
                        time_s: r.time_s,
                        thinking_used: r.thinking_used || false,
                        category_scores: r.category_scores || null,
                        questions: qr,
                    };
                    if (r.external) {
                        Object.assign(exportData, {
                            valid_response_count: r.valid_response_count,
                            empty_content_count: r.empty_content_count,
                            truncated_count: r.truncated_count,
                            timeout_count: r.timeout_count,
                            http_error_count: r.http_error_count,
                            connection_error_count: r.connection_error_count,
                            invalid_response_count: r.invalid_response_count,
                            parse_error_count: r.parse_error_count,
                            wrong_count: r.wrong_count,
                            valid_response_rate: r.valid_response_rate,
                            valid_answer_accuracy: r.valid_answer_accuracy,
                            reliability_warning: r.reliability_warning,
                        });
                    }
                    content = JSON.stringify(exportData, null, 2);
                    mime = 'application/json';
                } else if (format === 'csv') {
                    const esc = s => '"' + (s || '').replace(/"/g, '""') + '"';
                    const lines = [r.external
                        ? 'id,category,status,correct,expected,predicted,finish_reason,reasoning_fields,prompt_tokens,completion_tokens,error_message,question,raw_response,time_s'
                        : 'id,category,correct,expected,predicted,question,raw_response,time_s'];
                    for (const q of qr) {
                        if (r.external) {
                            lines.push([
                                q.id, esc(q.category || ''), esc(q.status || ''), q.correct,
                                esc(q.expected), esc(q.predicted), esc(q.finish_reason || ''),
                                esc((q.reasoning_fields_nonempty || []).join('|')),
                                q.prompt_tokens || 0, q.completion_tokens || 0,
                                esc(q.error_message || ''), esc(q.question),
                                esc(q.raw_response), q.time_s,
                            ].join(','));
                        } else {
                            lines.push([q.id, esc(q.category || ''), q.correct, esc(q.expected), esc(q.predicted), esc(q.question), esc(q.raw_response), q.time_s].join(','));
                        }
                    }
                    content = lines.join('\n');
                    mime = 'text/csv';
                } else {
                    const lines = [
                        `Model: ${r.model_id}`,
                        `Benchmark: ${r.benchmark.toUpperCase()}`,
                        `Accuracy: ${(r.accuracy * 100).toFixed(1)}% (${r.correct}/${r.total})`,
                        `Time: ${r.time_s}s`,
                        '',
                    ];
                    if (r.external) {
                        lines.splice(4, 0,
                            `Valid responses: ${r.valid_response_count}/${r.total} (${(r.valid_response_rate * 100).toFixed(1)}%)`,
                            `Valid-answer accuracy: ${(r.valid_answer_accuracy * 100).toFixed(1)}%`,
                            `Empty: ${r.empty_content_count}; Truncated: ${r.truncated_count}; Timeout: ${r.timeout_count}; HTTP errors: ${r.http_error_count}; Connection errors: ${r.connection_error_count}; Invalid responses: ${r.invalid_response_count}; Parse errors: ${r.parse_error_count}`
                        );
                    }
                    for (const q of qr) {
                        const label = r.external ? (q.status || 'invalid_response').toUpperCase() : (q.correct ? 'CORRECT' : 'WRONG');
                        lines.push(`--- Q${q.id} [${label}] ---`);
                        if (q.category) lines.push(`Category: ${q.category}`);
                        if (r.external && q.finish_reason) lines.push(`Finish reason: ${q.finish_reason}`);
                        if (r.external && (q.reasoning_fields_nonempty || []).length) lines.push(`Reasoning fields: ${q.reasoning_fields_nonempty.join(', ')}`);
                        if (r.external && q.error_message) lines.push(`Error: ${q.error_message}`);
                        lines.push(`Question: ${q.question || ''}`);
                        lines.push(`Expected: ${q.expected}`);
                        lines.push(`Predicted: ${q.predicted}`);
                        lines.push(`Raw response: ${q.raw_response || '(empty)'}`);
                        lines.push(`Time: ${q.time_s}s`);
                        lines.push('');
                    }
                    content = lines.join('\n');
                    mime = 'text/plain';
                }

                const blob = new Blob([content], { type: mime });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.click();
                URL.revokeObjectURL(url);
            },

            // Log viewer functions
            filteredLogContent() {
                const LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
                const minIdx = LEVELS.indexOf(this.logMinLevel);
                if (minIdx <= 0) return this.logContent;
                const levelRe = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \S+ - (TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL) - /;
                let visible = true;
                return this.logContent.split('\n').filter(line => {
                    const m = line.match(levelRe);
                    if (m) visible = LEVELS.indexOf(m[1]) >= minIdx;
                    return visible;
                }).join('\n');
            },

            levelButtonClass(lvl) {
                const LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
                const idx = LEVELS.indexOf(lvl);
                const minIdx = LEVELS.indexOf(this.logMinLevel);
                // Levels at or above the minimum are all shown dark so the
                // included range is obvious; the selected minimum keeps the ring.
                if (idx < minIdx) return 'bg-neutral-100 text-neutral-300';
                if (idx === minIdx) return 'bg-neutral-900 text-white';
                return 'bg-neutral-700 text-white';
            },

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

            // Memory guard tier → live hard ceiling (GB) for the selected tier.
            // Mirrors ProcessMemoryEnforcer._get_hard_limit_bytes:
            //   static_ceiling  = total - tier.static_reserve
            //   dynamic_ceiling = omlx_phys + free + inactive + active * ratio
            //   final = min(static, dynamic, metal_cap)
            // The static / dynamic inputs come from the global-settings
            // response and reflect the moment that response was fetched.
            // Warning shown below the breakdown when the kernel
            // iogpu.wired_limit_mb is lower than what oMLX asked Metal
            // to allow at start. Returns an HTML string with the exact
            // sysctl command the user can paste into Terminal, or "" when
            // the kernel cap is fine.
            // True when the kernel iogpu.wired_limit_mb (or Apple default
            // working set) caps oMLX below its desired static ceiling.
            get memoryGuardShowWiredLimitWarning() {
                const sys = this.globalSettings.system || {};
                const kernelBytes = sys.iogpu_wired_limit_bytes || 0;
                const requestedBytes = sys.omlx_wired_limit_request_bytes || 0;
                if (kernelBytes <= 0 || requestedBytes <= 0) return false;
                return kernelBytes < requestedBytes;
            },

            // Red bold warning text (no copy button). The button is a
            // separate sibling in the template so Alpine can wire @click.
            get memoryGuardWiredLimitWarningHTML() {
                if (!this.memoryGuardShowWiredLimitWarning) return '';
                const sys = this.globalSettings.system;
                const kernelGB = (sys.iogpu_wired_limit_bytes / (1024 ** 3)).toFixed(1);
                const template = window.t('settings.resource.guard_tier.wired_limit_warning');
                return template.replaceAll(
                    '{kernel}',
                    `<strong>${kernelGB} GB</strong>`,
                );
            },

            // The sysctl command rendered in the dark bold <code> chip.
            get memoryGuardWiredLimitCommand() {
                if (!this.memoryGuardShowWiredLimitWarning) return '';
                const requested = this.globalSettings.system.omlx_wired_limit_request_bytes;
                const requestedMB = Math.ceil(requested / (1024 ** 2));
                return `sudo sysctl iogpu.wired_limit_mb=${requestedMB}`;
            },

            // 2-second "Copied!" affordance after the clipboard button is
            // pressed. Reset by the same setTimeout so it's harmless if
            // the user clicks rapidly.
            wiredLimitCopied: false,

            copyWiredLimitCommand() {
                const text = this.memoryGuardWiredLimitCommand;
                if (!text) return;
                const onSuccess = () => {
                    this.wiredLimitCopied = true;
                    setTimeout(() => { this.wiredLimitCopied = false; }, 2000);
                };
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(onSuccess).catch(() => {
                        onSuccess();
                    });
                } else {
                    onSuccess();
                }
            },

            // Description text shown next to the Memory guard tier dropdown.
            // safe / balanced / aggressive get a "free + inactive + N% of
            // active (via macOS reclaim_method)" sentence. custom shows the
            // user-supplied ceiling.
            get memoryGuardTierDescription() {
                const tier = this.globalSettings.memory?.memory_guard_tier || 'balanced';
                const tierLabel = window.t('settings.resource.guard_tier.' + tier);
                if (tier === 'custom') {
                    const gb = Number(
                        this.globalSettings.memory?.memory_guard_custom_ceiling_gb || 0
                    ).toFixed(1);
                    return window
                        .t('settings.resource.guard_tier.description_custom')
                        .replace('{custom_gb}', gb);
                }
                const pct = { safe: 20, balanced: 50, aggressive: 80 }[tier] ?? 50;
                const method = window.t(
                    'settings.resource.guard_tier.reclaim_method.' + tier
                );
                return window
                    .t('settings.resource.guard_tier.description_template')
                    .replace('{tier}', tierLabel)
                    .replace('{active_pct}', pct)
                    .replace('{reclaim_method}', method);
            },

            // Breakdown line. For ratio tiers: `Free X, inactive Y, active Z
            // × N% = R → ceiling C`. For custom: `Custom ceiling X GB →
            // effective ceiling C` (after clamp by static / metal cap).
            get memoryGuardBreakdownHTML() {
                const sys = this.globalSettings.system || {};
                const GB = 1024 ** 3;
                const tier = this.globalSettings.memory?.memory_guard_tier || 'balanced';
                const fmt = (gb) => Number(gb).toFixed(1);
                const bold = (gb) => `<strong>${fmt(gb)} GB</strong>`;

                // Static / metal cap for the final clamp shown to the user.
                // The small-system threshold must track
                // ProcessMemoryEnforcer._SMALL_SYSTEM_THRESHOLD (24 GB): under
                // it the server reserves a flat 4 GB regardless of tier. This
                // read 16 and so understated the static ceiling by up to 4 GB
                // on every 16-23 GB Mac.
                const totalGB = (sys.total_memory_bytes || 0) / GB;
                const staticReserveGB =
                    tier === 'custom'
                        ? 2
                        : totalGB < 24
                            ? 4
                            : { safe: 8, balanced: 6, aggressive: 4 }[tier] ?? 6;
                const staticCeiling = Math.max(0, totalGB - staticReserveGB);
                const metalCapGB = (sys.iogpu_wired_limit_bytes || 0) / GB;

                // Helper: is the kernel iogpu.wired_limit_mb the smallest
                // of the three candidates? When yes we swap "→ ceiling" for
                // "/ effective ceiling X (kernel limit)" so the user knows
                // why the value isn't what their tier math suggested.
                const kernelBinds = (candidates, finalCeiling) =>
                    metalCapGB > 0 &&
                    Math.abs(metalCapGB - finalCeiling) < 1e-6 &&
                    candidates.every((c) => c >= metalCapGB - 1e-6);

                if (tier === 'custom') {
                    const custom = Number(
                        this.globalSettings.memory?.memory_guard_custom_ceiling_gb || 0
                    );
                    const candidates = [custom, staticCeiling];
                    if (metalCapGB > 0) candidates.push(metalCapGB);
                    const ceiling = Math.max(0, Math.min(...candidates));
                    const tmpl = kernelBinds([custom, staticCeiling], ceiling)
                        ? 'settings.resource.guard_tier.breakdown_custom_kernel_limit'
                        : 'settings.resource.guard_tier.breakdown_custom';
                    return window
                        .t(tmpl)
                        .replace('{custom_gb}', bold(custom))
                        .replace('{ceiling}', bold(ceiling));
                }

                const freeGB = (sys.free_memory_bytes || 0) / GB;
                const inactiveGB = (sys.inactive_memory_bytes || 0) / GB;
                const activeGB = (sys.active_memory_bytes || 0) / GB;
                const ratio = { safe: 0.2, balanced: 0.5, aggressive: 0.8 }[tier] ?? 0.5;
                const pct = Math.round(ratio * 100);
                const reclaim = activeGB * ratio;
                const omlxGB = (sys.omlx_phys_footprint_bytes || 0) / GB;
                const dynamicCeiling = omlxGB + freeGB + inactiveGB + reclaim;
                const candidates = [dynamicCeiling, staticCeiling];
                if (metalCapGB > 0) candidates.push(metalCapGB);
                const ceiling = Math.max(0, Math.min(...candidates));
                const tmpl = kernelBinds([dynamicCeiling, staticCeiling], ceiling)
                    ? 'settings.resource.guard_tier.breakdown_kernel_limit'
                    : 'settings.resource.guard_tier.breakdown';
                return window
                    .t(tmpl)
                    .replace('{free}', bold(freeGB))
                    .replace('{inactive}', bold(inactiveGB))
                    .replace('{active}', bold(activeGB))
                    .replace(/{active_pct}/g, pct)
                    .replace('{reclaim}', bold(reclaim))
                    .replace('{ceiling}', bold(ceiling));
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
            normalizeHotCacheMaxSize(value) {
                const normalized = String(value ?? '').trim();
                if (!normalized || normalized.toLowerCase() === 'auto') return '0';
                return normalized;
            },

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

            // Filter + sort models
            get sortedModels() {
                return [...this.filterModelsByName(this.models, this.modelSearch)].sort((a, b) => {
                    // Favorites always sort first, regardless of the active column.
                    const favDiff = (b.is_favorite ? 1 : 0) - (a.is_favorite ? 1 : 0);
                    if (favDiff !== 0) return favDiff;

                    let aVal, bVal;

                    switch (this.sortBy) {
                        case 'id':
                            aVal = (a.display_name || a.id || '').toLowerCase();
                            bVal = (b.display_name || b.id || '').toLowerCase();
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
                        case 'is_hidden':
                            aVal = a.is_hidden ? 1 : 0;
                            bVal = b.is_hidden ? 1 : 0;
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
                this.persistSort('omlx_models_sort_by', this.sortBy, 'omlx_models_sort_order', this.sortOrder);
            },

            resetSort() {
                this.sortBy = MODELS_SORT_DEFAULT.by;
                this.sortOrder = MODELS_SORT_DEFAULT.order;
                try {
                    localStorage.removeItem('omlx_models_sort_by');
                    localStorage.removeItem('omlx_models_sort_order');
                } catch (e) { /* storage disabled */ }
            },

            get isModelsSortDefault() {
                return this.sortBy === MODELS_SORT_DEFAULT.by
                    && this.sortOrder === MODELS_SORT_DEFAULT.order;
            },

            // ---- Manager (Browse Models > Local) filter + sort ----

            // Cross-reference the richer /api/models entry (has model_type,
            // settings) for a manager row keyed by its model name.
            managerModelInfo(name) {
                return this.models.find(m => m.id === name);
            },

            filterModelsByName(list, query) {
                const q = (query || '').trim().toLowerCase();
                if (!q) return list;
                return list.filter(m => {
                    const id = (m.id || m.name || '').toLowerCase();
                    const display = (m.display_name || '').toLowerCase();
                    const alias = (
                        (m.settings && m.settings.model_alias)
                        || (this.managerModelInfo(m.name) && this.managerModelInfo(m.name).settings
                            && this.managerModelInfo(m.name).settings.model_alias)
                        || ''
                    ).toLowerCase();
                    return id.includes(q) || display.includes(q) || alias.includes(q);
                });
            },

            get sortedManagerModels() {
                const list = this.filterModelsByName(this.hfModels, this.managerSearch);
                return [...list].sort((a, b) => {
                    // Favorites always sort first, regardless of the active column.
                    const aFav = this.managerModelInfo(a.name)?.is_favorite ? 1 : 0;
                    const bFav = this.managerModelInfo(b.name)?.is_favorite ? 1 : 0;
                    if (aFav !== bFav) return bFav - aFav;

                    let aVal, bVal;
                    switch (this.managerSortBy) {
                        case 'name':
                            aVal = (a.display_name || a.name || '').toLowerCase();
                            bVal = (b.display_name || b.name || '').toLowerCase();
                            break;
                        case 'type':
                            aVal = (this.managerModelInfo(a.name)?.model_type || 'llm').toLowerCase();
                            bVal = (this.managerModelInfo(b.name)?.model_type || 'llm').toLowerCase();
                            break;
                        case 'size':
                            aVal = a.size || 0;
                            bVal = b.size || 0;
                            break;
                        default:
                            return 0;
                    }
                    if (aVal < bVal) return this.managerSortOrder === 'asc' ? -1 : 1;
                    if (aVal > bVal) return this.managerSortOrder === 'asc' ? 1 : -1;
                    return 0;
                });
            },

            toggleManagerSort(column) {
                if (this.managerSortBy === column) {
                    this.managerSortOrder = this.managerSortOrder === 'asc' ? 'desc' : 'asc';
                } else {
                    this.managerSortBy = column;
                    this.managerSortOrder = 'asc';
                }
                this.persistSort('omlx_manager_sort_by', this.managerSortBy, 'omlx_manager_sort_order', this.managerSortOrder);
            },

            resetManagerSort() {
                this.managerSortBy = MANAGER_SORT_DEFAULT.by;
                this.managerSortOrder = MANAGER_SORT_DEFAULT.order;
                try {
                    localStorage.removeItem('omlx_manager_sort_by');
                    localStorage.removeItem('omlx_manager_sort_order');
                } catch (e) { /* storage disabled */ }
            },

            get isManagerSortDefault() {
                return this.managerSortBy === MANAGER_SORT_DEFAULT.by
                    && this.managerSortOrder === MANAGER_SORT_DEFAULT.order;
            },

            persistSort(byKey, byVal, orderKey, orderVal) {
                try {
                    localStorage.setItem(byKey, byVal);
                    localStorage.setItem(orderKey, orderVal);
                } catch (e) { /* storage disabled */ }
            },

            // Deeplink from a manager row to that model's settings card (modal).
            openModelSettingsFromManager(name) {
                const model = this.managerModelInfo(name);
                if (model) this.openModelSettings(model);
            },

            // Theme select
            setTheme(theme) {
                this.theme = theme;
                localStorage.setItem(THEME_STORAGE_KEY, this.theme);
                this.applyTheme();
            },

            setEnhancedReadability(enabled) {
                this.enhancedReadability = enabled;
                localStorage.setItem(ENHANCED_READABILITY_KEY, enabled ? 'on' : 'off');
                if (enabled) {
                    document.documentElement.setAttribute('data-enhanced-readability', '');
                } else {
                    document.documentElement.removeAttribute('data-enhanced-readability');
                }
            },

            applyTheme() {
                // Clean up existing listener
                if (this.systemThemeListener) {
                    window.matchMedia('(prefers-color-scheme: dark)').removeEventListener('change', this.systemThemeListener);
                    this.systemThemeListener = null;
                }

                if (this.theme === 'auto') {
                    // Detect system theme
                    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    this.activeTheme = prefersDark ? 'dark' : 'light';
                    document.documentElement.setAttribute('data-theme', this.activeTheme);

                    // Add listener for system theme changes
                    this.systemThemeListener = (e) => {
                        this.activeTheme = e.matches ? 'dark' : 'light';
                        document.documentElement.setAttribute('data-theme', this.activeTheme);
                    };
                    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', this.systemThemeListener);
                } else {
                    // Use explicit theme
                    this.activeTheme = this.theme;
                    document.documentElement.setAttribute('data-theme', this.activeTheme);
                }
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
                        alert(Array.isArray(data.detail) ? data.detail.map(e => (e && typeof e === 'object') ? (e.msg || JSON.stringify(e)) : String(e)).join(', ') : (data.detail || 'Failed to save'));
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
                        this.oqAllModels = data.all_models || [];
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
                    const payload = {
                        model_path: this.oqSelectedModelPath,
                        oq_level: this.oqLevel,
                        group_size: 64,
                        sensitivity_model_path: this.oqSensitivityModelPath,
                        text_only: this.oqTextOnly,
                        dtype: this.oqDtype,
                        preserve_mtp: this.oqSelectedModelHasMtp() ? this.oqPreserveMtp : false,
                        mtp_assistant_model_path: this.oqMtpAssistantCandidates().some(m => m.path === this.oqMtpAssistantPath)
                            ? this.oqMtpAssistantPath : '',
                    };
                    if (this.oqEnhanced) {
                        payload.enhanced = true;
                        payload.imatrix_reuse_cache = this.oqeReuseImatrixCache;
                        payload.imatrix_cache_path = this.oqeImatrixCachePath.trim();
                        payload.imatrix_strict = this.oqeStrictImatrix;
                    }
                    const response = await fetch('/admin/api/oq/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (response.ok) {
                        const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                        const name = model ? model.name : this.oqSelectedModelPath;
                        this.oqSuccess = `Quantization started: ${name} → oQ${this.oqLevel}${this.oqEnhanced ? 'e' : ''}`;
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
                const label = task.progress_detail || task.phase || task.status;
                return `${pct}% · ${label}`;
            },

            formatOQElapsed(task) {
                if (!task.started_at) return '';
                const now = task.completed_at || (Date.now() / 1000);
                const elapsed = now - task.started_at;
                const mins = Math.floor(elapsed / 60);
                const secs = Math.floor(elapsed % 60);
                return `${mins}:${String(secs).padStart(2, '0')}`;
            },

            oqSensitivityModelCandidates() {
                if (!this.oqSelectedModelPath) return [];
                const source = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                if (!source) return [];
                return this.oqAllModels.filter(m =>
                    m.path !== this.oqSelectedModelPath &&
                    m.is_quantized &&
                    m.model_type === source.model_type
                );
            },

            oqMtpAssistantCandidates() {
                // Gemma 4 ships its MTP head as a separate gemma4_assistant
                // checkpoint; offer to merge it into the quantized output.
                // Qwen3.5/3.6 recipients can instead graft the native mtp.*
                // head out of a same-geometry donor checkpoint (e.g. the
                // base model of a fine-tune). Loose filter here; strict
                // tokenizer/geometry validation happens server-side at
                // submit.
                if (!this.oqSelectedModelPath) return [];
                const source = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                if (!source) return [];
                if (source.model_type === 'gemma4') {
                    return this.oqAllModels.filter(m => m.model_type === 'gemma4_assistant');
                }
                const family = this.oqMtpFamily(source.model_type);
                if (!family) return [];
                if (this.oqSelectedModelHasMtp() && this.oqPreserveMtp) return [];
                return this.oqAllModels.filter(m =>
                    m.path !== source.path &&
                    m.has_mtp_heads &&
                    this.oqMtpFamily(m.model_type) === family &&
                    (!m.hidden_size || !source.hidden_size || m.hidden_size === source.hidden_size)
                );
            },

            oqMtpFamily(modelType) {
                if (!modelType) return null;
                if (modelType.startsWith('qwen3_6')) return 'qwen3_6';
                if (modelType.startsWith('qwen3_5')) return 'qwen3_5';
                return null;
            },

            oqSelectedModelType() {
                const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                return model?.model_type || '';
            },

            oqLevelLabel(level) {
                return `oQ${level}${this.oqEnhanced ? 'e' : ''}`;
            },

            oqSelectedModelIsVLM() {
                const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                return model?.is_vlm || false;
            },

            oqSelectedModelHasMtp() {
                const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                return model?.has_mtp_heads || false;
            },

            oqEstimatedMemory() {
                // Use precise estimate from API if available
                if (this.oqEstimate) {
                    // If sensitivity model selected, memory ≈ sensitivity model size × 1.5
                    if (this.oqSensitivityModelPath) {
                        const sensModel = this.oqAllModels.find(m => m.path === this.oqSensitivityModelPath);
                        if (sensModel) {
                            const bytes = Math.round(sensModel.size * 1.5) + 5 * 1024 * 1024 * 1024;
                            if (bytes > 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
                            return (bytes / (1024 * 1024)).toFixed(0) + ' MB';
                        }
                    }
                    return this.oqEstimate.memory_streaming_formatted || '';
                }
                // Fallback to rough model-level estimate
                const model = this.oqModels.find(m => m.path === this.oqSelectedModelPath);
                if (!model) return '';
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
                            preserve_mtp: this.oqSelectedModelHasMtp() && this.oqPreserveMtp ? 'true' : 'false',
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
            // oQ Uploader Functions
            // =================================================================

            async validateUploadToken() {
                if (!this.uploadHfToken || this.uploadTokenValidating) return;
                this.uploadTokenValidating = true;
                this.uploadError = '';
                try {
                    const response = await fetch('/admin/api/upload/validate-token', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ hf_token: this.uploadHfToken }),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (response.ok) {
                        this.uploadHfUsername = data.username || '';
                        this.uploadHfOrgs = data.orgs || [];
                        this.uploadHfNamespace = this.uploadHfUsername;
                        this.uploadTokenValidated = true;
                        localStorage.setItem('omlx-hf-upload-token', this.uploadHfToken);
                        this.loadUploadOqModels();
                    } else {
                        this.uploadError = data.detail || window.t('models.uploader.invalid_token');
                        this.uploadTokenValidated = false;
                    }
                } catch (err) {
                    this.uploadError = 'Connection error. Server may be unavailable.';
                } finally {
                    this.uploadTokenValidating = false;
                }
            },

            async loadUploadOqModels() {
                try {
                    const response = await fetch('/admin/api/upload/oq-models');
                    if (response.ok) {
                        const data = await response.json();
                        this.uploadOqModels = data.oq_models || [];
                        this.uploadAllModels = data.all_models || [];
                        this.uploadOqModelsLoaded = true;
                    }
                } catch (err) {
                    console.error('Failed to load oQ models for upload:', err);
                }
            },

            openUploadModal(model) {
                this.uploadModalModelPath = model.path;
                this.uploadModalModelName = model.name;
                this.uploadModalRepoId = (this.uploadHfNamespace || this.uploadHfUsername) + '/' + model.name;
                this.uploadReadmeSource = '';
                this.uploadAutoReadme = true;
                this.uploadRedownloadNotice = false;
                this.uploadPrivate = false;
                this.uploadStarting = false;
                this.uploadModalOpen = true;
            },

            async startUpload() {
                if (!this.uploadModalRepoId || this.uploadStarting) return;
                this.uploadStarting = true;
                this.uploadError = '';
                try {
                    const response = await fetch('/admin/api/upload/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model_path: this.uploadModalModelPath,
                            repo_id: this.uploadModalRepoId,
                            hf_token: this.uploadHfToken,
                            readme_source_path: this.uploadReadmeSource,
                            auto_readme: this.uploadAutoReadme,
                            redownload_notice: this.uploadRedownloadNotice && this.uploadReadmeSource === '',
                            private: this.uploadPrivate,
                        }),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (response.ok) {
                        this.uploadModalOpen = false;
                        this.uploadSuccess = `Upload queued: ${this.uploadModalModelName}`;
                        await this.loadUploadTasks();
                        this.startUploadRefresh();
                        setTimeout(() => { this.uploadSuccess = ''; }, 5000);
                    } else {
                        this.uploadError = data.detail || 'Failed to start upload';
                    }
                } catch (err) {
                    this.uploadError = 'Connection error. Server may be unavailable.';
                } finally {
                    this.uploadStarting = false;
                }
            },

            async loadUploadTasks() {
                try {
                    const response = await fetch('/admin/api/upload/tasks');
                    if (response.ok) {
                        const data = await response.json();
                        this.uploadTasks = data.tasks || [];
                        const hasActive = this.uploadTasks.some(t =>
                            ['pending', 'uploading'].includes(t.status));
                        if (!hasActive) {
                            this.stopUploadRefresh();
                        }
                    }
                } catch (err) {
                    console.error('Failed to load upload tasks:', err);
                }
            },

            async cancelUploadTask(taskId) {
                try {
                    await fetch(`/admin/api/upload/cancel/${taskId}`, { method: 'POST' });
                    await this.loadUploadTasks();
                } catch (err) {
                    console.error('Failed to cancel upload task:', err);
                }
            },

            async removeUploadTask(taskId) {
                try {
                    await fetch(`/admin/api/upload/task/${taskId}`, { method: 'DELETE' });
                    await this.loadUploadTasks();
                } catch (err) {
                    console.error('Failed to remove upload task:', err);
                }
            },

            startUploadRefresh() {
                this.stopUploadRefresh();
                this._uploadRefreshTimer = setInterval(() => {
                    this.loadUploadTasks();
                }, 2000);
            },

            stopUploadRefresh() {
                if (this._uploadRefreshTimer) {
                    clearInterval(this._uploadRefreshTimer);
                    this._uploadRefreshTimer = null;
                }
            },

            formatUploadElapsed(task) {
                if (!task.started_at) return '';
                const now = task.completed_at || (Date.now() / 1000);
                const elapsed = now - task.started_at;
                const mins = Math.floor(elapsed / 60);
                const secs = Math.floor(elapsed % 60);
                return `${mins}:${String(secs).padStart(2, '0')}`;
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
                        const data = await response.json();
                        this.hfTokenInvalid = !!data.hf_token_invalid;
                        // Attach original rank so the # column survives column-header re-sorts
                        this.hfRecommended = {
                            trending: (data.trending || []).map((m, i) => ({ ...m, rank: i + 1 })),
                            popular: (data.popular || []).map((m, i) => ({ ...m, rank: i + 1 })),
                        };
                        this.hfRecommendedLoaded = true;
                        this.hfPage.trending = 1;
                        this.hfPage.popular = 1;
                        // Default sort for trending/popular is original rank
                        this.hfTableSort = 'rank';
                        this.hfTableSortDir = 'asc';
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

            // Table sort helpers for Browse Models
            sortModels(list) {
                const sortBy = this.hfTableSort;
                const dir = this.hfTableSortDir === 'asc' ? 1 : -1;
                return [...list].sort((a, b) => {
                    if (sortBy === 'rank') {
                        return dir * ((a.rank || 0) - (b.rank || 0));
                    } else if (sortBy === 'name') {
                        return dir * (a.name || '').localeCompare(b.name || '');
                    } else if (sortBy === 'downloads') {
                        return dir * ((a.downloads || 0) - (b.downloads || 0));
                    } else if (sortBy === 'likes') {
                        return dir * ((a.likes || 0) - (b.likes || 0));
                    } else if (sortBy === 'size') {
                        return dir * ((a.size || 0) - (b.size || 0));
                    } else if (sortBy === 'params') {
                        return dir * ((a.params || 0) - (b.params || 0));
                    }
                    return 0;
                });
            },

            toggleTableSort(column) {
                if (this.hfTableSort === column) {
                    this.hfTableSortDir = this.hfTableSortDir === 'asc' ? 'desc' : 'asc';
                } else {
                    this.hfTableSort = column;
                    // Name and rank read more naturally as ascending by default
                    this.hfTableSortDir = (column === 'name' || column === 'rank') ? 'asc' : 'desc';
                }
            },

            syncTableSortToDropdown() {
                const map = {
                    largest:      { col: 'size',      dir: 'desc' },
                    smallest:     { col: 'size',      dir: 'asc'  },
                    most_params:  { col: 'params',    dir: 'desc' },
                    least_params: { col: 'params',    dir: 'asc'  },
                    downloads:    { col: 'downloads', dir: 'desc' },
                    trending:     { col: 'downloads', dir: 'desc' },
                    created:      { col: 'downloads', dir: 'desc' },
                    updated:      { col: 'downloads', dir: 'desc' },
                };
                const m = map[this.hfSearchSort];
                if (m) {
                    this.hfTableSort = m.col;
                    this.hfTableSortDir = m.dir;
                }
            },

            // Pagination helpers
            getPagedModels(tab) {
                const page = this.hfPage[tab] || 1;
                const size = this.hfPageSize;
                let list;
                if (tab === 'trending') list = this.hfRecommended.trending || [];
                else if (tab === 'popular') list = this.hfRecommended.popular || [];
                else list = this.hfSearchResults || [];
                // Apply table sorting
                const sorted = this.sortModels(list);
                return sorted.slice((page - 1) * size, page * size);
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
            },

            // Search
            async searchHFModels() {
                if (!this.hfSearchQuery.trim()) return;
                this.hfSearchLoading = true;
                this.hfRecommendedTab = 'search';
                this.hfPage.search = 1;
                // Sync table sort with dropdown choice so the frontend re-sort
                // does not override what the backend returned
                this.syncTableSortToDropdown();
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 15000);
                try {
                    const params = new URLSearchParams({
                        q: this.hfSearchQuery,
                        sort: this.hfSearchSort,
                        limit: '100',
                        mlx_only: this.hfMlxOnly,
                    });
                    // Add filter parameters if set. Sizes use binary GiB to match
                    // _format_model_size on the backend.
                    const GIB = 1024 * 1024 * 1024;
                    if (this.hfSearchMinParams) params.set('min_params', (parseFloat(this.hfSearchMinParams) * 1e9).toString());
                    if (this.hfSearchMaxParams) params.set('max_params', (parseFloat(this.hfSearchMaxParams) * 1e9).toString());
                    if (this.hfSearchMaxSize) params.set('max_size', (parseFloat(this.hfSearchMaxSize) * GIB).toString());
                    if (this.hfSearchMinSize) params.set('min_size', (parseFloat(this.hfSearchMinSize) * GIB).toString());
                    // Wire largest/smallest sort params to backend
                    if (this.hfSearchSort === 'largest') {
                        params.set('sort_by_size', 'true');
                        params.set('sort_ascending', 'false');
                    } else if (this.hfSearchSort === 'smallest') {
                        params.set('sort_by_size', 'true');
                        params.set('sort_ascending', 'true');
                    }


                    const response = await fetch(`/admin/api/hf/search?${params}`, { signal: controller.signal });
                    if (response.ok) {
                        const data = await response.json();
                        this.hfTokenInvalid = !!data.hf_token_invalid;
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
                }
            },

            clearHFSearchFilters() {
                this.hfSearchMinParams = '';
                this.hfSearchMaxParams = '';
                this.hfSearchMaxSize = '';
                this.hfSearchMinSize = '';
                if (this.hfSearchQuery.trim()) this.immediateSearch();
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
                }
            },
        }
    }
