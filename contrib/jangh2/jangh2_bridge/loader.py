"""Opt-in, model-specific JANGH2 loader for the already-installed Qwen4 model.

No package files or checkpoint files are edited. The only persistent hook is an
in-process wrapper around mlx_vlm.utils.load_model, restricted to one path.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import importlib
import inspect
import json
import logging
from pathlib import Path
import threading
from typing import Any, Mapping

from .performance import PerformanceOptions
from .checkpoint import inspect_checkpoint
from .ple import PLEState, stock_ple_context
from .metadata import canonical_map, choose_ple_prefix, is_mtp, is_ple_storage, safe_shard_path, reconcile_ple_module_parameters

log = logging.getLogger('jangh2_bridge')
_LOAD_LOCK = threading.RLock()


@contextmanager
def ple_prefix_context(language: Any, weight_map: Mapping[str,str]):
    """Pass physical tensor prefixes to the existing mmap reader, temporarily."""
    cls = getattr(language,'DiskBackedShardedEmbedding',None)
    if cls is None:
        raise RuntimeError('Installed Qwen4 implementation has no SSD PLE reader')
    original = cls.__init__
    signature = inspect.signature(original)
    if 'prefix' not in signature.parameters:
        raise RuntimeError('Unrecognised SSD PLE constructor; refusing an unsafe patch')
    @functools.wraps(original)
    def init(self,*args,**kwargs):
        bound = signature.bind(self,*args,**kwargs)
        bound.arguments['prefix'] = choose_ple_prefix(bound.arguments['prefix'],weight_map)
        return original(*bound.args,**bound.kwargs)
    cls.__init__ = init
    try:
        yield
    finally:
        cls.__init__ = original


@contextmanager
def vendor_runtime_context(language):
    names=('_PLE_RUNTIME_MODEL_PATH','_PLE_RUNTIME_MODE','_MTP_RUNTIME')
    saved={name:getattr(language,name) for name in names if hasattr(language,name)}
    try:
        yield
    finally:
        for name,value in saved.items():
            setattr(language,name,value)



@contextmanager
def ple_runtime_context(language, checkpoint, *, max_open_files=8, row_cache_bytes=0, device_cache_bytes=0):
    """Select a verified mmap construction path for either Qwen4 implementation."""
    configure_ple = getattr(language, 'configure_ple_runtime', None)
    if not callable(configure_ple):
        with stock_ple_context(language, checkpoint, max_open_files=max_open_files,
                               row_cache_bytes=row_cache_bytes, device_cache_bytes=device_cache_bytes) as state:
            log.info('JANGH2: PLE API=stock-indexed-mmap; resident constructor intercepted')
            yield state
        return
    with vendor_runtime_context(language):
        resolved = configure_ple(checkpoint.root, mode='mmap')
        get_mode = getattr(language, 'get_ple_runtime_mode', None)
        mode = get_mode() if callable(get_mode) else resolved
        if mode != 'mmap':
            raise RuntimeError(f'Vendor PLE refused mmap mode: {mode!r}; refusing resident allocation')
        configure_mtp = getattr(language, 'configure_mtp_runtime', None)
        if callable(configure_mtp):
            configure_mtp(checkpoint.root, enabled=False)
        elif hasattr(language, 'get_mtp_runtime') and language.get_mtp_runtime().enabled:
            raise RuntimeError('Cannot disable the installed MTP runtime')
        with ple_prefix_context(language, checkpoint.weight_map):
            log.info('JANGH2: PLE API=vendor-mmap')
            yield PLEState('vendor-mmap')


def compare_parameters(expected: Mapping[str,Any], supplied: Mapping[str,Any]) -> None:
    missing,extra = sorted(set(expected)-set(supplied)),sorted(set(supplied)-set(expected))
    bad = [(k,tuple(expected[k].shape),tuple(supplied[k].shape))
           for k in set(expected)&set(supplied) if tuple(expected[k].shape)!=tuple(supplied[k].shape)]
    if missing or extra or bad:
        raise ValueError('Strict JANGH2 parameter audit failed. '
                         f'Missing ({len(missing)}): {missing[:12]}; '
                         f'Unexpected ({len(extra)}): {extra[:12]}; '
                         f'Shape mismatches ({len(bad)}): {sorted(bad)[:8]}. '
                         'No unmatched weights have been silently ignored.')


def recenter_residual_norms(model,weights,norm_class,xp) -> int:
    count = 0
    for name,module in model.named_modules():
        if not isinstance(module,norm_class):
            continue
        key = name+'.weight'
        if key not in weights:
            raise ValueError(f'Missing residual RMSNorm tensor: {key}')
        value = weights[key]
        if not xp.issubdtype(value.dtype,xp.floating):
            raise ValueError(f'Non-floating residual RMSNorm tensor: {key}')
        residual = value.astype(xp.float32)-1.0
        narrow = residual.astype(value.dtype)
        exact = xp.array_equal(narrow.astype(xp.float32),residual)
        # Keep BF16 when exact so vendor fused kernels keep their supported
        # dtype, but do not lose gamma information for values below 0.5.
        weights[key] = narrow if bool(exact.item() if hasattr(exact, "item") else exact) else residual
        count += 1
    return count


def _sanitize(model,weights,language,config,mx):
    convention = config.get('jang_config',{}).get('norm_convention')
    if convention != 'runtime_plus1_applied':
        raise ValueError('This checkpoint bridge requires explicit jang_config.norm_convention='
                         'runtime_plus1_applied; refusing to guess the RMSNorm convention')
    norm_class = getattr(language,'Qwen4ExpRMSNorm',None)
    if norm_class is None:
        raise RuntimeError('Installed Qwen4 module has no Qwen4ExpRMSNorm class')
    owner = importlib.import_module(type(model).__module__)
    helper_name = '_normalize_ones_centered_rmsnorm_weights'
    heuristic = getattr(owner,helper_name,None)
    if heuristic is not None:
        setattr(owner,helper_name,lambda model,weights: None)
    try:
        # The model's own transforms handle Conv1d and PLE shared scales. We
        # supply canonical names first, so defaults cannot overwrite real data.
        before = set(weights)
        weights = model.sanitize(weights)
        vision = getattr(model,'vision_tower',None)
        if vision is not None and hasattr(vision,'sanitize'):
            weights = vision.sanitize(weights)
        weights = canonical_map(weights)
        lost = before - set(weights)
        allowed = {k for k in lost if 'position_ids' in k or is_mtp(k)}
        if lost-allowed:
            raise ValueError(f'Unexpected weights dropped by vendor sanitizer: {sorted(lost-allowed)[:12]}')
        count = recenter_residual_norms(model,weights,norm_class,mx)
        log.info('JANGH2: recentered %d residual RMSNorm tensors using explicit checkpoint metadata',count)
        return weights
    finally:
        if heuristic is not None:
            setattr(owner,helper_name,heuristic)


def load_jangh2(model_path: str | Path, *, backend: str='reference', chunk_size: int | None=None,
                lazy: bool=False, **kwargs):
    """Load this known format without passing JANGH2 metadata to affine layers."""
    import mlx.core as mx
    import mlx_vlm.utils as utils
    from mlx.utils import tree_flatten
    from .experts import replace_experts, validate_loaded_experts, quantize_affine

    if kwargs.get('quantize_activations') or kwargs.get('adapter_path'):
        raise ValueError('Activation quantization / adapters are outside this experimental bridge')
    if kwargs.get('strict',True) is not True:
        raise ValueError('The experimental JANGH2 loader requires strict=True')
    with _LOAD_LOCK:
        checkpoint = inspect_checkpoint(model_path)
        root = checkpoint.root
        # Use the library's config reader for generation defaults as well.
        config = utils.load_config(root)
        text = config.get('text_config',{})
        if text.get('tie_word_embeddings',False):
            raise ValueError('Tied embeddings are not supported by this checkpoint-specific bridge')
        architecture = importlib.import_module('mlx_vlm.models.qwen4_exp')
        language = importlib.import_module('mlx_vlm.models.qwen4_exp.language')
        if chunk_size is None:
            chunk_size = 8 if backend == 'reference-fast' else 1
        log.info('JANGH2: architecture=%s, backend=%s, chunk_size=%d',architecture.__file__,backend,chunk_size)
        log.info('JANGH2: checkpoint audit: %s',json.dumps(checkpoint.report(),sort_keys=True))
        options = PerformanceOptions.from_environment(fast_defaults=backend == 'metal')
        if options.prefill == 'grouped' or options.ple_device_mib:
            from .profiles import require_qualification
            require_qualification()
        with ple_runtime_context(language, checkpoint, max_open_files=options.ple_open_files,
                                 row_cache_bytes=options.ple_cache_mib*1024*1024,
                                 device_cache_bytes=options.ple_device_mib*1024*1024) as ple_state:
            log.info('JANGH2: PLE=mmap (%s); base-model load, MTP weights excluded',ple_state.kind)
            config.setdefault('text_config',config.pop('llm_config',{}))
            config.setdefault('vision_config',{})
            model_config = architecture.ModelConfig.from_dict(config)
            model_config = utils.update_module_configs(model_config,architecture,config,
                                                       ['text','vision','perceiver','projector','audio'])
            generation_defaults = getattr(utils,'apply_generation_config_defaults',None)
            if generation_defaults is not None:
                model_config = generation_defaults(model_config,config)
            if hasattr(model_config,'model_path'):
                model_config.model_path = str(root)
            model = None
            try:
                model = architecture.Model(model_config)
                ple_state.validate_complete()
                # Crucial ordering: replace the routed experts BEFORE nn.quantize.
                replaced = replace_experts(model,config,backend=backend,chunk_size=chunk_size)
                by_file: dict[str,list[str]] = {}
                for raw_key,filename in checkpoint.weight_map.items():
                    if not is_ple_storage(raw_key) and not is_mtp(raw_key):
                        by_file.setdefault(filename,[]).append(raw_key)
                raw = {}
                for number,(filename,keys) in enumerate(sorted(by_file.items()),1):
                    # mx.load creates lazy array loads. Never evaluate the entire
                    # returned dict: a mixed shard may also contain the SSD table.
                    arrays = mx.load(str(safe_shard_path(root,filename)))
                    for key in keys:
                        if key not in arrays:
                            raise ValueError(f'Indexed tensor missing while loading {filename}: {key}')
                        raw[key] = arrays[key]
                    del arrays
                    log.info('JANGH2: staged resident shard %d/%d: %s',number,len(by_file),filename)
                weights = canonical_map(raw)
                del raw
                weights = _sanitize(model,weights,language,config,mx)
                validate_loaded_experts(model,weights)
                quantize_affine(model,config,weights)
                expected = dict(tree_flatten(model.parameters()))
                weights, remapped = reconcile_ple_module_parameters(weights, expected)
                if remapped:
                    log.info('JANGH2: reconciled %d exact PLE runtime parameter aliases', remapped)
                compare_parameters(expected,weights)
                model.load_weights(list(weights.items()),strict=True)
                # Release references before evaluation; all parameters now belong
                # to the model. Only resident parameters are materialized.
                del weights
                if not lazy:
                    mx.eval(model.parameters())
                model.model_path = root
                model._jangh2_bridge_model = True
                model._jangh2_ple_state = ple_state
                model.eval()
                log.info('JANGH2: strict load complete; %d expert blocks use %s',len(replaced),backend)
                return model
            except BaseException:
                log.exception('JANGH2 bridge load failed; this is the primary failure before any generic LLM fallback')
                if model is not None:
                    close = getattr(model,'close',None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            log.exception('Failed to close PLE resources after unsuccessful load')
                raise


def install(target_model: str | Path, *, backend: str='reference', chunk_size: int | None=None):
    """Install a process-local hook for one exact local checkpoint path."""
    import mlx_vlm.utils as utils
    target = Path(target_model).expanduser().resolve()
    if not (target/'config.json').is_file():
        raise FileNotFoundError(target/'config.json')
    if chunk_size is None:
        chunk_size = 8 if backend == 'reference-fast' else 1
    if backend not in ('reference','reference-fast','native','metal') or not 1 <= chunk_size <= 64:
        raise ValueError('Invalid backend or route chunk size')
    if getattr(utils.load_model,'_jangh2_bridge_target',None) is not None:
        raise RuntimeError('A JANGH2 bridge is already installed in this process')
    original = utils.load_model
    @functools.wraps(original)
    def load_model(model_path,lazy=False,**kwargs):
        with _LOAD_LOCK:
            if Path(model_path).expanduser().resolve() != target:
                return original(model_path,lazy=lazy,**kwargs)
            return load_jangh2(target,backend=backend,chunk_size=chunk_size,lazy=lazy,**kwargs)
    load_model._jangh2_bridge_target = str(target)
    utils.load_model = load_model
    def uninstall():
        if utils.load_model is load_model:
            utils.load_model = original
    return uninstall
