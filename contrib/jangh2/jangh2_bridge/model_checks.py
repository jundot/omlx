"""Whole-model comparison helpers. No backend-specific imports at module import."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np


def compare_logits(actual,expected,*,relative_l2_limit=.02,max_abs_limit=1.0,
                   mean_kl_limit=.01,top1_min=.9):
    a=np.asarray(actual,np.float64);b=np.asarray(expected,np.float64)
    if a.shape!=b.shape or a.ndim<2 or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Nonfinite, empty or differently shaped whole-model logits')
    a=a.reshape(-1,a.shape[-1]);b=b.reshape(-1,b.shape[-1]);delta=a-b
    def logsm(x):
        z=x-x.max(axis=-1,keepdims=True);return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))
    la,lb=logsm(a),logsm(b)
    result={'relative_l2':float(np.linalg.norm(delta)/(np.linalg.norm(b)+1e-20)),
            'max_abs':float(np.abs(delta).max()),'rms_error':float(np.sqrt(np.mean(delta**2))),
            'mean_kl_reference_to_candidate':float(np.mean(np.sum(np.exp(lb)*(lb-la),axis=-1))),
            'top1_agreement':float(np.mean(a.argmax(axis=-1)==b.argmax(axis=-1))),
            'positions':a.shape[0],'vocab_size':a.shape[1],
            'thresholds':{'relative_l2':relative_l2_limit,'max_abs':max_abs_limit,
                          'mean_kl':mean_kl_limit,'top1_min':top1_min}}
    if (result['relative_l2']>relative_l2_limit or result['max_abs']>max_abs_limit
            or result['mean_kl_reference_to_candidate']>mean_kl_limit or result['top1_agreement']<top1_min):
        raise AssertionError(f'Whole-model logit comparison failed: {result}')
    return result


@contextmanager
def temporary_expert_forward(model,forward):
    """Standalone qualification only. Share immutable packed weights; no second 35GB load."""
    classes={type(m) for _,m in model.named_modules() if getattr(m,'is_jangtq2',False)}
    if not classes:raise ValueError('No JANGH2 expert modules to compare')
    originals={cls:cls.__call__ for cls in classes}
    try:
        for cls in classes:cls.__call__=forward
        yield
    finally:
        for cls,original in originals.items():cls.__call__=original


def reference_call(self,x,ids):
    import mlx.core as mx
    from .reference_ops import expert_forward
    return expert_forward(x,ids,{p:getattr(self,p).spec() for p in ('gate_proj','up_proj','down_proj')},
                          mx,chunk_size=8,evaluation='per_chunk')


def ple_readers(model):
    state=getattr(model,'_jangh2_ple_state',None)
    return list(getattr(state,'readers',[]))


def ple_stats(model):return [r.mapping_stats() for r in ple_readers(model)]


def reset_device_rows(model):
    for reader in ple_readers(model):
        if reader._device_cache is not None:
            reader._device_cache.close();reader._device_cache=None


def logits(language,ids,cache,mx):
    result=language(ids,cache=cache)
    value=getattr(result,'logits',result)
    if not isinstance(value,mx.array) or value.ndim!=3:
        raise TypeError('Unrecognised Qwen4 language-model output; expected [batch,tokens,vocabulary] logits')
    return value
