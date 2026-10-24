"""Reversible, target-model-only prompt policy for the existing oMLX server.

Does not replace the API response parser, template, system prompt, sampler,
expert math, caches, stream ownership or generated content. All hooks are
validated before being installed, and leave unrelated models untouched.
"""
from __future__ import annotations

import copy
import functools
from importlib import import_module
import inspect
import logging
from pathlib import Path
import threading

from .chat_policy import ChatPolicy, prompt_reasoning_state, token_reasoning_state

log = logging.getLogger('jangh2_bridge')
_LOCK = threading.RLock()
_ACTIVE = False


def install(target_model, *, policy: ChatPolicy | None = None):
    global _ACTIVE
    policy = policy or ChatPolicy.from_environment()
    target = Path(target_model).expanduser().resolve()
    engine_class = import_module('omlx.engine.vlm').VLMBatchedEngine
    scheduler_class = import_module('omlx.scheduler').Scheduler
    server = import_module('omlx.server')

    def targeted(engine):
        name = getattr(engine, '_model_name', None)
        return isinstance(name, (str, Path)) and Path(name).expanduser().resolve() == target

    patches = []

    def wrap_template(name, *, vision=False):
        original = getattr(engine_class, name, None)
        if not callable(original):
            raise RuntimeError(f'Unsupported oMLX VLM API: missing {name}')
        sig = inspect.signature(original)
        if not {'self', 'messages', 'chat_template_kwargs'}.issubset(sig.parameters):
            raise RuntimeError(f'Unsupported oMLX VLM signature: {name}{sig}')

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            if not targeted(self):
                return original(self, *args, **kwargs)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            options = policy.merge(bound.arguments.get('chat_template_kwargs'),
                                   engine_thinking=getattr(self, '_enable_thinking', None))
            bound.arguments['chat_template_kwargs'] = options or None
            # oMLX's template helper may remove partial-message keys in place.
            # Isolate each request without altering its text, tools or history.
            bound.arguments['messages'] = copy.deepcopy(bound.arguments['messages'])
            result = original(*bound.args, **bound.kwargs)
            desired = options.get('enable_thinking')
            if desired is not None:
                state = (token_reasoning_state(self._tokenizer, result[0]) if vision
                         else prompt_reasoning_state(result))
                if state is not None and state != desired:
                    raise ValueError('Qwen4 template did not honour enable_thinking; '
                                     'check the local chat template and request kwargs. '
                                     'Refusing to silently discard the requested mode.')
            return result
        patches.append((engine_class, name, original, wrapped))

    wrap_template('_apply_chat_template')
    wrap_template('_prepare_vision_inputs', vision=True)

    original_detect = getattr(scheduler_class, '_detect_needs_think_prefix', None)
    if not callable(original_detect):
        raise RuntimeError('Unsupported oMLX scheduler: missing reasoning-prefix detection')

    @functools.wraps(original_detect)
    def detect(self, request):
        adapter = getattr(self, 'model', None)
        model = getattr(adapter, '_vlm_model', None)
        if getattr(model, '_jangh2_bridge_model', False):
            state = token_reasoning_state(self.tokenizer, getattr(request, 'prompt_token_ids', None))
            if state is not None:
                return state
        return original_detect(self, request)
    patches.append((scheduler_class, '_detect_needs_think_prefix', original_detect, detect))

    # Grammar compilation precedes engine template application. Resolve the same
    # defaults there, preventing off-mode prompts and thinking-on grammars diverging.
    original_compile = getattr(server, '_compile_grammar_for_request', None)
    if not callable(original_compile):
        raise RuntimeError('Unsupported oMLX server: missing grammar request integration')
    sig = inspect.signature(original_compile)
    if not {'engine', 'chat_template_kwargs'}.issubset(sig.parameters):
        raise RuntimeError('Unsupported oMLX grammar request signature')

    @functools.wraps(original_compile)
    def compile_grammar(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        engine = bound.arguments['engine']
        if targeted(engine):
            options = policy.merge(bound.arguments.get('chat_template_kwargs'),
                                   engine_thinking=getattr(engine, '_enable_thinking', None))
            bound.arguments['chat_template_kwargs'] = options or None
        return original_compile(*bound.args, **bound.kwargs)
    patches.append((server, '_compile_grammar_for_request', original_compile, compile_grammar))

    with _LOCK:
        if _ACTIVE:
            raise RuntimeError('JANGH2 chat compatibility is already installed')
        installed = []
        try:
            for owner, name, old, new in patches:
                setattr(owner, name, new)
                installed.append((owner, name, old, new))
        except BaseException:
            for owner, name, old, new in reversed(installed):
                if getattr(owner, name) is new:
                    setattr(owner, name, old)
            raise
        _ACTIVE = True
    log.info('JANGH2: chat mode default=%s, reasoning effort default=%s; '
             'explicit request/admin options take priority; no prose filtering',
             policy.thinking, policy.reasoning_effort or 'template')

    undone = False
    def undo():
        global _ACTIVE
        nonlocal undone
        with _LOCK:
            if undone:
                return
            for owner, name, old, new in reversed(patches):
                if getattr(owner, name) is new:
                    setattr(owner, name, old)
            undone = True
            _ACTIVE = False
    return undo
