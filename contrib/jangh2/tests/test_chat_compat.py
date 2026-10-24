import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from jangh2_bridge.chat_policy import ChatPolicy
from test_chat_policy import Tokenizer

TARGET='/models/Allosaurus'

class ChatCompatTests(unittest.TestCase):
    def setUp(self):
        class Engine:
            def __init__(self,name=TARGET):
                self._model_name=name;self._enable_thinking=None;self._tokenizer=Tokenizer()
            def _apply_chat_template(self,messages,tools=None,chat_template_kwargs=None):
                options=chat_template_kwargs or {};messages[0].pop('partial',None)
                self.options=options;self.tools=tools
                enabled=options.get('enable_thinking',True)
                return '<|im_start|>assistant\n<think>\n'+('' if enabled else '\n</think>\n\n')
            def _prepare_vision_inputs(self,messages,images,chat_template_kwargs=None,tools=None):
                s=self._apply_chat_template(messages,tools,chat_template_kwargs)
                return self._tokenizer.encode(s),images,{},'hash'
        class Scheduler:
            def __init__(self,marked=True):
                self.tokenizer=Tokenizer()
                self.model=SimpleNamespace(_vlm_model=SimpleNamespace(_jangh2_bridge_model=marked))
            def _detect_needs_think_prefix(self,request):
                # Narrow legacy check deliberately misses longer prefills.
                return '<think>' in self.tokenizer.decode(request.prompt_token_ids[-3:])
        def compile_grammar(engine,structured_outputs=None,response_format=None,
                            chat_template_kwargs=None,reasoning_parser=None):
            return chat_template_kwargs
        self.Engine=Engine;self.Scheduler=Scheduler
        self.server=ModuleType('omlx.server');self.server._compile_grammar_for_request=compile_grammar
        modules={'omlx.engine.vlm':SimpleNamespace(VLMBatchedEngine=Engine),
                 'omlx.scheduler':SimpleNamespace(Scheduler=Scheduler),'omlx.server':self.server}
        self.scope=patch.dict(sys.modules,modules);self.scope.start()
        sys.modules.pop('jangh2_bridge.chat_compat',None)
        self.module=importlib.import_module('jangh2_bridge.chat_compat');self.undo=None
    def tearDown(self):
        if self.undo:self.undo()
        self.scope.stop();sys.modules.pop('jangh2_bridge.chat_compat',None)
    def install(self,mode='auto'):
        self.undo=self.module.install(TARGET,policy=ChatPolicy(mode))
    def test_open_prefix_no_longer_misses_longer_prefill(self):
        scheduler=self.Scheduler();request=SimpleNamespace(prompt_token_ids=scheduler.tokenizer.encode(
            '<|im_start|>assistant\n<think>\nWe need respond to user'))
        self.assertFalse(scheduler._detect_needs_think_prefix(request))
        self.install()
        self.assertTrue(scheduler._detect_needs_think_prefix(request))
        self.undo();self.undo=None
        self.assertFalse(scheduler._detect_needs_think_prefix(request))
    def test_off_applies_without_mutating_messages_or_options(self):
        self.install('off');engine=self.Engine();messages=[{'role':'user','content':'hello','partial':True}]
        options={'preserve_thinking':True}
        text=engine._apply_chat_template(messages,tools=[{'name':'test'}],chat_template_kwargs=options)
        self.assertIn('</think>',text);self.assertTrue(messages[0]['partial'])
        self.assertEqual(options,{'preserve_thinking':True});self.assertEqual(engine.tools,[{'name':'test'}])
    def test_explicit_on_not_silently_disabled(self):
        self.install('off');engine=self.Engine()
        result=engine._apply_chat_template([{'content':'hi'}],chat_template_kwargs={'enable_thinking':True})
        self.assertNotIn('</think>',result)
    def test_default_off_applies_to_vision_and_grammar(self):
        self.install('off');engine=self.Engine();messages=[{'content':'hi'}]
        ids,*_=engine._prepare_vision_inputs(messages,['image'])
        self.assertIn('</think>',engine._tokenizer.decode(ids))
        self.assertFalse(self.server._compile_grammar_for_request(engine)['enable_thinking'])
    def test_unrelated_model_and_scheduler_unchanged(self):
        self.install('off');engine=self.Engine('/models/other')
        self.assertNotIn('</think>',engine._apply_chat_template([{'content':'hi'}]))
        self.assertIsNone(self.server._compile_grammar_for_request(engine))
        s=self.Scheduler(False);req=SimpleNamespace(prompt_token_ids=s.tokenizer.encode('<|im_start|>assistant\n<think>\nhello'))
        self.assertFalse(s._detect_needs_think_prefix(req))
    def test_invalid_kwargs_fail_before_template_runs(self):
        self.install('off')
        with self.assertRaisesRegex(ValueError,'boolean'):
            self.Engine()._apply_chat_template([{'content':'hi'}],chat_template_kwargs={'enable_thinking':'false'})
    def test_install_is_atomic_for_unsupported_signature(self):
        original=self.Engine._apply_chat_template
        self.server._compile_grammar_for_request=lambda engine:None
        with self.assertRaises(RuntimeError):self.install()
        self.assertIs(self.Engine._apply_chat_template,original)
    def test_undo_idempotent_and_does_not_clobber_later_hooks(self):
        original=self.Engine._apply_chat_template;self.install()
        subsequent=lambda *a,**k:'new hook'
        self.Engine._apply_chat_template=subsequent
        self.undo();self.undo()
        self.assertIs(self.Engine._apply_chat_template,subsequent)
    def test_cannot_double_install(self):
        self.install()
        with self.assertRaises(RuntimeError):self.module.install(TARGET)
    def test_mode_ignoring_template_is_rejected(self):
        self.Engine._apply_chat_template=lambda self,messages,tools=None,chat_template_kwargs=None:'<|im_start|>assistant\n<think>\n'
        self.install('off')
        with self.assertRaisesRegex(ValueError,'honour'):
            self.Engine()._apply_chat_template([{'content':'hi'}])

if __name__=='__main__':unittest.main()
