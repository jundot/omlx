import unittest
from unittest.mock import patch
from jangh2_bridge.chat_policy import ChatPolicy, prompt_reasoning_state, token_reasoning_state

class Tokenizer:
    def encode(self, text, **kwargs): return list(text.encode())
    def decode(self, ids, **kwargs): return bytes(ids).decode(errors='replace')

class ChatPolicyTests(unittest.TestCase):
    def test_auto_does_not_change_defaults(self):
        self.assertEqual(ChatPolicy().merge(None), {})
    def test_off_is_default_not_override(self):
        self.assertEqual(ChatPolicy('off').merge(None), {'enable_thinking':False})
        self.assertEqual(ChatPolicy('off').merge({'enable_thinking':True}), {'enable_thinking':True})
    def test_engine_setting_precedes_environment_default(self):
        self.assertTrue(ChatPolicy('off').merge(None, engine_thinking=True)['enable_thinking'])
    def test_preserves_tool_and_history_options(self):
        options={'enable_thinking':True,'preserve_thinking':True,'custom':'x'}
        self.assertEqual(ChatPolicy('off').merge(options), options)
        self.assertIsNot(ChatPolicy('off').merge(options), options)
    def test_explicit_reasoning_effort_takes_priority(self):
        self.assertEqual(ChatPolicy('on','low').merge({'reasoning_effort':'xhigh'}),
                         {'enable_thinking':True,'reasoning_effort':'xhigh'})
    def test_none_options_are_unspecified(self):
        self.assertFalse(ChatPolicy('off').merge({'enable_thinking':None})['enable_thinking'])
    def test_invalid_values_refused(self):
        for kwargs in ({'thinking':'yes'}, {'reasoning_effort':'high'}):
            with self.assertRaises(ValueError): ChatPolicy(**kwargs)
        for value in ('false',0,1,[],{}):
            with self.subTest(value=value),self.assertRaises(ValueError):
                ChatPolicy().merge({'enable_thinking':value})
    def test_environment(self):
        with patch.dict('os.environ',{'OMLX_JANGH_THINKING':'off','OMLX_JANGH_REASONING_EFFORT':'medium'},clear=True):
            self.assertEqual(ChatPolicy.from_environment(),ChatPolicy('off','medium'))
    def test_open_think_prefix(self):
        self.assertTrue(prompt_reasoning_state('<|im_start|>assistant\n<think>\n'))
    def test_prefilled_reasoning_beyond_three_tokens(self):
        t=Tokenizer(); text='<|im_start|>assistant\n<think>\nLet us check this carefully.'
        self.assertTrue(token_reasoning_state(t,t.encode(text)))
    def test_closed_empty_reasoning_is_not_open(self):
        self.assertFalse(prompt_reasoning_state('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
    def test_earlier_history_and_user_literals_do_not_activate(self):
        self.assertFalse(prompt_reasoning_state('<|im_start|>user\nExplain <think>\n<|im_end|>\n<|im_start|>assistant\n'))
        self.assertFalse(prompt_reasoning_state('<|im_start|>assistant\n<think>old</think>old<|im_end|>\n<|im_start|>assistant\n'))
    def test_last_turn_must_be_assistant(self):
        self.assertFalse(prompt_reasoning_state('<|im_start|>assistant\n<think>old<|im_end|>\n<|im_start|>user\nnew'))
    def test_unknown_prompt_returns_unknown(self):
        self.assertIsNone(prompt_reasoning_state('unstructured prompt'))
    def test_closed_assistant_is_not_a_generation_prefix(self):
        self.assertFalse(prompt_reasoning_state('<|im_start|>assistant\n<think>x<|im_end|>'))
    def test_truncated_boundary_falls_back(self):
        t=Tokenizer()
        self.assertIsNone(token_reasoning_state(t,t.encode('<|im_start|>assistant\n<think>'+'a'*6000),max_tail_tokens=512))
    def test_missing_decode_is_unknown(self):
        self.assertIsNone(token_reasoning_state(object(),[1,2,3]))
    def test_no_tokens_is_false(self):
        self.assertFalse(token_reasoning_state(Tokenizer(),[]))

if __name__=='__main__': unittest.main()
