import json
import unittest
from jangh2_bridge.api_benchmark import consume_events

class AnswerMetricTests(unittest.TestCase):
    def test_first_answer_is_separate_from_reasoning(self):
        events=[{'choices':[{'delta':{'reasoning_content':'thinking'}}]},
                {'choices':[{'delta':{'content':'answer'}}]},
                {'choices':[{'delta':{},'finish_reason':'length'}]}]
        data=''.join('data: '+json.dumps(v)+'\n\n' for v in events)+'data: [DONE]\n\n'
        times=iter([1.,4.,5.,6.,7.])
        result=consume_events(data.splitlines(keepends=True),started=0,clock=lambda:next(times))
        self.assertEqual(result['first_reasoning_seconds'],1.)
        self.assertEqual(result['first_content_seconds'],4.)
        self.assertEqual(result['first_output_seconds'],1.)
        self.assertEqual(result['finish_reasons'],['length'])
        self.assertTrue(result['truncated'])
    def test_thinking_only_reports_absence_of_answer(self):
        data='data: {"choices":[{"delta":{"reasoning_content":"x"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
        result=consume_events(data.splitlines(keepends=True),started=0,clock=lambda:1.)
        self.assertIsNone(result['first_content_seconds'])
        self.assertTrue(result['truncated'])
    def test_content_only_is_not_truncated(self):
        data='data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        result=consume_events(data.splitlines(keepends=True),started=0,clock=lambda:2.)
        self.assertEqual(result['first_content_seconds'],2.)
        self.assertIsNone(result['first_reasoning_seconds'])
        self.assertFalse(result['truncated'])

if __name__=='__main__':unittest.main()
