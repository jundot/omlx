import json
import unittest
from jangh2_bridge.api_benchmark import consume_events


def event(value):
    return 'data: '+json.dumps(value)+'\n\n'


class StreamMeasurementTests(unittest.TestCase):
    def test_usage_drives_tokens_not_stream_chunks(self):
        rows=[
            ': ping\n','\n',
            event({'choices':[{'delta':{'role':'assistant'}}]}),
            event({'choices':[{'delta':{'reasoning_content':'think'}}]}),
            event({'choices':[{'delta':{'content':'many tokens in one chunk'}}]}),
            event({'choices':[], 'usage':{'completion_tokens':19,'prompt_tokens':7}}),
            'data: [DONE]\n\n',
        ]
        times=iter([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
        result=consume_events((line.encode() for text in rows for line in text.splitlines(keepends=True)),
                              started=0,clock=lambda: next(times))
        self.assertEqual(result['completion_tokens'],19)
        self.assertEqual(result['nonempty_output_chunks'],2)
        self.assertTrue(result['completed'])
        self.assertNotIn('api_key',result)
        self.assertNotIn('generated_text',result)
        self.assertGreater(result['end_to_end_tokens_per_second'],0)

    def test_no_usage_does_not_invent_a_token_count(self):
        data=event({'choices':[{'delta':{'content':'hello world'}}]})+'data: [DONE]\n\n'
        result=consume_events(iter(data.encode().splitlines(keepends=True)),started=0,clock=lambda:1.)
        self.assertIsNone(result['completion_tokens'])
        self.assertIsNone(result['end_to_end_tokens_per_second'])

    def test_error_event_and_incomplete_stream_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'broken'):
            consume_events(iter(event({'error':{'message':'broken'}}).encode().splitlines(keepends=True)),started=0)
        with self.assertRaisesRegex(RuntimeError,'before.*DONE'):
            consume_events(iter(event({'choices':[]}).encode().splitlines(keepends=True)),started=0)

    def test_multiline_data_and_unicode_text(self):
        rows=[b'data: {"choices":\n', b'data: [{"delta":{"content":"hello \\u2603"}}]}\n',b'\n',b'data: [DONE]\n',b'\n']
        result=consume_events(iter(rows),started=0,clock=lambda:1.)
        self.assertEqual(result['content_characters'],7)
        self.assertEqual(result['nonempty_output_chunks'],1)

    def test_malformed_json_and_no_output_fail(self):
        for data in (b'data: rubbish\n\n',b'data: [DONE]\n\n'):
            with self.assertRaises((RuntimeError,ValueError)):
                consume_events(iter(data.splitlines(keepends=True)),started=0)
