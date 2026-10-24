import json
import unittest
from jangh2_bridge.output_audit import collect_stream, inspect_reply

class OutputAuditTests(unittest.TestCase):
    def test_channels_are_kept_separate(self):
        chunks=[{'choices':[{'delta':{'reasoning_content':'We need respond to user'}}]},
                {'choices':[{'delta':{'content':'hello'},'finish_reason':'stop'}]}]
        data=''.join('data: '+json.dumps(v)+'\n\n' for v in chunks)+'data: [DONE]\n\n'
        r=collect_stream(data.splitlines(keepends=True))
        self.assertEqual(r['content'],'hello')
        self.assertEqual(r['reasoning'],'We need respond to user')
        result=inspect_reply(r,expected='hello',thinking=True)
        self.assertTrue(result['passed']);self.assertFalse(result['self_talk_in_answer'])
    def test_self_talk_is_reported_not_deleted(self):
        r={'content':'We need respond to user. hello','reasoning':'','finish_reason':'stop'}
        result=inspect_reply(r,expected='hello')
        self.assertFalse(result['passed']);self.assertTrue(result['self_talk_in_answer'])
        self.assertEqual(r['content'],'We need respond to user. hello')
    def test_json_not_silently_repaired(self):
        base={'reasoning':'','finish_reason':'stop'}
        self.assertTrue(inspect_reply({**base,'content':'{"ready":true,"count":3}'},expected={'ready':True,'count':3})['passed'])
        self.assertFalse(inspect_reply({**base,'content':'```json\n{"ready":true,"count":3}\n```'},expected={'ready':True,'count':3})['passed'])
    def test_truncated_answer_not_success(self):
        self.assertFalse(inspect_reply({'content':'hello','reasoning':'','finish_reason':'length'},expected='hello')['passed'])
    def test_rejects_incomplete_stream_and_server_error(self):
        for data in ('data: {"choices":[]}\n\n','data: {"error":{"message":"failed"}}\n\n'):
            with self.assertRaises(RuntimeError):collect_stream(data.splitlines(keepends=True))
    def test_bounded_response(self):
        data='data: {"choices":[{"delta":{"content":"123456789"}}]}\n\ndata: [DONE]\n\n'
        with self.assertRaises(RuntimeError):collect_stream(data.splitlines(keepends=True),max_characters=4)
    def test_off_mode_reports_unexpected_reasoning(self):
        r={'content':'hello','reasoning':'thoughts','finish_reason':'stop'}
        self.assertFalse(inspect_reply(r,expected='hello',thinking=False)['passed'])
    def test_unicode(self):
        r={'content':'bonjour','reasoning':'','finish_reason':'stop'}
        self.assertTrue(inspect_reply(r,expected='bonjour')['passed'])

if __name__=='__main__':unittest.main()
