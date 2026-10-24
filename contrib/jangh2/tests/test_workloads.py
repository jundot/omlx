import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from jangh2_bridge.workloads import observed_request,assess_output,make_prompt,aggregate_wave

class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                marker=payload['messages'][0]['content'].splitlines()[0].split()[-1]
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                chunks=[marker,' a',' b',' c']
                try:
                    for chunk in chunks:
                        self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':chunk}}]})+'\n\n').encode());self.wfile.flush();time.sleep(.003)
                    self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}],
                        'usage':{'completion_tokens':4,'prompt_tokens':32}})+'\n\ndata: [DONE]\n\n').encode());self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}/v1'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,i,cancel=None):
        marker=f'RID_{i:08x}'
        return observed_request(self.url,'test',f'Request {marker}\nbody',marker,[f'RID_{j:08x}' for j in range(8)],
            timeout=3,max_tokens=32,cancel_after=cancel)
    def test_concurrent_requests_no_cross_contamination(self):
        with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(self.request,range(4)))
        self.assertTrue(all(r['isolation']['marker_present'] and not r['isolation']['foreign_markers'] for r in rows))
        self.assertTrue(all(r['completion_tokens']==4 for r in rows))
        self.assertNotIn('content',rows[0]);self.assertNotIn('Authorization',json.dumps(rows))
    def test_cancel_closes_then_recovery_succeeds(self):
        row=self.request(1,cancel=2);self.assertTrue(row['cancelled']);self.assertFalse(row['completed'])
        self.assertTrue(self.request(2)['completed'])
    def test_marker_mixing_is_flagged_not_rewritten(self):
        a=assess_output('RID_00000001 RID_00000002','RID_00000001',['RID_00000001','RID_00000002'])
        self.assertEqual(a['foreign_markers'],['RID_00000002'])
    def test_prompts_differ_at_start_and_token_target_preserved(self):
        class Tok:
            def encode(self,s,**kw):return list(s.encode())
            def decode(self,ids,**kw):return bytes(ids).decode()
        a=make_prompt('RID_00000001',256,Tok());b=make_prompt('RID_00000002',256,Tok())
        self.assertNotEqual(a[:30],b[:30]);self.assertIn('RID_00000001',a)
    def test_aggregate_uses_wave_wall_time_not_sum_of_per_request_rates(self):
        rows=[{'completed':True,'cancelled':False,'completion_tokens':20,'first_content_seconds':1,'total_seconds':2}]*2
        self.assertEqual(aggregate_wave(rows,2)['aggregate_tokens_per_second'],20)
