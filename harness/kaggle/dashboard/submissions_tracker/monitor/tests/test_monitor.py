import copy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from collector import Collector, credential_values, merge_metadata, redact
from model import Catalog, DASH, iso, lifecycle, normalize, read_json, score
from server import handler_for
from probe import finite_json


class Lifecycle(unittest.TestCase):
    def test_running_needs_process_evidence(self):
        self.assertEqual(lifecycle({'status': {'server_ok': True}}, 1000, 1000, True)[0], 'stopped')

    def test_offline_does_not_invent_a_death_or_keep_running(self):
        m = {'runner_alive': True, 'solver_alive': True, 'server_alive': True}
        self.assertEqual(lifecycle(m, 1000, 1000, False)[0], 'unverified')
        self.assertEqual(lifecycle(m, 1000, 500, True)[0], 'unverified')

    def test_retained_terminal_result_survives_pod_deletion(self):
        self.assertEqual(lifecycle({'timeline': 'NOTEBOOK_DONE ok'}, 1000, None, False)[0], 'completed')
        self.assertEqual(lifecycle({'timeline': 'NOTEBOOK_DONE error'}, 1000, None, False)[0], 'failed')

    def test_recovering_and_blocked(self):
        m = {'runner_alive': True, 'supervisor': {'state': 'restarting', 'start': 1}}
        self.assertEqual(lifecycle(m, 1000, 1000, True)[0], 'recovering')
        m['supervisor']['state'] = 'stopped'
        self.assertEqual(lifecycle(m, 1000, 1000, True)[0], 'blocked')

    def test_midnight_is_an_epoch_not_a_guessed_date(self):
        self.assertEqual(iso(1793404800), '2026-10-31T00:00:00Z')


class Scoring(unittest.TestCase):
    def test_task_weighting_and_unanswered_denominator(self):
        p = {'tasks': 2, 'solutions': {'single': [[[1]]], 'multi': [[[2]], [[3]], [[4]]]}}
        rs = [{'task_id': 'single', 'pair': 0, 'attempt': 0, 'pred': [[1]], 't_end_h': .1}]
        s, curve = score(rs, p)
        self.assertEqual(s['score_pass2'], 50)
        self.assertEqual(curve[-1]['score'], 50)

    def test_multitest_records_and_completed_missing_grid(self):
        p = {'tasks': 1, 'solutions': {'x': [[[1]], [[2]]]}}
        rs = [dict(task_id='x', pair=0, attempt=0, multi_test=2, pred=None, t_end_h=.1),
              dict(task_id='x', pair=0, attempt=1, multi_test=2, pred=[[[1]], [[0]]], t_end_h=.2)]
        s, _ = score(rs, p)
        self.assertEqual(s['score_pass2'], 50)
        self.assertEqual(s['both_finished'], 1)
        self.assertEqual(s['completed_attempts'], 2)

    def test_two_ranked_answers_not_oracle_over_three(self):
        p = {'tasks': 1, 'solutions': {'x': [[[2]]]}}
        rs = [dict(task_id='x', attempt=a, pred=[[a]], train_pass=False, t_end_h=.1+a*.1) for a in range(3)]
        self.assertEqual(score(rs, p)[0]['score_pass2'], 0)
        rs[2]['train_pass'] = True
        self.assertEqual(score(rs, p)[0]['score_pass2'], 100)

    def test_catalog_rejects_same_id_with_different_grids(self):
        c = Catalog(); key = next(iter(c.workloads['eval120_kaggle']))
        ch = {key: copy.deepcopy(c.workloads['eval120_kaggle'][key])}
        self.assertEqual(c.identify(ch)['tasks'], 1)
        ch[key]['test'][0]['input'][0][0] = (ch[key]['test'][0]['input'][0][0]+1)%10
        self.assertIsNone(c.identify(ch))

    def test_unknown_workload_hides_score(self):
        c = Catalog()
        meta = {'name': 'unknown', 'attempts': [{'task_id': 'x', 'attempt': 0, 'pred': [[1]]}]}
        r = normalize(meta, {'id': 'p', 'observed_at': 1000}, c, 1000)
        self.assertIsNone(r['tasks']); self.assertEqual(r['summary'], {})

    def test_conditional_retry_counts_include_actual_extra_work(self):
        c = Catalog(); key = next(iter(c.workloads['eval120_kaggle']))
        ch = {key: c.workloads['eval120_kaggle'][key]}
        rs = [dict(task_id=key, attempt=a, pred=c.solutions[key], multi_test=len(c.solutions[key]), t_end_h=.1)
              for a in range(3)]
        meta = {'name': 'retry', 'challenges': ch, 'attempts': rs, 'runner_alive': True,
                'solver_alive': True, 'server_alive': True, 'telemetry_at': 1000,
                'env': {'ARC_EXTRA_IF_NO_TRAIN_PASS': '1'}, 'status': {'done': 3, 'queued': 0, 'running': 0}}
        r = normalize(meta, {'id': 'p', 'observed_at': 1000}, c, 1000)
        self.assertEqual(r['summary']['completed_attempts'], 3)
        self.assertEqual(r['summary']['expected_attempts'], 3)

    def test_live_holdout_accuracy_is_not_in_api(self):
        c = Catalog(); key = next(iter(c.workloads['eval120_kaggle']))
        meta = {'name': 'q_e073_holdout', 'challenges': {key: c.workloads['eval120_kaggle'][key]},
                'attempts': [dict(task_id=key, attempt=0, pred=c.solutions[key], multi_test=len(c.solutions[key]), t_end_h=.1)],
                'runner_alive': True, 'solver_alive': True, 'server_alive': True, 'telemetry_at': 1000}
        r = normalize(meta, {'id': 'p', 'observed_at': 1000}, c, 1000)
        self.assertTrue(r['holdout_locked']); self.assertEqual(r['series']['score'], [])
        for k in ('score_pass1', 'score_pass2', 'attempt0_acc', 'attempt1_acc'):
            self.assertNotIn(k, r['summary'])


class Retention(unittest.TestCase):
    def test_disconnection_retains_history_and_terminal_markers(self):
        local = {'attempts': [1, 2], 'end_reason': 'done', 'timeline': 'NOTEBOOK_DONE ok',
                 'env': {'ARC_EXTRA_IF_NO_TRAIN_PASS': '1'}}
        remote = {'attempts': [1], 'end_reason': '', 'timeline': 'solver started'}
        out = merge_metadata(local, remote)
        self.assertEqual(out['attempts'], [1, 2]); self.assertEqual(out['end_reason'], 'done')
        self.assertEqual(out['timeline'], 'NOTEBOOK_DONE ok')
        self.assertEqual(out['env']['ARC_EXTRA_IF_NO_TRAIN_PASS'], '1')

    def test_redacts_credentials_in_nested_logs(self):
        self.assertEqual(redact({'x': ['Bearer secret-example-123']}, ['secret-example-123']),
                         {'x': ['Bearer [redacted]']})
        self.assertEqual(credential_values([{'name': 'server', 'token': 'endpoint-secret-123'}]), ['endpoint-secret-123'])
        self.assertEqual(finite_json({'metric': float('nan'), 'values': [float('inf'), 0]}),
                         {'metric': None, 'values': [None, 0]})

    def test_all_llm_comments_survive_collection_unchanged(self):
        before = {f: (DASH/f).read_bytes() for f in ['events.json', 'findings.json']}
        with tempfile.TemporaryDirectory() as d:
            s = Collector(d, remote=False).collect()
        for f in before:
            self.assertEqual((DASH/f).read_bytes(), before[f])
        self.assertEqual(s['notes']['events'], json.loads(before['events.json']))
        self.assertEqual(s['notes']['findings'], json.loads(before['findings.json']))

    def test_failed_fetch_retains_score_and_marks_observation_unverified(self):
        with tempfile.TemporaryDirectory() as d:
            c = Collector(d); key = next(iter(c.catalog.workloads['eval120_kaggle']))
            ch = {key: c.catalog.workloads['eval120_kaggle'][key]}
            p = Path(d)/'pods'; p.mkdir()
            old = {'observed_at': time.time()-600, 'runs': [{'name': 'test', 'runner_alive': True,
                   'solver_alive': True, 'server_alive': True, 'challenges': ch,
                   'attempts': [{'task_id': key, 'attempt': 0, 'pred': c.catalog.solutions[key][0], 't_end_h': .1}]}]}
            # Use a multi-test wrapper when the chosen example has more than one test input.
            old['runs'][0]['attempts'][0].update(pred=c.catalog.solutions[key], multi_test=len(c.catalog.solutions[key]))
            (p/'fake.json').write_text(json.dumps(old))
            with patch('collector.discover', return_value=[{'id': 'fake', 'name': 'fake'}]), \
                 patch('collector.fetch', side_effect=RuntimeError('offline')), patch('collector.local_sources', return_value=[]):
                s = c.collect()
            r = next(x for x in s['runs'] if x.get('run_name') == 'test')
            self.assertEqual(r['state'], 'unverified'); self.assertEqual(r['summary']['score_pass2'], 100)
            self.assertIsNone(r['gen_tok_s']); self.assertFalse(s['pods'][0]['connected'])

    def test_collection_failure_keeps_last_atomic_generation(self):
        with tempfile.TemporaryDirectory() as d:
            c = Collector(d); c.snapshot = {'generated_at': 'last-good'}
            def fail():
                c.stop.set(); raise ValueError('bad source')
            c.request.set()
            with patch.object(c, 'collect', side_effect=fail):
                c.loop()
            self.assertEqual(c.get()['generated_at'], 'last-good')
            self.assertIn('previous snapshot retained', c.last_error)


class HTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Fake:
            request = threading.Event()
            def get(self):
                return {'generated_at': 'test-generation', 'runs': []}
            def health(self):
                return {'collector_alive': True}
        cls.collector = Fake()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(cls.collector))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.url = 'http://127.0.0.1:'+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_atomic_snapshot_etag(self):
        with urllib.request.urlopen(self.url+'/api/snapshot') as r:
            self.assertEqual(json.load(r)['generated_at'], 'test-generation'); etag = r.headers['ETag']
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(self.url+'/api/snapshot', headers={'If-None-Match': etag}))
        self.assertEqual(e.exception.code, 304)

    def test_no_workspace_file_serving(self):
        for path in ['/SECRETS.yaml', '/../../SECRETS.yaml', '/monitor/state/snapshot.json']:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(self.url+path)
            self.assertEqual(e.exception.code, 404)

    def test_refresh_and_origin(self):
        with urllib.request.urlopen(urllib.request.Request(self.url+'/api/refresh', data=b'', method='POST')) as r:
            self.assertEqual(r.status, 202)
        self.assertTrue(self.collector.request.is_set())
        req = urllib.request.Request(self.url+'/api/refresh', data=b'', method='POST', headers={'Origin': 'http://elsewhere.invalid'})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        self.assertEqual(e.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
