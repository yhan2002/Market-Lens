import io
import subprocess
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch
import market_data
import refresh_pipeline as pipeline
import refresh_request_agent as agent

class ReviewSafetyTest(unittest.TestCase):
    def test_probe_failure_cannot_claim_cached_freshness(self):
        with patch.object(market_data, '_download_direct_chart_rows', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, '최신 거래일 확인 불가'):
                market_data.probe_latest_completed_market_date(now=datetime(2026,9,30,12,tzinfo=timezone.utc), tickers=('NQ=F',), fallback_database=Mock())

    def test_agent_marks_probe_failure_failed_without_pipeline(self):
        with patch.object(agent,'is_refresh_busy',return_value=False), patch.object(agent,'load_config',return_value={}), patch.object(agent,'post_refresh',side_effect=[{'id':1},{}]) as post, patch.object(agent,'server_snapshot_as_of',return_value='2026-09-01'), patch.object(agent,'server_roro_as_of',return_value='2026-09-01'), patch.object(agent,'probe_latest_completed_market_date',side_effect=RuntimeError('최신 거래일 확인 불가')), patch.object(agent,'run_refresh_pipeline') as run, patch.object(agent,'_log_exception'):
            self.assertEqual(agent.main(),1)
            self.assertEqual(post.call_args.args[1]['status'],'FAILED')
            run.assert_not_called()

    def test_timeout_kills_only_owned_tree_and_reaps(self):
        child=Mock(pid=12345)
        child.wait.side_effect=[subprocess.TimeoutExpired('fixture',900),0]
        child.poll.return_value=0
        # Keep the host path class when simulating Windows on a POSIX test host.
        path_class = type(Path())
        with patch.object(pipeline.subprocess,'Popen',return_value=child), patch.object(pipeline.subprocess,'run',return_value=Mock(returncode=0)) as cleanup, patch.object(pipeline, 'Path', path_class), patch.object(pipeline.os,'name','nt'):
            with self.assertRaises(pipeline.RefreshFailed):pipeline._run('fixture',['never-run'],io.StringIO())
        self.assertEqual(child.wait.call_args_list[0].kwargs,{'timeout':900})
        self.assertEqual(cleanup.call_args.args[0][-4:],['/PID','12345','/T','/F'])
        self.assertEqual(child.wait.call_count,2)

    def test_normal_child_exit_is_preserved(self):
        with patch.object(pipeline.subprocess,'Popen',return_value=Mock(wait=Mock(return_value=7))):
            self.assertEqual(pipeline._run('fixture',['never-run'],io.StringIO()),7)
