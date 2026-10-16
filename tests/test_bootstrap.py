import subprocess
import unittest
from unittest.mock import patch
from rag_contest.bootstrap import ensure_environment, ensure_uv


class BootstrapTests(unittest.TestCase):
    @patch('rag_contest.bootstrap.Path.exists', return_value=True)
    @patch('rag_contest.bootstrap.subprocess.run')
    def test_existing_environment_no_pip(self, run, exists):
        python = ensure_environment('existing', uv='uv-test')
        run.assert_called_once_with([python, '--version'], check=True)

    @patch('rag_contest.bootstrap.Path.exists', return_value=False)
    @patch('rag_contest.bootstrap.subprocess.run')
    def test_new_environment_uses_uv_and_host_python(self, run, exists):
        ensure_environment('new', host_python='host-python', uv='uv-test')
        cmd = run.call_args_list[0].args[0]
        self.assertEqual(cmd[:4], ['uv-test', 'venv', '--python', 'host-python'])
        self.assertIn('--no-python-downloads', cmd)
        self.assertNotIn('--seed', cmd)

    @patch('rag_contest.bootstrap.Path.exists', return_value=False)
    @patch('rag_contest.bootstrap.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'uv'))
    def test_creation_failure_stops(self, run, exists):
        with self.assertRaises(subprocess.CalledProcessError):
            ensure_environment('new', uv='uv-test')
        self.assertEqual(run.call_count, 1)

    @patch('rag_contest.bootstrap.Path.is_file', return_value=False)
    @patch('rag_contest.bootstrap.shutil.which', return_value='existing-uv')
    @patch('rag_contest.bootstrap.subprocess.run')
    @patch('rag_contest.bootstrap.urlopen')
    def test_reuse_uv_without_download(self, download, run, which, is_file):
        self.assertEqual(ensure_uv('uv-bin'), 'existing-uv')
        download.assert_not_called()
