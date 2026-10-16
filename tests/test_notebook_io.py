import contextlib
import io
import sys
import unittest

from rag_contest.notebook_io import stream_command


class NotebookOutputTests(unittest.TestCase):
    def test_stdout_stderr_and_exit_status(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = stream_command([sys.executable, "-u", "-c",
                                     "import sys; print('progress'); print('failure detail', file=sys.stderr); sys.exit(7)"])
        self.assertEqual(status, 7)
        self.assertIn("progress", output.getvalue())
        self.assertIn("failure detail", output.getvalue())

    def test_quiet_process_shows_wait_time(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = stream_command([sys.executable, "-c", "import time; time.sleep(0.2)"], heartbeat_seconds=.05)
        self.assertEqual(status, 0)
        self.assertIn("ยังรอโปรแกรมส่งผลลัพธ์", output.getvalue())


if __name__ == "__main__":
    unittest.main()
