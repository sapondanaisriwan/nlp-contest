import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_contest.data import file_hash
from rag_contest.runtime import stage_assets, cpu_report
from rag_contest.pipeline import Config, timeout_report


class RuntimeTests(unittest.TestCase):
    def test_name_request_is_not_yesno(self):
        from rag_contest.data import classify
        self.assertEqual(classify('What is the name of the app for booking rides?'),'short')
        self.assertEqual(classify('Is the name of the app LiftGo?'),'yesno')
    def test_staging_repairs_changed_local_file_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'source', Path(temp)/'local'
            source.mkdir()
            (source/'weights').write_bytes(b'original')
            manifest={'hashes':{'weights':file_hash(source/'weights')}}
            with patch('rag_contest.runtime.verify_assets', return_value=manifest):
                stage_assets(source,target)
                (target/'weights').write_bytes(b'corrupt')
                stage_assets(source,target)
            self.assertEqual((target/'weights').read_bytes(),b'original')
            self.assertEqual((source/'weights').read_bytes(),b'original')

    def test_timeout_reports_missing_without_caching_fake_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            saved={'traces':[{'question_id':'1','answer_text':'yes','document_id':'7','status':'ok','calls':[]}]}
            checkpoint=out/'checkpoint.json'
            checkpoint.write_text(json.dumps(saved))
            before=checkpoint.read_bytes()
            result=timeout_report([{'question_id':'1','question_text':'a'},
                                   {'question_id':'2','question_text':'b'}],out,Config(),{})
            self.assertEqual(result['unresolved_ids'],['2'])
            self.assertFalse(result['complete'])
            self.assertFalse((out/'answer.csv').exists())
            self.assertEqual(checkpoint.read_bytes(),before)
            self.assertEqual(len(json.loads((out/'diagnostics.json').read_text())),2)

    def test_cpu_metadata(self):
        self.assertEqual(cpu_report(2)['configured_threads'],2)
