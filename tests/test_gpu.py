import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from rag_contest.pipeline import Config
from rag_contest.models import Generator
from rag_contest.gpu import require_cuda


class GpuTests(unittest.TestCase):
    def test_cpu_remains_default(self):
        self.assertEqual(Config().device, 'cpu')
        with self.assertRaises(ValueError): Config(device='auto')

    def test_cpu_build_rejected_for_cuda(self):
        with patch.dict('sys.modules', {'llama_cpp': SimpleNamespace(llama_supports_gpu_offload=lambda: False)}):
            with self.assertRaises(RuntimeError): require_cuda()

    def test_device_controls_offload_without_changing_model(self):
        llama = Mock()
        tokenizer = Mock()
        modules = {'llama_cpp': SimpleNamespace(Llama=llama),
                   'transformers': SimpleNamespace(AutoTokenizer=tokenizer)}
        spec = dict(spec=dict(file='same.gguf', tokenizer='Qwen/Qwen3-1.7B'))
        with patch.dict('sys.modules', modules), patch('rag_contest.gpu.require_cuda') as check:
            Generator('assets', spec, device='cpu')
            cpu = llama.call_args.kwargs
            check.assert_not_called()
            Generator('assets', spec, device='cuda')
            gpu = llama.call_args.kwargs
            check.assert_called_once()
        self.assertEqual(cpu['model_path'], gpu['model_path'])
        self.assertEqual(cpu['n_gpu_layers'], 0)
        self.assertEqual(gpu['n_gpu_layers'], -1)

    def test_no_nvidia_device_rejected(self):
        with patch.dict('sys.modules', {'llama_cpp': SimpleNamespace(llama_supports_gpu_offload=lambda: True)}), patch(
                'rag_contest.gpu.subprocess.run', return_value=SimpleNamespace(stdout='')):
            with self.assertRaises(RuntimeError): require_cuda()
