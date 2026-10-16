"""Explicit development-only CUDA checks; no silent CPU fallback."""
import subprocess


def require_cuda():
    import llama_cpp
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError('llama-cpp-python lacks GPU offload. Rebuild with GGML_CUDA=ON.')
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=20)
    if not result.stdout.strip():
        raise RuntimeError('No NVIDIA GPU found. Select GPU in Colab runtime settings.')
    print('GPU development only:', result.stdout.strip(), flush=True)
    return result.stdout.strip()
