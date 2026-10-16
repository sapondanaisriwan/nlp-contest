"""Explicit online preparation; inference only opens verified local files."""
import importlib.metadata
import json
import os
import platform
import socket
import time
from contextlib import contextmanager
from pathlib import Path

from .data import file_hash, write_json

MODELS = {
    "qwen3-1.7b-q8": {"repo": "Qwen/Qwen3-1.7B-GGUF", "file": "Qwen3-1.7B-Q8_0.gguf", "tokenizer": "Qwen/Qwen3-1.7B", "parameters_b": 1.7},
    "qwen2.5-1.5b-q8": {"repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "file": "qwen2.5-1.5b-instruct-q8_0.gguf", "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct", "parameters_b": 1.54},
    "qwen2.5-1.5b-q4": {"repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct", "parameters_b": 1.54},
    "qwen3-1.7b-hf": {"repo": "Qwen/Qwen3-1.7B", "tokenizer": "Qwen/Qwen3-1.7B", "parameters_b": 1.7},
    "qwen2.5-1.5b-hf": {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct", "parameters_b": 1.54},
    "baseline-0.5b-hf": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct", "parameters_b": .49},
}
EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


def environment():
    packages = {}
    for name in ("numpy", "pypdf", "psutil", "torch", "transformers", "sentence-transformers", "llama-cpp-python", "huggingface-hub", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Include transitive runtime dependencies, not unrelated preinstalled Colab apps.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pending = [name for name, version in packages.items() if version is not None]
    resolved = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in resolved:
            continue
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        resolved[name] = dist.version
        for text in dist.requires or []:
            requirement = Requirement(text)
            if not requirement.marker or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages, "resolved_dependencies": dict(sorted(resolved.items()))}


def asset_hashes(folder):
    return {str(p.relative_to(folder)).replace("\\", "/"): file_hash(p)
            for p in sorted(Path(folder).rglob("*")) if p.is_file() and ".cache" not in p.parts
            and p.name != "assets.json"}


def prepare(model_name, folder):
    from huggingface_hub import HfApi, snapshot_download
    start = time.perf_counter()
    folder = Path(folder)
    spec = MODELS[model_name]
    if (folder / "assets.json").exists():
        manifest = verify_assets(folder)
        if manifest["model_name"] != model_name:
            raise ValueError("Use a different asset folder for each candidate")
        return manifest
    api = HfApi()
    revisions = {}
    def fetch(repo, destination, patterns):
        revision = api.model_info(repo).sha
        revisions[repo] = revision
        snapshot_download(repo, revision=revision, local_dir=str(folder / destination), allow_patterns=patterns)
    tokenizer_files = ["*.json", "*.jinja", "*.txt", "*.model"]
    fetch(spec["tokenizer"], "tokenizer", tokenizer_files)
    fetch(spec["repo"], "generator", [spec["file"]] if "file" in spec else ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
    fetch(EMBEDDING, "embedding", ["*.json", "*.safetensors", "*.txt", "*.model"])
    manifest = {"model_name": model_name, "spec": spec, "revisions": revisions,
                "hashes": asset_hashes(folder), "prepare_seconds": time.perf_counter() - start}
    write_json(folder / "assets.json", manifest)
    return manifest


def verify_assets(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "assets.json").read_text(encoding="utf-8"))
    if manifest["model_name"] not in MODELS or manifest["spec"] != MODELS[manifest["model_name"]]:
        raise ValueError("Unknown or changed model specification")
    if asset_hashes(folder) != manifest["hashes"]:
        raise ValueError("Model assets changed or incomplete; use a fresh asset folder")
    return manifest


@contextmanager
def offline():
    """Defense in depth for Python clients; not an OS network sandbox."""
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    old_env = {k: os.environ.get(k) for k in keys}
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    def denied(*args, **kwargs):
        raise RuntimeError("Network disabled during inference; run prepare separately")
    try:
        for key in keys:
            os.environ[key] = "1"
        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        socket.create_connection = denied
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class Generator:
    def __init__(self, folder, manifest, threads=2, context=5120, seed=20261009, device="cpu"):
        if device not in ('cpu', 'cuda'):
            raise ValueError('Unknown inference device')
        from transformers import AutoTokenizer
        folder = Path(folder)
        spec = manifest["spec"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"), local_files_only=True, trust_remote_code=False)
        self.context, self.seed = context, seed
        self.gguf = "file" in spec
        if device == 'cuda' and not self.gguf:
            raise ValueError('GPU development requires GGUF to preserve the same quantized model')
        self.qwen3 = "Qwen3" in spec["tokenizer"]
        if self.gguf:
            from llama_cpp import Llama
            if device == 'cuda':
                from .gpu import require_cuda
                require_cuda()
            self.model = Llama(model_path=str(folder / "generator" / spec["file"]), n_ctx=context,
                               n_threads=threads, n_threads_batch=threads, n_batch=256,
                               n_gpu_layers=-1 if device == 'cuda' else 0, seed=seed, verbose=device == 'cuda')
        else:
            import torch
            from transformers import AutoModelForCausalLM
            torch.set_num_threads(threads)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            self.model = AutoModelForCausalLM.from_pretrained(str(folder / "generator"), local_files_only=True,
                                                            torch_dtype=torch.float32, trust_remote_code=False).to("cpu").eval()
            if sum(p.numel() for p in self.model.parameters()) > 2_000_000_000:
                raise ValueError("Generator exceeds 2B parameters")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate(self, messages, max_tokens=64):
        started = time.perf_counter()
        extra = {"enable_thinking": False} if self.qwen3 else {}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        if self.gguf:
            tokens = self.model.tokenize(prompt.encode(), add_bos=False, special=True)
            if len(tokens) + max_tokens > self.context:
                raise ValueError("Prompt exceeds configured context budget")
            self.model.reset()
            prepared_at = time.perf_counter()
            result = self.model.create_completion(tokens, max_tokens=max_tokens, temperature=0,
                                                  seed=self.seed, repeat_penalty=1.0,
                                                  stop=["<|im_end|>", "<|endoftext|>"], echo=False)
            return {"text": result["choices"][0]["text"], "truncated": result["choices"][0]["finish_reason"] == "length",
                    "prompt_tokens": len(tokens), "completion_tokens": result.get('usage', {}).get('completion_tokens'),
                    "prompt_prepare_seconds": prepared_at - started,
                    "inference_seconds": time.perf_counter() - prepared_at}
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if inputs.input_ids.shape[1] + max_tokens > self.context:
            raise ValueError("Prompt exceeds configured context budget")
        prepared_at = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        new = output[0, inputs.input_ids.shape[1]:]
        return {"text": self.decode(new), "truncated": len(new) >= max_tokens and int(new[-1]) != self.tokenizer.eos_token_id,
                "prompt_tokens": inputs.input_ids.shape[1], "completion_tokens": len(new),
                "prompt_prepare_seconds": prepared_at - started,
                "inference_seconds": time.perf_counter() - prepared_at}
