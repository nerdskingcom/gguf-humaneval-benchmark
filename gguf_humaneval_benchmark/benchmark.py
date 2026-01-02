# benchmark.py
#
# HumanEval benchmark runner for llama.cpp (GGUF) via OpenAI-compatible HTTP API.
#
# Key fixes:
#   - Correct HumanEval execution semantics: prompt + generated_code + test + check(entry_point)
#   - Works with --server-url even when --no-server (creates an HTTP client)
#   - Adds per-task error_type/error_detail for auditing failures
#   - Fixes readiness probe bug (self.model_name was undefined)

import os
import sys
import argparse
import json
import time
import subprocess
import threading
import tempfile
import requests


# --- CONFIGURATION ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "eval_utils", "bench_config.json")

# Load benchmark config
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
        BENCH_CFG = CONFIG.get("benchmark", {}) or {}
        LLAMA_CPP_CFG = CONFIG.get("llama_cpp", {}) or {}
        MODELS_CFG = CONFIG.get("models", {}) or {}
else:
    BENCH_CFG = {
        "samples": 164,
        "timeout_seconds": 25,
        "max_new_tokens_normal": 4096,
        "max_new_tokens_thinking": 4096,
        "start_timeout_seconds": 25,
        "finish_timeout_seconds": 25,
        "request_retries": 1,
        "retry_multiplier": 2.0,
    }
    LLAMA_CPP_CFG = {
        "server_path": "/home/x4245/llama.cpp/build/bin",
        "ctx_size": 4096,
        "n_gpu_layers": 99,
        "port": 8080,
        "host": "127.0.0.1",
        "temp": 0.0,
        "mirostat": 2,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }
    MODELS_CFG = {
        "gguf_directory": "/home/x4245/gptoss_training/models",
        "default_model": "nerdsking-python-coder-7B-i_Q8_0.gguf",
    }

# Set default values from config
DEFAULT_MODELS_DIR = MODELS_CFG.get("gguf_directory", "/home/x4245/gptoss_training/models")
DEFAULT_MODEL = MODELS_CFG.get("default_model", "nerdsking-python-coder-7B-i_Q8_0.gguf")
DEFAULT_SERVER_PATH = LLAMA_CPP_CFG.get("server_path", "/home/x4245/llama.cpp/build/bin")


class _LlamaHttpClient:
    """
    Minimal client for llama.cpp OpenAI-compatible /v1/completions endpoint.

    Supports streaming and the same timeout policy used by LlamaCppServer.generate().
    """

    def __init__(self, server_url: str, model_name: str = ""):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name or ""

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0):
        """Return: (text, audit_text, error_str_or_None)"""
        start_timeout = int(BENCH_CFG.get("start_timeout_seconds", 25))
        finish_timeout = int(BENCH_CFG.get("finish_timeout_seconds", 25))

        retries = int(BENCH_CFG.get("request_retries", 1))
        retry_multiplier = float(BENCH_CFG.get("retry_multiplier", 2.0))

        stop_list = ["###", "User:", "System:", "<|endoftext|>", "<|im_end|>"]

        def _stream_once(st_to: int, fin_to: int):
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop_list,
                "stream": True,
            }

            t0 = time.time()
            t_first = None
            out_text = ""

            # Prevent requests from timing out before our own timers.
            read_timeout = max(1, int(st_to + fin_to))

            try:
                resp = requests.post(
                    f"{self.server_url}/v1/completions",
                    json=payload,
                    stream=True,
                    timeout=(5, read_timeout),
                )
            except Exception as e:
                err = str(e)
                return "", err, err

            try:
                # Immediate non-OK responses: capture body for audit
                if resp.status_code == 503:
                    body = ""
                    try:
                        body = resp.text or ""
                    except Exception:
                        pass
                    return "", (body or "503 Service Unavailable"), "503 Service Unavailable"

                if resp.status_code != 200:
                    body = ""
                    try:
                        body = resp.text or ""
                    except Exception:
                        pass
                    try:
                        resp.raise_for_status()
                    except Exception as e:
                        err = str(e)
                        return "", (body or err), err

                # SSE streaming: lines like "data: {...}" and possibly "data: [DONE]"
                for raw_line in resp.iter_lines(decode_unicode=True):
                    now = time.time()

                    if t_first is None and (now - t0) > st_to:
                        audit = out_text or "Start timeout exceeded without any text"
                        return "", audit, f"Start timeout ({st_to}s) exceeded"

                    if t_first is not None and (now - t_first) > fin_to:
                        audit = out_text or "Finish timeout exceeded"
                        return "", audit, f"Finish timeout ({fin_to}s) exceeded"

                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()

                    if line == "[DONE]":
                        break

                    # Parse JSON chunk; ignore non-JSON keepalives
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    chunk = ""
                    if isinstance(obj, dict):
                        choices = obj.get("choices")
                        if isinstance(choices, list) and choices:
                            first = choices[0]
                            if isinstance(first, dict):
                                chunk = first.get("text", "") or ""

                    if chunk:
                        if t_first is None:
                            t_first = now
                        out_text += chunk

                return out_text, out_text, None

            except requests.exceptions.ReadTimeout:
                audit = out_text or "Read timeout"
                return "", audit, "Read timeout"
            except Exception as e:
                err = str(e)
                audit = out_text or err
                return "", audit, err
            finally:
                try:
                    resp.close()
                except Exception:
                    pass

        attempt = 0
        cur_st, cur_fin = start_timeout, finish_timeout
        last_audit, last_err = "", None

        while True:
            text, audit, err = _stream_once(cur_st, cur_fin)
            if err is None:
                return text, audit, None

            last_audit, last_err = audit, err

            if attempt >= retries:
                return "", last_audit, last_err

            attempt += 1
            cur_st = int(max(1, cur_st * retry_multiplier))
            cur_fin = int(max(1, cur_fin * retry_multiplier))
            time.sleep(0.5)


class LlamaCppServer:
    def __init__(
        self,
        model_path,
        server_path,
        ctx_size=4096,
        n_gpu_layers=99,
        port=8080,
        host="127.0.0.1",
        temp=0.0,
        mirostat=2,
        cache_type_k="f16",
        cache_type_v="f16",
    ):
        self.model_path = model_path
        self.server_path = server_path
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.port = port
        self.host = host
        self.temp = temp
        self.mirostat = mirostat
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.server_process = None
        self.server_url = f"http://{host}:{port}"
        self.model_name = os.path.basename(model_path)  # FIX: used by readiness probe

        # HTTP client wrapper for generation once started
        self._client = _LlamaHttpClient(self.server_url, model_name=self.model_name)

    def start(self):
        """Start llama.cpp server"""
        server_exe = None

        # Check if server_path is a directory or file
        if os.path.isdir(self.server_path):
            possible_paths = [
                os.path.join(self.server_path, "server"),
                os.path.join(self.server_path, "llama-server"),
                os.path.join(self.server_path, "llama_server"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    server_exe = path
                    break
        else:
            if os.path.exists(self.server_path):
                server_exe = self.server_path

        if not server_exe:
            print(f"❌ Llama.cpp server not found. Searched in: {self.server_path}")
            print("Possible server executable names: server, llama-server, llama_server")
            print("Please specify the correct path in bench_config.json")
            print("\nTo build llama.cpp server:")
            print("  cd /home/x4245/llama.cpp")
            print("  mkdir -p build && cd build")
            print("  cmake .. -DLLAMA_CUBLAS=ON")
            print("  cmake --build . --config Release")
            print("\nThe server executable will be at: build/bin/server")
            return False

        print(f"✅ Found server executable: {server_exe}")

        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False

        cmd = [
            server_exe,
            "-m",
            self.model_path,
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.n_gpu_layers),
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--temp",
            str(self.temp),
            "--parallel",
            "1",
        ]

        if self.mirostat > 0:
            cmd.extend(["--mirostat", str(self.mirostat)])

        cmd.extend(["--cache-type-k", self.cache_type_k])
        cmd.extend(["--cache-type-v", self.cache_type_v])

        print("🚀 Starting llama.cpp server...")
        print(f"   Model: {os.path.basename(self.model_path)}")
        print(f"   Server: {server_exe}")
        print(f"   URL: {self.server_url}")
        print(f"   GPU Layers: {self.n_gpu_layers}")
        print(f"   Context: {self.ctx_size}")
        print(f"   Temperature: {self.temp}")
        print(f"   Mirostat: {self.mirostat}")

        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        def monitor_output():
            try:
                while True:
                    line = self.server_process.stderr.readline()
                    if not line:
                        break
                    if "HTTP server listening" in line:
                        print(f"✅ Server ready at {self.server_url}")
                        break
                    elif "error" in line.lower():
                        print(f"❌ Server error: {line.strip()}")
            except Exception:
                pass

        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()

        # Wait for server to be ready (health check + a short completions warmup probe)
        max_wait = 60
        for i in range(max_wait):
            try:
                response = requests.get(f"{self.server_url}/v1/models", timeout=1)
                if response.status_code == 200:
                    print(f"✅ Server health check passed at {self.server_url}/v1/models")

                    # Warmup probe: /v1/models may be available before /v1/completions is ready.
                    probe_deadline = time.time() + 120
                    while time.time() < probe_deadline:
                        try:
                            ping_payload = {
                                "prompt": "ping",
                                "max_tokens": 1,
                                "temperature": 0.0,
                                "stream": False,
                            }
                            r2 = requests.post(
                                f"{self.server_url}/v1/completions", json=ping_payload, timeout=5
                            )
                            if r2.status_code == 200:
                                return True
                        except Exception:
                            pass
                        time.sleep(0.5)

                    # If warmup never succeeded, still allow start if models endpoint is OK
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                if i % 5 == 0:
                    print(f"   Waiting for server... {i+1}s")
            except Exception:
                time.sleep(1)

        print("❌ Server failed to start within timeout")
        if self.server_process and self.server_process.stderr:
            try:
                stderr_output = self.server_process.stderr.read()
                if stderr_output:
                    print("Server stderr output:")
                    print(stderr_output)
            except Exception:
                pass
        return False

    def stop(self):
        """Stop llama.cpp server"""
        if self.server_process:
            print("🛑 Stopping llama.cpp server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️  Server didn't stop gracefully, forcing kill...")
                self.server_process.kill()
            self.server_process = None

    def generate(self, prompt, max_tokens=4096, temperature=0.0):
        return self._client.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


def format_prompt(prompt_text: str, thinking: bool = False) -> str:
    """
    IMPORTANT: HumanEval prompt must be preserved "as is".
    This wrapper only adds a small instruction header.
    """
    if thinking:
        header = (
            "System: You are a Python coding assistant. Think silently, then return ONLY Python code "
            "inside a single fenced code block:```python ... ```\n\n"
            "User: "
        )
    else:
        header = (
            "System: You are a Python coding assistant. Return ONLY Python code "
            "inside a single fenced code block:```python ... ```\n\n"
            "User: "
        )

    # Keep prompt verbatim
    formatted = header + (prompt_text or "") + "\n\nAssistant: "
    return formatted


def load_humaneval_jsonl(jsonl_path: str):
    """Load HumanEval tasks from a local JSONL file (uses prompt verbatim, no stripping)."""
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "task_id" in obj and "prompt" in obj and "test" in obj:
                tasks.append(obj)
    return tasks


def extract_code(text: str) -> str:
    """Extract only the content inside a ```python fenced block."""
    if not isinstance(text, str) or not text:
        return ""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()


def _build_humaneval_script(prompt: str, generated_code: str, test_code: str, entry_point: str) -> str:
    """
    Correct HumanEval semantics:
      - execute prompt (which defines the function signature + docstring)
      - execute candidate completion (generated_code)
      - execute test code (defines check)
      - call check(entry_point)
    """
    ep = (entry_point or "").strip()
    if not ep:
        # If dataset is malformed, make it fail loudly.
        ep = "None"

    return (
        "# --- HumanEval runner (auto-generated) ---\n"
        "import sys\n\n"
        + (prompt or "")
        + "\n\n"
        + (generated_code or "")
        + "\n\n"
        + (test_code or "")
        + "\n\n"
        "try:\n"
        f"    check({ep})\n"
        "except Exception as e:\n"
        "    print('EVAL_ERROR:', repr(e))\n"
        "    raise\n"
    )


def test_code(sample, generated_code: str):
    """
    Execute HumanEval test in a separate process.

    Returns:
        (passed: bool, error_type: str|None, error_detail: str|None)
    """
    if not generated_code or not generated_code.strip():
        return False, "no_code", "Empty extracted code"

    prompt = sample.get("prompt", "") or ""
    test = sample.get("test", "") or ""
    entry_point = sample.get("entry_point", "") or ""
    timeout = int(BENCH_CFG.get("timeout_seconds", 25))

    script = _build_humaneval_script(prompt, generated_code, test, entry_point)

    with tempfile.TemporaryDirectory(prefix="humaneval_") as td:
        script_path = os.path.join(td, "run_task.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout", f"Timeout after {timeout}s"
        except Exception as e:
            return False, "runner_error", repr(e)

        if proc.returncode == 0:
            return True, None, None

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        detail = "\n".join([x for x in [out, err] if x])[:8000]
        return False, "runtime_error", detail or f"Non-zero exit code: {proc.returncode}"


def main():
    parser = argparse.ArgumentParser(description="HumanEval benchmark with llama.cpp GGUF models")

    # Model selection
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"GGUF model file (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing GGUF models (default: {DEFAULT_MODELS_DIR})",
    )

    # Server parameters
    parser.add_argument(
        "--server-path",
        type=str,
        default=DEFAULT_SERVER_PATH,
        help=f"Path to llama.cpp server (default: {DEFAULT_SERVER_PATH})",
    )
    parser.add_argument("--port", type=int, default=LLAMA_CPP_CFG.get("port", 8080), help="Server port")
    parser.add_argument("--host", type=str, default=LLAMA_CPP_CFG.get("host", "127.0.0.1"), help="Server host")
    parser.add_argument("--ctx-size", type=int, default=LLAMA_CPP_CFG.get("ctx_size", 4096), help="Context size")
    parser.add_argument(
        "--n-gpu-layers", type=int, default=LLAMA_CPP_CFG.get("n_gpu_layers", 99), help="GPU layers to offload"
    )
    parser.add_argument("--temp", type=float, default=LLAMA_CPP_CFG.get("temp", 0.0), help="Temperature")
    parser.add_argument(
        "--mirostat",
        type=int,
        default=LLAMA_CPP_CFG.get("mirostat", 2),
        choices=[0, 1, 2],
        help="Mirostat sampling",
    )
    parser.add_argument(
        "--cache-type-k",
        type=str,
        default=LLAMA_CPP_CFG.get("cache_type_k", "f16"),
        choices=["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="Key cache type",
    )
    parser.add_argument(
        "--cache-type-v",
        type=str,
        default=LLAMA_CPP_CFG.get("cache_type_v", "f16"),
        choices=["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="Value cache type",
    )

    # Benchmark parameters
    parser.add_argument(
        "--humaneval-jsonl",
        type=str,
        default=None,
        help=(
            "Path to a local HumanEval.jsonl file (uses full prompt verbatim). "
            "If omitted, uses ./HumanEval.jsonl if present; otherwise falls back to HuggingFace openai_humaneval."
        ),
    )
    parser.add_argument("--samples", type=int, default=BENCH_CFG.get("samples", 164), help="Number of tasks to run")
    parser.add_argument("--thinking", action="store_true", help="Use thinking prompt (still expects fenced code)")
    parser.add_argument("--no-server", action="store_true", help="Don't start/stop server (use existing server)")
    parser.add_argument("--server-url", type=str, default=None, help="Use existing server URL (instead of starting new)")

    args = parser.parse_args()

    # Resolve model path
    if os.path.isabs(args.model):
        model_path = args.model
    else:
        model_path = os.path.join(args.models_dir, args.model)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print(f"   Available models in {args.models_dir}:")
        try:
            models = [f for f in os.listdir(args.models_dir) if f.endswith(".gguf")]
            if models:
                for f in models:
                    print(f"   - {f}")
            else:
                print(f"   No GGUF files found in {args.models_dir}")
        except Exception as e:
            print(f"   Error listing directory: {e}")
        return 1

    print("\n" + "=" * 60)
    print("=== LLAMA.CPP HUMANEVAL BENCHMARK ===")
    print("=" * 60)
    print(f"📁 Model: {os.path.basename(model_path)}")
    print(f"📂 Path: {model_path}")
    print(f"🔧 Server: {args.server_path}")
    print(f"🔧 Config: ctx={args.ctx_size}, gpu_layers={args.n_gpu_layers}")
    print(f"🌡️  Temp: {args.temp}, Mirostat: {args.mirostat}")
    print(f"📊 Samples: {args.samples}")
    print(f"💭 Thinking mode: {args.thinking}")
    print("=" * 60)

    # Setup server/client
    server = None
    client = None

    if args.server_url:
        print(f"📡 Using existing server: {args.server_url}")
        client = _LlamaHttpClient(args.server_url, model_name=os.path.basename(model_path))
    elif args.no_server:
        print("❌ --no-server was set but no --server-url provided.")
        return 1
    else:
        server = LlamaCppServer(
            model_path=model_path,
            server_path=args.server_path,
            ctx_size=args.ctx_size,
            n_gpu_layers=args.n_gpu_layers,
            port=args.port,
            host=args.host,
            temp=args.temp,
            mirostat=args.mirostat,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
        )
        if not server.start():
            print("\n❌ Failed to start server. Check:")
            print(f"   1. Server exists at: {args.server_path}")
            print(f"   2. Model exists at: {model_path}")
            print(f"   3. Port {args.port} is not in use")
            return 1
        client = _LlamaHttpClient(server.server_url, model_name=os.path.basename(model_path))

    # Import HumanEval dataset
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets: pip install datasets")
        if server:
            server.stop()
        return 1

    print("\n📊 Loading HumanEval dataset...")
    try:
        local_jsonl = args.humaneval_jsonl
        if local_jsonl is None and os.path.exists("HumanEval.jsonl"):
            local_jsonl = "HumanEval.jsonl"

        if local_jsonl is not None and os.path.exists(local_jsonl):
            ds = load_humaneval_jsonl(local_jsonl)
            print(f"   Loaded local HumanEval JSONL: {local_jsonl} ({len(ds)} tasks)")
        else:
            ds = load_dataset("openai_humaneval", split="test")
            print(f"   Loaded HuggingFace dataset: openai_humaneval ({len(ds)} tasks)")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        if server:
            server.stop()
        return 1

    if args.samples and args.samples < len(ds):
        ds = ds.select(range(args.samples)) if hasattr(ds, "select") else ds[: args.samples]
        print(f"   Using subset: {args.samples}/{len(ds)} tasks")
    else:
        args.samples = len(ds)
        print(f"   Using full dataset: {args.samples} tasks")

    # Run evaluation
    print("\n🚀 Starting evaluation...")
    start_time = time.time()

    passed = 0
    results = []

    for i, sample in enumerate(ds):
        task_id = sample["task_id"]
        prompt_text = sample["prompt"]

        formatted_prompt = format_prompt(prompt_text, args.thinking)
        max_tokens = (
            BENCH_CFG.get("max_new_tokens_thinking", 4096)
            if args.thinking
            else BENCH_CFG.get("max_new_tokens_normal", 4096)
        )

        print(f"\n[{i+1}/{args.samples}] {task_id}")

        response, audit_text, api_error = client.generate(formatted_prompt, max_tokens=max_tokens, temperature=args.temp)

        if api_error:
            print(f"   ❌ FAIL - API error: {api_error}")
            results.append(
                {
                    "task_id": task_id,
                    "passed": False,
                    "prompt": prompt_text,
                    "prompt_formatted": formatted_prompt,
                    "full_response": response if isinstance(response, str) else json.dumps(response, ensure_ascii=False),
                    "generated_code": "",
                    "error_type": "api_error",
                    "error_detail": str(api_error),
                    "raw_answer_audit": audit_text,
                }
            )
            continue

        if not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)

        code = extract_code(response)

        if not code:
            print("   ❌ FAIL - No code generated")
            results.append(
                {
                    "task_id": task_id,
                    "passed": False,
                    "prompt": prompt_text,
                    "prompt_formatted": formatted_prompt,
                    "full_response": response,
                    "generated_code": "",
                    "error_type": "no_code",
                    "error_detail": "No ```python fenced block found",
                    "raw_answer_audit": audit_text if (audit_text is not None and str(audit_text).strip() != "") else response,
                }
            )
            continue

        ok, etype, edetail = test_code(sample, code)

        if ok:
            passed += 1
            print("   ✅ PASS")
        else:
            print("   ❌ FAIL")

        results.append(
            {
                "task_id": task_id,
                "passed": ok,
                "prompt": prompt_text,
                "prompt_formatted": formatted_prompt,
                "full_response": response,
                "generated_code": code,
                "error_type": etype,
                "error_detail": edetail,
            }
        )

        current_acc = (passed / (i + 1)) * 100.0
        print(f"   Progress: {passed}/{i+1} ({current_acc:.1f}%)")

    total_duration = time.time() - start_time

    if args.samples > 0:
        accuracy = (passed / args.samples) * 100.0
        avg_time_per_task = total_duration / args.samples
    else:
        accuracy = 0.0
        avg_time_per_task = 0.0

    print("\n" + "=" * 60)
    print("🏁 BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"📊 Model: {os.path.basename(model_path)}")
    print(f"✅ Accuracy (Pass@1): {accuracy:.2f}% ({passed}/{args.samples})")
    print(f"⏱️  Total Time: {total_duration:.2f}s")
    print(f"⚡ Avg per Task: {avg_time_per_task:.2f}s")
    print("=" * 60)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path).replace(".gguf", "")
    out_fname = f"results_llamacpp_{model_name}_{timestamp}.json"

    with open(out_fname, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": os.path.basename(model_path),
                "model_path": model_path,
                "parameters": {
                    "ctx_size": args.ctx_size,
                    "n_gpu_layers": args.n_gpu_layers,
                    "temp": args.temp,
                    "mirostat": args.mirostat,
                    "cache_type_k": args.cache_type_k,
                    "cache_type_v": args.cache_type_v,
                    "server_path": args.server_path,
                    "port": args.port,
                    "host": args.host,
                    "server_url": args.server_url,
                    "no_server": bool(args.no_server),
                },
                "benchmark": {
                    "samples": args.samples,
                    "thinking": args.thinking,
                    "accuracy": accuracy,
                    "passed": passed,
                    "total": args.samples,
                    "total_time": total_duration,
                    "avg_time_per_task": avg_time_per_task,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"📁 Results saved to: {out_fname}")

    if server:
        server.stop()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
