# Nerdsking HumanEval Benchmark for llama.cpp (GGUF)

A **strict, auditable HumanEval benchmark runner** for **GGUF models** served via **llama.cpp**, using its **OpenAI-compatible HTTP API**.

This project focuses on **correct execution semantics** and **reproducibility**:
- Prompts are preserved verbatim (no stripping or truncation).
- Only fenced Python code is accepted.
- Each task is executed using strict HumanEval semantics.
- Full outputs and failure reasons are saved for auditing.

<hr>

##  Install from PyPI
<code>
pip install gguf-humaneval-benchmark
</code>code>
####  Verify installation
<code>
gguf-humaneval-benchmark --help
</code>
You should see the CLI help for the HumanEval benchmark runner.

<hr>


## Key Features

### ✅ Correct HumanEval Semantics (Strict)
For every task, execution follows **exactly**:

1. Execute the original **prompt** (function signature + docstring)  
2. Execute the **model-generated code**  
3. Execute the **test harness**  
4. Call `check(entry_point)`

---

### ✅ Prompt Integrity
- HumanEval prompts are used **verbatim**
- No stripping, rewriting, or truncation
- Only a minimal **instruction header** is prepended
- Raw prompts are stored in the output JSON

---

### ✅ Strict Code Extraction
Only code inside a **single fenced block** is accepted:

```python
```python
# code here
```
```

If no such block exists → **automatic failure (`no_code`)**.

---

### ✅ Full Failure Attribution
Each failed task records:
- `error_type`
- `error_detail`
- `full_response`
- `generated_code`

---

### ✅ llama.cpp Native Support
- Automatic server start/stop **or** reuse an existing server
- Uses `/v1/completions` OpenAI-compatible API
- Streaming-safe and timeout-safe
- **GGUF-only by design**

---

## Repository Structure

```
.
├── benchmark.py
├── HumanEval.jsonl
├── LICENSE
├── README.md
└── eval_utils/
    ├── __init__.py
    ├── bench_config.json
    └── code_bench.py
```

---

## Dependencies

### Required
- Python 3.10+
- llama.cpp (with server support)
- GGUF model

### Python packages
```bash
pip install requests datasets
```

---

## Building llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release
```

---

## Usage

### Automatic server management
```bash
python benchmark.py --model model.gguf --server-path /path/to/llama.cpp/build/bin
```

### Use existing server
```bash
python benchmark.py --server-url http://127.0.0.1:8080 --no-server
```

### Use local HumanEval
```bash
python benchmark.py --humaneval-jsonl HumanEval.jsonl
```

---

## Output

A JSON file is generated containing:
- Full configuration
- Per-task results
- Raw model outputs
- Error attribution
- Timing metrics

---

## License / citation

-If you use this code in research or benchmarking, please cite:

https://github.com/nerdskingcom/gguf-humaneval-benchmark, IPMN/IMNECHO / Nerdsking.com
