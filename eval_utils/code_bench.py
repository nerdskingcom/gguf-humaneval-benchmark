# eval_utils/code_bench.py
#
# PURPOSE:
#   HumanEval evaluation utilities with strict fenced-code extraction and
#   prompt+completion execution semantics.
#
# DEPENDENCIES:
#   - Python 3.10+
#   - Standard library only (subprocess/tempfile/json)
#
# FILE VERSION: 2.0.0
# DATE/TIME: 2025-12-29T04:33:16
#
# NOTES:
#   - Backend-agnostic: does not start llama.cpp.
#   - Execute semantics: prompt + generated_code + test.
#
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "bench_config.json")


def load_bench_cfg(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("benchmark", {}) or {}
    return {}


BENCH_CFG: Dict[str, Any] = load_bench_cfg()


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    error_type: Optional[str] = None
    error_detail: Optional[str] = None
    full_response: str = ""
    generated_code: str = ""
    prompt: str = ""
    prompt_formatted: str = ""


def extract_fenced_python(full_response: str) -> str:
    """Strictly extract the content inside a ```python fenced block."""
    if not isinstance(full_response, str) or not full_response:
        return ""
    start = full_response.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = full_response.find("```", start)
    if end == -1:
        return ""
    return full_response[start:end].strip()


def format_prompt(prompt_text: str, thinking: bool = False) -> str:
    """Format a safe instruction wrapper + the HumanEval prompt as-is."""
    if thinking:
        header = (
            "You are a Python coding assistant.\n"
            "Think silently, then return ONLY Python code inside a single fenced code block:```python ... ```\n"
            "Do not include explanations, prose, or extra code blocks.\n\n"
        )
    else:
        header = (
            "You are a Python coding assistant.\n"
            "Return ONLY Python code inside a single fenced code block:```python ... ```\n"
            "Do not include explanations, prose, or extra code blocks.\n\n"
        )
    return header + (prompt_text or "")


def _build_humaneval_script(prompt: str, generated_code: str, test_code: str, entry_point: str) -> str:
    return (
        "# --- HumanEval runner (auto-generated) ---\n"
        "import sys\n"
        "\n"
        + (prompt or "") + "\n\n"
        + (generated_code or "") + "\n\n"
        + (test_code or "") + "\n\n"
        + "try:\n"
        + f"    check({entry_point})\n"
        + "except Exception as e:\n"
        + "    print('EVAL_ERROR:', repr(e))\n"
        + "    raise\n"
    )


def run_humaneval_test(
    *,
    prompt: str,
    generated_code: str,
    test_code: str,
    entry_point: str,
    timeout_seconds: int = 25,
    python_exe: str = "python",
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Execute the HumanEval test in a separate process."""
    script = _build_humaneval_script(prompt, generated_code, test_code, entry_point)
    with tempfile.TemporaryDirectory(prefix="humaneval_") as td:
        script_path = os.path.join(td, "run_task.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout", f"Timeout after {timeout_seconds}s"
        except Exception as e:
            return False, "runner_error", repr(e)

        if proc.returncode == 0:
            return True, None, None

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        detail = "\n".join([x for x in [out, err] if x])[:8000]
        return False, "runtime_error", detail or f"Non-zero exit code: {proc.returncode}"


def load_humaneval_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def evaluate_tasks_with_responses(
    tasks: List[Dict[str, Any]],
    responses_by_task_id: Dict[str, str],
    *,
    timeout_seconds: Optional[int] = None,
    thinking: bool = False,
    python_exe: str = "python",
) -> List[EvalResult]:
    if timeout_seconds is None:
        timeout_seconds = int(BENCH_CFG.get("timeout_seconds", 25))

    results: List[EvalResult] = []
    for t in tasks:
        task_id = t.get("task_id", "")
        prompt = t.get("prompt", "") or ""
        test_code = t.get("test", "") or ""
        entry_point = t.get("entry_point", "") or ""

        full_response = responses_by_task_id.get(task_id, "") or ""
        generated_code = extract_fenced_python(full_response)
        prompt_formatted = format_prompt(prompt, thinking=thinking)

        if not generated_code:
            results.append(EvalResult(
                task_id=task_id,
                passed=False,
                error_type="no_code",
                error_detail="No ```python fenced block found",
                full_response=full_response,
                generated_code="",
                prompt=prompt,
                prompt_formatted=prompt_formatted,
            ))
            continue

        ok, etype, edetail = run_humaneval_test(
            prompt=prompt,
            generated_code=generated_code,
            test_code=test_code,
            entry_point=entry_point,
            timeout_seconds=timeout_seconds,
            python_exe=python_exe,
        )
        results.append(EvalResult(
            task_id=task_id,
            passed=ok,
            error_type=etype,
            error_detail=edetail,
            full_response=full_response,
            generated_code=generated_code,
            prompt=prompt,
            prompt_formatted=prompt_formatted,
        ))
    return results