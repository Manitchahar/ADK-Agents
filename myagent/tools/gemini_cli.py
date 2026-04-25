"""Bridge tool: invoke the Gemini CLI as a coding worker from ADK.

Long-running variant: ``run_gemini_cli`` returns immediately with a job id while
the Gemini CLI subprocess executes in the background. The model (or the user)
calls ``check_gemini_cli_job`` to retrieve the final result. This keeps the
session history valid when the user sends a new message before the worker
finishes (the previous synchronous design produced "Unexpected role 'user'
after role 'tool'" errors on NVIDIA NIM via LiteLLM).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from google.adk.tools import FunctionTool, LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

_DEFAULT_TIMEOUT = int(os.environ.get("GEMINI_CLI_TIMEOUT", "300"))
_DEFAULT_MODEL = os.environ.get("GEMINI_CLI_MODEL", "gemini-2.5-pro")
_DEFAULT_WORKING_DIRECTORY = os.environ.get(
    "HERMES_WORKSPACE_ROOT",
    str(Path(__file__).resolve().parents[2]),
)
_APPROVAL_MODES = {"default", "auto_edit", "yolo", "plan"}

# Module-level thread pool: jobs must survive across ADK per-invocation event
# loops. Using asyncio.create_task on the per-request loop orphans the task
# when the tool returns.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.environ.get("GEMINI_CLI_MAX_WORKERS", "4")),
    thread_name_prefix="gemini_cli",
)

# job_id -> {"future": Future, "started_at": float, "prompt": str, "result": dict | None}
_JOBS: dict[str, dict[str, Any]] = {}
_JOB_TTL_SECONDS = 3600
_THREAD_LOCAL = threading.local()


def _cancelled_result(exit_code: int = -2) -> dict[str, Any]:
    return {
        "status": "cancelled",
        "error": "Gemini CLI job cancelled",
        "exit_code": exit_code,
        "stderr_tail": "",
    }


def _current_job() -> dict[str, Any] | None:
    job_id = getattr(_THREAD_LOCAL, "job_id", None)
    if job_id is None:
        return None
    return _JOBS.get(job_id)


def _job_status(job: dict[str, Any]) -> str:
    result = job.get("result")
    if isinstance(result, dict):
        return str(result.get("status") or "finished")
    if job.get("cancel_requested"):
        return "cancelling"
    return "running"


def _sync_finished_job(job_id: str, job: dict[str, Any]) -> None:
    future = job.get("future")
    if job.get("result") is not None or future is None or not future.done():
        return
    try:
        job["result"] = future.result()
        job["finished_at"] = time.time()
    except Exception as exc:  # pragma: no cover - defensive worker isolation
        job["result"] = {
            "status": "error",
            "error": f"Gemini CLI worker failed: {type(exc).__name__}: {exc}",
        }
        job["finished_at"] = time.time()


def _extract_json(stdout: str) -> dict[str, Any] | None:
    """Gemini CLI sometimes prefixes stdout with warnings; find the JSON object."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _execute_gemini_cli(
    prompt: str,
    cwd: Path,
    approval_mode: str,
    model: str,
) -> dict[str, Any]:
    cmd = [
        "gemini",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--approval-mode",
        approval_mode,
        "-m",
        model,
    ]
    job = _current_job()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return {"status": "error", "error": "gemini executable not found in PATH"}

    if job is not None:
        job["process"] = proc

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=_DEFAULT_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_bytes, stderr_bytes = proc.communicate()
        if job is not None:
            job["process"] = None
            if job.get("cancel_requested"):
                return _cancelled_result(proc.returncode or -2)
        return {
            "status": "error",
            "error": f"Gemini CLI timed out after {_DEFAULT_TIMEOUT}s",
            "exit_code": -1,
        }
    finally:
        if job is not None:
            job["process"] = None

    if job is not None and job.get("cancel_requested"):
        return _cancelled_result(proc.returncode or -2)

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    parsed = _extract_json(stdout)

    result: dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 else "error",
        "exit_code": proc.returncode,
        "stderr_tail": stderr[-2000:] if stderr else "",
    }
    if parsed:
        result["response"] = parsed.get("response", "")
        result["session_id"] = parsed.get("session_id", "")
        stats = parsed.get("stats", {})
        files = stats.get("files", {}) if isinstance(stats, dict) else {}
        result["lines_added"] = files.get("totalLinesAdded", 0)
        result["lines_removed"] = files.get("totalLinesRemoved", 0)
    else:
        result["status"] = "error"
        result["error"] = "Could not parse JSON from Gemini CLI stdout"
        result["stdout_tail"] = stdout[-2000:]
    return result


def _gc_jobs() -> None:
    now = time.time()
    stale = [
        jid
        for jid, j in _JOBS.items()
        if j.get("result") is not None
        and now - j.get("finished_at", j["started_at"]) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        _JOBS.pop(jid, None)


async def run_gemini_cli(
    prompt: str,
    working_directory: str | None = None,
    approval_mode: str = "default",
    model: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Start a Gemini CLI worker session in the background and return a job id.

    This is a long-running tool. It returns immediately with status='running'
    and a job_id. Call ``check_gemini_cli_job(job_id)`` to retrieve the final
    result. While the job runs, you can keep talking to the user; the result
    will be available when ``check_gemini_cli_job`` returns status='ok' or
    status='error'.

    Use this to delegate local-machine work: system inspection, shell commands,
    file edits, refactors, repo analysis, terminal tasks.

    Args:
        prompt: The task description for the worker. Be specific.
        working_directory: Absolute path to the repo/dir the worker should
            operate in. Defaults to this project root when omitted.
        approval_mode: 'default', 'plan' (read-only planning), 'auto_edit'
            (auto-approve edits), or 'yolo' (auto-approve everything).
        model: Optional model override (e.g. 'gemini-2.5-pro').

    Returns:
        Dict with keys: status='running', job_id, started_at, hint.
    """
    if approval_mode not in _APPROVAL_MODES:
        return {
            "status": "error",
            "error": f"approval_mode must be one of {sorted(_APPROVAL_MODES)}",
        }

    cwd = Path(working_directory or _DEFAULT_WORKING_DIRECTORY).expanduser().resolve()
    if not cwd.is_dir():
        return {
            "status": "error",
            "error": f"working_directory does not exist or is not a directory: {cwd}",
        }

    _gc_jobs()
    job_id = (
        tool_context.function_call_id
        if tool_context is not None and tool_context.function_call_id
        else uuid.uuid4().hex
    )
    started_at = time.time()

    job: dict[str, Any] = {
        "future": None,
        "started_at": started_at,
        "prompt": prompt[:200],
        "result": None,
        "cancel_requested": False,
        "process": None,
        "working_directory": str(cwd),
        "approval_mode": approval_mode,
        "model": model or _DEFAULT_MODEL,
    }
    _JOBS[job_id] = job

    def _runner() -> dict[str, Any]:
        _THREAD_LOCAL.job_id = job_id
        try:
            result = _execute_gemini_cli(
                prompt=prompt,
                cwd=cwd,
                approval_mode=approval_mode,
                model=job["model"],
            )
            if job.get("cancel_requested") and result.get("status") != "cancelled":
                result = _cancelled_result(int(result.get("exit_code", -2)))
            job["result"] = result
            job["finished_at"] = time.time()
            return result
        finally:
            if hasattr(_THREAD_LOCAL, "job_id"):
                del _THREAD_LOCAL.job_id

    future = _EXECUTOR.submit(_runner)
    job["future"] = future

    return {
        "status": "running",
        "job_id": job_id,
        "started_at": started_at,
        "hint": (
            "Job started in background. Call check_gemini_cli_job(job_id) to "
            "retrieve the result. You may chat with the user while it runs."
        ),
    }


async def check_gemini_cli_job(
    job_id: str,
    wait_seconds: float = 0.0,
) -> dict[str, Any]:
    """Return the result of a background Gemini CLI job, non-blocking by default.

    Polls the job once and returns immediately. If still running, end your
    turn with a brief note to the user; you can call this again on the next
    turn. Long blocking waits inside a single tool call cause LiteLLM "user
    after tool" history errors when the user sends a new message mid-poll.

    Args:
        job_id: The job_id returned by run_gemini_cli.
        wait_seconds: Optional seconds to wait for completion, clamped 0..120.
            Default 0 (return immediately). Avoid >5 in chat scenarios.

    Returns:
        If finished: the original Gemini CLI result dict (status='ok' or
        'error', plus response, exit_code, session_id, lines_added,
        lines_removed, stderr_tail, etc).
        If still running after waiting:
        {"status":"running","job_id":..,"elapsed_s":..}.
        If unknown: {"status":"error","error":"unknown job_id"}.
    """
    job = _JOBS.get(job_id)
    if job is None:
        return {"status": "error", "error": f"unknown job_id: {job_id}"}

    future = job.get("future")
    wait_seconds = max(0.0, min(float(wait_seconds), 120.0))
    if job["result"] is None and future is not None and wait_seconds > 0:
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=wait_seconds)
        except TimeoutError:
            pass
        except concurrent.futures.CancelledError:
            pass

    _sync_finished_job(job_id, job)

    if job["result"] is not None:
        job["notified"] = True
        return {"job_id": job_id, **job["result"]}

    elapsed = time.time() - job["started_at"]
    return {
        "status": "running",
        "job_id": job_id,
        "elapsed_s": round(elapsed, 1),
        "prompt_preview": job["prompt"],
    }


async def list_gemini_cli_jobs(include_finished: bool = True) -> dict[str, Any]:
    """List known Gemini CLI worker jobs without blocking.

    Args:
        include_finished: When false, return only running/cancelling jobs.

    Returns:
        Dict with status='ok' and a jobs list containing job_id, status,
        elapsed_s, prompt_preview, working_directory, approval_mode, model, and
        finished_at when available.
    """
    _gc_jobs()
    now = time.time()
    jobs: list[dict[str, Any]] = []
    for job_id, job in sorted(_JOBS.items(), key=lambda item: item[1]["started_at"]):
        status = _job_status(job)
        if not include_finished and status not in {"running", "cancelling"}:
            continue
        entry = {
            "job_id": job_id,
            "status": status,
            "elapsed_s": round(now - job["started_at"], 1),
            "prompt_preview": job["prompt"],
            "working_directory": job.get("working_directory", ""),
            "approval_mode": job.get("approval_mode", ""),
            "model": job.get("model", ""),
        }
        if job.get("finished_at") is not None:
            entry["finished_at"] = job["finished_at"]
        jobs.append(entry)
    return {"status": "ok", "count": len(jobs), "jobs": jobs}


async def collect_gemini_cli_job_notifications() -> dict[str, Any]:
    """Return newly finished Gemini CLI jobs and mark them as notified.

    This is a lightweight helper for the supervisor agent. It lets Rocky mention
    completed background work at the start of a later user turn, even if the
    user has started talking about something else.
    """
    _gc_jobs()
    notifications: list[dict[str, Any]] = []
    for job_id, job in sorted(_JOBS.items(), key=lambda item: item[1]["started_at"]):
        _sync_finished_job(job_id, job)
        result = job.get("result")
        if result is None or job.get("notified"):
            continue
        response = str(result.get("response") or result.get("error") or "").strip()
        notifications.append(
            {
                "job_id": job_id,
                "status": str(result.get("status") or "finished"),
                "prompt_preview": job["prompt"],
                "response_preview": response[:500],
                "elapsed_s": round(
                    job.get("finished_at", time.time()) - job["started_at"], 1
                ),
            }
        )
        job["notified"] = True
    return {"status": "ok", "count": len(notifications), "notifications": notifications}


async def cancel_gemini_cli_job(job_id: str) -> dict[str, Any]:
    """Best-effort cancellation for a Gemini CLI worker job.

    Running subprocesses are asked to terminate. Queued futures are cancelled
    before they start. Always call check_gemini_cli_job afterwards for the final
    terminal status.
    """
    job = _JOBS.get(job_id)
    if job is None:
        return {"status": "error", "error": f"unknown job_id: {job_id}"}
    if job.get("result") is not None:
        return {
            "status": "ok",
            "job_id": job_id,
            "message": "Job already finished.",
            "job_status": _job_status(job),
        }

    job["cancel_requested"] = True
    future = job.get("future")
    if future is not None and future.cancel():
        job["result"] = _cancelled_result()
        job["finished_at"] = time.time()
        return {"status": "cancelled", "job_id": job_id}

    process = job.get("process")
    if process is not None and process.poll() is None:
        process.terminate()
    return {
        "status": "cancelling",
        "job_id": job_id,
        "message": "Cancellation requested. Poll this job for its final status.",
    }


gemini_cli_tool = LongRunningFunctionTool(func=run_gemini_cli)
gemini_cli_check_tool = FunctionTool(func=check_gemini_cli_job)
gemini_cli_list_tool = FunctionTool(func=list_gemini_cli_jobs)
gemini_cli_cancel_tool = FunctionTool(func=cancel_gemini_cli_job)
gemini_cli_notifications_tool = FunctionTool(func=collect_gemini_cli_job_notifications)
