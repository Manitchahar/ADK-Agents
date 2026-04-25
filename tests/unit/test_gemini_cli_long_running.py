"""Unit test for the long-running gemini_cli tool."""

import time

import pytest


@pytest.fixture
def gemini_cli(monkeypatch):
    # Import inside fixture so patches apply to module-level state cleanly.
    from myagent.tools import gemini_cli as mod

    # Reset job registry between tests.
    mod._JOBS.clear()

    def _fake_execute(prompt, cwd, approval_mode, model):
        time.sleep(0.05)
        return {
            "status": "ok",
            "exit_code": 0,
            "response": f"echo: {prompt}",
            "session_id": "fake-session",
            "lines_added": 0,
            "lines_removed": 0,
            "stderr_tail": "",
        }

    monkeypatch.setattr(mod, "_execute_gemini_cli", _fake_execute)
    return mod


@pytest.mark.asyncio
async def test_run_returns_running_immediately(gemini_cli):
    result = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = result["job_id"]

    try:
        assert result["status"] == "running"
        assert "hint" in result
    finally:
        gemini_cli._JOBS[job_id]["future"].result(timeout=5)


@pytest.mark.asyncio
async def test_check_waits_then_returns_ok(gemini_cli):
    started = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = started["job_id"]

    done = await gemini_cli.check_gemini_cli_job(job_id=job_id, wait_seconds=5)
    assert done["status"] == "ok"
    assert done["response"] == "echo: hello"
    assert done["job_id"] == job_id


@pytest.mark.asyncio
async def test_fast_completion_is_not_lost(gemini_cli, monkeypatch):
    def _immediate_execute(prompt, cwd, approval_mode, model):
        return {
            "status": "ok",
            "exit_code": 0,
            "response": "done immediately",
            "session_id": "fast-session",
            "lines_added": 0,
            "lines_removed": 0,
            "stderr_tail": "",
        }

    monkeypatch.setattr(gemini_cli, "_execute_gemini_cli", _immediate_execute)

    started = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = started["job_id"]
    gemini_cli._JOBS[job_id]["future"].result(timeout=5)

    done = await gemini_cli.check_gemini_cli_job(job_id=job_id)
    assert done["status"] == "ok"
    assert done["response"] == "done immediately"


@pytest.mark.asyncio
async def test_check_with_zero_wait_returns_running(gemini_cli):
    started = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = started["job_id"]

    pending = await gemini_cli.check_gemini_cli_job(
        job_id=job_id,
        wait_seconds=0,
    )
    assert pending["status"] == "running"
    assert pending["job_id"] == job_id

    gemini_cli._JOBS[job_id]["future"].result(timeout=5)


@pytest.mark.asyncio
async def test_check_unknown_job(gemini_cli):
    result = await gemini_cli.check_gemini_cli_job(job_id="nope")
    assert result["status"] == "error"
    assert "unknown" in result["error"]


@pytest.mark.asyncio
async def test_list_jobs_reports_running_and_finished(gemini_cli):
    started = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = started["job_id"]

    running = await gemini_cli.list_gemini_cli_jobs()
    assert running["status"] == "ok"
    assert any(job["job_id"] == job_id for job in running["jobs"])

    gemini_cli._JOBS[job_id]["future"].result(timeout=5)
    done = await gemini_cli.check_gemini_cli_job(job_id=job_id)
    assert done["status"] == "ok"

    finished = await gemini_cli.list_gemini_cli_jobs()
    listed = next(job for job in finished["jobs"] if job["job_id"] == job_id)
    assert listed["status"] == "ok"


@pytest.mark.asyncio
async def test_collect_notifications_reports_finished_once(gemini_cli):
    started = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = started["job_id"]
    gemini_cli._JOBS[job_id]["future"].result(timeout=5)

    first = await gemini_cli.collect_gemini_cli_job_notifications()
    assert first["status"] == "ok"
    assert first["count"] == 1
    assert first["notifications"][0]["job_id"] == job_id
    assert first["notifications"][0]["status"] == "ok"

    second = await gemini_cli.collect_gemini_cli_job_notifications()
    assert second["status"] == "ok"
    assert second["count"] == 0


@pytest.mark.asyncio
async def test_checked_job_is_not_notified_again(gemini_cli):
    started = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = started["job_id"]
    gemini_cli._JOBS[job_id]["future"].result(timeout=5)

    done = await gemini_cli.check_gemini_cli_job(job_id=job_id)
    assert done["status"] == "ok"

    notifications = await gemini_cli.collect_gemini_cli_job_notifications()
    assert notifications["status"] == "ok"
    assert notifications["count"] == 0


@pytest.mark.asyncio
async def test_cancel_unknown_job(gemini_cli):
    result = await gemini_cli.cancel_gemini_cli_job(job_id="nope")
    assert result["status"] == "error"
    assert "unknown" in result["error"]


@pytest.mark.asyncio
async def test_cancel_running_job_marks_cancelled(gemini_cli, monkeypatch):
    def _slow_execute(prompt, cwd, approval_mode, model):
        time.sleep(0.1)
        return {
            "status": "ok",
            "exit_code": 0,
            "response": "should not surface",
            "session_id": "cancel-session",
            "lines_added": 0,
            "lines_removed": 0,
            "stderr_tail": "",
        }

    monkeypatch.setattr(gemini_cli, "_execute_gemini_cli", _slow_execute)

    started = await gemini_cli.run_gemini_cli(prompt="hello", approval_mode="yolo")
    job_id = started["job_id"]
    cancelled = await gemini_cli.cancel_gemini_cli_job(job_id=job_id)
    assert cancelled["status"] in {"cancelled", "cancelling"}

    done = await gemini_cli.check_gemini_cli_job(job_id=job_id)
    if done["status"] == "cancelling":
        gemini_cli._JOBS[job_id]["future"].result(timeout=5)
        done = await gemini_cli.check_gemini_cli_job(job_id=job_id)
    assert done["status"] == "cancelled"


@pytest.mark.asyncio
async def test_invalid_approval_mode(gemini_cli):
    result = await gemini_cli.run_gemini_cli(prompt="x", approval_mode="bogus")
    assert result["status"] == "error"
    assert "approval_mode" in result["error"]


@pytest.mark.asyncio
async def test_tool_flags(gemini_cli):
    assert gemini_cli.gemini_cli_tool.is_long_running is True
    assert gemini_cli.gemini_cli_check_tool.is_long_running is False
    assert gemini_cli.gemini_cli_list_tool.is_long_running is False
    assert gemini_cli.gemini_cli_cancel_tool.is_long_running is False
    assert gemini_cli.gemini_cli_notifications_tool.is_long_running is False
