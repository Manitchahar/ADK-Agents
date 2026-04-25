# Rocky

Rocky is a Google ADK prototype for a Max/Hermes/OpenClaw-like personal
operator. It uses ADK as the orchestration layer and Gemini CLI as the coding
worker, so the main agent stays responsive while long-running repo work happens
in isolated background jobs.

The goal is not to clone any single project. The goal is a clean ADK showcase:
multi-agent routing, MCP tools, long-running tool calls, memory hooks, eval
scaffolding, and a deployable FastAPI surface in one small repo.

## Demo story

Ask Rocky for a coding or local-machine task:

```text
Fix the failing auth tests in ~/src/myapp and tell me what changed.
```

Rocky routes the work to `local_operator`, which starts a Gemini CLI background
job and returns a job id. Rocky can keep chatting while the worker runs. You can
ask for status, list active jobs, or cancel a job without freezing the root
agent.

```text
What jobs are running?
Cancel job 1234abcd.
Check the auth-test job again.
```

Ask Rocky to remember stable context:

```text
Remember that Rocky is my ADK demo project for recruiters and ADK developers.
What do you remember about Rocky?
Forget the recruiter note.
```

The `memory_keeper` sub-agent stores durable facts in markdown wiki pages under
`~/.rocky/wiki` by default, giving the prototype a Claw/Hermes-style persistent
memory feel without requiring external infrastructure.

Ask Rocky for current public information:

```text
Search what changed in the latest Google ADK release and give me the highlights.
```

Rocky routes current-events and public web questions to `search_operator`, which
uses ADK's built-in Google Search grounding tool and a Gemini search model.

## Structure

- `myagent/agent.py`: defines Rocky and its specialist sub-agents.
- `myagent/tools/gemini_cli.py`: starts, checks, lists, and cancels Gemini CLI
  worker jobs for local code and terminal tasks.
- `myagent/fast_api_app.py`: exposes Rocky through ADK's FastAPI app.
- `myagent/app_utils/telemetry.py`: optional prompt/response metadata logging.
- `tests/unit/`: deterministic code tests for model selection, sub-agent wiring,
  and Gemini CLI job management.
- `tests/eval/`: ADK eval scaffolding for behavior-level checks.
- `myagent/__init__.py`: package entrypoint.

## Current Agent

The repo currently exposes a `root_agent` named `rocky` with:

- provider-aware model selection via `HERMES_ROOT_MODEL`, NVIDIA NIM,
  Gemini API, and OpenRouter fallbacks
- a supervisor prompt that delegates instead of doing heavy work directly
- `local_operator` for local machine, repository, terminal, and coding tasks via
  Gemini CLI background jobs
- `workspace_operator` for Google Workspace MCP tools
- `memory_keeper` for Claw-style markdown wiki memory over stable user/project
  context
- `search_operator` for current public facts through ADK's built-in Google
  Search grounding tool
- an exported ADK `App` wrapper for app-level runtime features
- memory preload plus automatic session-to-memory ingestion when a memory
  backend is configured
- Google Workspace MCP forced to encrypted file storage so spawned tool
  processes do not depend on desktop keychain access

## ADK features this demonstrates

| ADK feature | How Rocky uses it |
| --- | --- |
| Root + sub-agents | Rocky supervises `local_operator`, `workspace_operator`, `memory_keeper`, and `search_operator`. |
| `LongRunningFunctionTool` | Gemini CLI coding tasks start in the background instead of blocking the chat turn. |
| Function tools | Job check/list/cancel tools expose worker control to the agent. |
| MCP toolsets | Google Workspace tools are wired through ADK MCP support. |
| Built-in Google Search | `search_operator` uses ADK's `google_search` grounding tool for current web facts. |
| Session state callbacks | Tool results are summarized into namespaced `rocky:*` state keys. |
| Memory tools | `memory_keeper` can remember, recall, list, and forget markdown wiki entries. |
| Memory hooks | ADK memory preload/save paths are also present for longer-term context. |
| FastAPI app | ADK runtime can serve Rocky through web/API surfaces. |
| Eval scaffolding | Behavior cases live under `tests/eval/`. |

## Gemini CLI worker controls

Rocky's local operator has five job tools:

| Tool | Purpose |
| --- | --- |
| `run_gemini_cli` | Start a Gemini CLI worker subprocess and return immediately with a `job_id`. |
| `check_gemini_cli_job` | Poll a job once without blocking the root agent. |
| `list_gemini_cli_jobs` | Show active/recent jobs with status and prompt previews. |
| `cancel_gemini_cli_job` | Request cancellation for a queued/running worker job. |
| `collect_gemini_cli_job_notifications` | Report newly finished background jobs on a later turn. |

Useful environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HERMES_ROOT_MODEL` | auto provider stack | Main/sub-agent model override. |
| `HERMES_ROOT_FALLBACK_MODELS` | empty | Comma-separated extra fallback models after an explicit root model. |
| `NVIDIA_NIM_API_KEY` / `NVIDIA_API_KEY` | unset | Enables the primary NVIDIA NIM tool-calling stack. |
| `NVIDIA_NIM_API_BASE` | `https://integrate.api.nvidia.com/v1` | NVIDIA NIM endpoint for LiteLLM. |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | unset | Enables Gemini API fallbacks for the root model and search model. |
| `OPENROUTER_API_KEY` | unset | Enables OpenRouter as the last-resort fallback provider. |
| `HERMES_SEARCH_MODEL` | `gemini-flash-latest` | Gemini model used by `search_operator` for Google Search grounding. |
| `HERMES_GEMINI_FALLBACK_MODEL` | `gemini-2.5-flash` | Native Gemini fallback when LiteLLM is unavailable. |
| `GEMINI_CLI_MODEL` | `gemini-2.5-pro` | Gemini CLI worker model. |
| `GEMINI_CLI_TIMEOUT` | `300` | Per-worker timeout in seconds. |
| `GEMINI_CLI_MAX_WORKERS` | `4` | Maximum concurrent Gemini CLI workers. |
| `HERMES_WORKSPACE_ROOT` | repo root | Default working directory for Gemini CLI jobs. |
| `WORKSPACE_MCP_PATH` | `~/.gemini/extensions/google-workspace` | Workspace MCP server directory. |
| `ROCKY_MEMORY_DIR` | `~/.rocky/wiki` | Markdown wiki directory for Claw-style memory. |

Default root model order:

| Order | Provider | Models |
| --- | --- | --- |
| 1 | NVIDIA NIM | `qwen/qwen3.5-397b-a17b`, then `nvidia/nemotron-3-super-120b-a12b`, then `moonshotai/kimi-k2-instruct-0905` |
| 2 | Gemini API | `gemini-2.5-pro`, then Gemini 3 Flash preview, then Flash fallbacks |
| 3 | OpenRouter | Kimi K2, Qwen3 Coder, GPT OSS, then OpenRouter free fallback |

Mistral Small stays available in the NVIDIA list, but it is no longer first
because Rocky needs stronger instruction following and tool-call orchestration at
the supervisor layer.

## Run

Install dependencies, then load the agent package from `myagent`:

```bash
uv sync
adk web --port 8501
```

Run the API server:

```bash
uv run uvicorn myagent.fast_api_app:app --port 8000
```

Run code checks:

```bash
uv run pytest tests/unit
uv run ruff check .
```

Run ADK evals after configuring model credentials:

```bash
agents-cli eval run --evalset tests/eval/evalsets/basic.evalset.json \
  --config tests/eval/eval_config.json
```

Local `.env` files and ADK runtime state are intentionally ignored and should not be committed.

## Notes

- `llms.txt` is included as reference material for ADK documentation links.
