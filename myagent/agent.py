import os

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import google_search
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types
from mcp import StdioServerParameters
from pydantic import Field

from .model_fallbacks import (
    GOOGLE_TOOL_CALL_MODELS,
    NVIDIA_TOOL_CALL_MODELS,
    OPENROUTER_LAST_RESORT_MODELS,
)
from .tools.gemini_cli import (
    gemini_cli_cancel_tool,
    gemini_cli_check_tool,
    gemini_cli_list_tool,
    gemini_cli_notifications_tool,
    gemini_cli_tool,
    mark_gemini_cli_job_notifications_reported,
    peek_gemini_cli_job_notifications,
)
from .tools.memory import (
    forget_memory_tool,
    list_memory_topics_tool,
    recall_memory_tool,
    remember_memory_tool,
)

try:
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
except ImportError:  # pragma: no cover - exercised by non-extension ADK installs.
    LiteLLMClient = None
    LiteLlm = None


if LiteLlm is not None:

    class SerializableLiteLlm(LiteLlm):
        """LiteLlm variant that ADK Web can include in its graph response."""

        llm_client: LiteLLMClient = Field(default_factory=LiteLLMClient, exclude=True)
else:
    SerializableLiteLlm = None

_workspace_mcp = os.path.expanduser(
    os.environ.get("WORKSPACE_MCP_PATH", "~/.gemini/extensions/google-workspace")
)

mcp_tools = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="node",
                args=["dist/index.js"],
                env={
                    **os.environ,
                    "GEMINI_CLI_WORKSPACE_FORCE_FILE_STORAGE": "true",
                },
                cwd=_workspace_mcp,
            ),
        ),
    ),
]

_LAST_TOOL_SUMMARY_KEY = "rocky:last_tool_result_summary"
_PENDING_JOB_NOTIFICATION_IDS_KEY = "rocky:pending_job_notification_ids"
_PENDING_JOB_NOTIFICATION_TEXT_KEY = "rocky:pending_job_notification_text"


def _assistant_content(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def _summarize_tool_result(tool_name: str, result: dict) -> str:
    status = result.get("status")
    response = str(result.get("response") or result.get("error") or "").strip()
    if len(response) > 500:
        response = f"{response[:500]}..."
    if response:
        return f"{tool_name} returned {status or 'a result'}: {response}"
    return f"{tool_name} returned {status or 'a result'}."


async def _remember_tool_result(tool, args, tool_context, tool_response):
    if isinstance(tool_response, dict):
        tool_context.state[_LAST_TOOL_SUMMARY_KEY] = _summarize_tool_result(
            tool.name, tool_response
        )
    return None


async def _remember_tool_error(tool, args, tool_context, error):
    tool_context.state[_LAST_TOOL_SUMMARY_KEY] = (
        f"{tool.name} failed: {type(error).__name__}: {error}"
    )
    return None


async def _recover_model_error(callback_context, llm_request, error):
    # Defensive net for NVIDIA NIM via LiteLLM. With LongRunningFunctionTool the
    # function_response is recorded immediately, so this should rarely fire.
    error_text = str(error)
    if "Unexpected role 'user' after role 'tool'" not in error_text:
        return None
    return LlmResponse(
        content=_assistant_content(
            "I hit a stale tool-turn ordering issue. Start a fresh turn now and "
            "I can continue normally."
        )
    )


def _format_job_notifications(notifications: list[dict]) -> str:
    lines = []
    for notification in notifications:
        response = str(notification.get("response_preview") or "").strip()
        if len(response) > 240:
            response = f"{response[:240]}..."
        detail = f"job {notification['job_id']} finished with {notification['status']}"
        if response:
            detail = f"{detail}: {response}"
        lines.append(f"- {detail}")
    return "\n".join(lines)


async def _inject_job_notifications(callback_context, llm_request):
    notifications = peek_gemini_cli_job_notifications()
    if not notifications:
        return None

    job_ids = [str(notification["job_id"]) for notification in notifications]
    callback_context.state[_PENDING_JOB_NOTIFICATION_IDS_KEY] = job_ids
    callback_context.state[_PENDING_JOB_NOTIFICATION_TEXT_KEY] = (
        _format_job_notifications(notifications)
    )
    llm_request.append_instructions(
        [
            "Pending Gemini CLI job notifications for this turn:\n"
            f"{_format_job_notifications(notifications)}\n\n"
            "Your next response must be plain text, not a tool call and not a "
            "transfer. Briefly mention these completed background jobs first, "
            "including each job id, then answer the user's current message if "
            "you can do so directly. If the current message needs tools, say "
            "you can continue with it on the next turn after this status note."
        ]
    )
    return None


async def _mark_injected_job_notifications_reported(callback_context, llm_response):
    job_ids = callback_context.state.get(_PENDING_JOB_NOTIFICATION_IDS_KEY)
    if not job_ids:
        return None
    notification_text = callback_context.state.get(_PENDING_JOB_NOTIFICATION_TEXT_KEY)
    if not notification_text:
        return None

    content = getattr(llm_response, "content", None)
    text = "\n".join(
        part.text or "" for part in getattr(content, "parts", []) if part.text
    )
    reported_ids = [job_id for job_id in job_ids if job_id in text]
    if reported_ids:
        mark_gemini_cli_job_notifications_reported(reported_ids)
        remaining_ids = [job_id for job_id in job_ids if job_id not in reported_ids]
        callback_context.state[_PENDING_JOB_NOTIFICATION_IDS_KEY] = remaining_ids
        if not remaining_ids:
            callback_context.state[_PENDING_JOB_NOTIFICATION_TEXT_KEY] = ""
        return None

    mark_gemini_cli_job_notifications_reported(job_ids)
    callback_context.state[_PENDING_JOB_NOTIFICATION_IDS_KEY] = []
    callback_context.state[_PENDING_JOB_NOTIFICATION_TEXT_KEY] = ""
    if text.strip():
        return LlmResponse(
            content=_assistant_content(
                f"Background job update:\n{notification_text}\n\n{text.strip()}"
            )
        )
    return LlmResponse(
        content=_assistant_content(
            "Background job update:\n"
            f"{notification_text}\n\n"
            "I can continue with your current request on the next turn."
        )
    )


async def _save_session_to_memory(callback_context):
    invocation_context = getattr(callback_context, "_invocation_context", None)
    if invocation_context is None or invocation_context.memory_service is None:
        return None

    await invocation_context.memory_service.add_session_to_memory(
        invocation_context.session
    )
    return None


def _root_model():
    candidates = _root_model_candidates()
    model = candidates[0]
    _configure_model_environment(candidates)

    if SerializableLiteLlm is None:
        return Gemini(model=_first_gemini_candidate(candidates))

    if model.startswith("gemini/") and len(candidates) == 1:
        return Gemini(model=model.removeprefix("gemini/"))
    lite_llm_kwargs: dict[str, object] = {
        "fallbacks": list(candidates[1:]),
        "drop_params": True,
    }
    return SerializableLiteLlm(model=model, **lite_llm_kwargs)


def _has_env(*names: str) -> bool:
    return any(os.environ.get(name) for name in names)


def _split_model_env(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(model.strip() for model in value.split(",") if model.strip())


def _dedupe(models: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(models))


def _root_model_candidates() -> tuple[str, ...]:
    explicit_model = os.environ.get("HERMES_ROOT_MODEL")
    explicit_fallbacks = _split_model_env(os.environ.get("HERMES_ROOT_FALLBACK_MODELS"))
    if explicit_model:
        return _dedupe([explicit_model, *explicit_fallbacks])

    candidates: list[str] = []
    if _has_env("NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"):
        candidates.extend(f"nvidia_nim/{model}" for model in NVIDIA_TOOL_CALL_MODELS)

    if _has_env("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"):
        candidates.extend(f"gemini/{model}" for model in GOOGLE_TOOL_CALL_MODELS)

    if _has_env("OPENROUTER_API_KEY"):
        candidates.extend(OPENROUTER_LAST_RESORT_MODELS)

    if not candidates:
        candidates.append(
            f"gemini/{os.environ.get('HERMES_GEMINI_FALLBACK_MODEL', 'gemini-2.5-flash')}"
        )
    return _dedupe(candidates)


def _first_gemini_candidate(candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate.startswith("gemini/"):
            return candidate.removeprefix("gemini/")
    return os.environ.get("HERMES_GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")


def _configure_model_environment(candidates: tuple[str, ...]) -> None:
    if any(model.startswith("nvidia_nim/") for model in candidates):
        os.environ.setdefault(
            "NVIDIA_NIM_API_BASE",
            "https://integrate.api.nvidia.com/v1",
        )
        if "NVIDIA_NIM_API_KEY" not in os.environ and "NVIDIA_API_KEY" in os.environ:
            os.environ["NVIDIA_NIM_API_KEY"] = os.environ["NVIDIA_API_KEY"]
    if any(model.startswith("gemini/") for model in candidates):
        if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" in os.environ:
            os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def _search_model():
    return Gemini(model=os.environ.get("HERMES_SEARCH_MODEL", "gemini-flash-latest"))


GEMINI_CLI_WORKFLOW = """You have exactly five Gemini CLI job tools. Use these names verbatim, never invent variations:

- `run_gemini_cli` — starts a background worker, returns immediately with
  {status:'running', job_id:'...'}.
- `check_gemini_cli_job` — non-blocking single poll. Returns the final result
  or {status:'running'} immediately.
- `list_gemini_cli_jobs` — lists active/recent jobs without blocking.
- `cancel_gemini_cli_job` — best-effort cancellation for a specific job_id.
- `collect_gemini_cli_job_notifications` — returns newly finished jobs that
    have not yet been reported in chat.

Workflow:

0. For status/check/cancel requests, call `list_gemini_cli_jobs` or
   `check_gemini_cli_job` instead of starting a duplicate job.
1. Call `run_gemini_cli` with your task. Note the job_id.
2. Call `check_gemini_cli_job(job_id=...)` once.
3. If status='ok' or 'error', summarize the result and end the turn.
4. If status='running', reply with a short text like
   "Still working — ask me again in a bit." Then end the turn. Do not call
   `check_gemini_cli_job` again in the same turn. The user will prompt you
   to check again.
5. Never call `run_gemini_cli` a second time for the same task.
"""


LOCAL_OPERATOR_PROMPT = f"""You are `local_operator`, Rocky's specialist for local machine, repository, terminal, and code work.
You keep the root Rocky agent responsive by never blocking on long coding work.

{GEMINI_CLI_WORKFLOW}

## Execution Bias
If the user gives an actionable request, act in this turn. Do not end with a
plan or generic instructions when a tool can move the task forward.

Mutable facts need live checks: files, git state, clocks, versions, services,
processes, installed packages, RAM, disk, OS state, and Workspace data. Check
them with tools before answering.

Keep going until the task is done or genuinely blocked. If a tool result is weak
or incomplete, try a better prompt, path, command, or source before concluding.

## Local Machine
For any request about this laptop, terminal, operating system, hardware, RAM,
disk, installed software, current directory, files, repositories, shell commands,
logs, tests, or code, call run_gemini_cli before answering.

Never tell the user to run a local command themselves when Gemini CLI can inspect
it. Never say you lack access to local system metrics while run_gemini_cli is
available.

Omit working_directory unless the user gives a specific path; the tool defaults
to the project root. For read-only local inspection, including RAM, disk, and
process checks, use approval_mode='yolo' and ask Gemini CLI to run only read-only
commands. For code edits use approval_mode='auto_edit'. For read-only code
investigation use approval_mode='plan'.

## Workspace
Treat the project root as the default workspace. For repo-specific work, ask
Gemini CLI to inspect relevant project guidance files such as AGENTS.md,
CLAUDE.md, README files, and local config before making changes.

Use Hermes-style context priority for project guidance: prefer .hermes.md or
HERMES.md when present, then AGENTS.md, then CLAUDE.md, then Cursor rules. Treat
persona/profile/memory files as context layers, not as replacements for higher
priority system and safety rules.
"""


WORKSPACE_OPERATOR_PROMPT = """You are `workspace_operator`, Rocky's specialist for Google Workspace tasks.

Use the available Workspace MCP tools for Gmail, Calendar, Docs, Sheets, Slides,
Chat, and related Workspace data/actions. Do not guess about Workspace data when
a tool can check it.

## Google Workspace
When the user asks about Gmail, Calendar, Docs, Sheets, Slides, Chat, or other
Workspace data/actions, use the Workspace tools instead of guessing. For write
operations, preview the intended change and ask for confirmation first.
"""


MEMORY_KEEPER_PROMPT = """You are `memory_keeper`, Rocky's specialist for stable user and project context.

You maintain a Claw/Hermes-style markdown wiki using these tools:

- `remember_memory` stores durable, non-secret facts under a topic.
- `recall_memory` searches stored wiki entries.
- `list_memory_topics` shows available wiki pages.
- `forget_memory` removes entries that match a phrase.

Use these tools for stable, high-signal facts: user preferences, project
locations, active goals, recurring workflows, and decisions. Do not store
secrets, credentials, API keys, access tokens, or short-lived scratch details.
When a user says "remember", "recall", "what do you know", or asks about stable
context, handle it directly through the wiki tools.
"""


SEARCH_OPERATOR_PROMPT = """You are `search_operator`, Rocky's specialist for web search and current public facts.

Use the built-in `google_search` grounding tool when the user asks about recent
events, current information, public documentation, product pages, releases,
prices, comparisons, or anything likely to have changed after model training.

Prefer concise answers with source attribution. If a question can be answered
from stable memory or local project files, do not use web search; Rocky should
route those tasks to memory_keeper or local_operator instead.
"""


ROCKY_SYSTEM_PROMPT = """You are Rocky, a Google ADK orchestrator inspired by Max, Hermes Agent, OpenClaw, and Hermes-style local agents.
You are the fast supervisor/router, not the heavy worker.

## Specialist agents
Delegate instead of doing everything yourself:

- `local_operator` handles local machine, terminal, repo, code, tests, processes,
  installed software, disk/RAM, and Gemini CLI worker jobs.
- `workspace_operator` handles Google Workspace tasks through MCP.
- `memory_keeper` handles Claw-style markdown memory, recall, and stable
  user/project context.
- `search_operator` handles web search and current public facts using built-in
  Google Search grounding.

Keep your own turns responsive. For long-running coding or inspection work,
transfer to `local_operator`; it will start a Gemini CLI background job and
return a job id instead of blocking.

At the start of each new user turn, call
`collect_gemini_cli_job_notifications` once. If any previous Gemini CLI jobs
finished, mention those results briefly first, then continue with the user's
new topic. This is the OpenClaw-style "tell me when the old job finished while
I kept chatting" behavior. If there are no notifications, continue normally.

## Safety
You have no independent goals. Do not pursue self-preservation, replication,
resource acquisition, persistence, or access expansion. Do not disable
safeguards or change tool policies unless the user explicitly asks and the
change is safe.

For destructive, irreversible, credential-related, or external write actions,
pause and ask for confirmation. For ordinary read-only inspection, act.

## Replies
Summarize tool results clearly and briefly. Include the important evidence from
the tool output. If a tool fails, report the useful error details and the next
best path.
"""


local_operator = Agent(
    model=_root_model(),
    name="local_operator",
    description="Handles local machine, terminal, repository, and coding work via Gemini CLI worker jobs.",
    instruction=LOCAL_OPERATOR_PROMPT,
    tools=[
        gemini_cli_tool,
        gemini_cli_check_tool,
        gemini_cli_list_tool,
        gemini_cli_cancel_tool,
        gemini_cli_notifications_tool,
    ],
    on_model_error_callback=_recover_model_error,
    before_model_callback=_inject_job_notifications,
    after_model_callback=_mark_injected_job_notifications_reported,
    after_tool_callback=_remember_tool_result,
    on_tool_error_callback=_remember_tool_error,
    after_agent_callback=_save_session_to_memory,
)

workspace_operator = Agent(
    model=_root_model(),
    name="workspace_operator",
    description="Handles Google Workspace data and actions through MCP tools.",
    instruction=WORKSPACE_OPERATOR_PROMPT,
    tools=mcp_tools,
    on_model_error_callback=_recover_model_error,
    before_model_callback=_inject_job_notifications,
    after_model_callback=_mark_injected_job_notifications_reported,
    after_tool_callback=_remember_tool_result,
    on_tool_error_callback=_remember_tool_error,
    after_agent_callback=_save_session_to_memory,
)

memory_keeper = Agent(
    model=_root_model(),
    name="memory_keeper",
    description="Handles recall and stable user/project context.",
    instruction=MEMORY_KEEPER_PROMPT,
    tools=[
        PreloadMemoryTool(),
        remember_memory_tool,
        recall_memory_tool,
        list_memory_topics_tool,
        forget_memory_tool,
    ],
    on_model_error_callback=_recover_model_error,
    before_model_callback=_inject_job_notifications,
    after_model_callback=_mark_injected_job_notifications_reported,
    after_tool_callback=_remember_tool_result,
    on_tool_error_callback=_remember_tool_error,
    after_agent_callback=_save_session_to_memory,
)

search_operator = Agent(
    model=_search_model(),
    name="search_operator",
    description="Handles web search and current public facts through built-in Google Search grounding.",
    instruction=SEARCH_OPERATOR_PROMPT,
    tools=[google_search],
    on_model_error_callback=_recover_model_error,
    before_model_callback=_inject_job_notifications,
    after_model_callback=_mark_injected_job_notifications_reported,
    after_tool_callback=_remember_tool_result,
    on_tool_error_callback=_remember_tool_error,
    after_agent_callback=_save_session_to_memory,
)

root_agent = Agent(
    model=_root_model(),
    name="rocky",
    description="Rocky: a responsive ADK supervisor with specialist local, Workspace, memory, and search agents.",
    instruction=ROCKY_SYSTEM_PROMPT,
    tools=[PreloadMemoryTool(), gemini_cli_notifications_tool],
    sub_agents=[local_operator, workspace_operator, memory_keeper, search_operator],
    on_model_error_callback=_recover_model_error,
    before_model_callback=_inject_job_notifications,
    after_model_callback=_mark_injected_job_notifications_reported,
    after_tool_callback=_remember_tool_result,
    on_tool_error_callback=_remember_tool_error,
    after_agent_callback=_save_session_to_memory,
)

app = App(
    name="myagent",
    root_agent=root_agent,
)
