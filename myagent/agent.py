import os

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

_workspace_mcp = os.path.expanduser(
    os.environ.get('WORKSPACE_MCP_PATH', '~/.gemini/extensions/google-workspace')
)

mcp_tools = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="node",
                args=["dist/index.js", "--use-dot-names"],
                env={
                    **os.environ,
                    "GEMINI_CLI_WORKSPACE_FORCE_FILE_STORAGE": "true",
                },
                cwd=_workspace_mcp,
            ),
        ),
    ),
]


async def _save_session_to_memory(callback_context):
    invocation_context = getattr(callback_context, '_invocation_context', None)
    if invocation_context is None or invocation_context.memory_service is None:
        return None

    await invocation_context.memory_service.add_session_to_memory(
        invocation_context.session
    )
    return None

root_agent = Agent(
    model=Gemini(model='gemma-4-31b-it'),
    name='root_agent',
    description='A helpful assistant that can use Google Workspace when needed.',
    instruction=(
        'You are a helpful general-purpose assistant with access to Google '
        'Workspace tools (Gmail, Calendar, Docs, Sheets, Slides, Chat). '
        'Answer general questions directly when no tool is needed. When the '
        'user asks about their Workspace data or wants you to act in '
        'Workspace, use the appropriate tools instead of guessing. For '
        'repeat interactions, use any available memory context to avoid asking '
        'the user to restate stable preferences or known facts. For '
        'time-sensitive or identity-dependent Workspace tasks, identify the '
        'user with people.getMe() and resolve dates with time.getCurrentDate() '
        'or time.getTimeZone() when needed. Keep calling tools until you can '
        'give a complete natural-language answer or finish the requested task; '
        'do not stop after only naming a tool. Preview proposed changes and '
        'ask for confirmation before executing write operations.'
    ),
    tools=[PreloadMemoryTool(), *mcp_tools],
    after_agent_callback=_save_session_to_memory,
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level ='minimal',
    
            include_thoughts=False,
        ),
    ),
)

app = App(
    name='myagent',
    root_agent=root_agent,
)
