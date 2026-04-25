"""Model fallback ordering for Rocky's ADK orchestration models."""

GOOGLE_TOOL_CALL_MODELS = (
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
)

GOOGLE_CHAT_FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-flash-latest",
)

GOOGLE_WORKING_MODELS = GOOGLE_TOOL_CALL_MODELS + GOOGLE_CHAT_FALLBACK_MODELS

OPENROUTER_LAST_RESORT_MODELS = (
    "openrouter/moonshotai/kimi-k2",
    "openrouter/qwen/qwen3-coder",
    "openrouter/openai/gpt-oss-120b",
    "openrouter/free",
)

# Ordered for multi-agent/tool orchestration quality. The top three are the
# preferred NIM stack for Rocky:
# 1. Qwen 3.5 397B: strongest general orchestrator/reasoner in the local list.
# 2. Nemotron 3 Super 120B: NVIDIA-native, reliable NIM fallback.
# 3. Kimi K2: fast, strong interactive agent fallback.
#
# Mistral Small remains available but is intentionally no longer first.
NVIDIA_TOOL_CALL_MODELS = (
    "qwen/qwen3.5-397b-a17b",
    "nvidia/nemotron-3-super-120b-a12b",
    "moonshotai/kimi-k2-instruct-0905",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "moonshotai/kimi-k2-thinking",
    "qwen/qwen3.5-122b-a10b",
    "mistralai/devstral-2-123b-instruct-2512",
    "z-ai/glm-5.1",
    "deepseek-ai/deepseek-v3.1-terminus",
    "moonshotai/kimi-k2-instruct",
    "z-ai/glm5",
    "openai/gpt-oss-120b",
    "stepfun-ai/step-3.5-flash",
    "minimaxai/minimax-m2.5",
    "minimaxai/minimax-m2.7",
    "mistralai/mistral-small-4-119b-2603",
)

# Working chat model, but it did not emit a tool_call in the default probe.
NVIDIA_CHAT_FALLBACK_MODELS = ("z-ai/glm4.7",)

NVIDIA_WORKING_MODELS = NVIDIA_TOOL_CALL_MODELS + NVIDIA_CHAT_FALLBACK_MODELS

TOOL_CALL_MODELS = GOOGLE_TOOL_CALL_MODELS + NVIDIA_TOOL_CALL_MODELS
WORKING_MODELS = (
    GOOGLE_WORKING_MODELS + NVIDIA_WORKING_MODELS + OPENROUTER_LAST_RESORT_MODELS
)
