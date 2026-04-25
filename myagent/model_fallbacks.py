"""Model fallback ordering for OpenAI-compatible NVIDIA NIM chat models."""

GOOGLE_TOOL_CALL_MODELS = (
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
)

GOOGLE_CHAT_FALLBACK_MODELS = (
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
)

GOOGLE_WORKING_MODELS = GOOGLE_TOOL_CALL_MODELS + GOOGLE_CHAT_FALLBACK_MODELS

OPENROUTER_LAST_RESORT_MODELS = ("openrouter/free",)

# Ordered by direct default tool-call probe results. These models returned
# OpenAI-style tool_calls when sent only model, messages, and tools.
NVIDIA_TOOL_CALL_MODELS = (
    "mistralai/mistral-small-4-119b-2603",
    "qwen/qwen3.5-122b-a10b",
    "mistralai/devstral-2-123b-instruct-2512",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "stepfun-ai/step-3.5-flash",
    "minimaxai/minimax-m2.5",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "z-ai/glm-5.1",
    "deepseek-ai/deepseek-v3.1-terminus",
    "moonshotai/kimi-k2-instruct",
    "z-ai/glm5",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.5-397b-a17b",
    "moonshotai/kimi-k2-thinking",
    "minimaxai/minimax-m2.7",
)

# Working chat model, but it did not emit a tool_call in the default probe.
NVIDIA_CHAT_FALLBACK_MODELS = ("z-ai/glm4.7",)

NVIDIA_WORKING_MODELS = NVIDIA_TOOL_CALL_MODELS + NVIDIA_CHAT_FALLBACK_MODELS

TOOL_CALL_MODELS = GOOGLE_TOOL_CALL_MODELS + NVIDIA_TOOL_CALL_MODELS
WORKING_MODELS = (
    GOOGLE_WORKING_MODELS + NVIDIA_WORKING_MODELS + OPENROUTER_LAST_RESORT_MODELS
)
