"""Unit tests for Rocky model selection."""

from google.adk.models import Gemini

from myagent import agent as agent_mod

_MODEL_ENV_VARS = (
    "HERMES_ROOT_MODEL",
    "HERMES_ROOT_FALLBACK_MODELS",
    "HERMES_GEMINI_FALLBACK_MODEL",
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "NVIDIA_NIM_API_BASE",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_CLOUD_PROJECT",
    "OPENROUTER_API_KEY",
)


def _clear_model_env(monkeypatch):
    for name in _MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_root_model_falls_back_to_gemini_without_litellm(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setattr(agent_mod, "SerializableLiteLlm", None)

    model = agent_mod._root_model()

    assert isinstance(model, Gemini)
    assert model.model == "gemini-2.5-flash"


def test_root_model_gemini_fallback_strips_litellm_prefix(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setattr(agent_mod, "SerializableLiteLlm", None)
    monkeypatch.setenv("HERMES_ROOT_MODEL", "gemini/gemini-2.5-flash")

    model = agent_mod._root_model()

    assert isinstance(model, Gemini)
    assert model.model == "gemini-2.5-flash"


def test_root_model_candidates_default_to_native_gemini(monkeypatch):
    _clear_model_env(monkeypatch)

    assert agent_mod._root_model_candidates() == ("gemini/gemini-2.5-flash",)


def test_root_model_candidates_prioritize_strong_nvidia_stack(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    candidates = agent_mod._root_model_candidates()

    assert candidates[:3] == (
        "nvidia_nim/qwen/qwen3.5-397b-a17b",
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        "nvidia_nim/moonshotai/kimi-k2-instruct-0905",
    )
    assert candidates.index("nvidia_nim/mistralai/mistral-small-4-119b-2603") > 10
    assert candidates.index("gemini/gemini-2.5-pro") > candidates.index(
        "nvidia_nim/mistralai/mistral-small-4-119b-2603"
    )
    assert candidates[-4:] == (
        "openrouter/moonshotai/kimi-k2",
        "openrouter/qwen/qwen3-coder",
        "openrouter/openai/gpt-oss-120b",
        "openrouter/free",
    )


def test_root_model_candidates_honor_explicit_override(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("HERMES_ROOT_MODEL", "gemini/gemini-2.5-pro")
    monkeypatch.setenv(
        "HERMES_ROOT_FALLBACK_MODELS",
        "nvidia_nim/qwen/qwen3.5-397b-a17b,openrouter/moonshotai/kimi-k2",
    )

    assert agent_mod._root_model_candidates() == (
        "gemini/gemini-2.5-pro",
        "nvidia_nim/qwen/qwen3.5-397b-a17b",
        "openrouter/moonshotai/kimi-k2",
    )


def test_root_model_passes_litellm_fallbacks(monkeypatch):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    model = agent_mod._root_model()

    if agent_mod.SerializableLiteLlm is None:
        assert isinstance(model, Gemini)
        assert model.model == "gemini-2.5-pro"
        return

    assert isinstance(model, agent_mod.SerializableLiteLlm)
    assert model.model == "nvidia_nim/qwen/qwen3.5-397b-a17b"
    assert model._additional_args["fallbacks"][0] == (
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    )
    assert "gemini/gemini-2.5-pro" in model._additional_args["fallbacks"]
    assert "openrouter/moonshotai/kimi-k2" in model._additional_args["fallbacks"]
    assert model._additional_args["drop_params"] is True


def test_search_model_uses_gemini_for_google_search(monkeypatch):
    monkeypatch.delenv("HERMES_SEARCH_MODEL", raising=False)

    model = agent_mod._search_model()

    assert isinstance(model, Gemini)
    assert model.model == "gemini-flash-latest"


def test_root_agent_uses_specialist_subagents():
    sub_agents = {
        sub_agent.name: sub_agent for sub_agent in agent_mod.root_agent.sub_agents
    }

    assert {
        "local_operator",
        "workspace_operator",
        "memory_keeper",
        "search_operator",
    } <= set(sub_agents)

    root_tool_names = {
        getattr(tool, "name", type(tool).__name__)
        for tool in agent_mod.root_agent.tools
    }
    local_tool_names = {
        getattr(tool, "name", type(tool).__name__)
        for tool in sub_agents["local_operator"].tools
    }
    memory_tool_names = {
        getattr(tool, "name", type(tool).__name__)
        for tool in sub_agents["memory_keeper"].tools
    }
    search_tool_names = {
        getattr(tool, "name", type(tool).__name__)
        for tool in sub_agents["search_operator"].tools
    }

    assert "run_gemini_cli" not in root_tool_names
    assert "google_search" not in root_tool_names
    assert {
        "run_gemini_cli",
        "check_gemini_cli_job",
        "list_gemini_cli_jobs",
        "cancel_gemini_cli_job",
        "collect_gemini_cli_job_notifications",
    } <= local_tool_names
    assert {
        "remember_memory",
        "recall_memory",
        "list_memory_topics",
        "forget_memory",
    } <= memory_tool_names
    assert {"google_search"} <= search_tool_names
