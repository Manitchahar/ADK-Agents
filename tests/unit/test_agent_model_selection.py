"""Unit tests for Rocky model selection."""

from google.adk.models import Gemini

from myagent import agent as agent_mod


def test_root_model_falls_back_to_gemini_without_litellm(monkeypatch):
    monkeypatch.setattr(agent_mod, "SerializableLiteLlm", None)
    monkeypatch.delenv("HERMES_ROOT_MODEL", raising=False)
    monkeypatch.delenv("HERMES_GEMINI_FALLBACK_MODEL", raising=False)

    model = agent_mod._root_model()

    assert isinstance(model, Gemini)
    assert model.model == "gemma-4-31b-it"


def test_root_model_gemini_fallback_strips_litellm_prefix(monkeypatch):
    monkeypatch.setattr(agent_mod, "SerializableLiteLlm", None)
    monkeypatch.setenv("HERMES_ROOT_MODEL", "gemini/gemini-2.5-flash")

    model = agent_mod._root_model()

    assert isinstance(model, Gemini)
    assert model.model == "gemini-2.5-flash"


def test_root_agent_uses_specialist_subagents():
    sub_agents = {
        sub_agent.name: sub_agent for sub_agent in agent_mod.root_agent.sub_agents
    }

    assert {"local_operator", "workspace_operator", "memory_keeper"} <= set(sub_agents)

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

    assert "run_gemini_cli" not in root_tool_names
    assert {
        "run_gemini_cli",
        "check_gemini_cli_job",
        "list_gemini_cli_jobs",
        "cancel_gemini_cli_job",
    } <= local_tool_names
    assert {
        "remember_memory",
        "recall_memory",
        "list_memory_topics",
        "forget_memory",
    } <= memory_tool_names
