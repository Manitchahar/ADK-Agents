# ADK-Agents

Minimal Google ADK agent example.

## Structure

- `myagent/agent.py`: defines the root ADK agent.
- `myagent/__init__.py`: package entrypoint.

## Current Agent

The repo currently exposes a single `root_agent` with:

- model: `gemma-4-31b-it` via `google.adk.models.Gemini`
- name: `root_agent`
- a general-purpose prompt that answers directly or uses Workspace tools
  depending on the query
- an exported ADK `App` wrapper for app-level runtime features
- memory preload plus automatic session-to-memory ingestion when a memory
	backend is configured
- default generation config compatible with Gemma on the Gemini API path
- Google Workspace MCP forced to encrypted file storage so spawned tool
	processes do not depend on desktop keychain access

## Run

Install the dependencies required by Google ADK in your environment, then load the agent package from `myagent`.

Local `.env` files and ADK runtime state are intentionally ignored and should not be committed.

## Notes

- `llms.txt` is included as reference material for ADK documentation links.
