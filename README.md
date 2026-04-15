# ADK-Agents

Minimal Google ADK agent example.

## Structure

- `myagent/agent.py`: defines the root ADK agent.
- `myagent/__init__.py`: package entrypoint.

## Current Agent

The repo currently exposes a single `root_agent` with:

- model: `gemma-4-31b-it`
- name: `root_agent`
- minimal thinking enabled through `GenerateContentConfig`

## Run

Install the dependencies required by Google ADK in your environment, then load the agent package from `myagent`.

## Notes

- `llms.txt` is included as reference material for ADK documentation links.
