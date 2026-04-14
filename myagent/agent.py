from google.adk.agents.llm_agent import Agent
from google.genai import types

root_agent = Agent(
    model='gemma-4-31b-it',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level='minimal',
            include_thoughts=False,
        )
    ),
)
