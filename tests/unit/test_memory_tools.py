# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Rocky's markdown memory tools."""

import pytest

from myagent.tools import memory


@pytest.fixture
def rocky_memory_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCKY_MEMORY_DIR", str(tmp_path / "wiki"))
    return tmp_path / "wiki"


@pytest.mark.asyncio
async def test_remember_and_recall_memory(rocky_memory_dir):
    remembered = await memory.remember_memory(
        topic="Projects / Rocky",
        fact="Rocky should feel like a Claw-style persistent agent.",
    )

    assert remembered["status"] == "ok"
    assert remembered["topic"] == "projects-rocky"
    assert (rocky_memory_dir / "projects-rocky.md").exists()

    recalled = await memory.recall_memory(query="claw persistent")

    assert recalled["status"] == "ok"
    assert recalled["count"] == 1
    assert "Claw-style" in recalled["memories"][0]["text"]


@pytest.mark.asyncio
async def test_memory_refuses_secret_like_facts(rocky_memory_dir):
    result = await memory.remember_memory(
        topic="credentials",
        fact="The API token is sk-test-123.",
    )

    assert result["status"] == "error"
    assert not rocky_memory_dir.exists()


@pytest.mark.asyncio
async def test_list_and_forget_memory(rocky_memory_dir):
    await memory.remember_memory(topic="User", fact="User likes concise demos.")
    await memory.remember_memory(topic="User", fact="User wants LinkedIn polish.")

    topics = await memory.list_memory_topics()
    assert topics["status"] == "ok"
    assert topics["topics"][0]["topic"] == "user"
    assert topics["topics"][0]["entries"] == 2

    forgotten = await memory.forget_memory(topic="User", phrase="LinkedIn")
    assert forgotten["status"] == "ok"
    assert forgotten["removed"] == 1

    recalled = await memory.recall_memory(query="LinkedIn")
    assert recalled["count"] == 0
