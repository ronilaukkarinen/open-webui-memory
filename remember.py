"""
title: Remember
author: Roni Laukkarinen (originally mhio and cooksleep)
version: 2.2.0
license: MIT

Memory retrieval tool designed to work with auto_memory_retrieval_and_storage.py.

This tool provides AI with the ability to actively recall and use stored memories
to enhance responses. It focuses purely on memory retrieval and does not handle
memory storage, modification, or deletion - those functions are handled by the
auto_memory_retrieval_and_storage.py filter.

Key features:
- Proactive memory recall for enhanced AI responses
- Retrieval of relevant stored information
- Integration with OpenWebUI's native memory system

This tool is designed to be used alongside auto_memory_retrieval_and_storage.py
which handles automatic memory creation, storage, and management.

NOTE: This tool only handles memory retrieval. All memory storage, updates,
and deletions are handled by auto_memory_retrieval_and_storage.py to prevent
conflicts and ensure consistency.
"""

import json
import aiohttp
from typing import Callable, Any, List
from open_webui.models.memories import Memories, MemoryModel
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown state", status="in_progress", done=False):
        """
        Send a status event to the event emitter.

        :param description: Event description
        :param status: Event status
        :param done: Whether the event is complete
        """
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    """
    Memory Recall Tool

    Use this tool to retrieve and recall stored memories to enhance AI responses.
    This tool works alongside auto_memory_retrieval_and_storage.py which handles
    automatic memory creation, storage, updates, and deletions.

    Key features:
    1. Memory recall: Retrieve stored memories to enhance responses with personal context
    2. Proactive memory usage: Don't wait for users to ask - actively check memories
    3. Contextual enhancement: Use retrieved memories to provide personalized responses

    IMPORTANT:
    - This tool ONLY retrieves memories, it does NOT store, update, or delete them
    - Memory storage/management is handled by auto_memory_retrieval_and_storage.py
    - Always check memories proactively to enhance your responses
    - Use memories to provide personalized, contextual responses

    If users ask about memory management (add/update/delete), explain that this
    is handled automatically by the system and they don't need to manually manage memories.
    """

    class Valves(BaseModel):
        USE_MEMORY: bool = Field(
            default=True, description="Enable or disable memory usage."
        )
        openai_api_url: str = Field(
            default="http://localhost:11434/v1",
            description="OpenAI API endpoint for memory analysis",
        )
        openai_api_key: str = Field(
            default="", description="API key for memory analysis"
        )
        model: str = Field(
            default="qwen2.5:7b",
            description="Model to use for memory relevance analysis",
        )
        DEBUG: bool = Field(default=True, description="Enable or disable debug mode.")

    def __init__(self):
        """Initialize the memory management tool."""
        self.valves = self.Valves()

    async def recall_memories(
        self, __user__: dict = None, __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Retrieves all stored memories from the user's memory vault.

        IMPORTANT: Proactively check memories to enhance your responses!
        Don't wait for users to ask what you remember.

        Returns memories in chronological order with index numbers.
        Use when you need to check stored information, reference previous
        preferences, or build context for responses.

        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter for tracking status
        :return: JSON string with indexed memories list
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        await emitter.emit(
            description="Retrieving stored memories.",
            status="recall_in_progress",
            done=False,
        )

        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memory stored."
            await emitter.emit(description=message, status="recall_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        content_list = [
            f"{index}. {memory.content}"
            for index, memory in enumerate(
                sorted(user_memories, key=lambda m: m.created_at), start=1
            )
        ]

        await emitter.emit(
            description=f"{len(user_memories)} memories loaded",
            status="recall_complete",
            done=True,
        )

        return f"Memories from the users memory vault: {content_list}"

    async def get_relevant_memories(
        self,
        query: str,
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Retrieve memories relevant to the current query using AI analysis.

        This function automatically finds and returns memories that are contextually
        relevant to the user's query, providing smart memory retrieval for enhanced responses.

        :param query: The user's current message/query
        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter for tracking status
        :return: String with relevant memories or empty string if none found
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            await emitter.emit(description="User ID not provided", status="error", done=True)
            return ""

        user_id = __user__.get("id")
        if not user_id:
            await emitter.emit(description="User ID not provided", status="error", done=True)
            return ""

        await emitter.emit(description="Searching for relevant memories...", status="in_progress", done=False)

        try:
            # Get all memories for the user
            existing_memories = Memories.get_memories_by_user_id(user_id=str(user_id))
            if not existing_memories:
                await emitter.emit(description="No memories found", status="complete", done=True)
                return ""

            # Convert to searchable format
            memory_contents = []
            for mem in existing_memories:
                if isinstance(mem, MemoryModel):
                    memory_contents.append(f"[Id: {mem.id}, Content: {mem.content}]")
                elif hasattr(mem, "content"):
                    memory_contents.append(f"[Id: {mem.id}, Content: {mem.content}]")

            if not memory_contents:
                await emitter.emit(description="No valid memories found", status="complete", done=True)
                return ""

            await emitter.emit(description=f"Analyzing {len(memory_contents)} memories for relevance...", status="in_progress", done=False)

            # Use AI to find relevant memories
            relevant_memories = await self._analyze_memory_relevance(query, memory_contents)

            if relevant_memories:
                # Clean up the memory format for response
                cleaned_memories = []
                for mem in relevant_memories:
                    if "Content:" in mem:
                        content = mem.split("Content:", 1)[1].rstrip("]").strip()
                        cleaned_memories.append(content)
                    else:
                        cleaned_memories.append(mem)

                result = "Relevant memories:\n" + "\n".join(f"- {mem}" for mem in cleaned_memories)
                await emitter.emit(description=f"Found {len(relevant_memories)} relevant memories", status="complete", done=True)
                return result
            else:
                await emitter.emit(description="No relevant memories found", status="complete", done=True)
                return ""

        except Exception as e:
            await emitter.emit(description=f"Error retrieving memories: {str(e)}", status="error", done=True)
            return ""

    async def _analyze_memory_relevance(self, query: str, memory_contents: List[str]) -> List[str]:
        """Analyze which memories are relevant to the query using AI."""
        try:
            # Create prompt for memory relevance analysis
            memory_prompt = f"""RESPOND ONLY WITH VALID JSON ARRAY. NO TEXT BEFORE OR AFTER.

User query: "{query}"
Available memories: {memory_contents}

Analyze which memories could help answer this query. Use VERY BROAD semantic understanding:

KEY PRINCIPLES:
1. BE EXTREMELY INCLUSIVE - if there's ANY possible connection, include it
2. Think beyond exact word matches - use conceptual relationships
3. Consider what the user is really asking for, not just the literal words
4. Rate generously - err on the side of including too much rather than too little

RATING SCALE:
- 8-10: Directly answers the query or highly relevant
- 5-7: Related information that provides useful context
- 3-4: Somewhat related, might be helpful
- 1-2: Minimal connection but could be useful

BE GENEROUS: If you're unsure, rate it higher rather than lower.

Return ONLY the JSON array:
[{{"memory": "complete memory string exactly as provided", "relevance": number}}]"""

            response = await self._query_openai_api(memory_prompt)

            try:
                memory_ratings = json.loads(response.strip())
                # Use low threshold to be inclusive
                threshold = 2

                relevant_memories = [
                    item["memory"]
                    for item in sorted(memory_ratings, key=lambda x: x["relevance"], reverse=True)
                    if item["relevance"] >= threshold
                ]

                return relevant_memories

            except json.JSONDecodeError:
                # Fallback: return some memories if JSON parsing fails
                return memory_contents[:5] if memory_contents else []

        except Exception:
            # Fallback: return some memories if AI analysis fails
            return memory_contents[:5] if memory_contents else []

    async def _query_openai_api(self, prompt: str) -> str:
        """Query the OpenAI API for memory analysis."""
        url = f"{self.valves.openai_api_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.openai_api_key}",
        }
        payload = {
            "model": self.valves.model,
            "messages": [
                {"role": "system", "content": "You are a JSON-only assistant. Return ONLY valid JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
        }

        async with aiohttp.ClientSession() as session:
            response = await session.post(url, headers=headers, json=payload)
            response.raise_for_status()
            json_content = await response.json()
            return str(json_content["choices"][0]["message"]["content"])
