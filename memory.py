"""
title: Memory
author: Roni Laukkarinen
description: Automatically identify, retrieve and store memories.
repository_url: https://github.com/ronilaukkarinen/open-webui-memory
version: 3.2.4
required_open_webui_version: >= 0.5.0
"""

import ast
import asyncio
import json
import time
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any

import aiohttp
from aiohttp import ClientError
from fastapi.requests import Request
from pydantic import BaseModel, Field

from open_webui.main import app as webui_app
from open_webui.models.users import Users, UserModel
from open_webui.routers.memories import (
    add_memory,
    AddMemoryForm,
    delete_memory_by_id,
    query_memory,
    QueryMemoryForm,
    get_memories,
)

STRINGIFIED_MESSAGE_TEMPLATE = "-{index}. {role}: ```{content}```"

IDENTIFY_MEMORIES_PROMPT = """\
You are helping maintain a collection of the User's Memories—like individual "journal entries," each automatically timestamped upon creation or update.
You will be provided with the last 2 or more messages from a conversation. Your job is to decide which details within the last User message (-2) are worth saving long-term as Memory entries.

** Key Instructions **
1. **HIGHEST PRIORITY - EXPLICIT REMEMBER REQUESTS**: If the User explicitly requests to "remember" or note down something in their latest message (-2), ALWAYS include it regardless of any other rules. This overrides all filtering rules below.
2. **CRITICAL - ONLY PROCESS MESSAGE (-2)**: Identify new or changed personal details from the User's **latest** message (-2) ONLY. You MUST completely IGNORE all older user messages (-3, -4, -5, etc.) even if they contain interesting information. Older user messages are provided ONLY for context to understand what the user is referring to in their latest message (-2). Do NOT extract memories from any message other than (-2).
2b. IMPORTANT: If the User's message (-2) is asking about existing memories (e.g., "What do you know about me?", "What are my preferences?", "Tell me about myself"), and the Assistant's response (-1) is just summarizing existing information, return an empty list `[]`. Do NOT store the Assistant's summary as new memories.
3. If the User's newest message contradicts an older statement (e.g., message -4 says "I love oranges" vs. message -2 says "I hate oranges"), extract only the updated info ("User hates oranges").
4. Think of each Memory as a single "fact" or statement. Never combine multiple facts into one Memory. If the User mentions multiple distinct items, break them into separate entries.
5. Your goal is to capture anything that might be valuable for the "assistant" to remember about the User, to personalize and enrich future interactions.
5b. CRITICAL: Do NOT extract memories from Assistant responses that are clearly just summarizing or listing existing knowledge about the user. Only extract from genuine new information provided by the User.
6. Avoid storing short-term situational details or temporary actions (e.g. user: "I'm reading this question right now", user: "I just woke up!", user: "Oh yeah, I saw that on TV the other day"). However, DO capture personal preferences, interests, opinions, and persistent facts about the user (e.g. "I like berries", "I enjoy hiking", "I prefer tea over coffee", "I work in marketing").
6b. CRITICAL: Do NOT store memories that only describe what the user is asking for help with in the current conversation. These are temporary interactions, not personal facts. Examples of what NOT to store: "User asked about configuration", "User is asking how to set up X", "User wants help with debugging", "User requested information about Y", "User has a problem with Z", "User needs assistance with W". However, DO store if they mention personal context like "User is learning piano" or "User is working on a React project".
6c. CRITICAL: Do NOT store assistant responses or explanations as memories about the user. Only store facts that the USER themselves provides about their personal life, preferences, work, interests, or persistent situations.
7. If the user writes in another language, translate the memory content to English while preserving the original meaning.
8. Return your result as a Python list of strings, **each string representing a separate Memory**. If no relevant info is found, **only** return an empty list (`[]`). No explanations, just the list. Do NOT wrap your response in markdown code blocks or any other formatting - return the raw Python list only.

---

### Examples

**Example 1 - 4 messages**
-4. user: ```I love oranges 😍```
-3. assistant: ```That's great! 🍊 I love oranges too!```
-2. user: ```Actually, I hate oranges 😂```
-1. assistant: ```omg you LIAR 😡```

**Analysis**
- The last user message states a new personal fact: "User hates oranges."
- This replaces the older statement about loving oranges.

**Correct Output**
["User hates oranges"]

**Example 2 - 2 messages**
-2. user: ```I work as a junior data analyst. Please remember that my big presentation is on March 15.```
-1. assistant: ```Got it! I'll make a note of that.```

**Analysis**
- The user provides two new pieces of information: their profession and the date of their presentation.

**Correct Output**
["User works as a junior data analyst", "User has a big presentation on March 15"]

**Example 3 - 5 messages**
-5. assistant: ```Nutella is amazing! 😍```
-4. user: ```Soo, remember how a week ago I had bought a new TV?```
-3. assistant: ```Yes, I remember that. What about it?```
-2. user: ```well, today it broke down 😭```
-1. assistant: ```Oh no! That's terrible!```

**Analysis**
- The only relevant message is the last User message (-2), which provides new information about the TV breaking down.
- The previous messages (-3, -4) provide context over what the user was talking about.
- The remaining message (-5) is irrelevant.

**Correct Output**
["User's TV they bought a week ago broke down today"]

**Example 4 - 3 messages**
-3. assistant: ```As an AI assistant, I can perform extremely complex calculations in seconds.```
-2. user: ```Oh yeah? I can do that with my eyes closed!```
-1. assistant: ```😂 Sure you can, Joe!```

**Analysis**
- The User message (-2) is clearly sarcastic and not meant to be taken literally. It does not contain any relevant information to store.
- The other messages (-3, -1) are not relevant as they're not about the User.

**Correct Output**
[]

**Example 5 - Simple Preference**
-2. user: ```I like berries```
-1. assistant: ```That's great! Berries are delicious and healthy. Do you have a favorite type of berry?```

**Analysis**
- The User (-2) is expressing a personal preference about food.
- This is valuable personal information that should be remembered for future interactions.
- Personal preferences like food likes/dislikes are important to capture.

**Correct Output**
["User likes berries"]

**Example 6 - Memory Summary Request**
-2. user: ```What do you know about me?```
-1. assistant: ```Based on our conversations, here's what I know: You enjoy sci-fi movies, work as a software engineer, prefer coffee over tea, and live in Seattle. You also mentioned liking hiking and having a dog named Max.```

**Analysis**
- The User (-2) is asking for a summary of existing memories, not providing new information.
- The Assistant (-1) is just reciting back previously stored information.
- This is NOT new information about the user - it's just a summary of existing knowledge.

**Correct Output**
[]

**Example 7 - Help Request (NOT to store)**
-2. user: ```I'm having trouble with my Python code. Can you help me debug this function?```
-1. assistant: ```I'd be happy to help you debug your Python code! Please share the function you're having trouble with.```

**Analysis**
- The User (-2) is asking for help with debugging, which is a temporary interaction.
- This is NOT personal information about the user - it's just a request for assistance.
- We should NOT store "User is having trouble with Python code" or "User asked for debugging help".

**Correct Output**
[]

**Example 8 - IGNORING OLDER MESSAGES (CRITICAL)**
-4. user: ```I love The Midnight band and I make synthwave music under the alias Streetgazer. I've also watched over 5000 movies.```
-3. assistant: ```That's amazing! The Midnight is fantastic, and your movie count is impressive.```
-2. user: ```Have you seen Stranger Things? It's one of my favorite shows.```
-1. assistant: ```Yes! Stranger Things is excellent. Given your love for synthwave and sci-fi, it's perfect for you.```

**Analysis**
- Message (-4) contains lots of valuable information about the user's music interests, alias, and movie count.
- However, we MUST ONLY process message (-2), which only mentions Stranger Things being a favorite show.
- We MUST IGNORE all the information in message (-4) even though it's valuable personal information.
- Only extract from the latest user message (-2).

**Correct Output**
["User has seen Stranger Things, which is one of their favorite shows"]\
"""

CONSOLIDATE_MEMORIES_PROMPT = """You are maintaining a set of "Memories" for a user, similar to journal entries. Each memory has:
- A "fact" (a string describing something about the user or a user-related event).
- A "created_at" timestamp (an integer or float representing when it was stored/updated).

**What You're Doing**
1. You're given a list of such Memories that the system believes might be related or overlapping.
2. Your goal is to produce a cleaned-up list of final facts, making sure we:
   - Only combine Memories if they are exact duplicates or direct conflicts about the same topic.
   - In case of duplicates, keep only the one with the latest (most recent) `created_at`.
   - In case of a direct conflict (e.g., the user's favorite color stated two different ways), keep only the most recent one.
   - If Memories are partially similar but not truly duplicates or direct conflicts, preserve them both. We do NOT want to lose details or unify "User likes oranges" and "User likes ripe oranges" into a single statement—those remain separate.
3. Return the final list as a simple Python list of strings—**each string is one separate memory/fact**—with no extra commentary.

**Remember**
- This is a journaling system meant to give the user a clear, time-based record of who they are and what they've done.
- We do not want to clump multiple distinct pieces of info into one memory.
- We do not throw out older facts unless they are direct duplicates or in conflict with a newer statement.
- If there is a conflict (e.g., "User's favorite color is red" vs. "User's favorite color is teal"), keep the more recent memory only.

---

### **Extended Example**

Below is an example list of 15 "Memories." Notice the variety of scenarios:
- Potential duplicates
- Partial overlaps
- Direct conflicts
- Ephemeral/past events

**Input** (a JSON-like array):

```
[
  {"fact": "User visited Paris for a business trip", "created_at": 1631000000},
  {"fact": "User visited Paris for a personal trip with their girlfriend", "created_at": 1631500000},
  {"fact": "User visited Paris for a personal trip with their girlfriend", "created_at": 1631600000},
  {"fact": "User works as a junior data analyst", "created_at": 1633000000},
  {"fact": "User's meeting with the project team is scheduled for Friday at 10 AM", "created_at": 1634000000},
  {"fact": "User's meeting with the project team is scheduled for Friday at 11 AM", "created_at": 1634050000},
  {"fact": "User likes to eat oranges", "created_at": 1635000000},
  {"fact": "User likes to eat ripe oranges", "created_at": 1635100000},
  {"fact": "User used to like red color, but not anymore", "created_at": 1635200000},
  {"fact": "User's favorite color is teal", "created_at": 1635500000},
  {"fact": "User's favorite color is red", "created_at": 1636000000},
  {"fact": "User traveled to Japan last year", "created_at": 1637000000},
  {"fact": "User traveled to Japan this month", "created_at": 1637100000},
  {"fact": "User also works part-time as a painter", "created_at": 1637200000},
  {"fact": "User had a dentist appointment last Tuesday", "created_at": 1637300000}
]
```

**Analysis**:
1. **Paris trips**
   - "User visited Paris for a personal trip with their girlfriend" appears **twice** (`created_at`: 1631500000 and 1631600000). They are exact duplicates but have different timestamps, so we keep only the most recent. The business trip is different, so keep it too.

2. **Meeting time**
   - There's a direct conflict about the meeting time (10 AM vs 11 AM). We keep the more recent statement.

3. **Likes oranges / ripe oranges**
   - These are partially similar, but not exactly the same or in conflict, so we keep both.

4. **Color**
   - We have "User used to like red," "User's favorite color is teal," and "User's favorite color is red."
   - The statement "User used to like red color, but not anymore" is not actually a direct conflict with "favorite color is teal." We keep them both.
   - The newest color memory is "User's favorite color is red" (timestamp 1636000000) which conflicts with the older "User's favorite color is teal" (timestamp 1635500000). We keep the more recent red statement.

5. **Japan**
   - "User traveled to Japan last year" vs "User traveled to Japan this month." They're not contradictory; one is old, one is new. Keep them both.

6. **Past events**
   - Dentist appointment is ephemeral, but we keep it since each memory is a separate time-based journal entry.

**Correct Output** (the final consolidated list of facts as strings):

```
[
  "User visited Paris for a business trip",
  "User visited Paris for a personal trip with their girlfriend",  <-- keep only the most recent from duplicates
  "User works as a junior data analyst",
  "User's meeting with the project team is scheduled for Friday at 11 AM",
  "User likes to eat oranges",
  "User likes to eat ripe oranges",
  "User used to like red color, but not anymore",
  "User's favorite color is red",  <-- overrides teal
  "User traveled to Japan last year",
  "User traveled to Japan this month",
  "User also works part-time as a painter",
  "User had a dentist appointment last Tuesday"
]
```

Make sure your final answer is just the array, with no added commentary.

---

### **Final Reminder**
- You're only seeing these Memories because our system guessed they might overlap. If they're not exact duplicates or direct conflicts, keep them all.
- Always produce a **Python list of strings**—each string is a separate memory/fact.
- Do not add any explanation or disclaimers—just the final list.\
"""




class Filter:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="https://api.openai.com",
            description="openai compatible endpoint",
        )
        model: str = Field(
            default="gpt-4o",
            description="Model to use to determine memory. An intelligent model is highly recommended, as it will be able to better understand the context of the conversation.",
        )
        api_key: str = Field(
            default="", description="API key for OpenAI compatible endpoint"
        )
        priority: int = Field(default=15, description="Priority level")
        related_memories_n: int = Field(
            default=5,
            description="Number of related memories to consider when updating memories",
        )
        related_memories_dist: float = Field(
            default=0.8,
            description="Distance of memories to consider for updates. Smaller number will be more closely related.",
        )
        save_assistant_response: bool = Field(
            default=False,
            description="Automatically save assistant responses as memories",
        )
        simplified_output: bool = Field(
            default=True,
            description="Show simplified 'Memory updated' message instead of detailed memory content.",
        )
        excluded_models: str = Field(
            default="",
            description="Comma-separated list of model names to exclude from memory processing. Use lowercase with hyphens (e.g., 'english-refiner,translator,obfuscator')"
        )
        model_specific_settings: str = Field(
            default='{"character_name": {"openai_api_url": "http://localhost:11434", "api_key": "ollama", "model": "qwen2.5:7b"}}',
            description='JSON object with per-model API settings. Format: {"character_name": {"openai_api_url": "http://localhost:11434", "api_key": "ollama", "model": "qwen2.5:7b"}}.'
        )
        disable_for_image_generation: bool = Field(
            default=True,
            description="Disable memory injection for image generation requests to prevent interference with image prompts"
        )

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        openai_api_url: Optional[str] = Field(
            default=None,
            description="User-specific openai compatible endpoint (overrides global)",
        )
        model: Optional[str] = Field(
            default=None,
            description="User-specific model to use (overrides global). An intelligent model is highly recommended, as it will be able to better understand the context of the conversation.",
        )
        api_key: Optional[str] = Field(
            default=None, description="User-specific API key (overrides global)"
        )
        messages_to_consider: int = Field(
            default=4,
            description="Number of messages to consider for memory processing, starting from the last message. Includes assistant responses.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"inlet:{__name__}")
        print(f"inlet:user:{__user__}")

        # Check if current model should be excluded
        if self.valves.excluded_models:
            if self._should_exclude_model(body, self.valves.excluded_models):
                print("MEMORY FILTER: Skipping memory processing for excluded model")
                return body

        # Check if image generation is disabled and this is an image generation request
        if self.valves.disable_for_image_generation:
            if self._is_image_generation_request(body):
                print("MEMORY FILTER: Skipping memory injection for image generation request")
                return body
            else:
                # Debug logging to help understand request structure
                print(f"MEMORY DEBUG: Request body keys: {list(body.keys())}")
                if 'messages' in body and len(body['messages']) > 0:
                    last_message = body['messages'][-1]
                    print(f"MEMORY DEBUG: Last message keys: {list(last_message.keys()) if isinstance(last_message, dict) else 'Not a dict'}")
                    if isinstance(last_message, dict) and 'content' in last_message:
                        content = last_message['content']
                        print(f"MEMORY DEBUG: Content type: {type(content)}, Content preview: {str(content)[:100] if isinstance(content, str) else content}")
                print(f"MEMORY DEBUG: Model: {body.get('model', 'Not found')}")
                print(f"MEMORY DEBUG: Tools: {body.get('tools', 'Not found')}")
                print(f"MEMORY DEBUG: Images: {body.get('images', 'Not found')}")
                print(f"MEMORY DEBUG: Backend: {body.get('backend', 'Not found')}")
                print(f"MEMORY DEBUG: Response format: {body.get('response_format', 'Not found')}")
                print(f"MEMORY DEBUG: Image generation: {body.get('image_generation', 'Not found')}")

        # Always inject all memories for context
        if __user__:
            try:
                user = Users.get_user_by_id(__user__["id"])
                memories = await self.get_all_memories(user)
                if memories:
                    self.inject_memories_into_conversation(body, memories)
            except Exception as e:
                print(f"Error retrieving memories: {e}")

        return body

    async def get_all_memories(self, user) -> list:
        """Retrieve ALL memories for the user with timestamps for context."""
        try:
            # Get all memories for the user
            memories_result = await get_memories(user=user)

            memories_with_time = []
            if memories_result and hasattr(memories_result, 'data'):
                for memory in memories_result.data:
                    # Format timestamp as human-readable date
                    created_at = getattr(memory, 'created_at', time.time())
                    if isinstance(created_at, (int, float)):
                        formatted_time = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                    else:
                        # Handle string timestamps or other formats
                        try:
                            formatted_time = datetime.fromisoformat(str(created_at).replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
                        except:
                            formatted_time = str(created_at)[:16]  # Fallback

                    memory_with_time = f"{memory.content} (on {formatted_time})"
                    memories_with_time.append(memory_with_time)

                print(f"Retrieved {len(memories_with_time)} memories for user")
                return memories_with_time
            elif isinstance(memories_result, list):
                for memory in memories_result:
                    if hasattr(memory, 'content'):
                        # Format timestamp as human-readable date
                        created_at = getattr(memory, 'created_at', time.time())
                        if isinstance(created_at, (int, float)):
                            formatted_time = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                        else:
                            # Handle string timestamps or other formats
                            try:
                                formatted_time = datetime.fromisoformat(str(created_at).replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
                            except:
                                formatted_time = str(created_at)[:16]  # Fallback

                        memory_with_time = f"{memory.content} (on {formatted_time})"
                        memories_with_time.append(memory_with_time)

                print(f"Retrieved {len(memories_with_time)} memories for user")
                return memories_with_time
            return []
        except Exception as e:
            print(f"Error getting memories: {e}")
            return []

    def inject_memories_into_conversation(self, body: dict, memories: list):
        """Inject all memories into the conversation with clear AI instructions."""
        if not memories or not body.get("messages"):
            return

        # Create memory context with clear instructions for the AI
        memory_context = f"""<MEMORY_CONTEXT>
You have access to the user's personal memories below. These are facts about the user that may be relevant to your conversation.

IMPORTANT INSTRUCTIONS:
- Only reference memories when they are directly relevant to the current conversation
- Do not list, enumerate, or mention memories unless specifically asked
- Do not say things like "I remember you mentioned..." or "Based on your memories..."
- Use the information naturally as context to provide better, more personalized responses
- If memories are not relevant to the current topic, ignore them completely

USER MEMORIES:
{chr(10).join(f"- {memory}" for memory in memories)}
</MEMORY_CONTEXT>

"""

        # Add memory context to the first user message if it exists
        if body["messages"]:
            # Find the first user message
            for i, message in enumerate(body["messages"]):
                if message.get("role") == "user":
                    original_content = message.get("content", "")
                    # Only inject if not already present
                    if not original_content.startswith("<MEMORY_CONTEXT>"):
                        message["content"] = memory_context + original_content
                        print(f"Injected {len(memories)} memories into conversation")
                    else:
                        print(f"Memory context already present, skipping injection")
                    break

    def _is_image_generation_request(self, body: dict) -> bool:
        """
        Detect if the current request is for image generation.
        Based on Open WebUI's official patterns for image generation requests.
        """
        if not body:
            return False
        
        # Method 1: Check for image generation endpoint in URL (if available)
        # This would be checked in the calling context if URL is available
        
        # Method 2: Check for image generation specific parameters
        image_params = ['size', 'n', 'response_format']
        has_image_params = any(param in body for param in image_params)
        
        # Method 3: Check for prompt-only structure (common in image generation)
        has_prompt_only = 'prompt' in body and len([k for k in body.keys() if k not in ['prompt', 'model']]) <= 2
        
        # Method 4: Check metadata flags
        metadata = body.get('metadata', {})
        metadata_flags = metadata.get('image_generation') or metadata.get('generate_image')
        
        # Method 5: Check features and options
        features = body.get('features', {})
        options = body.get('options', {})
        feature_flags = features.get('image_generation') or options.get('generate_image')
        
        # Method 6: Check for direct image generation flags
        direct_flags = body.get('image_generation', False) or body.get('generate_image', False)
        
        # Method 7: Check for image generation models
        model_check = any(model in body.get("model", "").lower() for model in ["dall-e", "stable-diffusion", "imagen", "midjourney"])
        
        # Method 8: Check for ComfyUI or other image generation backends
        backend_check = body.get("backend") in ["comfyui", "automatic1111", "stable-diffusion"]
        
        # Combine all detection methods
        is_image_request = (
            has_image_params or
            (has_prompt_only and 'prompt' in body) or
            metadata_flags or
            feature_flags or
            direct_flags or
            model_check or
            backend_check
        )
        
        if is_image_request:
            detection_methods = []
            if has_image_params: detection_methods.append("image_params")
            if has_prompt_only and 'prompt' in body: detection_methods.append("prompt_only")
            if metadata_flags: detection_methods.append("metadata_flags")
            if feature_flags: detection_methods.append("feature_flags")
            if direct_flags: detection_methods.append("direct_flags")
            if model_check: detection_methods.append("model_check")
            if backend_check: detection_methods.append("backend_check")
            
            print(f"MEMORY FILTER: Detected image generation request via: {', '.join(detection_methods)}")
            
        return is_image_request

    def _should_exclude_model(self, body: dict, excluded_models: str) -> bool:
        """
        Check if the current model should be excluded from memory processing.
        Supports multiple model identifier fields for robust filtering.
        """
        if not excluded_models:
            return False

        # Parse excluded models list
        excluded_models_list = [model.strip().strip('"\'') for model in excluded_models.split(",")]

        # Check multiple possible model identifier fields
        current_model = body.get("model", "")
        model_id = body.get("model_id", "")

        # Check for OpenWebUI model/character names in nested structures
        model_name = ""
        model_title = ""

        # Look for model info in nested chat structure
        if "chat" in body and isinstance(body["chat"], dict):
            if "models" in body["chat"] and isinstance(body["chat"]["models"], list):
                for model_info in body["chat"]["models"]:
                    if isinstance(model_info, dict):
                        if "name" in model_info:
                            model_name = model_info.get("name", "")
                        if "title" in model_info:
                            model_title = model_info.get("title", "")

        # Also check direct model info
        if "model_info" in body and isinstance(body["model_info"], dict):
            model_name = body["model_info"].get("name", model_name)
            model_title = body["model_info"].get("title", model_title)

        # Debug logging
        print(f"MEMORY FILTER DEBUG: Checking exclusion for request")
        print(f"MEMORY FILTER DEBUG: body.model: '{current_model}'")
        print(f"MEMORY FILTER DEBUG: body.model_id: '{model_id}'")
        print(f"MEMORY FILTER DEBUG: model_name from nested: '{model_name}'")
        print(f"MEMORY FILTER DEBUG: model_title from nested: '{model_title}'")
        print(f"MEMORY FILTER DEBUG: Available body keys: {list(body.keys())}")
        print(f"MEMORY FILTER DEBUG: Excluded models list: {excluded_models_list}")

        # Check all possible model identifiers against exclusion list
        models_to_check = [current_model, model_id, model_name, model_title]
        for model_identifier in models_to_check:
            if model_identifier and model_identifier in excluded_models_list:
                print(f"MEMORY FILTER: Excluding model: {model_identifier}")
                return True

        print(f"MEMORY FILTER DEBUG: No exclusion match found, proceeding with memory processing")
        return False

    def _get_current_model_name(self, body: dict) -> str:
        """
        Extract the current Open WebUI model name from the request body.
        Returns empty string if no model name is found.
        """
        # Check multiple possible model identifier fields
        current_model = body.get("model", "")

        # Check for OpenWebUI model/character names in nested structures
        model_name = ""
        model_title = ""

        # Look for model info in nested chat structure
        if "chat" in body and isinstance(body["chat"], dict):
            if "models" in body["chat"] and isinstance(body["chat"]["models"], list):
                for model_info in body["chat"]["models"]:
                    if isinstance(model_info, dict):
                        if "name" in model_info:
                            model_name = model_info.get("name", "")
                        if "title" in model_info:
                            model_title = model_info.get("title", "")

        # Also check direct model info
        if "model_info" in body and isinstance(body["model_info"], dict):
            model_name = body["model_info"].get("name", model_name)
            model_title = body["model_info"].get("title", model_title)

        # Priority: model_name > model_title > current_model
        for candidate in [model_name, model_title, current_model]:
            if candidate:
                return candidate

        return ""

    def _get_model_specific_settings(self, model_name: str) -> dict:
        """
        Parse model-specific settings and return settings for the specified model.
        Returns empty dict if no specific settings found or if parsing fails.
        """
        if not self.valves.model_specific_settings or not model_name:
            return {}

        try:
            settings_dict = json.loads(self.valves.model_specific_settings)
            return settings_dict.get(model_name, {})
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing model_specific_settings: {e}")
            return {}

    def _get_effective_settings(self, body: dict) -> tuple[str, str, str]:
        """
        Get effective API settings (URL, model, key) for the current request.
        Uses model-specific settings if available, otherwise falls back to global/user settings.
        Returns tuple of (api_url, model, api_key)
        """
        # Get current model name
        current_model_name = self._get_current_model_name(body)
        print(f"MEMORY DEBUG: Detected model name: '{current_model_name}'")

        # Get model-specific settings
        model_settings = self._get_model_specific_settings(current_model_name)
        print(f"MEMORY DEBUG: Model-specific settings found: {bool(model_settings)}")
        if model_settings:
            print(f"MEMORY DEBUG: Model settings: {model_settings}")

        # Start with global/user settings as fallback
        api_url = self.user_valves.openai_api_url or self.valves.openai_api_url
        model = self.user_valves.model or self.valves.model
        api_key = self.user_valves.api_key or self.valves.api_key

        # Override with model-specific settings if available
        if model_settings:
            api_url = model_settings.get("openai_api_url", api_url)
            model = model_settings.get("model", model)
            api_key = model_settings.get("api_key", api_key)
            print(f"MEMORY DEBUG: Using model-specific settings for '{current_model_name}': url={api_url}, LLM model={model}")
        else:
            print(f"MEMORY DEBUG: Using global settings for model '{current_model_name}' (no specific settings found): LLM model={model}")

        return api_url, model, api_key

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        user = Users.get_user_by_id(__user__["id"])
        self.user_valves: Filter.UserValves = __user__.get("valves", self.UserValves())

        # Check if current model should be excluded from memory processing
        if self.valves.excluded_models:
            if self._should_exclude_model(body, self.valves.excluded_models):
                print("MEMORY FILTER: Skipping memory processing for excluded model")
                return body

        # Check if image generation is disabled and this is an image generation request
        if self.valves.disable_for_image_generation:
            if self._is_image_generation_request(body):
                print("MEMORY FILTER: Skipping memory processing for image generation request")
                return body

        # Process user message for memories
        if len(body["messages"]) >= 2:
            stringified_messages = []
            for i in range(1, self.user_valves.messages_to_consider + 1):
                try:
                    # Check if we have enough messages to safely access this index
                    if i <= len(body["messages"]):
                        message = body["messages"][-i]
                        content = message["content"]
                        
                        # Skip assistant messages unless the valve is enabled
                        if message["role"] == "assistant" and not self.valves.save_assistant_response:
                            continue
                        
                        # Remove memory context if it was injected from any user message
                        if message["role"] == "user" and "<MEMORY_CONTEXT>" in content:
                            # More robust removal - handle both start and embedded contexts
                            if content.startswith("<MEMORY_CONTEXT>"):
                                content = content.split("</MEMORY_CONTEXT>\n\n", 1)[-1]
                            else:
                                # Handle cases where context might be embedded
                                import re
                                content = re.sub(r'<MEMORY_CONTEXT>.*?</MEMORY_CONTEXT>\n\n', '', content, flags=re.DOTALL)
                        stringified_message = STRINGIFIED_MESSAGE_TEMPLATE.format(
                            index=i,
                            role=message["role"],
                            content=content,
                        )
                        stringified_messages.append(stringified_message)
                    else:
                        break
                except Exception as e:
                    print(f"Error stringifying messages: {e}")
            prompt_string = "\n".join(stringified_messages)
            try:
                print(f"MEMORY DEBUG: About to call identify_memories with prompt length: {len(prompt_string)}")
                memories = await self.identify_memories(prompt_string, body)
                print(f"MEMORY DEBUG: identify_memories returned: '{memories}'")
            except Exception as e:
                # Show error notification for memory identification failures
                error_msg = str(e)
                await __event_emitter__(
                    {
                        "type": "notification",
                        "data": {
                            "type": "error",
                            "content": f"Memory identification failed: {error_msg}",
                        },
                    }
                )
                print(f"MEMORY ERROR: Memory identification failed: {error_msg}")
                return body

            # Clean and normalize the response
            memories = memories.strip()

            # Remove markdown code block wrapper if present (fallback parsing)
            if memories.startswith("```") and memories.endswith("```"):
                memories = memories[3:-3].strip()
                print("MEMORY DEBUG: Cleaned markdown code blocks from response")

            # Additional cleanup for code blocks that might contain language tags
            if memories.startswith("```python") and memories.endswith("```"):
                memories = memories[9:-3].strip()
                print("MEMORY DEBUG: Cleaned python code blocks from response")
            elif memories.startswith("```json") and memories.endswith("```"):
                memories = memories[7:-3].strip()
                print("MEMORY DEBUG: Cleaned json code blocks from response")

            # Strip any remaining code block markers
            memories = memories.replace("```", "").strip()

            # Remove common prefixes that might be added by AI
            prefixes_to_remove = [
                "**Correct Output**",
                "**Output**",
                "**Response**",
                "**Result**",
                "Output:",
                "Response:",
                "Result:"
            ]
            for prefix in prefixes_to_remove:
                if memories.startswith(prefix):
                    memories = memories[len(prefix):].strip()
                    print(f"MEMORY DEBUG: Removed prefix '{prefix}' from response")

            if (
                memories.startswith("[")
                and memories.endswith("]")
                and len(memories) != 2
            ):
                try:
                    result = await self.process_memories(memories, user, body, __user__)
                    # Only show success notification if user wants status updates
                    if self.user_valves.show_status and result:
                        # Status message
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "Memory updated",
                                    "done": True,
                                },
                            }
                        )

                        # Notification
                        if self.valves.simplified_output:
                            await __event_emitter__(
                                {
                                    "type": "notification",
                                    "data": {
                                        "type": "success",
                                        "content": "Memory stored successfully",
                                    },
                                }
                            )
                        else:
                            # Parse the memories list to create a detailed notification
                            try:
                                # Safely parse memories for notification counting
                                memories_cleaned = memories.strip()
                                
                                # Remove common AI response prefixes
                                prefixes_to_remove = ["**Correct Output**", "**Output**", "**Response**", "**Result**", "Output:", "Response:", "Result:"]
                                for prefix in prefixes_to_remove:
                                    if memories_cleaned.startswith(prefix):
                                        memories_cleaned = memories_cleaned[len(prefix):].strip()
                                
                                memory_list = ast.literal_eval(memories_cleaned)
                                memory_count = len(memory_list)
                                await __event_emitter__(
                                    {
                                        "type": "notification",
                                        "data": {
                                            "type": "success",
                                            "content": f"Stored {memory_count} new memor{'ies' if memory_count != 1 else 'y'}",
                                        },
                                    }
                                )
                            except Exception as parse_error:
                                # Fallback notification
                                await __event_emitter__(
                                    {
                                        "type": "notification",
                                        "data": {
                                            "type": "success",
                                            "content": "Memory stored successfully",
                                        },
                                    }
                                )
                except Exception as e:
                    # Always show error notifications regardless of user settings
                    error_msg = str(e)
                    await __event_emitter__(
                        {
                            "type": "notification",
                            "data": {
                                "type": "error",
                                "content": f"Memory storage failed: {error_msg}",
                            },
                        }
                    )
                    print(f"MEMORY ERROR: Memory processing failed: {error_msg}")
            else:
                print("Auto Memory: no new memories identified")

        # Process assistant response if auto-save is enabled
        print(f"ASSISTANT DEBUG: save_assistant_response valve = {self.valves.save_assistant_response}")
        if self.valves.save_assistant_response and len(body["messages"]) > 0:
            last_message = body["messages"][-1]
            print(f"ASSISTANT DEBUG: Last message role = {last_message.get('role', 'unknown')}")
            # Only save if the last message is actually from assistant
            if last_message.get("role") == "assistant":
                print(f"ASSISTANT DEBUG: Proceeding to save assistant response")
                try:
                    print(f"ASSISTANT DEBUG: Adding assistant memory: {last_message['content'][:100]}...")
                    memory_obj = await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=last_message["content"]),
                        user=user,
                    )
                    print(f"Assistant Memory Added: {memory_obj}")

                    if self.user_valves.show_status:
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {
                                    "type": "success",
                                    "content": "Assistant memory saved",
                                },
                            }
                        )
                except Exception as e:
                    print(f"Error adding assistant memory {str(e)}")

                    if self.user_valves.show_status:
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {
                                    "type": "error",
                                    "content": f"Error saving assistant memory: {str(e)}",
                                },
                            }
                        )
            else:
                print(f"ASSISTANT DEBUG: Skipping - not an assistant message")
        else:
            print(f"ASSISTANT DEBUG: Auto-save disabled or no messages")
        return body

    async def identify_memories(self, input_text: str, body: dict = None) -> str:
        memories = await self.query_openai_api(
            system_prompt=IDENTIFY_MEMORIES_PROMPT,
            prompt=input_text,
            body=body,
        )
        print(f"MEMORY DEBUG: LLM response for memory identification: '{memories}'")
        print(f"MEMORY DEBUG: Input text was: '{input_text}'")
        return memories

    async def query_openai_api(self, system_prompt: str, prompt: str, body: dict = None) -> str:

        # Use model-specific settings if available, otherwise use global/user values
        if body:
            api_url, model, api_key = self._get_effective_settings(body)
        else:
            # Fallback to original logic when body is not provided (backward compatibility)
            api_url = self.user_valves.openai_api_url or self.valves.openai_api_url
            model = self.user_valves.model or self.valves.model
            api_key = self.user_valves.api_key or self.valves.api_key

        url = f"{api_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        # Debug logging for API calls
        print(f"MEMORY API DEBUG: Making request to {url}")
        print(f"MEMORY API DEBUG: Using LLM model for memory identification: {model}")

        try:
            async with aiohttp.ClientSession() as session:
                print(f"MEMORY API DEBUG: Sending POST request...")
                response = await session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30))
                print(f"MEMORY API DEBUG: Response status: {response.status}")
                response.raise_for_status()
                
                # Get response text first for validation
                response_text = await response.text()
                print(f"MEMORY API DEBUG: Response text length: {len(response_text)}")
                
                # Validate response is not empty
                if not response_text or response_text.strip() == "":
                    print(f"MEMORY API ERROR: Empty response received")
                    raise Exception("Empty response from API")
                
                # Try to parse JSON with proper error handling
                try:
                    json_content = json.loads(response_text)
                    print(f"MEMORY API DEBUG: JSON parsed successfully")
                except json.JSONDecodeError as e:
                    print(f"MEMORY API ERROR: JSON decode error: {e}")
                    print(f"MEMORY API ERROR: Response text: {response_text[:500]}...")
                    raise Exception(f"Invalid JSON response: {str(e)}")
                
                # Validate expected structure
                if "choices" not in json_content or not json_content["choices"]:
                    print(f"MEMORY API ERROR: Missing choices in response: {json_content}")
                    raise Exception("Invalid API response structure - missing choices")
                
                if "message" not in json_content["choices"][0] or "content" not in json_content["choices"][0]["message"]:
                    print(f"MEMORY API ERROR: Missing message content in response")
                    raise Exception("Invalid API response structure - missing message content")
                
                return json_content["choices"][0]["message"]["content"]
                
        except aiohttp.ClientError as e:
            # Fixed error handling
            error_msg = str(e)
            print(f"MEMORY API ERROR: HTTP error: {error_msg}")
            raise Exception(f"Http error: {error_msg}")
        except asyncio.TimeoutError:
            print(f"MEMORY API ERROR: Request timeout")
            raise Exception("API request timeout after 30 seconds")
        except Exception as e:
            print(f"MEMORY API ERROR: Unexpected error: {str(e)}")
            raise Exception(f"API error: {str(e)}")

    async def process_memories(
        self,
        memories: str,
        user: UserModel,
        body: dict = None,
        __user__: Optional[dict] = None,
    ) -> bool:
        """Given a list of memories as a string, go through each memory, check for duplicates, then store the remaining memories."""
        try:
            # Safely parse the memories string with better error handling
            memories_cleaned = memories.strip()
            
            # Remove common AI response prefixes
            prefixes_to_remove = ["**Correct Output**", "**Output**", "**Response**", "**Result**", "Output:", "Response:", "Result:"]
            for prefix in prefixes_to_remove:
                if memories_cleaned.startswith(prefix):
                    memories_cleaned = memories_cleaned[len(prefix):].strip()
            
            # Validate the cleaned response looks like a list
            if not memories_cleaned.startswith("[") or not memories_cleaned.endswith("]"):
                print(f"Auto Memory: Response doesn't look like a list: {repr(memories_cleaned[:100])}")
                raise ValueError(f"AI response is not a valid list format: {memories_cleaned[:100]}...")

            # Try parsing with comprehensive error handling
            try:
                memory_list = ast.literal_eval(memories_cleaned)
                if not isinstance(memory_list, list):
                    print(f"Auto Memory: Parsed result is not a list: {type(memory_list)}")
                    raise ValueError(f"Expected list, got {type(memory_list)}")
            except (ValueError, SyntaxError) as e:
                print(f"Auto Memory: ast.literal_eval failed: {e}")
                print(f"Auto Memory: Failed to parse: {repr(memories_cleaned[:200])}")
                
                # Try JSON parsing as fallback
                try:
                    memory_list = json.loads(memories_cleaned)
                    if not isinstance(memory_list, list):
                        raise ValueError(f"Expected list, got {type(memory_list)}")
                    print(f"Auto Memory: Fallback JSON parsing succeeded")
                except json.JSONDecodeError as json_e:
                    print(f"Auto Memory: JSON fallback also failed: {json_e}")
                    raise Exception(f"Unable to parse AI response as list. AST error: {e}, JSON error: {json_e}")
            
            print(f"Auto Memory: identified {len(memory_list)} new memories")

            # Validate all items are strings
            for i, memory in enumerate(memory_list):
                if not isinstance(memory, str):
                    print(f"Auto Memory: Memory item {i} is not a string: {type(memory)}")
                    memory_list[i] = str(memory)

            # Pre-process to remove exact duplicates within the same batch
            unique_memories = []
            for memory in memory_list:
                memory_lower = memory.lower().strip()
                if not any(existing.lower().strip() == memory_lower for existing in unique_memories):
                    unique_memories.append(memory)

            if len(unique_memories) < len(memory_list):
                print(f"Auto Memory: removed {len(memory_list) - len(unique_memories)} exact duplicates from batch")

            # Instead of processing each memory individually, consolidate the entire batch
            # This handles cases where multiple memories in the same conversation contradict each other
            if len(unique_memories) > 1:
                # Check if any memories in this batch contradict each other
                try:
                    batch_consolidation_prompt = f"""
You have {len(unique_memories)} new memories from the same conversation that need to be consolidated:

{json.dumps([{"fact": memory, "created_at": time.time()} for memory in unique_memories])}

If any of these memories contradict each other (like "loves X" then "hates X"), keep only the LAST one mentioned (most recent in the conversation). If they're all compatible, keep them all.

Return the final list as a Python list of strings - just the memory text, no extra formatting.
"""
                    
                    if __user__:
                        self.user_valves = __user__.get("valves", self.UserValves())
                    
                    batch_result = await self.query_openai_api(
                        system_prompt="You are consolidating memories from a single conversation. Keep the most recent when there are contradictions.",
                        prompt=batch_consolidation_prompt,
                        body=body,
                    )
                    
                    # Safely parse batch consolidation result using the same safe parsing
                    batch_result_cleaned = batch_result.strip()
                    
                    # Remove common AI response prefixes
                    prefixes_to_remove = ["**Correct Output**", "**Output**", "**Response**", "**Result**", "Output:", "Response:", "Result:"]
                    for prefix in prefixes_to_remove:
                        if batch_result_cleaned.startswith(prefix):
                            batch_result_cleaned = batch_result_cleaned[len(prefix):].strip()
                    
                    # Parse batch consolidation result safely
                    try:
                        batch_memories = ast.literal_eval(batch_result_cleaned)
                        if not isinstance(batch_memories, list):
                            raise ValueError(f"Expected list, got {type(batch_memories)}")
                    except (ValueError, SyntaxError) as e:
                        # Try JSON fallback
                        try:
                            batch_memories = json.loads(batch_result_cleaned)
                            if not isinstance(batch_memories, list):
                                raise ValueError(f"Expected list, got {type(batch_memories)}")
                        except json.JSONDecodeError:
                            print(f"Auto Memory: Batch consolidation parsing failed: {e}")
                            # Fall back to individual processing
                            batch_memories = unique_memories
                    print(f"Auto Memory: Batch consolidation reduced {len(unique_memories)} memories to {len(batch_memories)}")
                    
                    # Now process the consolidated batch
                    for memory in batch_memories:
                        await self.store_memory(memory, user, body, __user__)
                    return True
                    
                except (ValueError, SyntaxError) as e:
                    if "string did not match the expected pattern" in str(e):
                        print(f"Auto Memory: Batch consolidation AI response parsing failed - malformed list format. Response was: {repr(batch_result[:200])}")
                    else:
                        print(f"Auto Memory: Batch consolidation parsing failed: {e}")
                    # Fall back to individual processing
                except Exception as e:
                    print(f"Auto Memory: Batch consolidation failed: {e}")
                    # Fall back to individual processing
            
            # Process individually if batch consolidation failed or only one memory
            for memory in unique_memories:
                await self.store_memory(memory, user, body, __user__)
            return True
        except Exception as e:
            print(f"Auto Memory: Unexpected error in process_memories: {e}")
            import traceback
            print(f"Auto Memory: Traceback: {traceback.format_exc()}")
            return False

    async def find_similar_memories_text(self, memory: str, user) -> list:
        """Find similar memories using text-based similarity as fallback when vector search fails."""
        try:
            print(f"Auto Memory: Starting text-based similarity search for '{memory}'")
            # Get all existing memories
            memories_result = await get_memories(user=user)
            print(f"Auto Memory: get_memories returned type: {type(memories_result)}")
            print(f"Auto Memory: get_memories result: {memories_result}")
            
            if not memories_result:
                print(f"Auto Memory: No existing memories found")
                return []
            
            # Handle different return formats
            if hasattr(memories_result, 'data'):
                existing_memories = memories_result.data
            elif isinstance(memories_result, list):
                existing_memories = memories_result
            else:
                print(f"Auto Memory: Invalid result format from get_memories")
                return []
            
            print(f"Auto Memory: Found {len(existing_memories)} existing memories to compare against")
            similar_memories = []
            memory_lower = memory.lower().strip()
            
            for existing_memory in existing_memories:
                existing_content = getattr(existing_memory, 'content', '').lower().strip()
                
                # Calculate simple text similarity
                similarity = self.calculate_text_similarity(memory_lower, existing_content)
                
                # Use the same distance threshold (convert similarity to distance)
                distance = 1.0 - similarity
                print(f"Auto Memory: Comparing '{memory}' vs '{existing_content[:50]}...' - similarity: {similarity:.3f}, distance: {distance:.3f}")
                
                if distance < self.valves.related_memories_dist:
                    similar_memories.append({
                        'id': getattr(existing_memory, 'id', ''),
                        'fact': getattr(existing_memory, 'content', ''),
                        'metadata': {'created_at': getattr(existing_memory, 'created_at', time.time())},
                        'distance': distance
                    })
                    print(f"Auto Memory: Added similar memory (distance: {distance:.3f}): '{existing_content[:50]}...'")
            
            # Sort by distance (most similar first) and limit results
            similar_memories.sort(key=lambda x: x['distance'])
            final_results = similar_memories[:self.valves.related_memories_n]
            print(f"Auto Memory: Text-based search returning {len(final_results)} similar memories")
            return final_results
            
        except Exception as e:
            print(f"Auto Memory: Error in text-based similarity search: {e}")
            import traceback
            print(f"Auto Memory: Traceback: {traceback.format_exc()}")
            return None

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity focusing on word overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Check if one text contains the other
        if text1 in text2 or text2 in text1:
            return 0.8
        
        # Simple word overlap (case-insensitive)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # For "User loves strawberries" vs "User hates strawberries":
        # intersection = {"user", "strawberries"} = 2 words
        # union = {"user", "loves", "strawberries", "hates"} = 4 words  
        # similarity = 2/4 = 0.5 (which converts to distance 0.5, well under 0.6 threshold)
        
        return len(intersection) / len(union) if union else 0.0

    async def store_memory(
        self,
        memory: str,
        user,
        body: dict = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """Given a memory, retrieve related memories. Update conflicting memories and consolidate memories as needed. Then store remaining memories."""
        # Initialize user_valves if __user__ is provided
        if __user__:
            self.user_valves = __user__.get("valves", self.UserValves())
        
        try:
            # Use improved text-based similarity search instead of vector search
            print(f"Auto Memory: Using improved text-based similarity search for '{memory}'")
            related_memories = await self.find_similar_memories_text(memory, user)
            
            if related_memories is None:
                print(f"Auto Memory: Text-based search failed for '{memory}'. Storing without duplicate detection.")
                await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=memory),
                    user=user,
                )
                print(f"Added memory without duplicate detection: {memory}")
                return True
        except Exception as e:
            return f"Unable to query related memories: {e}"
        try:
            # Handle different return formats: vector search results vs text-based results
            if isinstance(related_memories, list):
                if len(related_memories) > 0 and isinstance(related_memories[0], dict):
                    # Text-based search results (already in structured format)
                    structured_data = related_memories
                    print(f"Auto Memory: Found {len(related_memories)} related memories using text-based similarity")
                else:
                    # Empty list from text-based search - no similar memories found
                    structured_data = []
                    print(f"Auto Memory: No similar memories found using text-based similarity for '{memory}'")
            elif hasattr(related_memories, 'ids') and hasattr(related_memories, 'documents'):
                # Vector search SearchResult format
                ids = related_memories.ids[0] if related_memories.ids else []
                documents = related_memories.documents[0] if related_memories.documents else []
                metadatas = related_memories.metadatas[0] if related_memories.metadatas else []
                distances = related_memories.distances[0] if related_memories.distances else []
                
                # Combine each document and its associated data into a list of dictionaries
                structured_data = [
                    {
                        "id": ids[i],
                        "fact": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i],
                    }
                    for i in range(len(documents))
                ]
            else:
                # Old list format fallback
                related_list = [obj for obj in related_memories]
                ids = related_list[0][1][0]
                documents = related_list[1][1][0]
                metadatas = related_list[2][1][0]
                distances = related_list[3][1][0]
                
                # Combine each document and its associated data into a list of dictionaries
                structured_data = [
                    {
                        "id": ids[i],
                        "fact": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i],
                    }
                    for i in range(len(documents))
                ]
            # Filter for distance within threshhold
            filtered_data = [
                item
                for item in structured_data
                if item["distance"] < self.valves.related_memories_dist
            ]

            # Debug logging to understand why duplicates aren't being caught
            if filtered_data:
                print(f"Auto Memory: Found {len(filtered_data)} related memories for '{memory}'")
                for item in filtered_data:
                    print(f"  - Distance: {item['distance']:.4f}, Memory: '{item['fact']}'")
            else:
                print(f"Auto Memory: No related memories found for '{memory}' (threshold: {self.valves.related_memories_dist})")
            fact_list = [
                {"fact": item["fact"], "created_at": item["metadata"]["created_at"]}
                for item in filtered_data
            ]
            fact_list.append({"fact": memory, "created_at": time.time()})
            print(f"Fact list for consolidation: {fact_list}")
        except Exception as e:
            return f"Unable to restructure and filter related memories: {e}"
        # Consolidate conflicts or overlaps
        try:
            print(f"Auto Memory: Starting consolidation API call for memory: '{memory}'")
            print(f"Auto Memory: Consolidation input fact_list: {fact_list}")
            print(f"Auto Memory: Found {len(filtered_data)} similar memories for consolidation")
            if filtered_data:
                for item in filtered_data:
                    print(f"  - Similar memory (distance {item['distance']:.4f}): '{item['fact']}'")
            print(f"Auto Memory: API URL: {self.user_valves.openai_api_url if hasattr(self, 'user_valves') and self.user_valves.openai_api_url else self.valves.openai_api_url}")
            print(f"Auto Memory: Model: {self.user_valves.model if hasattr(self, 'user_valves') and self.user_valves.model else self.valves.model}")
            print(f"Auto Memory: API Key configured: {bool(self.user_valves.api_key if hasattr(self, 'user_valves') and self.user_valves.api_key else self.valves.api_key)}")
            
            consolidated_memories = await self.query_openai_api(
                system_prompt=CONSOLIDATE_MEMORIES_PROMPT,
                prompt=json.dumps(fact_list),
                body=body,
            )
            print(f"Auto Memory: Consolidation API call successful. Response: {consolidated_memories}")
            print(f"Auto Memory: Raw consolidation response: {repr(consolidated_memories)}")
        except Exception as e:
            print(f"Auto Memory: Consolidation API call failed with error: {e}")
            import traceback
            print(f"Auto Memory: Consolidation traceback: {traceback.format_exc()}")
            # If consolidation fails, try basic duplicate handling
            if len(filtered_data) > 0:
                print(f"Auto Memory: Consolidation failed but found {len(filtered_data)} similar memories. Performing basic duplicate handling.")
                # Check if this is a simple duplicate (very high similarity)
                for item in filtered_data:
                    if item["distance"] < 0.1:  # Very similar - likely duplicate
                        print(f"Auto Memory: Found very similar memory (distance: {item['distance']:.3f}). Skipping storage to avoid duplicate.")
                        return "Memory already exists"
                
                # Check if this contradicts existing memories about the same topic
                # Use AI to detect contradictions when consolidation API fails
                if len(filtered_data) > 0:
                    try:
                        # Create a simple contradiction check prompt
                        contradiction_prompt = f"""
You are checking if two memories about the same person contradict each other.

New memory: "{memory}"
Existing similar memories: {[item["fact"] for item in filtered_data]}

Do any of these memories directly contradict the new memory? A contradiction means they cannot both be true at the same time (like "loves X" vs "hates X", or "lives in A" vs "lives in B").

Return only a JSON object with this format:
{{"has_contradiction": true/false, "contradicting_memory": "exact text of contradicting memory or null"}}

Examples:
- "User loves pizza" vs "User hates pizza" = contradiction
- "User lives in New York" vs "User lives in Boston" = contradiction  
- "User likes apples" vs "User likes oranges" = NOT a contradiction (can like both)
- "User is 25 years old" vs "User is 30 years old" = contradiction
"""
                        
                        contradiction_result = await self.query_openai_api(
                            system_prompt="You are a precise contradiction detector. Return only valid JSON.",
                            prompt=contradiction_prompt,
                            body=body,
                        )
                        
                        result = json.loads(contradiction_result.strip())
                        
                        if result.get("has_contradiction") and result.get("contradicting_memory"):
                            # Find and delete the contradicting memory
                            for item in filtered_data:
                                if item["fact"] == result["contradicting_memory"]:
                                    print(f"Auto Memory: AI detected contradiction. Deleting old memory: '{item['fact']}'")
                                    await delete_memory_by_id(item["id"], user)
                                    break
                        elif result.get("has_contradiction"):
                            # If contradiction detected but no specific memory identified, 
                            # delete all similar memories (they're likely all conflicting)
                            print(f"Auto Memory: AI detected general contradiction. Deleting all similar memories.")
                            for item in filtered_data:
                                print(f"Auto Memory: Deleting conflicting memory: '{item['fact']}'")
                                await delete_memory_by_id(item["id"], user)
                                    
                    except Exception as e:
                        print(f"Auto Memory: Contradiction detection failed: {e}")
                        # Fall back to storing without contradiction detection
            
            # Store the new memory
            print(f"Auto Memory: Storing memory without consolidation: '{memory}'")
            await add_memory(
                request=Request(scope={"type": "http", "app": webui_app}),
                form_data=AddMemoryForm(content=memory),
                user=user,
            )
            print(f"Auto Memory: Successfully stored memory: '{memory}'")
            return True
        try:
            # Parse the consolidated memories first with better error handling
            consolidated_cleaned = consolidated_memories.strip()
            
            # Remove common AI response prefixes
            prefixes_to_remove = ["**Correct Output**", "**Output**", "**Response**", "**Result**", "Output:", "Response:", "Result:"]
            for prefix in prefixes_to_remove:
                if consolidated_cleaned.startswith(prefix):
                    consolidated_cleaned = consolidated_cleaned[len(prefix):].strip()
            
            memory_list = ast.literal_eval(consolidated_cleaned)

            # Only proceed with deletion/addition if consolidation actually changed something
            original_facts = [item["fact"] for item in fact_list]
            original_facts_set = set(original_facts)
            memory_list_set = set(memory_list)
            
            # Check if consolidation changed anything meaningfully
            memories_were_consolidated = len(memory_list) < len(original_facts)
            content_was_changed = original_facts_set != memory_list_set and len(filtered_data) > 0
            
            if memories_were_consolidated or content_was_changed:
                # Real consolidation happened - duplicates/conflicts resolved
                print(f"Consolidation detected: {len(original_facts)} -> {len(memory_list)} memories")
                print(f"Original facts: {original_facts}")
                print(f"Consolidated facts: {memory_list}")

                # Delete the old related memories (but not the new memory being added)
                if len(filtered_data) > 0:
                    for id in [item["id"] for item in filtered_data]:
                        await delete_memory_by_id(id, user)
                        print(f"Deleted old memory: {id}")

                # Add the new consolidated memories
                for item in memory_list:
                    await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=item),
                        user=user,
                    )
                    print(f"Added consolidated memory: {item}")
            else:
                # No real consolidation - just add the new memory without touching existing ones
                print("No consolidation needed - just adding new memory without affecting existing ones")
                await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=memory),
                    user=user,
                )
        except (ValueError, SyntaxError) as e:
            # Handle specific parsing errors with more helpful messages
            if "string did not match the expected pattern" in str(e):
                print(f"Auto Memory: Consolidation AI response parsing failed - malformed list format. Response was: {repr(consolidated_memories[:200])}")
                # Fall back to storing without consolidation
                print(f"Auto Memory: Storing memory without consolidation due to parsing error: '{memory}'")
                await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=memory),
                    user=user,
                )
                print(f"Auto Memory: Successfully stored memory without consolidation: '{memory}'")
                return True
            else:
                return f"Memory consolidation parsing error: {str(e)}"
        except Exception as e:
            return f"Unable to consolidate memories: {e}"
