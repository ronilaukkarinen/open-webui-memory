## 🧠 Open WebUI Memory function

### ChatGPT-like automatic memory retrieval and storage for [Open WebUI](https://github.com/open-webui/open-webui)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Version](https://img.shields.io/badge/version-3.0.3-blue?style=for-the-badge) ![Open WebUI](https://img.shields.io/badge/Open%20WebUI-Compatible-orange?style=for-the-badge)

Automatically identify, retrieve and store memories from user conversations in Open WebUI. This filter intelligently processes chat messages to extract meaningful information about users and stores it as memories for future reference.

![image](https://github.com/user-attachments/assets/a76ec505-282a-4f40-b7c7-c9855a86610a)

This is a fork of [davixk's/nokodo's work](https://github.com/Davixk/open-webui-extensions).

## ✨ What it does

Auto Memory listens in on your conversations and detects facts, preferences, key moments, or anything useful for the assistant to remember about you.
It stores these as separate memories, so future AI interactions stay personal and context-aware—without you micromanaging.

You get:
* Seamless journaling of your important info
* Smarter, context-rich AI assistance
* No more "please remember X" (unless you want to!)

## 🧠 Memory extraction logic

- New or changed facts from User's latest message are saved.
- Explicit "please remember..." requests always create a Memory.
- Avoids duplicates & merges conflicts by keeping only the latest.
- Filters out ephemeral/trivial/short-term details.

## Installation

1. Go to **Settings → Functions** and add the contents of `auto_memory_retrieval_and_storage.py` file, save
2. If you want to use OpenAI API, you need to add your API key to the **OpenAI API key** field. For pollinations, use `https://text.pollinations.ai/openai`.
3. If you want to use a local model, use Ollama API:
    - Set **OpenAI API URL** to `http://localhost:11434/v1`
    - Set **OpenAI API key** to `ollama`
    - Set **Model** to the model you want to use (you can see the list with command `ollama list`)

## Examples

**User input**: "I live in Central street 45 and I love sushi"<br>
**Stored memories**:<br>
- Location: "User lives in Central street 45"<br>
- Preference: "User loves sushi"<br>

**User input**: "Remember that my doctor's appointment is next Tuesday at 3pm" <br>
**Stored memory**: "Doctor's appointment scheduled for next Tuesday at 2025-01-14 15:00:00"<br>

**Context retrieval**: When user asks "What's my address?", the filter automatically retrieves and provides the stored location information.
