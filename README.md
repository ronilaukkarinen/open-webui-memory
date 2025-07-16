## 🧠 Open WebUI Memory function

### ChatGPT-like automatic memory retrieval and storage for [Open WebUI](https://github.com/open-webui/open-webui)

![Open WebUI](https://img.shields.io/badge/Open%20WebUI-222222?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjZmZmZmZmIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGhlaWdodD0iMWVtIiBzdHlsZT0iZmxleDpub25lO2xpbmUtaGVpZ2h0OjEiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjFlbSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xNy42OTcgMTJjMCA0Ljk3LTMuOTYyIDktOC44NDkgOUMzLjk2MiAyMSAwIDE2Ljk3IDAgMTJzMy45NjItOSA4Ljg0OC05YzQuODg3IDAgOC44NDkgNC4wMyA4Ljg0OSA5em0tMy42MzYgMGMwIDIuOTI4LTIuMzM0IDUuMzAxLTUuMjEzIDUuMzAxLTIuODc4IDAtNS4yMTItMi4zNzMtNS4yMTItNS4zMDFTNS45NyA2LjY5OSA4Ljg0OCA2LjY5OWMyLjg4IDAgNS4yMTMgMi4zNzMgNS4yMTMgNS4zMDF6Ij48L3BhdGg+PHBhdGggZD0iTTI0IDNoLTMuMzk0djE4SDI0VjN6Ij48L3BhdGg+PC9zdmc+Cg==)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Version](https://img.shields.io/badge/version-3.2.0-blue?style=for-the-badge)

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

1. Go to **Settings → Functions** and add the contents of `memory.py` file, save
2. Configure your AI model for memory identification:

### OpenAI API (Recommended)

- Set **OpenAI API key** to your OpenAI API key
- Set **Model** to `gpt-4o` (default)
- Leave **OpenAI API URL** as default (`https://api.openai.com`)

### Alternative API Services

- **Pollinations**: Set **OpenAI API URL** to `https://text.pollinations.ai/openai`

### Local Models with Ollama

- Set **OpenAI API URL** to `http://localhost:11434/v1`
- Set **OpenAI API key** to `ollama`
- Set **Model** to one of the recommended models:

#### ✅ Recommended Ollama Models (tested for memory identification):

- `mistral:7b-instruct` - Excellent instruction following
- `qwen2.5:7b` - Good balance of performance and capability
- `llama3.1:8b` - Works but may need more specific prompting

#### ❌ Not Recommended:

- GGUF models
- Models without instruction tuning typically perform poorly

**Note**: Memory identification requires models that can follow complex instructions and output structured data. If using local models, ensure they're instruction-tuned variants.

## Examples

**User input**: "I live in Central street 45 and I love sushi"<br>
**Stored memories**:<br>
- Location: "User lives in Central street 45"<br>
- Preference: "User loves sushi"<br>

**User input**: "Remember that my doctor's appointment is next Tuesday at 3pm" <br>
**Stored memory**: "Doctor's appointment scheduled for next Tuesday at 2025-01-14 15:00:00"<br>

**Context retrieval**: When user asks "What's my address?", the filter automatically retrieves and provides the stored location information.
