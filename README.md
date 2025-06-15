## 🧠 Auto Memory Retrieval and Storage Open WebUI function

### ChatGPT-like automatic memory retrieval and storage for [Open WebUI](https://github.com/open-webui/open-webui)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Version](https://img.shields.io/badge/version-2.0.3-blue?style=for-the-badge) ![Open WebUI](https://img.shields.io/badge/Open%20WebUI-Compatible-orange?style=for-the-badge)

Automatically identify, retrieve and store memories from user conversations in Open WebUI. Successor to [Auto Memory](https://github.com/ronilaukkarinen/open-webui-auto-memory). This filter intelligently processes chat messages to extract meaningful information about users and stores it as memories for future reference.

![image](https://github.com/user-attachments/assets/bf3a5d3a-6846-4d45-b46d-ffb80fcebed6)

## Features

- **Automatic memory identification**: Extracts important user information from conversations
- **Smart memory retrieval**: Finds relevant memories based on current context
- **Memory operations**: Support for creating, updating, and deleting memories
- **Flexible API support**: Works with OpenAI API, Ollama, and other compatible endpoints
- **Contextual processing**: Uses AI to determine memory relevance and importance
- **User privacy controls**: Configurable settings for memory processing visibility

## Installation

1. Go to **Settings → Functions** and add the contents of `auto_memory_retrieval_and_storage.py` file, save
2. If you want to use OpenAI API, you need to add your API key to the **OpenAI API key** field
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
