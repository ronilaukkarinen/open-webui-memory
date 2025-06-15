### 1.3.4: 2025-06-15

* Implement simple word-in-memory matching for maximum flexibility across any subject
* Improve AI prompt to understand user intent and context better
* Add examples for queries like "Do you know my phone?" and "according specs"
* Focus on semantic relevance and user intent rather than exact word matching
* Support any subject matter without domain-specific hardcoded terms

### 1.3.3: 2025-06-15

* Fix status indicators not showing by switching from reasoning_content to status messages
* Restore visible memory processing progress indicators
* Ensure status messages display properly during memory retrieval and analysis
* Fix issue where memory system was working but progress wasn't visible to users

### 1.3.2: 2025-06-15

* Remove hardcoded job indicators and context-specific word lists
* Simplify memory filtering to use semantic word matching only
* Let AI handle all context-aware filtering instead of hardcoded rules
* Clean up AI prompt to use general guidelines instead of specific examples
* Improve code maintainability by removing brittle hardcoded values

### 1.3.1: 2025-06-15

* Fix reasoning content display to use proper OpenAI API format
* Resolve issue with raw HTML showing instead of collapsible reasoning blocks
* Fix "Thinking..." indicator getting stuck by using correct reasoning_content field
* Ensure reasoning content accumulates properly across processing steps

### 1.3.0: 2025-06-15

* Add beautiful collapsible reasoning display for memory processing steps
* Use native Open WebUI reasoning format with <details type="reasoning"> blocks
* Show detailed processing steps in expandable/collapsible format
* Replace status notifications with elegant reasoning progress display

### 1.2.1: 2025-06-15

* Improve context-aware memory filtering to distinguish similar words in different contexts
* Enhance AI relevance analysis with stricter context matching rules
* Increase relevance threshold from 5 to 7 for more precise memory selection

### 1.2.0: 2025-06-15

* Remove tags system entirely for simplified memory storage
* Add valve to control assistant memory storage (disabled by default)

### 1.1.1: 2025-06-15

* Add detailed real-time processing information from logs to status updates

### 1.1.0: 2025-06-15

* Enhance memory search with query word extraction and filtering

### 1.0.0: 2025-06-15

* Initial release