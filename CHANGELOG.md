### 3.0.8: 2025-07-04

* Remove legacy prompt
* Fix AI saving duplicate memories and not recognizing there are already memories about the same topic

### 3.0.7: 2025-06-30

* Fixes to memory prompt

### 3.0.6: 2025-06-24

* Always save the memory in English for consistency
* Make sure no markdown code blocks are interfering the memory output

### 3.0.5: 2025-06-21

* Ensure memory gets saved

### 3.0.4: 2025-06-20

* Fix memory storing for local models
* Add proper instuctions for recommended models

### 3.0.3: 2025-06-19

* Add memory timestamp to memory context
* Do not extract pre-existing memories from assistant replies

### 3.0.2: 2025-06-19

* Improve debugging and error logging
* Add notifications from API call errors

### 3.0.1: 2025-06-18

* Re-implement excluded_models
* Add model-based settings

### 3.0.0: 2025-06-17

* Rename to Memory
* Completely rewrite the over-complex codebase
* Add "Analyzing message for new memories..." status indicator
* Add corner notifications for successful memory operations
* Use notification system for memory confirmations
* Show error notifications in corner for failed operations
* Standardize status messages to always show "Memory updated"
* Add memory retrieval feature that injects ALL user memories into conversations as context
* Implement dual functionality: memory saving and memory retrieval working together
* Add proper memory context filtering to prevent analysis loops
* Safe consolidation of memories, do not delete related memories without hesitation
* Fix duplicate memory storing issue

### 2.4.4: 2025-06-17

* Remove pre-filtering entirely - analyze all memories for relevance
* Fix issue where relevant memories were filtered out before analysis
* Ensure all memories get evaluated by AI relevance system

### 2.4.3: 2025-06-17

* Add assertive context formatting to override conflicting system instructions
* Improve memory context injection with stronger directives

### 2.4.2: 2025-06-17

* Remove arbitrary relevance threshold - include all AI-rated memories
* Improve memory relevance analysis prompt for better semantic matching
* Enhance context format to make memory facts clearer to AI
* Fix issue where relevant memories weren't being used in responses

### 2.4.1: 2025-06-17

* Add extensive debugging to memory context injection
* Improve memory content extraction and validation
* Fix memory context being ignored when retrieved memories exist

### 2.4.0: 2025-06-17

* Remove delayed_memory_analysis valve - always retrieve memories for questions
* Fix "Analyzing for new memories..." getting stuck after reply
* Ensure proper status completion in all code paths
* Enable memory retrieval for all user questions including simple ones

### 2.3.0: 2025-06-17

* Remove remember.py and integrate functionality into auto_memory_retrieval_and_storage.py
* Fix timing display bug - show total time without decimals 
* Optimize performance and refactor code to reduce duplication
* Add MemoryBase class for shared memory operations

### 2.2.0: 2025-06-17

* Add valve: Only store and analyze memories after the reply is complete (faster responses, no memory retrieval by default)
* Add Remember tool

### 2.1.1: 2025-06-17

* Fix model exclusion
* Add proper instructions for model exclusion
* Add LLM exclusion feature

### 2.1.0: 2025-06-17

* Performance: Remove foreign language support
* Add valve to exclude models from memory processing
* Improve prompt: Be more generous with what to store as a long term memory
* Add error notifications
* Add more status updates

### 2.0.4: 2025-06-16

* Remove unnecessary limitations
* Refine prompts
* Add smart pre-filtering for large memory collections

### 2.0.3: 2025-06-16

* Fix memories not stored, change to less limiting memory identification
* Fix "Memory updated" event emitter not shown
* Disable memory limitation in amounts
* Fix memory identification to use all memories for analysis

### 2.0.2: 2025-06-15

* Fix regressions in supporting the foreign language
* Fix performance and speed issues

### 2.0.1: 2025-06-15

* Simplify memory updated status text
* Prefer English
* Add foreign language support

### 2.0.0: 2025-06-15

* Improve memory loading indicator
* Remove complex stop words, filter everything
* Remove memory amount limitation
* Fixes to memory context
* Add more precision in memory notification

### 1.3.7: 2025-06-15

* Update existing memories when adding details instead of creating duplicates
* Include memory IDs in AI context for proper UPDATE operations
* Consolidate related information into single comprehensive memories

### 1.3.6: 2025-06-15

* Prevent duplicate memory creation when information already exists
* Add rules to not store questions, queries, or assistant responses

### 1.3.5: 2025-06-15

* Add detailed memory operation status indicators (saved/updated/deleted counts)

### 1.3.4: 2025-06-15

* Remove hardcoded terms for truly flexible matching across any subject
* Improve AI prompt to understand user intent and context better

### 1.3.3: 2025-06-15

* Fix status indicators by switching from reasoning_content to status messages

### 1.3.2: 2025-06-15

* Remove hardcoded job indicators and context-specific word lists
* Simplify memory filtering to use semantic word matching only

### 1.3.1: 2025-06-15

* Fix reasoning content display to use proper OpenAI API format

### 1.3.0: 2025-06-15

* Add collapsible reasoning display for memory processing steps

### 1.2.1: 2025-06-15

* Improve context-aware memory filtering with stricter matching rules
* Increase relevance threshold from 5 to 7 for more precise memory selection

### 1.2.0: 2025-06-15

* Remove tags system entirely for simplified memory storage
* Add valve to control assistant memory storage (disabled by default)

### 1.1.1: 2025-06-15

* Add detailed real-time processing information to status updates

### 1.1.0: 2025-06-15

* Enhance memory search with query word extraction and filtering

### 1.0.0: 2025-06-15

* Initial release
