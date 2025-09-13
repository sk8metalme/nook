# Nook Gemini-to-Claude Migration Context

## Current Implementation Status: **Phase 2 Complete - Core Client Development ✅**

### Claude Integration Implementation

1. **Core Claude Client Module**: `/nook/functions/common/python/claude_client.py` ✅
   - Complete Claude API client with retry logic using tenacity
   - Configuration management through `ClaudeClientConfig`
   - Key methods: `generate_content()`, `create_chat()`, `send_message()`
   - Robust error handling with exponential backoff for rate limits and timeouts
   - Model: "claude-3-5-sonnet-20241022"

2. **Client Factory Module**: `/nook/functions/common/python/client_factory.py` ✅
   - Unified interface for switching between Gemini and Claude clients
   - Environment-based client selection via `AI_CLIENT_TYPE`
   - Seamless configuration mapping between providers
   - Backward compatibility maintained

### Updated Dependencies

```
# Existing Gemini dependencies
google-genai==1.2.0
tenacity==9.0.0

# New Claude dependencies
anthropic>=0.25.0
```

### Environment Configuration

**Gemini Configuration (Legacy)**:
- `GEMINI_API_KEY`: Required for Gemini client
- Model: "gemini-2.0-flash-exp"

**Claude Configuration (New)**:
- `ANTHROPIC_API_KEY`: Required for Claude client
- `AI_CLIENT_TYPE`: Set to "claude" to use Claude client (defaults to "gemini")
- Model: "claude-3-5-sonnet-20241022"
- Timeout: 60000ms default (maintained for compatibility)

### Migration Status by Function

#### ✅ **Migrated Functions**
1. **Paper Summarizer** (`/nook/functions/paper_summarizer/paper_summarizer.py`)
   - **Status**: Fully migrated to use `client_factory.create_client()`
   - Uses factory pattern for client creation
   - Complex system instructions for structured output maintained
   - Processes multiple papers with threading (unchanged)

#### 🔄 **Pending Migration Functions**
2. **Web Viewer** (`/nook/functions/viewer/viewer.py`)
   - **Status**: Requires migration
   - Interactive chat functionality via `/chat/{topic_id}` endpoint
   - Uses `chat_with_search()` - needs Claude equivalent implementation
   - Processes markdown content and external links

3. **Content Aggregators** - **Need Migration**:
   - **Reddit Explorer** (`/nook/functions/reddit_explorer/reddit_explorer.py`)
   - **Tech Feed** (`/nook/functions/tech_feed/tech_feed.py`)
   - **Hacker News** (`/nook/functions/hacker_news/hacker_news.py`)
   - All currently use direct Gemini client imports
   - Use `generate_content()` for content summarization
   - Similar patterns across all aggregator functions

### Key Technical Characteristics

#### Claude Client Features ✅
- **Error Handling**: 5 retry attempts with exponential backoff (maintained)
- **Rate Limiting**: Specific retry logic for RateLimitError and APITimeoutError
- **Configuration Flexibility**: Runtime parameter overrides (maintained)
- **Chat Sessions**: Stateful conversation management
- **System Instructions**: Support for system prompts (Claude format)

#### Gemini Client Features (Legacy)
- **Safety Settings**: Custom safety configuration with BLOCK_NONE
- **Search Integration**: Google search tool integration for enhanced responses
- **Model**: "gemini-2.0-flash-exp"

## Migration Drivers

### Migration Implementation Details

#### ✅ **Completed Requirements**
- ✅ Claude API client implementation (replaces CLI approach)
- ✅ Maintained existing functionality interfaces
- ✅ Preserved error handling and retry mechanisms
- ✅ Updated configuration management with factory pattern
- ✅ Ensured backward compatibility via environment switching
- ✅ Comprehensive test suite implemented

#### 🔄 **Remaining Tasks**
- Migrate remaining 4 functions to use client factory
- Update viewer function to handle Claude chat format
- Update deployment configuration for environment variables
- Production testing and validation

### Technical Implementation Notes

#### API Approach vs CLI Approach
- **Decision**: Using Anthropic Python SDK directly instead of Claude CLI
- **Rationale**: Better integration with AWS Lambda, more reliable error handling
- **Result**: Full API compatibility with existing patterns

#### Environment Switching
- `AI_CLIENT_TYPE=gemini` → Uses existing Gemini client
- `AI_CLIENT_TYPE=claude` → Uses new Claude client
- Default behavior: Falls back to Gemini for backward compatibility

#### Test Coverage
- ✅ Unit tests for Claude client (90%+ coverage)
- ✅ Integration tests for basic functionality
- ✅ Factory pattern tests for client switching
- ✅ Configuration compatibility tests