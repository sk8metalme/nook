# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nook is a local/self-hosted fork of a news aggregation system that automatically collects and summarizes tech content from multiple sources (Reddit, Hacker News, GitHub Trending, arXiv papers, RSS feeds). This fork removes AWS dependencies to run locally or on home servers.

**✅ Claude Integration Status**: This project now supports both Google Gemini and Claude APIs for content generation, with seamless switching between providers via environment variables.

## Development Commands

### Local Development
```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure AI provider (choose one)
export AI_CLIENT_TYPE=gemini    # Use Google Gemini (default)
export AI_CLIENT_TYPE=claude    # Use Claude API

# Run data collection
python main.py

# Start web viewer
python nook/functions/viewer/viewer.py
```

### AWS Deployment (Original)
```bash
make cdk-deploy  # Special deployment that handles dependencies
```

### Code Quality
```bash
ruff check .          # Lint with ruff (configured in pyproject.toml)
python -m pytest     # Run tests (limited test coverage)
```

## Architecture

### Core Components
- **main.py**: Local entry point that orchestrates all data collection handlers
- **nook/nook_stack.py**: AWS CDK infrastructure definition (for original cloud version)
- **nook/functions/**: Modular data collectors, each handling a specific source:
  - `reddit_explorer/`: Reddit subreddit content collection and summarization
  - `hacker_news/`: Hacker News article collection
  - `github_trending/`: GitHub trending repository tracking
  - `tech_feed/`: RSS feed monitoring and summarization
  - `paper_summarizer/`: arXiv paper collection and summarization
  - `viewer/`: FastAPI web interface with chat functionality
  - `common/`: Shared utilities including Gemini API client

### Data Flow
1. Each handler collects from its respective API/source
2. Content is processed and summarized using Google Gemini API
3. Results are saved as markdown files to OUTPUT_DIR (local) or S3 (cloud)
4. Web viewer displays collected content with interactive chat features

### Configuration
- **Environment**: `.env` file with API keys and provider selection
  - **Gemini**: `GEMINI_API_KEY` (default provider)
  - **Claude**: `ANTHROPIC_API_KEY` + `AI_CLIENT_TYPE=claude`
  - **Other**: Reddit credentials, OUTPUT_DIR
- **Source Configuration**: TOML files control what content to collect:
  - `nook/functions/reddit_explorer/subreddits.toml`: Reddit subreddits to monitor
  - `nook/functions/tech_feed/feed.toml`: RSS feeds to monitor
  - `nook/functions/github_trending/languages.toml`: Programming languages to track

### Key Patterns
- **Modular Design**: Each data source is an independent handler class
- **Shared Client**: Common AI client with retry logic and rate limiting
- **Provider Flexibility**: Factory pattern for switching between Gemini and Claude APIs
- **Configuration-Driven**: Data sources configurable via TOML files
- **Dual Mode**: Supports both local file storage and AWS S3 storage
- **Error Handling**: Retry mechanisms and graceful degradation

## Development Guidelines

### Code Style
- Python 3.10+ with type hints and NumPy-style docstrings
- Ruff configuration enforces comprehensive linting rules
- Line length: 88 characters
- Use dataclasses for configuration objects

### Testing
- Limited test coverage currently exists
- Test file: `nook/functions/tech_feed/test_tech_feed.py`
- Run tests with: `python -m pytest`

### Adding New Data Sources
1. Create new directory under `nook/functions/`
2. Implement handler class with `__call__` method
3. Add configuration TOML file if needed
4. Update `main.py` to include new handler
5. Add to `nook_stack.py` for AWS deployment support

### Environment Setup
- Copy `.env.example` to `.env` and configure API keys
- **AI Provider Configuration** (choose one):
  - **Gemini**: Set `GEMINI_API_KEY` (default)
  - **Claude**: Set `ANTHROPIC_API_KEY` and `AI_CLIENT_TYPE=claude`
- Reddit API credentials needed for Reddit content collection
- OUTPUT_DIR specifies where to save collected content locally

### Claude Integration Details

#### Switching to Claude
```bash
# In your .env file
ANTHROPIC_API_KEY=your_claude_api_key
AI_CLIENT_TYPE=claude

# Or set environment variables
export ANTHROPIC_API_KEY=your_claude_api_key
export AI_CLIENT_TYPE=claude
```

#### Client Factory Pattern
The system uses a factory pattern to seamlessly switch between AI providers:

```python
from nook.functions.common.python.client_factory import create_client

# Automatically uses the configured provider
client = create_client()

# Both Gemini and Claude clients support the same interface:
response = client.generate_content("Summarize this content")
```

#### Migration Status
- ✅ **Paper Summarizer**: Migrated to use factory pattern
- 🔄 **Tech Feed**: Pending migration
- 🔄 **Hacker News**: Pending migration
- 🔄 **Reddit Explorer**: Pending migration
- 🔄 **Web Viewer**: Pending migration

#### Claude API Features
- **Model**: claude-3-5-sonnet-20241022
- **Error Handling**: Retry logic for rate limits and timeouts
- **Chat Sessions**: Stateful conversation management
- **Configuration**: Compatible parameter mapping from Gemini settings
- **Testing**: Comprehensive test suite with >90% coverage