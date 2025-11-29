# Reddit User Research CLI

A Python CLI tool for analyzing Reddit users to understand their personas, interests, and engagement patterns. Built with local LLMs for privacy-focused analysis.

## Features

- **Comment Analysis**: Fetch top comments from Reddit users using PRAW
- **Local LLM Processing**: Generate user personas using Ollama (runs entirely offline)
- **Structured Personas**: Analyze users based on the 12 Jungian Archetypes
- **Engagement Insights**: Get recommendations for how to reach similar users
- **RAG-Powered Q&A**: Ask questions about your audience with AI-synthesized answers
- **Semantic Search**: Find similar comments and personas using vector embeddings
- **Rate Limit Handling**: Respects Reddit API limits automatically
- **Resume Support**: Skip already processed users

## Tech Stack

- **Reddit API**: PRAW (Python Reddit API Wrapper)
- **Local LLM**: Ollama with Qwen models (generation + embeddings)
- **Vector Database**: Qdrant for semantic search
- **Embeddings**: nomic-embed-text (768-dim, runs locally via Ollama)
- **CLI Framework**: Click with Rich for beautiful output
- **Data Storage**: Markdown files + Qdrant vector collections

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Docker** - For running Qdrant vector database
4. **Reddit API Credentials** - Get from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) ([PRAW setup guide](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html))

### Setup

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/tommyshellberg/personna.git
   cd personna
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up Ollama:**
   ```bash
   # Start Ollama service
   ollama serve

   # Pull models (in another terminal)
   ollama pull qwen3:8b           # For persona generation & RAG
   ollama pull nomic-embed-text   # For embeddings (semantic search)
   ```

3. **Start Qdrant (for RAG features):**
   ```bash
   ./scripts/setup_qdrant.sh start

   # Verify it's running
   python scripts/test_qdrant_ready.py
   ```

4. **Configure Reddit API:**
   ```bash
   cp .env.example .env
   # Edit .env with your Reddit API credentials
   ```

5. **Create user list:**
   ```bash
   # Create a file with Reddit usernames (one per line)
   echo -e "spez\nAutomoderator\nreddit" > data/input/my_users.txt
   ```

## Usage

### Fetch User Comments

```bash
python main.py fetch data/input/my_users.txt
```

This will:
- Read usernames from your file
- Fetch their top 100 comments via Reddit API
- Save organized markdown files to `data/output/`
- Skip users that already have files

### Generate User Personas

```bash
python main.py personas
```

This will:
- Analyze existing comment files in `data/output/`
- Generate detailed personas using local LLM
- Save persona analysis as `{username}_persona.md`

### Full Pipeline

```bash
# Fetch comments then generate personas
python main.py fetch data/input/my_users.txt
python main.py personas
```

### RAG Features (Semantic Search & Q&A)

After collecting comments and personas, embed them for semantic search:

```bash
# Embed all data into Qdrant
python main.py embed

# Or embed specific collections
python main.py embed --collection comments
python main.py embed --collection personas

# Or embed a single user
python main.py embed --user spez
```

Example output:
```
Embedding data into Qdrant...
Embedding 15 comment files...
✓ spez: 100 comments
✓ productivityguru: 87 comments
✓ remoteworker42: 95 comments
Embedded 282 comments
Embedding 15 persona files...
✓ spez persona
✓ productivityguru persona
✓ remoteworker42 persona
Embedded 15 personas
Embedding complete!
```

Search your data semantically:

```bash
# Search comments
python main.py search "productivity tips"

# Search personas
python main.py search "creative personality" --collection personas

# Limit results
python main.py search "work from home" --limit 5
```

Example output:
```
Searching comments for: productivity tips

Found 10 results:

1. (similarity: 0.847)
   u/productivityguru in r/productivity (+234)
   The Pomodoro technique changed my life! I used to struggle with focus...

2. (similarity: 0.823)
   u/remoteworker42 in r/getdisciplined (+156)
   Time blocking is underrated. I schedule everything including breaks...

3. (similarity: 0.801)
   u/techfounder in r/entrepreneur (+89)
   Best productivity hack: batch similar tasks together. Email twice a day...
```

Ask questions about your audience (RAG-powered):

```bash
# Get AI-synthesized answers grounded in your data
python main.py ask "What communication styles resonate with my audience?"
python main.py ask "What topics do they care most about?"
python main.py ask "How do they talk about work-life balance?"
```

Example output:
```
Question: How does my audience talk about burnout?

Retrieving relevant context...
Generating answer...

Answer:
Your audience discusses burnout with a mix of vulnerability and practical advice.
Several users share personal experiences:

- u/remoteworker42 in r/antiwork notes: "Burnout is real. I learned the hard
  way that no job is worth your mental health."

- u/productivityguru frames it through a solutions lens: "The key is setting
  boundaries early. I now have hard stops at 6pm."

Common themes include:
1. Recognition that burnout affects high performers
2. Emphasis on boundaries and saying "no"
3. Preference for systemic solutions over individual coping
4. Supportive, non-judgmental tone when others share struggles

This suggests your audience values authentic discussion of challenges paired
with actionable strategies.
```

### Command Options

```bash
# Custom config file
python main.py fetch users.txt --config my_config.yaml

# Custom output directory  
python main.py fetch users.txt --output-dir custom_output/

# Force regenerate existing personas
python main.py personas --skip-existing false

# See all options
python main.py --help
python main.py fetch --help
python main.py personas --help
```

## Configuration

Edit `config/settings.yaml` to customize:

- **Reddit API**: Rate limits, comment count
- **Ollama**: Model choice, temperature, context size
- **Analysis**: Minimum comment length, analysis types

## Output Format

### Comment Files (`{username}.md`)
- Organized by subreddit
- Includes scores, dates, and permalinks
- Easy to read markdown format

### Persona Files (`{username}_persona.md`)
- **Demographics**: Age range, likely occupation
- **Communication Style**: Tone, language patterns
- **Jungian Archetype**: Personality classification
- **Interests**: Topics and passions
- **Engagement Strategy**: How to reach them
- **Subreddit Activity**: Where they're most active

## Example Workflow

1. **Collect usernames** from your Reddit posts' positive interactions
2. **Run analysis**: `python main.py fetch users.txt && python main.py personas`
3. **Review personas** to understand your audience
4. **Find patterns** across users to identify target demographics
5. **Plan engagement** based on recommended strategies

## Use Cases

### Content Creator: Understanding Your Audience

You're a content creator who wants to understand what resonates with your followers.

```bash
# 1. Export usernames of people who engaged positively with your posts
# 2. Analyze them
python main.py fetch data/input/engaged_users.txt
python main.py personas
python main.py embed

# 3. Discover what they care about
python main.py ask "What topics does my audience discuss most passionately?"
python main.py ask "What frustrations do they frequently mention?"
python main.py ask "What solutions are they looking for?"
```

**Insight**: Discover that your audience frequently discusses "imposter syndrome" in tech careers, giving you ideas for future content.

### Startup Founder: Market Research

You're validating a product idea and want to understand your target users.

```bash
# 1. Gather users from relevant subreddits (r/productivity, r/SaaS, etc.)
# 2. Build your audience database
python main.py fetch data/input/target_market.txt
python main.py personas
python main.py embed

# 3. Research their needs
python main.py ask "What tools do they currently use and complain about?"
python main.py ask "What would they pay for to solve their problems?"
python main.py search "wish there was" --limit 20
```

**Insight**: Find that users repeatedly mention wanting "a simple way to track client feedback without another SaaS subscription."

### Community Manager: Engagement Strategy

You manage a community and want to improve engagement.

```bash
# 1. Analyze your most active community members
python main.py fetch data/input/power_users.txt
python main.py personas
python main.py embed

# 2. Understand communication preferences
python main.py ask "What communication style do these users prefer?"
python main.py ask "What makes them engage vs lurk?"
python main.py search "love this community" --collection comments
```

**Insight**: Learn that your power users value "direct, no-BS advice" and engage most when posts include actionable steps.

### Researcher: Audience Segmentation

You're studying online communities and want to identify user archetypes.

```bash
# 1. Sample users from a subreddit
# 2. Generate personas with archetype analysis
python main.py fetch data/input/subreddit_sample.txt
python main.py personas
python main.py embed

# 3. Explore segments
python main.py search "The Creator" --collection personas
python main.py search "The Sage" --collection personas
python main.py ask "What differentiates Creator archetypes from Sage archetypes in this community?"
```

**Insight**: Discover that "Creator" archetypes in r/entrepreneur focus on building, while "Sage" archetypes focus on advising others.

## Privacy & Ethics

- **Local processing**: All LLM analysis runs on your machine
- **No data collection**: Tool doesn't store or transmit personal data
- **Respect rate limits**: Follows Reddit's API guidelines
- **Public data only**: Analyzes publicly available comments

## Development

### Testing Components

Use the scripts in `scripts/` to test individual components:

```bash
# Test Reddit API connection
python scripts/test_reddit_auth.py

# Test single user comment fetching
python scripts/test_single_user.py

# Test Ollama LLM connection
python scripts/test_ollama_simple.py

# Test persona generation
python scripts/test_persona.py
```

### Project Structure

```
reddit-user-research/
├── src/                      # Main application code
│   ├── cli.py               # Command-line interface (5 commands)
│   ├── reddit_client.py     # Reddit API wrapper
│   ├── persona_generator.py # LLM persona generation
│   ├── vector_store.py      # Qdrant vector operations
│   └── markdown_parser.py   # Parse comment/persona files
├── tests/                   # Test suite (42 tests)
│   ├── conftest.py          # Shared fixtures
│   ├── test_vector_store.py # VectorStore unit tests
│   ├── test_vector_store_integration.py
│   ├── test_markdown_parser.py
│   └── test_cli.py
├── config/                  # Configuration files
│   └── settings.yaml        # App settings (Reddit, Ollama, Qdrant)
├── scripts/                 # Setup and testing scripts
│   ├── setup_qdrant.sh      # Start/stop Qdrant Docker
│   ├── test_qdrant_ready.py # Verify Qdrant connectivity
│   └── test_embedding.py    # Verify Ollama embeddings
├── data/
│   ├── input/               # User lists
│   └── output/              # Generated markdown files
└── requirements.txt         # Python dependencies
```

## Roadmap

- [x] **Vector Storage**: Qdrant for semantic search across comments and personas
- [x] **RAG Q&A**: Ask questions with AI-synthesized answers
- [ ] **Version 2**: Parse Reddit post URLs to auto-extract commenters
- [ ] **Sentiment Analysis**: Analyze comment sentiment patterns
- [ ] **Batch Reports**: Generate aggregate insights across all users
- [ ] **User Clustering**: Auto-group users by interests and behavior

## Contributing

This project demonstrates Python API integration, local LLM usage, and data analysis patterns. Feel free to:

- Add new persona analysis dimensions
- Improve prompt engineering for better insights
- Add support for other social platforms
- Enhance the CLI with more options

## License

See LICENSE file.