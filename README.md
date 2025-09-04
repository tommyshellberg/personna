# Reddit User Research CLI

A Python CLI tool for analyzing Reddit users to understand their personas, interests, and engagement patterns. Built with local LLMs for privacy-focused analysis.

## Features

- **Comment Analysis**: Fetch top comments from Reddit users using PRAW
- **Local LLM Processing**: Generate user personas using Ollama (runs entirely offline)
- **Structured Personas**: Analyze users based on the 12 Jungian Archetypes
- **Engagement Insights**: Get recommendations for how to reach similar users
- **Rate Limit Handling**: Respects Reddit API limits automatically
- **Resume Support**: Skip already processed users

## Tech Stack

- **Reddit API**: PRAW (Python Reddit API Wrapper)
- **Local LLM**: Ollama with Qwen models
- **CLI Framework**: Click with Rich for beautiful output
- **Data Storage**: Markdown files with structured analysis

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Reddit API Credentials** - Get from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) ([PRAW setup guide](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html))

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
   
   # Pull a model (in another terminal)
   ollama pull qwen3:8b
   ```

3. **Configure Reddit API:**
   ```bash
   cp .env.example .env
   # Edit .env with your Reddit API credentials
   ```

4. **Create user list:**
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
├── src/                    # Main application code
│   ├── cli.py             # Command-line interface
│   ├── reddit_client.py   # Reddit API wrapper
│   └── persona_generator.py # LLM persona generation
├── config/                # Configuration files
│   └── settings.yaml      # App settings
├── data/
│   ├── input/             # User lists (gitignored except example)
│   └── output/            # Generated reports (gitignored)
├── scripts/               # Testing and development scripts
└── requirements.txt       # Python dependencies
```

## Roadmap

- **Version 2**: Parse Reddit post URLs to auto-extract commenters
- **Vector Storage**: Add Qdrant for similarity search across users
- **Sentiment Analysis**: Analyze comment sentiment patterns
- Use Langchain library for agentic AI and external LLM calls.
- **Batch Reports**: Generate aggregate insights across all users

## Contributing

This project demonstrates Python API integration, local LLM usage, and data analysis patterns. Feel free to:

- Add new persona analysis dimensions
- Improve prompt engineering for better insights
- Add support for other social platforms
- Enhance the CLI with more options

## License

See LICENSE file.