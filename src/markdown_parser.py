"""Parse Reddit comments and persona markdown files."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def parse_comments_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a Reddit comments markdown file into structured data.

    Args:
        file_path: Path to the comments markdown file.

    Returns:
        List of comment dictionaries with body, score, subreddit, etc.
    """
    content = Path(file_path).read_text(encoding="utf-8")
    comments = []
    current_subreddit = None

    # Pattern for subreddit headers: ## r/subreddit (N comments)
    subreddit_pattern = re.compile(r"^## r/(\w+)")

    # Pattern for comment blocks
    comment_pattern = re.compile(
        r"### Comment \(Score: (-?\d+)\)\n"
        r"\*\*Date:\*\* (\d{4}-\d{2}-\d{2})\n"
        r"\*\*Link:\*\* \[View on Reddit\]\((https://[^\)]+)\)\n\n"
        r"(.*?)(?=\n---|\Z)",
        re.DOTALL
    )

    # Split by lines to track subreddit context
    lines = content.split("\n")
    for line in lines:
        subreddit_match = subreddit_pattern.match(line)
        if subreddit_match:
            current_subreddit = subreddit_match.group(1)

    # Now parse all comments with their full context
    # Re-process to properly associate subreddits
    current_subreddit = None
    sections = re.split(r"(## r/\w+[^\n]*\n)", content)

    for i, section in enumerate(sections):
        # Check if this is a subreddit header
        subreddit_match = re.match(r"## r/(\w+)", section)
        if subreddit_match:
            current_subreddit = subreddit_match.group(1)
            continue

        # Parse comments in this section
        if current_subreddit:
            for match in comment_pattern.finditer(section):
                score = int(match.group(1))
                date_str = match.group(2)
                permalink = match.group(3)
                body = match.group(4).strip()

                # Convert date string to timestamp
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                created_utc = int(date_obj.timestamp())

                comments.append({
                    "body": body,
                    "score": score,
                    "subreddit": current_subreddit,
                    "created_utc": created_utc,
                    "permalink": permalink
                })

    return comments


def parse_persona_file(file_path: Path) -> Dict[str, Any]:
    """Parse a persona markdown file into structured data.

    Args:
        file_path: Path to the persona markdown file.

    Returns:
        Dictionary with username, archetype, top_subreddits, persona_text.
    """
    content = Path(file_path).read_text(encoding="utf-8")

    # Extract username from header: # User Persona: u/Username
    username_match = re.search(r"# User Persona: u/(\w+)", content)
    username = username_match.group(1) if username_match else ""

    # Extract archetype: **The Archetype** – description
    archetype_match = re.search(r"\*\*([^*]+)\*\* [–-] ", content)
    archetype = archetype_match.group(1) if archetype_match else ""

    # Extract top subreddits from Most Active Communities line
    subreddits_match = re.search(
        r"\*\*Most Active Communities:\*\*\s*([^\n]+)",
        content
    )
    top_subreddits = []
    if subreddits_match:
        subreddits_text = subreddits_match.group(1)
        # Extract r/subreddit patterns
        top_subreddits = re.findall(r"r/(\w+)", subreddits_text)

    return {
        "username": username,
        "archetype": archetype,
        "top_subreddits": top_subreddits,
        "persona_text": content
    }
