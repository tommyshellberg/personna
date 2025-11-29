"""Unit tests for markdown parser."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_comments_markdown():
    """Sample comments markdown content."""
    return '''# Reddit Comments Analysis: u/TestUser

**Generated:** 2025-09-03 12:17:16
**Total Comments:** 3

## r/productivity (2 comments)

### Comment (Score: 24)
**Date:** 2025-03-01
**Link:** [View on Reddit](https://reddit.com/r/productivity/comments/abc123/title/comment1/)

This is the first comment about productivity tips.

---

### Comment (Score: 7)
**Date:** 2025-03-02
**Link:** [View on Reddit](https://reddit.com/r/productivity/comments/def456/title/comment2/)

Second comment here with multiple
lines of text.

---

## r/remotework (1 comments)

### Comment (Score: 100)
**Date:** 2025-03-03
**Link:** [View on Reddit](https://reddit.com/r/remotework/comments/ghi789/title/comment3/)

Working from home is great!

---
'''


@pytest.fixture
def sample_persona_markdown():
    """Sample persona markdown content."""
    return '''# User Persona: u/TestUser

## User Persona Summary
TestUser is a productivity enthusiast who values work-life balance.

## Demographics & Background
- **Likely Age Range:** 25–35
- **Possible Occupation/Field:** Software Engineer

## Communication Style
- **Tone:** Helpful and encouraging
- **Language Patterns:** Clear, concise

## Interests & Topics
- Productivity tools
- Remote work
- Time management

## Jungian Archetype
**The Sage** – seeks truth and knowledge, shares wisdom with others.

## Subreddit Activity Analysis
- **Most Active Communities:** r/productivity, r/remotework, r/getdisciplined.
- **Community Role:** Helpful contributor

## Engagement Recommendations
- **Content Types:** How-to guides, productivity tips
'''


class TestParseCommentsFile:
    """Tests for parsing comments markdown files."""

    def test_parse_comments_returns_list(self, sample_comments_markdown, tmp_path):
        """parse_comments_file should return a list of comments."""
        from src.markdown_parser import parse_comments_file

        # Write sample to temp file
        file_path = tmp_path / "TestUser.md"
        file_path.write_text(sample_comments_markdown)

        comments = parse_comments_file(file_path)

        assert isinstance(comments, list)
        assert len(comments) == 3

    def test_parse_comments_extracts_body(self, sample_comments_markdown, tmp_path):
        """parse_comments_file should extract comment body text."""
        from src.markdown_parser import parse_comments_file

        file_path = tmp_path / "TestUser.md"
        file_path.write_text(sample_comments_markdown)

        comments = parse_comments_file(file_path)

        assert comments[0]["body"] == "This is the first comment about productivity tips."
        assert "multiple\nlines of text" in comments[1]["body"]

    def test_parse_comments_extracts_metadata(self, sample_comments_markdown, tmp_path):
        """parse_comments_file should extract score, date, subreddit, permalink."""
        from src.markdown_parser import parse_comments_file

        file_path = tmp_path / "TestUser.md"
        file_path.write_text(sample_comments_markdown)

        comments = parse_comments_file(file_path)

        # First comment
        assert comments[0]["score"] == 24
        assert comments[0]["subreddit"] == "productivity"
        assert comments[0]["created_utc"] is not None  # Parsed from date
        assert "reddit.com" in comments[0]["permalink"]

    def test_parse_comments_handles_multiple_subreddits(self, sample_comments_markdown, tmp_path):
        """parse_comments_file should correctly assign subreddits."""
        from src.markdown_parser import parse_comments_file

        file_path = tmp_path / "TestUser.md"
        file_path.write_text(sample_comments_markdown)

        comments = parse_comments_file(file_path)

        subreddits = [c["subreddit"] for c in comments]
        assert subreddits == ["productivity", "productivity", "remotework"]


class TestParsePersonaFile:
    """Tests for parsing persona markdown files."""

    def test_parse_persona_returns_dict(self, sample_persona_markdown, tmp_path):
        """parse_persona_file should return a dictionary."""
        from src.markdown_parser import parse_persona_file

        file_path = tmp_path / "TestUser_Persona.md"
        file_path.write_text(sample_persona_markdown)

        persona = parse_persona_file(file_path)

        assert isinstance(persona, dict)

    def test_parse_persona_extracts_username(self, sample_persona_markdown, tmp_path):
        """parse_persona_file should extract username from header."""
        from src.markdown_parser import parse_persona_file

        file_path = tmp_path / "TestUser_Persona.md"
        file_path.write_text(sample_persona_markdown)

        persona = parse_persona_file(file_path)

        assert persona["username"] == "TestUser"

    def test_parse_persona_extracts_archetype(self, sample_persona_markdown, tmp_path):
        """parse_persona_file should extract Jungian archetype."""
        from src.markdown_parser import parse_persona_file

        file_path = tmp_path / "TestUser_Persona.md"
        file_path.write_text(sample_persona_markdown)

        persona = parse_persona_file(file_path)

        assert persona["archetype"] == "The Sage"

    def test_parse_persona_extracts_top_subreddits(self, sample_persona_markdown, tmp_path):
        """parse_persona_file should extract top subreddits."""
        from src.markdown_parser import parse_persona_file

        file_path = tmp_path / "TestUser_Persona.md"
        file_path.write_text(sample_persona_markdown)

        persona = parse_persona_file(file_path)

        assert "productivity" in persona["top_subreddits"]
        assert "remotework" in persona["top_subreddits"]

    def test_parse_persona_includes_full_text(self, sample_persona_markdown, tmp_path):
        """parse_persona_file should include full persona text."""
        from src.markdown_parser import parse_persona_file

        file_path = tmp_path / "TestUser_Persona.md"
        file_path.write_text(sample_persona_markdown)

        persona = parse_persona_file(file_path)

        assert "persona_text" in persona
        assert "productivity enthusiast" in persona["persona_text"]
