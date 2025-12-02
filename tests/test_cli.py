"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_config(tmp_path):
    """Create a temporary config file."""
    config_content = """
reddit:
  rate_limit_seconds: 5
  max_comments_per_user: 100

ollama:
  base_url: "http://localhost:11434"
  model: "qwen3:8b"
  temperature: 0.3

qdrant:
  host: "localhost"
  port: 6333
  collections:
    comments: "reddit_comments"
    personas: "user_personas"
  vector_size: 768

embedding:
  model: "nomic-embed-text"
  batch_size: 50
"""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(config_content)
    return config_path


class TestEmbedCommand:
    """Tests for the embed CLI command."""

    def test_embed_command_exists(self, runner):
        """embed command should be available."""
        from src.cli import cli

        result = runner.invoke(cli, ["embed", "--help"])
        assert result.exit_code == 0
        assert "Embed" in result.output or "embed" in result.output

    def test_embed_with_collection_flag(self, runner, mock_config, tmp_path):
        """embed should accept --collection flag."""
        from src.cli import cli

        # Create dummy data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with patch("src.cli.VectorStore") as mock_vs:
            mock_instance = MagicMock()
            mock_vs.return_value = mock_instance

            result = runner.invoke(cli, [
                "embed",
                "--config", str(mock_config),
                "--input-dir", str(data_dir),
                "--collection", "comments"
            ])

            # Should not fail due to missing flag
            assert "--collection" not in result.output or result.exit_code == 0

    def test_embed_initializes_collections(self, runner, mock_config, tmp_path):
        """embed should initialize Qdrant collections."""
        from src.cli import cli

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a test comment file
        (data_dir / "TestUser.md").write_text("""# Reddit Comments Analysis: u/TestUser

**Generated:** 2025-01-01 12:00:00
**Total Comments:** 1

## r/test (1 comments)

### Comment (Score: 10)
**Date:** 2025-01-01
**Link:** [View on Reddit](https://reddit.com/r/test/comments/abc/title/comment1/)

Test comment body.

---
""")

        with patch("src.cli.VectorStore") as mock_vs:
            mock_instance = MagicMock()
            mock_vs.return_value = mock_instance

            result = runner.invoke(cli, [
                "embed",
                "--config", str(mock_config),
                "--input-dir", str(data_dir),
                "--collection", "comments"
            ])

            # Should call initialize_collections
            mock_instance.initialize_collections.assert_called()


class TestSearchCommand:
    """Tests for the search CLI command."""

    def test_search_command_exists(self, runner):
        """search command should be available."""
        from src.cli import cli

        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0

    def test_search_requires_query(self, runner, mock_config):
        """search should require a query argument."""
        from src.cli import cli

        result = runner.invoke(cli, ["search", "--config", str(mock_config)])

        # Should fail or show usage without query
        assert result.exit_code != 0 or "query" in result.output.lower()


class TestAskCommand:
    """Tests for the ask CLI command."""

    def test_ask_command_exists(self, runner):
        """ask command should be available."""
        from src.cli import cli

        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0

    def test_ask_requires_question(self, runner, mock_config):
        """ask should require a question argument."""
        from src.cli import cli

        result = runner.invoke(cli, ["ask", "--config", str(mock_config)])

        # Should fail or show usage without question
        assert result.exit_code != 0 or "question" in result.output.lower()


class TestInterestCommand:
    """Tests for the interest CLI command."""

    @pytest.fixture
    def mock_config_with_sentiment(self, tmp_path):
        """Create a config file with sentiment settings."""
        config_content = """
reddit:
  rate_limit_seconds: 5
  max_comments_per_user: 100

ollama:
  base_url: "http://localhost:11434"
  model: "qwen3:8b"
  temperature: 0

sentiment:
  batch_size: 20
  threshold: 0.3
"""
        config_path = tmp_path / "settings.yaml"
        config_path.write_text(config_content)
        return config_path

    def test_interest_command_exists(self, runner):
        """interest command should be available."""
        from src.cli import cli

        result = runner.invoke(cli, ["interest", "--help"])
        assert result.exit_code == 0
        assert "interest" in result.output.lower() or "Find" in result.output

    def test_interest_requires_post_url(self, runner, mock_config_with_sentiment):
        """interest should require a post_url argument."""
        from src.cli import cli

        result = runner.invoke(cli, ["interest", "--config", str(mock_config_with_sentiment)])

        # Should fail without URL
        assert result.exit_code != 0

    def test_interest_writes_usernames_to_output(self, runner, mock_config_with_sentiment, tmp_path):
        """interest command writes interested usernames to output file."""
        from src.cli import cli
        import json

        output_file = tmp_path / "interested.txt"

        # Mock Reddit client
        mock_submission = {
            'id': 'abc123',
            'title': 'Test Post',
            'selftext': 'Test body',
            'subreddit': 'test',
            'score': 100,
            'url': 'https://reddit.com/r/test/comments/abc123/title'
        }

        mock_comments = [
            {'id': 'c1', 'author': 'fan_user', 'body': 'This is amazing!', 'score': 10, 'created_utc': 1700000000, 'permalink': '/r/test/...'},
            {'id': 'c2', 'author': 'neutral_user', 'body': 'Interesting', 'score': 5, 'created_utc': 1700001000, 'permalink': '/r/test/...'},
            {'id': 'c3', 'author': 'critic_user', 'body': 'Not impressed', 'score': 2, 'created_utc': 1700002000, 'permalink': '/r/test/...'},
        ]

        # Mock sentiment results - only fan_user is above threshold
        mock_sentiment_response = {
            "response": json.dumps([
                {"id": "c1", "score": 0.8, "rationale": "Very positive"},
                {"id": "c2", "score": 0.1, "rationale": "Neutral"},
                {"id": "c3", "score": -0.5, "rationale": "Negative"},
            ])
        }

        with patch("src.cli.RedditClient") as mock_rc, \
             patch("src.cli.SentimentAnalyzer") as mock_sa:

            # Setup Reddit mock
            mock_reddit = MagicMock()
            mock_reddit.get_submission.return_value = mock_submission
            mock_reddit.get_top_level_comments.return_value = mock_comments
            mock_rc.return_value = mock_reddit

            # Setup Sentiment mock
            from src.sentiment_analyzer import SentimentResult
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_all.return_value = [
                SentimentResult(comment_id='c1', username='fan_user', score=0.8, rationale='Very positive'),
                SentimentResult(comment_id='c2', username='neutral_user', score=0.1, rationale='Neutral'),
                SentimentResult(comment_id='c3', username='critic_user', score=-0.5, rationale='Negative'),
            ]
            mock_sa.return_value = mock_analyzer

            result = runner.invoke(cli, [
                "interest",
                "https://reddit.com/r/test/comments/abc123/title",
                "--config", str(mock_config_with_sentiment),
                "--output", str(output_file),
                "--threshold", "0.3"
            ])

            assert result.exit_code == 0

            # Verify output file
            assert output_file.exists()
            usernames = output_file.read_text().strip().split('\n')
            assert 'fan_user' in usernames
            assert 'neutral_user' not in usernames  # Below threshold
            assert 'critic_user' not in usernames  # Negative

    def test_interest_filters_by_min_score(self, runner, mock_config_with_sentiment, tmp_path):
        """interest should filter comments by minimum Reddit score."""
        from src.cli import cli
        import json

        output_file = tmp_path / "interested.txt"

        mock_submission = {'id': 'abc123', 'title': 'Test', 'selftext': '', 'subreddit': 'test', 'score': 100, 'url': 'https://reddit.com/...'}

        # One comment has low Reddit score
        mock_comments = [
            {'id': 'c1', 'author': 'user1', 'body': 'Great!', 'score': 10, 'created_utc': 1700000000, 'permalink': '/...'},
            {'id': 'c2', 'author': 'user2', 'body': 'Also great!', 'score': 0, 'created_utc': 1700001000, 'permalink': '/...'},  # Low score
        ]

        with patch("src.cli.RedditClient") as mock_rc, \
             patch("src.cli.SentimentAnalyzer") as mock_sa:

            mock_reddit = MagicMock()
            mock_reddit.get_submission.return_value = mock_submission
            mock_reddit.get_top_level_comments.return_value = mock_comments
            mock_rc.return_value = mock_reddit

            from src.sentiment_analyzer import SentimentResult
            mock_analyzer = MagicMock()
            # Only called with user1 (user2 filtered out by min-score)
            mock_analyzer.analyze_all.return_value = [
                SentimentResult(comment_id='c1', username='user1', score=0.8, rationale='Positive'),
            ]
            mock_sa.return_value = mock_analyzer

            result = runner.invoke(cli, [
                "interest",
                "https://reddit.com/...",
                "--config", str(mock_config_with_sentiment),
                "--output", str(output_file),
                "--min-score", "1"  # Filter out score=0
            ])

            # Verify analyzer was only called with filtered comments
            call_args = mock_analyzer.analyze_all.call_args
            comments_passed = call_args[1]['comments'] if call_args[1] else call_args[0][0]
            assert len(comments_passed) == 1
            assert comments_passed[0]['author'] == 'user1'
