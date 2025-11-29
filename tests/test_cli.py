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
