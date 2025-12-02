"""Tests for SentimentAnalyzer - batched sentiment analysis via Ollama."""

import pytest
import json
from unittest.mock import MagicMock, patch
from src.sentiment_analyzer import SentimentAnalyzer, SentimentResult


class TestSentimentAnalyzer:
    """Tests for batched sentiment analysis."""

    @pytest.fixture
    def sentiment_config(self):
        """Config for SentimentAnalyzer."""
        return {
            "ollama": {
                "model": "qwen3:8b",
                "temperature": 0
            },
            "sentiment": {
                "batch_size": 20,
                "threshold": 0.3
            }
        }

    @pytest.fixture
    def sample_comments(self):
        """Sample comments for analysis."""
        return [
            {"id": "c1", "author": "user1", "body": "This is exactly what I needed!"},
            {"id": "c2", "author": "user2", "body": "Meh, seen this before"},
            {"id": "c3", "author": "user3", "body": "How does this compare to existing solutions?"},
        ]

    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama response with sentiment scores."""
        return {
            "response": json.dumps([
                {"id": "c1", "score": 0.8, "rationale": "Enthusiastic positive"},
                {"id": "c2", "score": -0.4, "rationale": "Dismissive"},
                {"id": "c3", "score": 0.1, "rationale": "Neutral inquiry"},
            ])
        }

    def test_analyze_batch_returns_sentiment_results(
        self, sentiment_config, sample_comments, mock_ollama_response
    ):
        """analyze_batch returns list of SentimentResult objects."""
        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            mock_ollama.generate.return_value = mock_ollama_response
            analyzer = SentimentAnalyzer(sentiment_config)

            results = analyzer.analyze_batch(
                comments=sample_comments,
                post_title="I built an app",
                post_body="Here's my new app..."
            )

            assert len(results) == 3
            assert all(isinstance(r, SentimentResult) for r in results)

    def test_analyze_batch_extracts_correct_scores(
        self, sentiment_config, sample_comments, mock_ollama_response
    ):
        """Sentiment scores are correctly extracted from LLM response."""
        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            mock_ollama.generate.return_value = mock_ollama_response
            analyzer = SentimentAnalyzer(sentiment_config)

            results = analyzer.analyze_batch(
                comments=sample_comments,
                post_title="Test post",
                post_body=""
            )

            assert results[0].score == 0.8
            assert results[0].comment_id == "c1"
            assert results[0].username == "user1"
            assert results[0].rationale == "Enthusiastic positive"

            assert results[1].score == -0.4
            assert results[2].score == 0.1

    def test_analyze_batch_calls_ollama_with_correct_model(
        self, sentiment_config, sample_comments, mock_ollama_response
    ):
        """Ollama is called with the configured model."""
        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            mock_ollama.generate.return_value = mock_ollama_response
            analyzer = SentimentAnalyzer(sentiment_config)

            analyzer.analyze_batch(
                comments=sample_comments,
                post_title="Test",
                post_body=""
            )

            mock_ollama.generate.assert_called_once()
            call_kwargs = mock_ollama.generate.call_args[1]
            assert call_kwargs['model'] == 'qwen3:8b'
            assert call_kwargs['options']['temperature'] == 0

    def test_analyze_batch_includes_post_context_in_prompt(
        self, sentiment_config, sample_comments, mock_ollama_response
    ):
        """Prompt includes post title and body for context."""
        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            mock_ollama.generate.return_value = mock_ollama_response
            analyzer = SentimentAnalyzer(sentiment_config)

            analyzer.analyze_batch(
                comments=sample_comments,
                post_title="My Amazing App",
                post_body="This app does wonderful things"
            )

            call_kwargs = mock_ollama.generate.call_args[1]
            prompt = call_kwargs['prompt']
            assert "My Amazing App" in prompt
            assert "This app does wonderful things" in prompt

    def test_analyze_batch_handles_json_in_markdown_code_block(
        self, sentiment_config, sample_comments
    ):
        """Handles LLM response wrapped in markdown code block."""
        response_with_markdown = {
            "response": """```json
[
    {"id": "c1", "score": 0.5, "rationale": "Positive"}
]
```"""
        }
        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            mock_ollama.generate.return_value = response_with_markdown
            analyzer = SentimentAnalyzer(sentiment_config)

            results = analyzer.analyze_batch(
                comments=[sample_comments[0]],
                post_title="Test",
                post_body=""
            )

            assert len(results) == 1
            assert results[0].score == 0.5

    def test_analyze_batch_strips_think_tags_from_reasoning_models(
        self, sentiment_config, sample_comments
    ):
        """Strips <think> tags from reasoning model responses (e.g., Qwen)."""
        response_with_think = {
            "response": """<think>
Let me analyze this comment. The user seems enthusiastic...
The sentiment is clearly positive with a score around 0.9.
</think>

[
    {"id": "c1", "score": 0.9, "rationale": "Enthusiastic endorsement"}
]"""
        }
        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            mock_ollama.generate.return_value = response_with_think
            analyzer = SentimentAnalyzer(sentiment_config)

            results = analyzer.analyze_batch(
                comments=[sample_comments[0]],
                post_title="Test",
                post_body=""
            )

            assert len(results) == 1
            assert results[0].score == 0.9
            assert results[0].rationale == "Enthusiastic endorsement"


class TestAnalyzeAll:
    """Tests for analyzing multiple batches of comments."""

    @pytest.fixture
    def sentiment_config(self):
        return {
            "ollama": {"model": "qwen3:8b", "temperature": 0},
            "sentiment": {"batch_size": 2, "threshold": 0.3}  # Small batch for testing
        }

    @pytest.fixture
    def five_comments(self):
        """Five comments to test batching."""
        return [
            {"id": f"c{i}", "author": f"user{i}", "body": f"Comment {i}"}
            for i in range(1, 6)
        ]

    def test_analyze_all_splits_into_batches(self, sentiment_config, five_comments):
        """Comments are split into batches of configured size."""
        # With batch_size=2, 5 comments should create 3 batches
        mock_response = {
            "response": json.dumps([
                {"id": "c1", "score": 0.5, "rationale": "ok"},
                {"id": "c2", "score": 0.5, "rationale": "ok"}
            ])
        }

        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            mock_ollama.generate.return_value = mock_response
            analyzer = SentimentAnalyzer(sentiment_config)

            # Mock to return appropriate response for each batch
            def mock_generate(**kwargs):
                # Return response with scores for whatever comments were sent
                return mock_response

            mock_ollama.generate.side_effect = mock_generate

            results = analyzer.analyze_all(
                comments=five_comments,
                post_title="Test",
                post_body=""
            )

            # Should have called generate 3 times (batches of 2, 2, 1)
            assert mock_ollama.generate.call_count == 3

    def test_analyze_all_returns_all_results(self, sentiment_config, five_comments):
        """All comments get sentiment scores regardless of batching."""
        def make_response(comment_ids):
            return {
                "response": json.dumps([
                    {"id": cid, "score": 0.5, "rationale": "ok"} for cid in comment_ids
                ])
            }

        with patch('src.sentiment_analyzer.ollama') as mock_ollama:
            # Return appropriate responses for each batch
            mock_ollama.generate.side_effect = [
                make_response(["c1", "c2"]),
                make_response(["c3", "c4"]),
                make_response(["c5"]),
            ]
            analyzer = SentimentAnalyzer(sentiment_config)

            results = analyzer.analyze_all(
                comments=five_comments,
                post_title="Test",
                post_body=""
            )

            assert len(results) == 5
            result_ids = {r.comment_id for r in results}
            assert result_ids == {"c1", "c2", "c3", "c4", "c5"}


class TestBatchSizeValidation:
    """Tests for batch_size configuration validation."""

    def test_valid_batch_size_accepted(self):
        """Valid batch sizes (1-100) are accepted."""
        config = {
            "ollama": {"model": "qwen3:8b"},
            "sentiment": {"batch_size": 50}
        }
        analyzer = SentimentAnalyzer(config)
        assert analyzer.batch_size == 50

    def test_batch_size_too_low_raises_error(self):
        """batch_size below 1 raises ValueError."""
        config = {
            "ollama": {"model": "qwen3:8b"},
            "sentiment": {"batch_size": 0}
        }
        with pytest.raises(ValueError, match="batch_size must be between 1 and 100"):
            SentimentAnalyzer(config)

    def test_batch_size_too_high_raises_error(self):
        """batch_size above 100 raises ValueError."""
        config = {
            "ollama": {"model": "qwen3:8b"},
            "sentiment": {"batch_size": 101}
        }
        with pytest.raises(ValueError, match="batch_size must be between 1 and 100"):
            SentimentAnalyzer(config)

    def test_batch_size_boundary_values(self):
        """Boundary values (1 and 100) are accepted."""
        config_min = {
            "ollama": {"model": "qwen3:8b"},
            "sentiment": {"batch_size": 1}
        }
        config_max = {
            "ollama": {"model": "qwen3:8b"},
            "sentiment": {"batch_size": 100}
        }
        analyzer_min = SentimentAnalyzer(config_min)
        analyzer_max = SentimentAnalyzer(config_max)
        assert analyzer_min.batch_size == 1
        assert analyzer_max.batch_size == 100
