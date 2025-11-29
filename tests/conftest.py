"""Shared pytest fixtures for Reddit RAG tests."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration matching settings.yaml structure."""
    return {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collections": {
                "comments": "reddit_comments",
                "personas": "user_personas"
            },
            "vector_size": 768
        },
        "embedding": {
            "model": "nomic-embed-text",
            "batch_size": 50
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "qwen3:8b",
            "temperature": 0.3
        }
    }


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for unit tests."""
    mock = MagicMock()

    # Mock get_collections to return empty list
    mock.get_collections.return_value = MagicMock(collections=[])

    # Mock collection_exists
    mock.collection_exists.return_value = False

    # Mock create_collection
    mock.create_collection.return_value = True

    # Mock upsert
    mock.upsert.return_value = MagicMock(status="completed")

    # Mock query_points with empty results
    mock.query_points.return_value = MagicMock(points=[])

    return mock


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for unit tests."""
    mock = MagicMock()

    # Mock embed to return 768-dim vector
    mock.embed.return_value = {
        "embeddings": [[0.1] * 768]
    }

    return mock


@pytest.fixture
def sample_comment() -> Dict[str, Any]:
    """Sample Reddit comment data."""
    return {
        "body": "This is a test comment about productivity and work-life balance.",
        "score": 42,
        "subreddit": "productivity",
        "created_utc": 1700000000,
        "permalink": "https://reddit.com/r/productivity/comments/abc123/test/def456"
    }


@pytest.fixture
def sample_comments() -> List[Dict[str, Any]]:
    """Multiple sample comments for batch testing."""
    return [
        {
            "body": "I love working from home, the flexibility is amazing.",
            "score": 150,
            "subreddit": "remotework",
            "created_utc": 1700000000,
            "permalink": "https://reddit.com/r/remotework/comments/abc123/test/comment1"
        },
        {
            "body": "Burnout is real. Take care of your mental health.",
            "score": 89,
            "subreddit": "antiwork",
            "created_utc": 1700001000,
            "permalink": "https://reddit.com/r/antiwork/comments/def456/test/comment2"
        },
        {
            "body": "The Pomodoro technique changed my life!",
            "score": 234,
            "subreddit": "productivity",
            "created_utc": 1700002000,
            "permalink": "https://reddit.com/r/productivity/comments/ghi789/test/comment3"
        }
    ]


@pytest.fixture
def sample_persona_text() -> str:
    """Sample persona markdown content."""
    return """# Persona Analysis: TestUser

## Jungian Archetype: The Creator

TestUser demonstrates creative problem-solving and a desire to build meaningful things.

## Key Themes
- Productivity optimization
- Work-life balance
- Remote work advocacy

## Communication Style
Direct, helpful, and supportive. Uses first-person examples.
"""


@pytest.fixture
def sample_embedding() -> List[float]:
    """Sample 768-dimensional embedding vector."""
    return [0.1] * 768
