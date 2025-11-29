"""Integration tests for VectorStore with real Ollama and Qdrant.

These tests require:
- Ollama running with nomic-embed-text model
- Qdrant running on localhost:6333

Run with: pytest tests/test_vector_store_integration.py -v
Skip with: pytest -m "not integration"
"""

import pytest
import time

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def integration_config():
    """Config for integration tests."""
    return {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collections": {
                "comments": "_test_comments",
                "personas": "_test_personas"
            },
            "vector_size": 768
        },
        "embedding": {
            "model": "nomic-embed-text",
            "batch_size": 50
        }
    }


@pytest.fixture
def vector_store(integration_config):
    """Create VectorStore and clean up test collections after."""
    from src.vector_store import VectorStore

    store = VectorStore(integration_config)

    yield store

    # Cleanup: delete test collections
    try:
        store.client.delete_collection("_test_comments")
    except Exception:
        pass
    try:
        store.client.delete_collection("_test_personas")
    except Exception:
        pass


class TestOllamaEmbedding:
    """Integration tests for Ollama embedding."""

    def test_embed_text_returns_real_vector(self, vector_store):
        """embed_text should return a real 768-dim vector from Ollama."""
        vector = vector_store.embed_text("This is a test about productivity.")

        assert isinstance(vector, list)
        assert len(vector) == 768
        assert all(isinstance(v, float) for v in vector)

    def test_embed_texts_batch_returns_multiple_vectors(self, vector_store):
        """embed_texts_batch should return vectors for multiple texts."""
        texts = [
            "I love working from home.",
            "Remote work is the future.",
            "Office culture is overrated."
        ]

        vectors = vector_store.embed_texts_batch(texts)

        assert len(vectors) == 3
        assert all(len(v) == 768 for v in vectors)

    def test_similar_texts_have_similar_vectors(self, vector_store):
        """Semantically similar texts should have similar embeddings."""
        import math

        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        # Similar texts
        v1 = vector_store.embed_text("I enjoy working remotely from home.")
        v2 = vector_store.embed_text("Remote work from my house is great.")

        # Different text
        v3 = vector_store.embed_text("The weather today is sunny and warm.")

        sim_similar = cosine_similarity(v1, v2)
        sim_different = cosine_similarity(v1, v3)

        # Similar texts should have higher similarity
        assert sim_similar > sim_different


class TestQdrantStorage:
    """Integration tests for Qdrant storage and retrieval."""

    def test_initialize_collections_creates_real_collections(self, vector_store):
        """initialize_collections should create real Qdrant collections."""
        vector_store.initialize_collections()

        # Verify collections exist
        assert vector_store.client.collection_exists("_test_comments")
        assert vector_store.client.collection_exists("_test_personas")

    def test_store_and_search_comment(self, vector_store):
        """Should store a comment and find it via search."""
        vector_store.initialize_collections()

        # Store a comment
        comment = {
            "body": "Working from home has improved my productivity significantly.",
            "score": 100,
            "subreddit": "remotework",
            "created_utc": 1700000000,
            "permalink": "https://reddit.com/r/remotework/comments/test123/comment1"
        }
        vector_store.store_comment(comment, username="TestUser")

        # Small delay for indexing
        time.sleep(0.5)

        # Search for similar content
        results = vector_store.search_similar(
            "productivity while working remotely",
            collection="comments",
            limit=5
        )

        assert len(results) >= 1
        assert results[0]["text"] == comment["body"]
        assert results[0]["username"] == "TestUser"
        assert results[0]["score"] >= 0.5  # Should be reasonably similar

    def test_store_and_search_persona(self, vector_store):
        """Should store a persona and find it via search."""
        vector_store.initialize_collections()

        # Store a persona
        persona_text = """
        # Persona: ProductivityGuru

        This user is passionate about productivity and time management.
        They frequently discuss tools like Notion, Todoist, and time-blocking.
        Their communication style is helpful and encouraging.
        """
        vector_store.store_persona(
            username="ProductivityGuru",
            persona_text=persona_text,
            archetype="The Sage",
            top_subreddits=["productivity", "getdisciplined"],
            comment_count=75
        )

        # Small delay for indexing
        time.sleep(0.5)

        # Search for similar content
        results = vector_store.search_similar(
            "productivity tools and time management",
            collection="personas",
            limit=5
        )

        assert len(results) >= 1
        assert results[0]["username"] == "ProductivityGuru"
        assert results[0]["archetype"] == "The Sage"

    def test_idempotent_comment_storage(self, vector_store):
        """Storing same comment twice should not create duplicates."""
        vector_store.initialize_collections()

        comment = {
            "body": "Duplicate test comment.",
            "score": 50,
            "subreddit": "test",
            "created_utc": 1700000000,
            "permalink": "https://reddit.com/r/test/comments/dup123/comment1"
        }

        # Store twice
        vector_store.store_comment(comment, username="User1")
        vector_store.store_comment(comment, username="User1")

        time.sleep(0.5)

        # Search should return exactly 1 result
        results = vector_store.search_similar(
            "Duplicate test comment",
            collection="comments",
            limit=10
        )

        # Filter to exact matches
        exact_matches = [r for r in results if r["text"] == comment["body"]]
        assert len(exact_matches) == 1
