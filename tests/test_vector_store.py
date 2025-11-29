"""Unit tests for VectorStore class."""

import pytest
from unittest.mock import patch, MagicMock
import ollama


class TestVectorStoreInit:
    """Tests for VectorStore initialization."""

    def test_init_with_config(self, sample_config):
        """VectorStore should initialize with config and store settings."""
        from src.vector_store import VectorStore

        store = VectorStore(sample_config)

        assert store.config == sample_config
        assert store.qdrant_host == "localhost"
        assert store.qdrant_port == 6333
        assert store.embedding_model == "nomic-embed-text"
        assert store.vector_size == 768

    def test_init_creates_qdrant_client(self, sample_config):
        """VectorStore should create a Qdrant client on init."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            store = VectorStore(sample_config)

            mock_client_class.assert_called_once_with(
                host="localhost",
                port=6333
            )
            assert store.client is not None


class TestEmbedText:
    """Tests for text embedding functionality."""

    def test_embed_text_returns_768_dim_vector(self, sample_config):
        """embed_text should return a 768-dimensional vector."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient"):
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                store = VectorStore(sample_config)
                vector = store.embed_text("Test text for embedding")

                assert isinstance(vector, list)
                assert len(vector) == 768
                mock_ollama.embed.assert_called_once_with(
                    model="nomic-embed-text",
                    input="Test text for embedding"
                )

    def test_embed_text_uses_configured_model(self, sample_config):
        """embed_text should use the model from config."""
        from src.vector_store import VectorStore

        sample_config["embedding"]["model"] = "custom-embed-model"

        with patch("src.vector_store.QdrantClient"):
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_ollama.embed.return_value = {"embeddings": [[0.2] * 768]}

                store = VectorStore(sample_config)
                store.embed_text("Test")

                mock_ollama.embed.assert_called_once_with(
                    model="custom-embed-model",
                    input="Test"
                )


class TestEmbedTextsBatch:
    """Tests for batch text embedding functionality."""

    def test_embed_texts_batch_returns_list_of_vectors(self, sample_config):
        """embed_texts_batch should return list of 768-dim vectors."""
        from src.vector_store import VectorStore

        texts = ["First text", "Second text", "Third text"]

        with patch("src.vector_store.QdrantClient"):
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_ollama.embed.return_value = {
                    "embeddings": [[0.1] * 768, [0.2] * 768, [0.3] * 768]
                }

                store = VectorStore(sample_config)
                vectors = store.embed_texts_batch(texts)

                assert isinstance(vectors, list)
                assert len(vectors) == 3
                assert all(len(v) == 768 for v in vectors)

    def test_embed_texts_batch_single_call(self, sample_config):
        """embed_texts_batch should make one API call for efficiency."""
        from src.vector_store import VectorStore

        texts = ["Text A", "Text B"]

        with patch("src.vector_store.QdrantClient"):
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_ollama.embed.return_value = {
                    "embeddings": [[0.1] * 768, [0.2] * 768]
                }

                store = VectorStore(sample_config)
                store.embed_texts_batch(texts)

                # Should be called once with all texts
                mock_ollama.embed.assert_called_once_with(
                    model="nomic-embed-text",
                    input=texts
                )


class TestInitializeCollections:
    """Tests for collection initialization."""

    def test_initialize_collections_creates_both_collections(self, sample_config):
        """initialize_collections should create comments and personas collections."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client.collection_exists.return_value = False
                mock_client_class.return_value = mock_client

                store = VectorStore(sample_config)
                store.initialize_collections()

                # Should check if collections exist
                assert mock_client.collection_exists.call_count == 2

                # Should create both collections
                assert mock_client.create_collection.call_count == 2

    def test_initialize_collections_skips_existing(self, sample_config):
        """initialize_collections should not recreate existing collections."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client.collection_exists.return_value = True
                mock_client_class.return_value = mock_client

                store = VectorStore(sample_config)
                store.initialize_collections()

                # Should check if collections exist
                assert mock_client.collection_exists.call_count == 2

                # Should NOT create any collections
                mock_client.create_collection.assert_not_called()

    def test_initialize_collections_uses_correct_vector_config(self, sample_config):
        """initialize_collections should use 768-dim vectors with cosine distance."""
        from src.vector_store import VectorStore
        from qdrant_client.http import models

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client.collection_exists.return_value = False
                mock_client_class.return_value = mock_client

                store = VectorStore(sample_config)
                store.initialize_collections()

                # Check the create_collection call arguments
                calls = mock_client.create_collection.call_args_list
                for call in calls:
                    vectors_config = call.kwargs.get("vectors_config")
                    assert vectors_config.size == 768
                    assert vectors_config.distance == models.Distance.COSINE


class TestStoreComment:
    """Tests for storing comments in Qdrant."""

    def test_store_comment_upserts_to_qdrant(self, sample_config, sample_comment):
        """store_comment should upsert comment with embedding to Qdrant."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                store = VectorStore(sample_config)
                store.store_comment(sample_comment, username="TestUser")

                # Should call upsert
                mock_client.upsert.assert_called_once()

                # Check collection name
                call_args = mock_client.upsert.call_args
                assert call_args.kwargs["collection_name"] == "reddit_comments"

    def test_store_comment_generates_id_from_permalink(self, sample_config, sample_comment):
        """store_comment should generate deterministic ID from permalink."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                store = VectorStore(sample_config)

                # Store same comment twice
                store.store_comment(sample_comment, username="TestUser")
                store.store_comment(sample_comment, username="TestUser")

                # Both calls should use same ID (idempotent)
                calls = mock_client.upsert.call_args_list
                id1 = calls[0].kwargs["points"][0].id
                id2 = calls[1].kwargs["points"][0].id
                assert id1 == id2

    def test_store_comment_includes_payload(self, sample_config, sample_comment):
        """store_comment should include all metadata in payload."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                store = VectorStore(sample_config)
                store.store_comment(sample_comment, username="TestUser")

                # Check payload
                call_args = mock_client.upsert.call_args
                point = call_args.kwargs["points"][0]
                payload = point.payload

                assert payload["text"] == sample_comment["body"]
                assert payload["username"] == "TestUser"
                assert payload["subreddit"] == sample_comment["subreddit"]
                assert payload["score"] == sample_comment["score"]
                assert payload["permalink"] == sample_comment["permalink"]
                assert "created_date" in payload


class TestStorePersona:
    """Tests for storing personas in Qdrant."""

    def test_store_persona_upserts_to_qdrant(self, sample_config, sample_persona_text):
        """store_persona should upsert persona with embedding to Qdrant."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                store = VectorStore(sample_config)
                store.store_persona(
                    username="TestUser",
                    persona_text=sample_persona_text,
                    archetype="The Creator",
                    top_subreddits=["productivity", "remotework"],
                    comment_count=50
                )

                # Should call upsert
                mock_client.upsert.assert_called_once()

                # Check collection name
                call_args = mock_client.upsert.call_args
                assert call_args.kwargs["collection_name"] == "user_personas"

    def test_store_persona_uses_username_as_id(self, sample_config, sample_persona_text):
        """store_persona should use username as the point ID."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                store = VectorStore(sample_config)
                store.store_persona(
                    username="TestUser",
                    persona_text=sample_persona_text,
                    archetype="The Creator",
                    top_subreddits=["productivity"],
                    comment_count=50
                )

                # Check that ID is derived from username
                call_args = mock_client.upsert.call_args
                point = call_args.kwargs["points"][0]
                # ID should be deterministic based on username
                assert point.id is not None

    def test_store_persona_includes_payload(self, sample_config, sample_persona_text):
        """store_persona should include all metadata in payload."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                store = VectorStore(sample_config)
                store.store_persona(
                    username="TestUser",
                    persona_text=sample_persona_text,
                    archetype="The Creator",
                    top_subreddits=["productivity", "remotework"],
                    comment_count=50
                )

                # Check payload
                call_args = mock_client.upsert.call_args
                point = call_args.kwargs["points"][0]
                payload = point.payload

                assert payload["username"] == "TestUser"
                assert payload["persona_text"] == sample_persona_text
                assert payload["archetype"] == "The Creator"
                assert payload["top_subreddits"] == ["productivity", "remotework"]
                assert payload["comment_count"] == 50
                assert "embedded_at" in payload


class TestUserHasComments:
    """Tests for checking if user has embedded comments."""

    def test_user_has_comments_returns_true_when_found(self, sample_config):
        """user_has_comments should return True if user has comments."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                # Simulate finding a comment
                mock_client.scroll.return_value = ([MagicMock()], None)

                store = VectorStore(sample_config)
                result = store.user_has_comments("TestUser")

                assert result is True
                mock_client.scroll.assert_called_once()

    def test_user_has_comments_returns_false_when_not_found(self, sample_config):
        """user_has_comments should return False if user has no comments."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                # Simulate no comments found
                mock_client.scroll.return_value = ([], None)

                store = VectorStore(sample_config)
                result = store.user_has_comments("NonexistentUser")

                assert result is False

    def test_user_has_comments_returns_false_on_error(self, sample_config):
        """user_has_comments should return False if Qdrant errors."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_client.scroll.side_effect = Exception("Connection error")

                store = VectorStore(sample_config)
                result = store.user_has_comments("TestUser")

                assert result is False


class TestUserHasPersona:
    """Tests for checking if user has embedded persona."""

    def test_user_has_persona_returns_true_when_found(self, sample_config):
        """user_has_persona should return True if user has persona."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                # Simulate finding a persona
                mock_client.retrieve.return_value = [MagicMock()]

                store = VectorStore(sample_config)
                result = store.user_has_persona("TestUser")

                assert result is True
                mock_client.retrieve.assert_called_once()

    def test_user_has_persona_returns_false_when_not_found(self, sample_config):
        """user_has_persona should return False if user has no persona."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                # Simulate no persona found
                mock_client.retrieve.return_value = []

                store = VectorStore(sample_config)
                result = store.user_has_persona("NonexistentUser")

                assert result is False

    def test_user_has_persona_returns_false_on_error(self, sample_config):
        """user_has_persona should return False if Qdrant errors."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama"):
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_client.retrieve.side_effect = Exception("Connection error")

                store = VectorStore(sample_config)
                result = store.user_has_persona("TestUser")

                assert result is False


class TestSearchSimilar:
    """Tests for semantic search functionality."""

    def test_search_similar_queries_qdrant(self, sample_config):
        """search_similar should query Qdrant with embedded query."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                # Mock search results
                mock_point = MagicMock()
                mock_point.id = 1
                mock_point.score = 0.95
                mock_point.payload = {"text": "Test comment", "username": "User1"}
                mock_client.query_points.return_value = MagicMock(points=[mock_point])

                store = VectorStore(sample_config)
                results = store.search_similar("productivity tips", collection="comments")

                # Should embed the query
                mock_ollama.embed.assert_called_with(
                    model="nomic-embed-text",
                    input="productivity tips"
                )

                # Should query Qdrant
                mock_client.query_points.assert_called_once()

    def test_search_similar_returns_formatted_results(self, sample_config):
        """search_similar should return results with score and payload."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}

                # Mock search results
                mock_point = MagicMock()
                mock_point.id = "abc-123"
                mock_point.score = 0.95
                mock_point.payload = {"text": "Great productivity tip!", "username": "User1"}
                mock_client.query_points.return_value = MagicMock(points=[mock_point])

                store = VectorStore(sample_config)
                results = store.search_similar("productivity", collection="comments")

                assert len(results) == 1
                assert results[0]["similarity"] == 0.95
                assert results[0]["text"] == "Great productivity tip!"
                assert results[0]["username"] == "User1"

    def test_search_similar_respects_limit(self, sample_config):
        """search_similar should pass limit to Qdrant query."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}
                mock_client.query_points.return_value = MagicMock(points=[])

                store = VectorStore(sample_config)
                store.search_similar("test", collection="comments", limit=5)

                # Check limit was passed
                call_args = mock_client.query_points.call_args
                assert call_args.kwargs["limit"] == 5

    def test_search_similar_uses_correct_collection(self, sample_config):
        """search_similar should query the specified collection."""
        from src.vector_store import VectorStore

        with patch("src.vector_store.QdrantClient") as mock_client_class:
            with patch("src.vector_store.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_ollama.embed.return_value = {"embeddings": [[0.1] * 768]}
                mock_client.query_points.return_value = MagicMock(points=[])

                store = VectorStore(sample_config)

                # Query comments collection
                store.search_similar("test", collection="comments")
                call_args = mock_client.query_points.call_args
                assert call_args.kwargs["collection_name"] == "reddit_comments"

                # Query personas collection
                store.search_similar("test", collection="personas")
                call_args = mock_client.query_points.call_args
                assert call_args.kwargs["collection_name"] == "user_personas"
