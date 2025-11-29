"""Vector store for Reddit RAG system using Qdrant."""

import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, List

from qdrant_client import QdrantClient
from qdrant_client.http import models
import ollama


class VectorStore:
    """Manages vector storage and retrieval using Qdrant."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize VectorStore with configuration.

        Args:
            config: Configuration dictionary with qdrant and embedding settings.
        """
        self.config = config

        # Extract Qdrant settings
        qdrant_config = config.get("qdrant", {})
        self.qdrant_host = qdrant_config.get("host", "localhost")
        self.qdrant_port = qdrant_config.get("port", 6333)
        self.vector_size = qdrant_config.get("vector_size", 768)

        # Extract collection names
        collections_config = qdrant_config.get("collections", {})
        self.comments_collection = collections_config.get("comments", "reddit_comments")
        self.personas_collection = collections_config.get("personas", "user_personas")

        # Extract embedding settings
        embedding_config = config.get("embedding", {})
        self.embedding_model = embedding_config.get("model", "nomic-embed-text")

        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed.

        Returns:
            768-dimensional embedding vector.
        """
        response = ollama.embed(model=self.embedding_model, input=text)
        return response["embeddings"][0]

    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for multiple texts in a single call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of 768-dimensional embedding vectors.
        """
        response = ollama.embed(model=self.embedding_model, input=texts)
        return response["embeddings"]

    def initialize_collections(self) -> None:
        """Create Qdrant collections if they don't exist.

        Creates both reddit_comments and user_personas collections
        with 768-dim vectors and cosine distance.
        """
        collections = [self.comments_collection, self.personas_collection]

        for collection_name in collections:
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )

    def _permalink_to_uuid(self, permalink: str) -> str:
        """Generate deterministic UUID from permalink for idempotent storage.

        Args:
            permalink: Reddit comment permalink.

        Returns:
            UUID string derived from permalink hash.
        """
        hash_bytes = hashlib.md5(permalink.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes))

    def store_comment(self, comment: Dict[str, Any], username: str) -> None:
        """Store a Reddit comment with its embedding in Qdrant.

        Args:
            comment: Comment dictionary with body, score, subreddit, etc.
            username: Reddit username who made the comment.
        """
        # Generate embedding
        vector = self.embed_text(comment["body"])

        # Generate deterministic ID from permalink
        point_id = self._permalink_to_uuid(comment["permalink"])

        # Convert timestamp to date string
        created_date = datetime.fromtimestamp(comment["created_utc"]).strftime("%Y-%m-%d")

        # Build payload
        payload = {
            "text": comment["body"],
            "username": username,
            "subreddit": comment["subreddit"],
            "score": comment["score"],
            "permalink": comment["permalink"],
            "created_date": created_date
        }

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.comments_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def _username_to_uuid(self, username: str) -> str:
        """Generate deterministic UUID from username for idempotent storage.

        Args:
            username: Reddit username.

        Returns:
            UUID string derived from username hash.
        """
        hash_bytes = hashlib.md5(username.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes))

    def store_persona(
        self,
        username: str,
        persona_text: str,
        archetype: str,
        top_subreddits: List[str],
        comment_count: int
    ) -> None:
        """Store a user persona with its embedding in Qdrant.

        Args:
            username: Reddit username.
            persona_text: Full markdown persona content.
            archetype: Jungian archetype (e.g., "The Creator").
            top_subreddits: List of most active subreddits.
            comment_count: Number of comments analyzed.
        """
        # Generate embedding from persona text
        vector = self.embed_text(persona_text)

        # Generate deterministic ID from username
        point_id = self._username_to_uuid(username)

        # Build payload
        payload = {
            "username": username,
            "persona_text": persona_text,
            "archetype": archetype,
            "top_subreddits": top_subreddits,
            "comment_count": comment_count,
            "embedded_at": int(datetime.now().timestamp())
        }

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.personas_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def user_has_comments(self, username: str) -> bool:
        """Check if a user already has comments embedded in Qdrant.

        Args:
            username: Reddit username to check.

        Returns:
            True if user has at least one comment embedded.
        """
        try:
            result = self.client.scroll(
                collection_name=self.comments_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="username",
                            match=models.MatchValue(value=username)
                        )
                    ]
                ),
                limit=1
            )
            return len(result[0]) > 0
        except Exception:
            return False

    def user_has_persona(self, username: str) -> bool:
        """Check if a user already has a persona embedded in Qdrant.

        Args:
            username: Reddit username to check.

        Returns:
            True if user has a persona embedded.
        """
        try:
            point_id = self._username_to_uuid(username)
            result = self.client.retrieve(
                collection_name=self.personas_collection,
                ids=[point_id]
            )
            return len(result) > 0
        except Exception:
            return False

    def search_similar(
        self,
        query: str,
        collection: str = "comments",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar content using semantic similarity.

        Args:
            query: Text query to search for.
            collection: Which collection to search ("comments" or "personas").
            limit: Maximum number of results to return.

        Returns:
            List of results with score and payload.
        """
        # Embed the query
        query_vector = self.embed_text(query)

        # Determine collection name
        if collection == "comments":
            collection_name = self.comments_collection
        elif collection == "personas":
            collection_name = self.personas_collection
        else:
            raise ValueError(f"Unknown collection: {collection}")

        # Query Qdrant
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        # Format results
        formatted = []
        for point in results.points:
            result = {
                "id": point.id,
                "similarity": point.score,  # Similarity score from Qdrant
                **point.payload  # Payload may include Reddit "score"
            }
            formatted.append(result)

        return formatted
