#!/usr/bin/env python3
"""Test Qdrant connectivity and basic operations.

Usage:
    python scripts/test_qdrant_ready.py
"""

import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models


def test_connection():
    """Test basic connection to Qdrant."""
    print("Testing Qdrant connection...")

    try:
        client = QdrantClient(host="localhost", port=6333)

        # Check if Qdrant is responding
        collections = client.get_collections()
        print(f"  Connected successfully!")
        print(f"  Existing collections: {len(collections.collections)}")

        for col in collections.collections:
            print(f"    - {col.name}")

        return client
    except Exception as e:
        print(f"  Failed to connect: {e}")
        return None


def test_create_collection(client: QdrantClient):
    """Test creating a temporary collection."""
    print("\nTesting collection creation...")

    test_collection = "_test_connectivity"

    try:
        # Create test collection
        client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=768,  # nomic-embed-text dimension
                distance=models.Distance.COSINE
            )
        )
        print(f"  Created test collection: {test_collection}")

        # Verify it exists
        info = client.get_collection(test_collection)
        print(f"  Vector size: {info.config.params.vectors.size}")
        print(f"  Distance metric: {info.config.params.vectors.distance}")

        return True
    except Exception as e:
        print(f"  Failed to create collection: {e}")
        return False
    finally:
        # Clean up test collection
        try:
            client.delete_collection(test_collection)
            print(f"  Cleaned up test collection")
        except Exception:
            pass


def test_vector_operations(client: QdrantClient):
    """Test basic vector storage and retrieval."""
    print("\nTesting vector operations...")

    test_collection = "_test_vectors"
    test_id = 1  # Qdrant requires integer or UUID, not arbitrary strings

    try:
        # Create collection
        client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        )

        # Insert test vector
        test_vector = [0.1] * 768
        client.upsert(
            collection_name=test_collection,
            points=[
                models.PointStruct(
                    id=test_id,
                    vector=test_vector,
                    payload={"text": "Hello world", "source": "test"}
                )
            ]
        )
        print("  Inserted test vector")

        # Search for similar vectors (qdrant-client 1.14+ uses query_points)
        results = client.query_points(
            collection_name=test_collection,
            query=test_vector,
            limit=1
        )

        if results.points and results.points[0].id == test_id:
            print("  Retrieved test vector successfully")
            print(f"  Score: {results.points[0].score}")
            print(f"  Payload: {results.points[0].payload}")
            return True
        else:
            print("  Failed to retrieve test vector")
            return False

    except Exception as e:
        print(f"  Vector operations failed: {e}")
        return False
    finally:
        # Clean up
        try:
            client.delete_collection(test_collection)
            print("  Cleaned up test collection")
        except Exception:
            pass


def main():
    """Run all connectivity tests."""
    print("=" * 50)
    print("Qdrant Connectivity Test")
    print("=" * 50)

    # Test connection
    client = test_connection()
    if not client:
        print("\n[FAIL] Cannot connect to Qdrant")
        print("Make sure Qdrant is running:")
        print("  ./scripts/setup_qdrant.sh start")
        sys.exit(1)

    # Test collection operations
    if not test_create_collection(client):
        print("\n[FAIL] Collection operations failed")
        sys.exit(1)

    # Test vector operations
    if not test_vector_operations(client):
        print("\n[FAIL] Vector operations failed")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("[PASS] All Qdrant tests passed!")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
