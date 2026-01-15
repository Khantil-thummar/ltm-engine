"""Qdrant vector store repository."""

import uuid
from datetime import datetime
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from ltm_engine.config import Settings

logger = structlog.get_logger(__name__)


class QdrantRepository:
    """
    Qdrant vector store repository.
    
    Handles vector storage and semantic similarity search.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection_name = settings.qdrant_collection_name
        self._dimensions = settings.embedding_dimensions

    async def init_collection(self) -> None:
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            await self._client.get_collection(self._collection_name)
            logger.info("Qdrant collection exists", collection=self._collection_name)
        except UnexpectedResponse:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self._dimensions,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            # Create payload indexes for filtering
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="agent_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="memory_type",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="status",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Created Qdrant collection", collection=self._collection_name)

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()

    async def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            await self._client.get_collections()
            return True
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return False

    async def upsert(
        self,
        vector_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or update a vector."""
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    async def upsert_batch(
        self,
        points: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Insert or update multiple vectors."""
        if not points:
            return

        qdrant_points = [
            qdrant_models.PointStruct(
                id=point[0],
                vector=point[1],
                payload=point[2],
            )
            for point in points
        ]

        await self._client.upsert(
            collection_name=self._collection_name,
            points=qdrant_points,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        agent_id: str | None = None,
        memory_types: list[str] | None = None,
        status: list[str] | None = None,
        min_confidence: float | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.
        
        Returns list of results with id, score, and payload.
        """
        # Build filter conditions
        must_conditions: list[qdrant_models.FieldCondition] = []

        if agent_id:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="agent_id",
                    match=qdrant_models.MatchValue(value=agent_id),
                )
            )

        if memory_types:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="memory_type",
                    match=qdrant_models.MatchAny(any=memory_types),
                )
            )

        if status:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="status",
                    match=qdrant_models.MatchAny(any=status),
                )
            )

        if min_confidence is not None:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="confidence",
                    range=qdrant_models.Range(gte=min_confidence),
                )
            )

        if time_start:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="created_at_ts",
                    range=qdrant_models.Range(gte=time_start.timestamp()),
                )
            )

        if time_end:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="created_at_ts",
                    range=qdrant_models.Range(lte=time_end.timestamp()),
                )
            )

        query_filter = None
        if must_conditions:
            query_filter = qdrant_models.Filter(must=must_conditions)

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in response.points
        ]

    async def get_by_id(self, vector_id: str) -> dict[str, Any] | None:
        """Get a vector by ID."""
        results = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[vector_id],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        point = results[0]
        return {
            "id": str(point.id),
            "vector": point.vector,
            "payload": point.payload,
        }

    async def delete(self, vector_id: str) -> None:
        """Delete a vector by ID."""
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=[vector_id],
            ),
        )

    async def delete_batch(self, vector_ids: list[str]) -> None:
        """Delete multiple vectors."""
        if not vector_ids:
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=vector_ids,
            ),
        )

    async def update_payload(
        self,
        vector_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Update the payload of a vector."""
        await self._client.set_payload(
            collection_name=self._collection_name,
            payload=payload,
            points=[vector_id],
        )

    async def count(
        self,
        agent_id: str | None = None,
        memory_type: str | None = None,
    ) -> int:
        """Count vectors with optional filters."""
        must_conditions: list[qdrant_models.FieldCondition] = []

        if agent_id:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="agent_id",
                    match=qdrant_models.MatchValue(value=agent_id),
                )
            )

        if memory_type:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="memory_type",
                    match=qdrant_models.MatchValue(value=memory_type),
                )
            )

        count_filter = None
        if must_conditions:
            count_filter = qdrant_models.Filter(must=must_conditions)

        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=count_filter,
            exact=True,
        )
        return result.count

    async def find_similar(
        self,
        query_vector: list[float],
        agent_id: str,
        memory_type: str,
        threshold: float = 0.85,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find highly similar vectors for conflict detection.
        
        Returns vectors above the similarity threshold.
        """
        must_conditions = [
            qdrant_models.FieldCondition(
                key="agent_id",
                match=qdrant_models.MatchValue(value=agent_id),
            ),
            qdrant_models.FieldCondition(
                key="memory_type",
                match=qdrant_models.MatchValue(value=memory_type),
            ),
            qdrant_models.FieldCondition(
                key="status",
                match=qdrant_models.MatchValue(value="active"),
            ),
        ]

        must_not_conditions = []
        if exclude_ids:
            must_not_conditions.append(
                qdrant_models.HasIdCondition(
                    has_id=exclude_ids,
                )
            )

        query_filter = qdrant_models.Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None,
        )

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=10,
            query_filter=query_filter,
            score_threshold=threshold,
            with_payload=True,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in response.points
        ]
