"""
Core routing engine with hybrid dense + sparse matching.

Combines cosine similarity from dense embeddings with BM25 keyword
scoring for high-accuracy intent classification at sub-50ms latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingEngine

logger = structlog.get_logger(__name__)


@dataclass
class Route:
    """
    A routing destination defined by example utterances.
    
    The router learns the semantic space of a route from these examples
    by computing the centroid of their embedding vectors.
    
    Attributes:
        name: Unique identifier for this route.
        utterances: Example queries that should match this route.
        metadata: Arbitrary key-value pairs (e.g., handler function, priority).
    """
    name: str
    utterances: list[str]
    metadata: dict = field(default_factory=dict)
    
    # Internal state (populated during router initialization)
    _centroid: Optional[np.ndarray] = field(default=None, repr=False)
    _embeddings: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class RouteDecision:
    """Result of routing a query."""
    name: str
    confidence: float  # 0.0 to 1.0
    dense_score: float
    sparse_score: float
    latency_ms: float
    metadata: dict = field(default_factory=dict)


class HybridRouter:
    """
    High-performance query router using hybrid dense + sparse matching.
    
    Combines two signals:
    1. Dense: Cosine similarity between query embedding and route centroid
    2. Sparse: BM25 keyword matching against route utterances
    
    The final score is a weighted combination of both signals,
    configurable via `dense_weight` and `sparse_weight`.
    """
    
    def __init__(
        self,
        routes: list[Route],
        embedding_engine: EmbeddingEngine | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        confidence_threshold: float = 0.4,
        fallback_route: str = "__fallback__",
    ) -> None:
        """
        Initialize the router with route definitions.
        
        Args:
            routes: List of Route objects defining routing destinations.
            embedding_engine: Optional custom embedding engine.
            dense_weight: Weight for dense (embedding) similarity score.
            sparse_weight: Weight for sparse (BM25) keyword score.
            confidence_threshold: Minimum confidence to make a routing decision.
            fallback_route: Route name returned when confidence is below threshold.
        """
        self._routes = {r.name: r for r in routes}
        self._engine = embedding_engine or EmbeddingEngine()
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._threshold = confidence_threshold
        self._fallback = fallback_route
        
        # Build indexes
        self._build_dense_index()
        self._build_sparse_index()
        
        logger.info(
            "HybridRouter initialized",
            routes=len(routes),
            threshold=confidence_threshold,
            weights=f"dense={dense_weight}, sparse={sparse_weight}",
        )
    
    def _build_dense_index(self) -> None:
        """Compute embeddings and centroids for all routes."""
        for route in self._routes.values():
            embeddings = self._engine.encode(route.utterances)
            route._embeddings = embeddings
            route._centroid = np.mean(embeddings, axis=0)
            
            # Normalize the centroid
            norm = np.linalg.norm(route._centroid)
            if norm > 0:
                route._centroid = route._centroid / norm
        
        logger.info("Dense index built", routes=len(self._routes))
    
    def _build_sparse_index(self) -> None:
        """Build BM25 index over all route utterances."""
        self._sparse_docs: list[tuple[str, list[str]]] = []
        all_tokenized: list[list[str]] = []
        
        for route in self._routes.values():
            for utterance in route.utterances:
                tokens = utterance.lower().split()
                all_tokenized.append(tokens)
                self._sparse_docs.append((route.name, tokens))
        
        self._bm25 = BM25Okapi(all_tokenized)
        logger.info("Sparse (BM25) index built", documents=len(all_tokenized))
    
    def _compute_dense_scores(self, query_embedding: np.ndarray) -> dict[str, float]:
        """Compute cosine similarity between query and each route centroid."""
        scores: dict[str, float] = {}
        
        for name, route in self._routes.items():
            if route._centroid is not None:
                similarity = float(np.dot(query_embedding, route._centroid))
                scores[name] = max(0.0, similarity)  # Clamp negatives
            else:
                scores[name] = 0.0
        
        return scores
    
    def _compute_sparse_scores(self, query: str) -> dict[str, float]:
        """Compute BM25 scores and aggregate per route."""
        query_tokens = query.lower().split()
        doc_scores = self._bm25.get_scores(query_tokens)
        
        # Aggregate scores per route (average of matching utterances)
        route_scores: dict[str, list[float]] = {name: [] for name in self._routes}
        
        for i, (route_name, _) in enumerate(self._sparse_docs):
            route_scores[route_name].append(doc_scores[i])
        
        # Average and normalize
        max_score = max(max(scores) for scores in route_scores.values() if scores) or 1.0
        
        normalized: dict[str, float] = {}
        for name, scores in route_scores.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            normalized[name] = avg / max_score if max_score > 0 else 0.0
        
        return normalized
    
    def route(self, query: str) -> RouteDecision:
        """
        Route a query to the best matching route.
        
        Computes both dense (embedding) and sparse (BM25) similarity
        scores, combines them with configured weights, and returns
        the highest-scoring route if it exceeds the confidence threshold.
        
        Args:
            query: The user's input query string.
            
        Returns:
            RouteDecision with the selected route and confidence score.
        """
        start = time.perf_counter()
        
        # Dense scoring
        query_embedding = self._engine.encode_single(query)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        dense_scores = self._compute_dense_scores(query_embedding)
        
        # Sparse scoring
        sparse_scores = self._compute_sparse_scores(query)
        
        # Combine scores
        combined: dict[str, tuple[float, float, float]] = {}
        for name in self._routes:
            d = dense_scores.get(name, 0.0)
            s = sparse_scores.get(name, 0.0)
            combined_score = (d * self._dense_weight) + (s * self._sparse_weight)
            combined[name] = (combined_score, d, s)
        
        # Find best match
        best_name = max(combined, key=lambda k: combined[k][0])
        best_score, best_dense, best_sparse = combined[best_name]
        
        latency = (time.perf_counter() - start) * 1000
        
        # Check threshold
        if best_score < self._threshold:
            logger.info(
                "Below threshold, returning fallback",
                best_route=best_name,
                score=round(best_score, 4),
                threshold=self._threshold,
                latency_ms=round(latency, 2),
            )
            return RouteDecision(
                name=self._fallback,
                confidence=best_score,
                dense_score=best_dense,
                sparse_score=best_sparse,
                latency_ms=latency,
            )
        
        route = self._routes[best_name]
        
        logger.info(
            "Query routed",
            route=best_name,
            confidence=round(best_score, 4),
            dense=round(best_dense, 4),
            sparse=round(best_sparse, 4),
            latency_ms=round(latency, 2),
        )
        
        return RouteDecision(
            name=best_name,
            confidence=best_score,
            dense_score=best_dense,
            sparse_score=best_sparse,
            latency_ms=latency,
            metadata=route.metadata,
        )
    
    def __call__(self, query: str) -> RouteDecision:
        """Shorthand: router("some query") instead of router.route("some query")."""
        return self.route(query)
    
    def add_route(self, route: Route) -> None:
        """Dynamically add a new route and rebuild indexes."""
        self._routes[route.name] = route
        self._build_dense_index()
        self._build_sparse_index()
        logger.info("Route added dynamically", name=route.name)
    
    def list_routes(self) -> list[str]:
        """Return all registered route names."""
        return list(self._routes.keys())
