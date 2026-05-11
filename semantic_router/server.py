"""
FastAPI server exposing the semantic router as a REST API.

Provides high-throughput query routing with sub-50ms latency,
health monitoring, and dynamic route management.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .router import HybridRouter, Route, RouteDecision

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Hybrid Semantic Router",
    description="Sub-50ms intent routing using dense embeddings + BM25",
    version="0.4.0",
)

# Global router instance (initialized via /configure or startup)
_router: HybridRouter | None = None


# ─── Request / Response Models ──────────────────────────────────────

class RouteRequest(BaseModel):
    """Request body for routing a query."""
    query: str = Field(..., min_length=1, max_length=2000)


class RouteResponse(BaseModel):
    """Response with routing decision."""
    route: str
    confidence: float
    dense_score: float
    sparse_score: float
    latency_ms: float
    metadata: dict[str, Any] = {}


class BatchRouteRequest(BaseModel):
    """Request body for batch routing."""
    queries: list[str] = Field(..., min_items=1, max_items=100)


class RouteDefinition(BaseModel):
    """Schema for defining a new route."""
    name: str
    utterances: list[str] = Field(..., min_items=2)
    metadata: dict[str, Any] = {}


class ConfigureRequest(BaseModel):
    """Request body for configuring the router."""
    routes: list[RouteDefinition]
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    confidence_threshold: float = 0.4


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    routes_loaded: int
    model: str


# ─── Default Routes (Demo) ─────────────────────────────────────────

DEFAULT_ROUTES = [
    Route(
        name="sales_inquiry",
        utterances=[
            "How much does the enterprise plan cost?",
            "I want to talk to sales",
            "Pricing details for your product",
            "Can I get a quote for 50 users?",
            "What are your subscription tiers?",
        ],
        metadata={"handler": "sales_pipeline", "priority": "high"},
    ),
    Route(
        name="tech_support",
        utterances=[
            "My server is down",
            "How do I reset my password?",
            "Getting a 500 error on the dashboard",
            "The API is returning timeout errors",
            "I can't log into my account",
        ],
        metadata={"handler": "support_queue", "priority": "urgent"},
    ),
    Route(
        name="product_feedback",
        utterances=[
            "I have a feature request",
            "The new update broke my workflow",
            "Love the new dark mode feature",
            "Can you add CSV export to the reports?",
            "The mobile app needs improvement",
        ],
        metadata={"handler": "product_board", "priority": "medium"},
    ),
    Route(
        name="general_question",
        utterances=[
            "What does your company do?",
            "Tell me about your platform",
            "How is this different from competitors?",
            "Where is your company headquartered?",
            "Do you have an API?",
        ],
        metadata={"handler": "faq_bot", "priority": "low"},
    ),
]


# ─── Startup ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialize the router with default routes on server startup."""
    global _router
    _router = HybridRouter(routes=DEFAULT_ROUTES)
    logger.info("Router initialized with default routes", count=len(DEFAULT_ROUTES))


# ─── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with router status."""
    return HealthResponse(
        status="healthy" if _router else "not_configured",
        version="0.4.0",
        routes_loaded=len(_router.list_routes()) if _router else 0,
        model="all-MiniLM-L6-v2",
    )


@app.post("/route", response_model=RouteResponse)
async def route_query(request: RouteRequest):
    """
    Route a single query to the best matching intent.
    
    Returns the route name, confidence score, and latency.
    Typical latency: 5-40ms on CPU.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    decision = _router.route(request.query)
    
    return RouteResponse(
        route=decision.name,
        confidence=round(decision.confidence, 4),
        dense_score=round(decision.dense_score, 4),
        sparse_score=round(decision.sparse_score, 4),
        latency_ms=round(decision.latency_ms, 2),
        metadata=decision.metadata,
    )


@app.post("/route/batch", response_model=list[RouteResponse])
async def route_batch(request: BatchRouteRequest):
    """
    Route multiple queries in a single request.
    
    Useful for batch classification of support tickets,
    email triage, or chat log analysis.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    results = []
    for query in request.queries:
        decision = _router.route(query)
        results.append(RouteResponse(
            route=decision.name,
            confidence=round(decision.confidence, 4),
            dense_score=round(decision.dense_score, 4),
            sparse_score=round(decision.sparse_score, 4),
            latency_ms=round(decision.latency_ms, 2),
            metadata=decision.metadata,
        ))
    
    return results


@app.post("/configure")
async def configure_router(request: ConfigureRequest):
    """
    Reconfigure the router with new routes at runtime.
    
    Rebuilds both dense and sparse indexes. Takes 1-3 seconds
    depending on the number of utterances.
    """
    global _router
    
    routes = [
        Route(
            name=r.name,
            utterances=r.utterances,
            metadata=r.metadata,
        )
        for r in request.routes
    ]
    
    start = time.perf_counter()
    _router = HybridRouter(
        routes=routes,
        dense_weight=request.dense_weight,
        sparse_weight=request.sparse_weight,
        confidence_threshold=request.confidence_threshold,
    )
    build_time = (time.perf_counter() - start) * 1000
    
    logger.info("Router reconfigured", routes=len(routes), build_ms=round(build_time, 2))
    
    return {
        "status": "configured",
        "routes": len(routes),
        "build_time_ms": round(build_time, 2),
    }


@app.get("/routes")
async def list_routes():
    """List all registered routes."""
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return {"routes": _router.list_routes()}


def start_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Launch the FastAPI server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
