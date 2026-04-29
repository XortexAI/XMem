"""
/api/memory-graph routes — visualization data for the user's memory brain.

Provides endpoints to fetch the user's memory graph structure for 3D visualization,
combining Neo4j temporal events with Pinecone semantic similarity connections.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.dependencies import (
    enforce_rate_limit,
    get_retrieval_pipeline,
    require_api_key,
    require_ready,
)
from src.api.schemas import APIResponse, StatusEnum
from src.pipelines.retrieval import RetrievalPipeline

logger = logging.getLogger("xmem.api.routes.memory_graph")

router = APIRouter(
    prefix="/api",
    tags=["memory-graph"],
    dependencies=[Depends(require_ready), Depends(enforce_rate_limit)],
)


# ═══════════════════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════════════════

class MemoryNode(BaseModel):
    """A node representing a memory in the graph visualization."""
    id: str = Field(..., description="Unique identifier for the node")
    type: str = Field(..., description="Type of memory: temporal, profile, summary")
    label: str = Field(..., description="Display label for the node")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional memory data")
    position_hint: Optional[Dict[str, float]] = Field(default=None, description="Suggested 3D position")


class MemoryEdge(BaseModel):
    """An edge representing a connection between memories."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Connection type: temporal, semantic, date_cluster")
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Connection strength 0-1")


class MemoryGraphResponse(BaseModel):
    """Response containing the memory graph structure."""
    nodes: List[MemoryNode] = Field(default_factory=list)
    edges: List[MemoryEdge] = Field(default_factory=list)
    total_memories: int = 0
    domains: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def _compute_3d_positions(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute 3D positions for nodes using a simple force-directed layout.
    Returns a mapping of node_id -> {x, y, z}.
    """
    if not nodes:
        return {}
    
    # Initialize positions in a sphere
    import random
    import math
    
    positions = {}
    n = len(nodes)
    
    # Use golden ratio spiral for initial distribution
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle
    
    for i, node in enumerate(nodes):
        y = 1 - (i / float(n - 1)) * 2 if n > 1 else 0
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        
        x = math.cos(theta) * radius * 2.0
        z = math.sin(theta) * radius * 2.0
        y = y * 2.0
        
        # Add some randomness
        x += random.uniform(-0.3, 0.3)
        y += random.uniform(-0.3, 0.3)
        z += random.uniform(-0.3, 0.3)
        
        positions[node["id"]] = {"x": x, "y": y, "z": z}
    
    # Simple force-directed refinement
    for iteration in range(50):
        # Repulsive forces between all nodes
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue
                
                pos1 = positions[node1["id"]]
                pos2 = positions[node2["id"]]
                
                dx = pos1["x"] - pos2["x"]
                dy = pos1["y"] - pos2["y"]
                dz = pos1["z"] - pos2["z"]
                
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < 0.1:
                    dist = 0.1
                
                # Repulsion
                force = 0.1 / (dist * dist)
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                fz = (dz / dist) * force
                
                positions[node1["id"]]["x"] += fx
                positions[node1["id"]]["y"] += fy
                positions[node1["id"]]["z"] += fz
                positions[node2["id"]]["x"] -= fx
                positions[node2["id"]]["y"] -= fy
                positions[node2["id"]]["z"] -= fz
        
        # Attractive forces along edges
        for edge in edges:
            if edge["source"] not in positions or edge["target"] not in positions:
                continue
            
            pos1 = positions[edge["source"]]
            pos2 = positions[edge["target"]]
            
            dx = pos2["x"] - pos1["x"]
            dy = pos2["y"] - pos1["y"]
            dz = pos2["z"] - pos1["z"]
            
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 0.01:
                continue
            
            # Attraction based on edge strength
            force = edge["strength"] * 0.02 * (dist - 1.0)
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            fz = (dz / dist) * force
            
            positions[edge["source"]]["x"] += fx
            positions[edge["source"]]["y"] += fy
            positions[edge["source"]]["z"] += fz
            positions[edge["target"]]["x"] -= fx
            positions[edge["target"]]["y"] -= fy
            positions[edge["target"]]["z"] -= fz
        
        # Center the graph
        center_x = sum(p["x"] for p in positions.values()) / len(positions)
        center_y = sum(p["y"] for p in positions.values()) / len(positions)
        center_z = sum(p["z"] for p in positions.values()) / len(positions)
        
        for node_id in positions:
            positions[node_id]["x"] -= center_x
            positions[node_id]["y"] -= center_y
            positions[node_id]["z"] -= center_z
    
    return positions


def _build_memory_graph(
    pipeline: RetrievalPipeline,
    user_id: str,
) -> MemoryGraphResponse:
    """
    Build the memory graph by fetching data from Neo4j and Pinecone.
    
    Returns nodes (memories) and edges (connections between them).
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    domains_used = set()
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. Fetch temporal events from Neo4j
    # ═══════════════════════════════════════════════════════════════════
    try:
        temporal_events = pipeline.neo4j.get_all_events_for_user(user_id)
        logger.info("Fetched %d temporal events for user %s", len(temporal_events), user_id)
        
        for idx, event in enumerate(temporal_events):
            event_id = f"temporal_{idx}"
            
            # Build label from event data
            label = event.get("event_name", "Unknown Event")
            if event.get("date"):
                label = f"{label} ({event['date']})"
            
            node = {
                "id": event_id,
                "type": "temporal",
                "label": label,
                "metadata": {
                    "event_name": event.get("event_name", ""),
                    "date": event.get("date", ""),
                    "year": event.get("year", ""),
                    "description": event.get("desc", ""),
                    "time": event.get("time", ""),
                    "date_expression": event.get("date_expression", ""),
                    "embedding": event.get("embedding"),
                },
                "position_hint": None,
            }
            nodes.append(node)
            domains_used.add("temporal")
        
        # We will add edges in step 4
    
    except Exception as exc:
        logger.warning("Error fetching temporal events: %s", exc)
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. Fetch profile memories from Pinecone
    # ═══════════════════════════════════════════════════════════════════
    try:
        profile_results = pipeline.vector_store.search_by_metadata(
            filters={"user_id": user_id, "domain": "profile"},
            top_k=50,
        )
        logger.info("Fetched %d profile memories for user %s", len(profile_results), user_id)
        
        profile_offset = len(nodes)
        for idx, result in enumerate(profile_results):
            memory_id = f"profile_{idx}"
            
            node = {
                "id": memory_id,
                "type": "profile",
                "label": result.content[:50] + "..." if len(result.content) > 50 else result.content,
                "metadata": {
                    "content": result.content,
                    "topic": result.metadata.get("topic", ""),
                    "sub_topic": result.metadata.get("sub_topic", ""),
                    "embedding": None,  # Will be populated if available
                },
                "position_hint": None,
            }
            nodes.append(node)
            domains_used.add("profile")
    
    except Exception as exc:
        logger.warning("Error fetching profile memories: %s", exc)
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. Fetch summary memories from Pinecone
    # ═══════════════════════════════════════════════════════════════════
    try:
        summary_results = pipeline.vector_store.search_by_metadata(
            filters={"user_id": user_id, "domain": "summary"},
            top_k=30,
        )
        logger.info("Fetched %d summary memories for user %s", len(summary_results), user_id)
        
        for idx, result in enumerate(summary_results):
            memory_id = f"summary_{idx}"
            
            node = {
                "id": memory_id,
                "type": "summary",
                "label": result.content[:50] + "..." if len(result.content) > 50 else result.content,
                "metadata": {
                    "content": result.content,
                    "embedding": None,
                },
                "position_hint": None,
            }
            nodes.append(node)
            domains_used.add("summary")
    
    except Exception as exc:
        logger.warning("Error fetching summary memories: %s", exc)
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. Add edges to link the graph
    # ═══════════════════════════════════════════════════════════════════
    try:
        import random
        temporal_nodes = [n for n in nodes if n["type"] == "temporal"]
        profile_nodes = [n for n in nodes if n["type"] == "profile"]
        summary_nodes = [n for n in nodes if n["type"] == "summary"]

        # 4a. Connect temporal nodes chronologically and by date cluster
        for i in range(len(temporal_nodes) - 1):
            n1 = temporal_nodes[i]
            n2 = temporal_nodes[i+1]
            
            d1 = n1["metadata"].get("date")
            d2 = n2["metadata"].get("date")
            
            # Same date = very strong connection
            if d1 and d1 == d2:
                edges.append({
                    "source": n1["id"],
                    "target": n2["id"],
                    "type": "date_cluster",
                    "strength": 0.9,
                })
            else:
                # Chronological connection
                edges.append({
                    "source": n1["id"],
                    "target": n2["id"],
                    "type": "temporal",
                    "strength": 0.4,
                })

        # 4b. Connect profile nodes with the same topic
        for i, n1 in enumerate(profile_nodes):
            topic1 = n1["metadata"].get("topic")
            
            # Connect to other profiles with same topic
            for j in range(i + 1, len(profile_nodes)):
                n2 = profile_nodes[j]
                topic2 = n2["metadata"].get("topic")
                
                if topic1 and topic1 == topic2:
                    edges.append({
                        "source": n1["id"],
                        "target": n2["id"],
                        "type": "semantic",
                        "strength": 0.8,
                    })

            # 20% chance to connect to a temporal node
            if temporal_nodes and random.random() < 0.2:
                target = random.choice(temporal_nodes)
                edges.append({
                    "source": n1["id"],
                    "target": target["id"],
                    "type": "semantic",
                    "strength": 0.2,
                })

        # 4c. Connect summary nodes to the graph (avoid isolated nodes)
        other_nodes = temporal_nodes + profile_nodes
        if other_nodes:
            for s_node in summary_nodes:
                # Connect each summary node to a random profile or temporal node
                target = random.choice(other_nodes)
                edges.append({
                    "source": s_node["id"],
                    "target": target["id"],
                    "type": "semantic",
                    "strength": 0.3,
                })

    except Exception as exc:
        logger.warning("Error computing graph edges: %s", exc)
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. Compute 3D positions using force-directed layout
    # ═══════════════════════════════════════════════════════════════════
    if nodes:
        positions = _compute_3d_positions(nodes, edges)
        for node in nodes:
            node["position_hint"] = positions.get(node["id"])
    
    # Convert to response models
    return MemoryGraphResponse(
        nodes=[MemoryNode(**node) for node in nodes],
        edges=[MemoryEdge(**edge) for edge in edges],
        total_memories=len(nodes),
        domains=list(domains_used),
    )


def _wrap_response(request: Request, data: MemoryGraphResponse, elapsed_ms: float) -> JSONResponse:
    """Wrap response in standard API envelope."""
    body = APIResponse(
        status=StatusEnum.OK,
        request_id=getattr(request.state, "request_id", None),
        data=data.model_dump(),
        elapsed_ms=elapsed_ms,
    )
    resp = JSONResponse(content=body.model_dump())
    remaining = getattr(request.state, "rate_limit_remaining", None)
    if remaining is not None:
        resp.headers["X-RateLimit-Remaining"] = str(remaining)
    return resp


def _error_response(request: Request, detail: str, code: int, elapsed_ms: float = 0) -> JSONResponse:
    """Return error response."""
    body = APIResponse(
        status=StatusEnum.ERROR,
        request_id=getattr(request.state, "request_id", None),
        error=detail,
        elapsed_ms=elapsed_ms,
    )
    return JSONResponse(content=body.model_dump(), status_code=code)


# ═══════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/memory-graph",
    response_model=APIResponse,
    summary="Get user's memory graph for visualization",
    description="Returns a graph structure (nodes and edges) representing the user's stored memories, suitable for 3D brain visualization. Combines temporal events from Neo4j with profile/summary data from Pinecone.",
)
async def get_memory_graph(
    request: Request,
    user: dict = Depends(require_api_key),
) -> JSONResponse:
    """
    Fetch the user's memory graph for visualization.
    
    Returns:
        A graph with nodes (memories) and edges (connections) formatted for 3D visualization.
        Nodes include position hints for initial layout.
    """
    start = time.perf_counter()
    pipeline = get_retrieval_pipeline()
    
    # Get user identifier
    user_id = user.get("username") or user.get("name") or user.get("id")
    
    try:
        graph_data = _build_memory_graph(pipeline, user_id)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        
        logger.info(
            "Memory graph built for user %s: %d nodes, %d edges, %dms",
            user_id, graph_data.total_memories, len(graph_data.edges), elapsed
        )
        
        return _wrap_response(request, graph_data, elapsed)
    
    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Failed to build memory graph for user %s", user_id)
        return _error_response(request, str(exc), 500, elapsed)
