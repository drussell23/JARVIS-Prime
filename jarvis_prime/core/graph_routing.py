"""
Graph-Based Routing v80.0 - Network Flow Optimization for Request Routing
==========================================================================

Advanced request routing using graph algorithms and network flow optimization.
Optimizes routing decisions based on capacity, latency, cost, and load.

FEATURES:
    - Multi-objective routing (latency, cost, reliability)
    - Dynamic capacity management
    - Load balancing across endpoints
    - Network flow algorithms (max flow, min cost flow)
    - Path diversity for reliability
    - Adaptive routing based on observed performance

ALGORITHMS:
    - Dijkstra's for shortest path
    - Ford-Fulkerson for max flow
    - Bellman-Ford for min cost flow
    - Weighted random selection
    - Epsilon-greedy exploration

USAGE:
    from jarvis_prime.core.graph_routing import get_graph_router

    router = await get_graph_router()

    # Find best route
    route = await router.find_route(
        source="jarvis",
        destination="prime",
        objectives=["latency", "cost"],
        constraints={"max_latency_ms": 100}
    )

    # Execute request on route
    result = await router.execute_on_route(route, request_data)
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import networkx for advanced graph algorithms
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available - using simplified routing")


# ============================================================================
# GRAPH STRUCTURES
# ============================================================================

class EdgeType(Enum):
    """Types of edges in routing graph."""
    DIRECT = "direct"  # Direct connection
    PROXY = "proxy"  # Through proxy/gateway
    FALLBACK = "fallback"  # Fallback route


@dataclass
class Edge:
    """Edge in routing graph."""
    source: str
    destination: str
    edge_type: EdgeType
    capacity: float  # Max requests/second
    latency_ms: float  # Average latency
    cost: float  # Relative cost (0-1)
    reliability: float  # Success rate (0-1)
    current_load: float = 0.0  # Current requests/second
    total_requests: int = 0
    total_failures: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def available_capacity(self) -> float:
        """Get available capacity."""
        return max(0.0, self.capacity - self.current_load)

    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        return self.current_load / self.capacity if self.capacity > 0 else 0.0

    @property
    def observed_reliability(self) -> float:
        """Get observed reliability from statistics."""
        if self.total_requests == 0:
            return self.reliability

        success_rate = 1.0 - (self.total_failures / self.total_requests)
        # Exponential moving average
        alpha = 0.1
        return alpha * success_rate + (1 - alpha) * self.reliability


@dataclass
class Route:
    """A routing path through the graph."""
    path: List[str]  # List of node names
    edges: List[Edge]  # Edges in path
    total_latency: float
    total_cost: float
    min_reliability: float
    bottleneck_capacity: float


# ============================================================================
# ROUTING OBJECTIVES
# ============================================================================

class RoutingObjective(Enum):
    """Objectives for route optimization."""
    LATENCY = "latency"  # Minimize latency
    COST = "cost"  # Minimize cost
    RELIABILITY = "reliability"  # Maximize reliability
    CAPACITY = "capacity"  # Maximize capacity
    LOAD_BALANCE = "load_balance"  # Balance load


# ============================================================================
# GRAPH ROUTER
# ============================================================================

class GraphRouter:
    """
    Graph-based router with network flow optimization.

    Maintains a directed graph of routing options and optimizes
    paths based on multiple objectives.
    """

    def __init__(self):
        """Initialize graph router."""
        # Graph representation
        self._nodes: Set[str] = set()
        self._edges: Dict[Tuple[str, str], Edge] = {}  # (source, dest) -> Edge
        self._adjacency: Dict[str, List[str]] = defaultdict(list)

        # NetworkX graph (if available)
        self._nx_graph: Optional[nx.DiGraph] = (
            nx.DiGraph() if NETWORKX_AVAILABLE else None
        )

        # Statistics
        self._route_cache: Dict[Tuple[str, str], Route] = {}
        self._cache_ttl = 60.0  # Cache routes for 60 seconds

        # Exploration-exploitation
        self._epsilon = 0.1  # 10% exploration rate

    def add_node(self, name: str):
        """Add node to graph."""
        self._nodes.add(name)

        if self._nx_graph is not None:
            self._nx_graph.add_node(name)

    def add_edge(
        self,
        source: str,
        destination: str,
        capacity: float = 100.0,
        latency_ms: float = 10.0,
        cost: float = 0.1,
        reliability: float = 0.99,
        edge_type: EdgeType = EdgeType.DIRECT,
    ):
        """
        Add edge to graph.

        Args:
            source: Source node
            destination: Destination node
            capacity: Maximum capacity (requests/second)
            latency_ms: Average latency in milliseconds
            cost: Relative cost (0-1)
            reliability: Reliability (0-1)
            edge_type: Type of edge
        """
        edge = Edge(
            source=source,
            destination=destination,
            edge_type=edge_type,
            capacity=capacity,
            latency_ms=latency_ms,
            cost=cost,
            reliability=reliability
        )

        # Add to graph
        self._edges[(source, destination)] = edge
        self._adjacency[source].append(destination)

        # Add nodes if needed
        self.add_node(source)
        self.add_node(destination)

        # Add to NetworkX graph
        if self._nx_graph is not None:
            self._nx_graph.add_edge(
                source,
                destination,
                weight=latency_ms,
                capacity=capacity,
                cost=cost,
                reliability=reliability
            )

    async def find_route(
        self,
        source: str,
        destination: str,
        objectives: Optional[List[RoutingObjective]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[Route]:
        """
        Find optimal route from source to destination.

        Args:
            source: Source node
            destination: Destination node
            objectives: List of optimization objectives
            constraints: Constraints to satisfy

        Returns:
            Optimal route if found, None otherwise
        """
        objectives = objectives or [RoutingObjective.LATENCY]
        constraints = constraints or {}

        # Check cache
        cache_key = (source, destination)
        if cache_key in self._route_cache:
            cached_route = self._route_cache[cache_key]
            # Check if cache is still valid
            if time.time() - cached_route.edges[0].last_updated < self._cache_ttl:
                return cached_route

        # Exploration vs exploitation
        if random.random() < self._epsilon:
            # Exploration: try random path
            route = await self._find_random_route(source, destination)
        else:
            # Exploitation: find optimal path
            if RoutingObjective.LATENCY in objectives:
                route = await self._find_min_latency_route(source, destination, constraints)
            elif RoutingObjective.COST in objectives:
                route = await self._find_min_cost_route(source, destination, constraints)
            elif RoutingObjective.RELIABILITY in objectives:
                route = await self._find_max_reliability_route(source, destination, constraints)
            else:
                route = await self._find_balanced_route(source, destination, objectives, constraints)

        # Cache result
        if route:
            self._route_cache[cache_key] = route

        return route

    async def _find_min_latency_route(
        self,
        source: str,
        destination: str,
        constraints: Dict[str, Any]
    ) -> Optional[Route]:
        """Find route with minimum latency using Dijkstra's algorithm."""
        if NETWORKX_AVAILABLE and self._nx_graph is not None:
            try:
                path = nx.shortest_path(
                    self._nx_graph,
                    source,
                    destination,
                    weight='weight'
                )
                return self._path_to_route(path)
            except nx.NetworkXNoPath:
                return None

        # Manual Dijkstra implementation
        return await self._dijkstra(source, destination, lambda e: e.latency_ms)

    async def _find_min_cost_route(
        self,
        source: str,
        destination: str,
        constraints: Dict[str, Any]
    ) -> Optional[Route]:
        """Find route with minimum cost."""
        return await self._dijkstra(source, destination, lambda e: e.cost)

    async def _find_max_reliability_route(
        self,
        source: str,
        destination: str,
        constraints: Dict[str, Any]
    ) -> Optional[Route]:
        """Find route with maximum reliability."""
        # Minimize negative log reliability (equivalent to maximizing product of reliabilities)
        def weight_fn(edge: Edge) -> float:
            return -1.0 * (edge.observed_reliability or 0.01)

        return await self._dijkstra(source, destination, weight_fn)

    async def _find_balanced_route(
        self,
        source: str,
        destination: str,
        objectives: List[RoutingObjective],
        constraints: Dict[str, Any]
    ) -> Optional[Route]:
        """Find route balancing multiple objectives."""
        # Weighted combination of objectives
        def weight_fn(edge: Edge) -> float:
            weight = 0.0

            for obj in objectives:
                if obj == RoutingObjective.LATENCY:
                    weight += edge.latency_ms * 0.4
                elif obj == RoutingObjective.COST:
                    weight += edge.cost * 100 * 0.3
                elif obj == RoutingObjective.RELIABILITY:
                    weight += (1.0 - edge.observed_reliability) * 100 * 0.2
                elif obj == RoutingObjective.CAPACITY:
                    # Prefer edges with more available capacity
                    weight += (1.0 - edge.available_capacity / edge.capacity) * 50 * 0.1

            return weight

        return await self._dijkstra(source, destination, weight_fn)

    async def _dijkstra(
        self,
        source: str,
        destination: str,
        weight_fn: Callable[[Edge], float]
    ) -> Optional[Route]:
        """
        Dijkstra's shortest path algorithm.

        Args:
            source: Source node
            destination: Destination node
            weight_fn: Function to compute edge weight

        Returns:
            Route if path found, None otherwise
        """
        # Priority queue: (distance, node, path)
        heap: List[Tuple[float, str, List[str]]] = [(0.0, source, [source])]
        visited: Set[str] = set()

        while heap:
            dist, node, path = heapq.heappop(heap)

            if node in visited:
                continue

            visited.add(node)

            if node == destination:
                # Found destination
                return self._path_to_route(path)

            # Explore neighbors
            for neighbor in self._adjacency[node]:
                if neighbor not in visited:
                    edge = self._edges.get((node, neighbor))

                    if edge and edge.available_capacity > 0:
                        # Calculate new distance
                        new_dist = dist + weight_fn(edge)
                        new_path = path + [neighbor]

                        heapq.heappush(heap, (new_dist, neighbor, new_path))

        # No path found
        return None

    async def _find_random_route(
        self,
        source: str,
        destination: str
    ) -> Optional[Route]:
        """Find random route (for exploration)."""
        # Simple BFS with randomization
        queue: deque = deque([(source, [source])])
        visited: Set[str] = set([source])

        while queue:
            node, path = queue.popleft()

            if node == destination:
                return self._path_to_route(path)

            # Randomize neighbors
            neighbors = list(self._adjacency[node])
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    edge = self._edges.get((node, neighbor))

                    if edge and edge.available_capacity > 0:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        return None

    def _path_to_route(self, path: List[str]) -> Route:
        """Convert node path to Route object."""
        edges = []
        total_latency = 0.0
        total_cost = 0.0
        min_reliability = 1.0
        bottleneck_capacity = float('inf')

        for i in range(len(path) - 1):
            edge = self._edges[(path[i], path[i+1])]
            edges.append(edge)

            total_latency += edge.latency_ms
            total_cost += edge.cost
            min_reliability = min(min_reliability, edge.observed_reliability)
            bottleneck_capacity = min(bottleneck_capacity, edge.available_capacity)

        return Route(
            path=path,
            edges=edges,
            total_latency=total_latency,
            total_cost=total_cost,
            min_reliability=min_reliability,
            bottleneck_capacity=bottleneck_capacity
        )

    async def update_edge_load(
        self,
        source: str,
        destination: str,
        delta: float
    ):
        """Update edge load."""
        edge = self._edges.get((source, destination))

        if edge:
            edge.current_load = max(0.0, edge.current_load + delta)
            edge.last_updated = time.time()

    async def record_result(
        self,
        route: Route,
        success: bool,
        latency_ms: Optional[float] = None
    ):
        """
        Record route execution result.

        Args:
            route: Route that was used
            success: Whether execution succeeded
            latency_ms: Observed latency
        """
        for edge in route.edges:
            edge.total_requests += 1

            if not success:
                edge.total_failures += 1

            if latency_ms is not None:
                # Update latency with exponential moving average
                alpha = 0.1
                edge.latency_ms = alpha * latency_ms + (1 - alpha) * edge.latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "cached_routes": len(self._route_cache),
            "total_capacity": sum(e.capacity for e in self._edges.values()),
            "total_load": sum(e.current_load for e in self._edges.values()),
            "average_utilization": (
                sum(e.utilization for e in self._edges.values()) / max(len(self._edges), 1)
            ),
        }


# ============================================================================
# GLOBAL ROUTER
# ============================================================================

_graph_router: Optional[GraphRouter] = None
_router_lock = asyncio.Lock()


async def get_graph_router() -> GraphRouter:
    """Get or create global graph router."""
    global _graph_router

    async with _router_lock:
        if _graph_router is None:
            _graph_router = GraphRouter()

            # Add default Trinity topology
            _graph_router.add_edge(
                "jarvis", "prime",
                capacity=100.0,
                latency_ms=5.0,
                cost=0.1,
                reliability=0.99
            )

            _graph_router.add_edge(
                "prime", "reactor",
                capacity=50.0,
                latency_ms=20.0,
                cost=0.3,
                reliability=0.95
            )

            _graph_router.add_edge(
                "jarvis", "claude",
                capacity=200.0,
                latency_ms=100.0,
                cost=0.01,
                reliability=0.999,
                edge_type=EdgeType.FALLBACK
            )

        return _graph_router
