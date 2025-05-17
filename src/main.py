#!/usr/bin/env python3

import logging
import os
import random
from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox
from numpy.random import poisson

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("logger")

# CITY: The chosen city that the graph represents
CITY = os.getenv("CITY", "San Cristóbal de La Laguna, Canary Islands, Spain")

# NUM_TIMESTEPS: The number of total timesteps
NUM_TIMESTEPS = int(os.getenv("NUM_TIMESTEPS", 1000))

# OX_LOG_CONSOLE: [1/0] If 1 osmnx will log everything.
OX_LOG_CONSOLE = bool(int(os.getenv("OX_LOG_CONSOLE", 0)))

# OX_USE_CACHE: [1/0] If 1 osmnx will use cache.
OX_USE_CACHE = bool(int(os.getenv("OX_USE_CACHE", 1)))


# R: The average entrance of vehicles in the network
# per time step per node.
R = int(os.getenv("RHO", 20))

# ROUTING_STRATEGY: ["shortest"/"random"] selects the routing strategy
# for every vehicle. Defaults to "random" if not set or value not in ["shortest", "random"].
ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "random")


def get_vehicle_router_strategy(raw_routing_strategy: "str") -> "int":
    return 0 if raw_routing_strategy == "shortest" else 1


# VEHICLE_ROUTER_STRATEGY: Is the interpretation of the ROUTING_STRATEGY
VEHICLE_ROUTER_STRATEGY = get_vehicle_router_strategy(ROUTING_STRATEGY)

# STATIONARY_RATIO: The ratio of the timesteps that have to run first
# without taking into account the congestion.
STATIONARY_RATIO = float(os.getenv("STATIONARY_RATIO", 20))

# STATIONARY_THRESHOLD: After that threshold we take into account
# the congestion values
STATIONARY_THRESHOLD = int(NUM_TIMESTEPS * STATIONARY_RATIO)

# T: The number of vehicles a node can process per
# timestep.
T = int(os.getenv("TAU", 5))


@dataclass
class Vehicle:
    """
    vehicle represents a item (package) inside the network.
    """

    start_nid: "int"
    shortest_path: "list[int]"


@dataclass
class Node:
    """
    structures the graph's node with a queue
    """

    nid: "int"
    neighbors: "list[Node]"
    queue: "deque[Vehicle]"

    @property
    def has_neighbors(self) -> "bool":
        return len(self.neighbors) > 0


class VehicleRouterStrategy:
    """
    VehicleRouterStrategy is a helper class to
    choose between different cases/strategies
    """

    SHORTEST_PATH = 0
    RANDOM_WALK = 1


class VehicleRouter:
    """
    VehicleRouter is the main class responsible in managing the
    traffic inside the network. Is the one processing each time
    step and all vehicles and nodes.
    """

    def __init__(self, g: "nx.Graph", strategy=VEHICLE_ROUTER_STRATEGY) -> "None":
        self.nodes: "list[Node]" = []
        self.g = g
        self.strategy = strategy

        # to reduce latency we cache up-front all the neigbors
        # only applicable for the VehicleRouterStrategy.RANDOM_WALK
        for nid in g.nodes():
            if len(list(g.neighbors(nid))) == 0:
                continue
            self.nodes.append(Node(nid=nid, neighbors=[], queue=deque()))

        for node in self.nodes:
            node.neighbors = self.get_node_neighbors(node)

    def generate_vehicles(self, node: "Node") -> "int":
        """
        generates a number of cars selected with poisson distribution
        for a given node.
        """
        num_vehicles = poisson(lam=R)
        for _ in range(num_vehicles):
            node.queue.append(
                Vehicle(
                    start_nid=node.nid,
                    shortest_path=self.generate_shortest_path(node),
                )
            )
        return num_vehicles

    def generate_shortest_path(self, start_node: "Node") -> "list[int]":
        """
        picks a random end_node in the graph and generates the
        shortest path starting from the given start Node.
        """
        shortest_path_not_found = True
        while shortest_path_not_found:
            end_node: "Node" = random.choice(self.nodes)
            # make sure that end_node differs from start node
            while end_node.nid == start_node.nid:
                end_node = random.choice(self.nodes)

            try:
                shortest_path = nx.shortest_path(
                    self.g, source=start_node.nid, target=end_node.nid
                )
                shortest_path_not_found = False

            # TODO: Fixme - Need to ensure we are using connected graphs
            except nx.exception.NetworkXNoPath as e:  # type: ignore
                logger.debug(f"VehicleRouter:: {str(e)}")
                pass

        return shortest_path

    def congestion(self, timestep_updates: "list[int]") -> "float":
        """
        the congestion metric eta / R * N
        """
        return float(np.mean(timestep_updates) / (R * len(self.nodes)))

    def timestep(self) -> "float":
        """
        runs a single timestep for the network
        """
        timestep_updates: "list[int]" = []
        for node in self.nodes:
            # don't process nodes with no neighbors
            if not node.has_neighbors:
                continue

            routes_started = self.generate_vehicles(node)
            routes_ended = self.process(node)
            timestep_updates.append(routes_started - routes_ended)

        return self.congestion(timestep_updates)

    def get_node_neighbors(self, node: "Node") -> "list[Node]":
        """
        returns the Node classes which are neighbors according
        to the nx graph
        """
        return [
            self.nodes[i]
            for i, n in enumerate(self.nodes)
            if n.nid in self.g.neighbors(node.nid)
        ]

    def get_next_node(self, shortest_path: "list[int]", nid: "int") -> "Node":
        """
        returns the next Node for the given shortest path according
        to the given (current) nid
        """
        next_nid = shortest_path[shortest_path.index(nid) + 1]
        return [node for node in self.nodes if node.nid == next_nid][0]

    def process(self, node: "Node") -> "int":
        """
        processes all vehicles for a given node based on the
        router's strategy
        """
        routes_ended = 0
        for _ in range(T):
            vehicle = node.queue.popleft()
            if vehicle.shortest_path[-1] == node.nid:
                routes_ended += 1
            else:
                if self.strategy == VehicleRouterStrategy.RANDOM_WALK:
                    random_neighbor = random.choice(node.neighbors)
                    random_neighbor.queue.append(vehicle)
                else:  # VehicleRouterStrategy.SHORTEST_PATH
                    next_node = self.get_next_node(vehicle.shortest_path, node.nid)
                    next_node.queue.append(vehicle)
        return routes_ended


def load_map_graph(
    city=CITY, network_type="drive", log_console=OX_LOG_CONSOLE, use_cache=OX_USE_CACHE
) -> "nx.Graph":
    """
    load_map_graph uses osmnx to load a given map and returns its
    biggest connected component.
    """
    logger.info(f"main:: Load map for city: {city}")
    ox.settings.use_cache = use_cache  # type: ignore
    ox.settings.log_console = log_console  # type: ignore
    initial_graph = ox.graph_from_place(city, network_type=network_type)
    largest_cc = max(nx.weakly_connected_components(initial_graph), key=len)
    final_graph: "nx.Graph" = initial_graph.subgraph(largest_cc).copy()  # type: ignore
    logger.info(f"main:: Map loaded successfully: num_nodes={len(final_graph.nodes())}")

    return final_graph


if __name__ == "__main__":
    router = VehicleRouter(g=load_map_graph())

    for step in range(NUM_TIMESTEPS):
        logger.info(f"main:: Running timestep: {step}")
        current_congestion = router.timestep()
        if step >= STATIONARY_THRESHOLD:
            logger.info(f"main:: congestion: {current_congestion}")

        else:
            logger.info("main:: not in stationary state yet. skipping..")
