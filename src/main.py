#!/usr/bin/env python3

import logging
import os
import random
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
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

# R_VALUES: A list of numbers of average entrance of vehicles
# in the network per time step per node.
R_VALUES = [0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1]

# R_REPETITIONS: The number of repetitions per R value
R_REPETITIONS = int(os.getenv("R_REPETITIONS", 10))

# ROUTING_STRATEGY: ["shortest"/"random"] selects the routing strategy
# for every vehicle. Defaults to "random" if not set or value not in ["shortest", "random"].
ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "shortest")


def get_vehicle_router_strategy(raw_routing_strategy: "str") -> "int":
    return 1 if raw_routing_strategy == "random" else 0


# VEHICLE_ROUTER_STRATEGY: Is the interpretation of the ROUTING_STRATEGY
VEHICLE_ROUTER_STRATEGY = get_vehicle_router_strategy(ROUTING_STRATEGY)

# STATIONARY_RATIO: The ratio of the timesteps that have to run first
# without taking into account the congestion.
STATIONARY_RATIO = float(os.getenv("STATIONARY_RATIO", 0.95))

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
    current_shortest_path_idx: "int"
    shortest_path: "list[int]"


@dataclass
class Node:
    """
    structures the graph's node with a queue
    """

    nid: "int"
    neighbors: "list[Node]"
    descendants: "list[Node]"
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


@dataclass
class Simulation:
    R: "float"
    congestion: "float"
    occupation_rates: "list[int]"


class VehicleRouter:
    """
    VehicleRouter is the main class responsible in managing the
    traffic inside the network. Is the one processing each time
    step and all vehicles and nodes.
    """

    def __init__(self, g: "nx.Graph", strategy=VEHICLE_ROUTER_STRATEGY) -> "None":
        self.nodes: "dict[str, Node]" = {}
        self.g = g
        self.strategy = strategy

        # to reduce latency we cache up-front all the neigbors
        # only applicable for the VehicleRouterStrategy.RANDOM_WALK
        for nid in g.nodes():
            if len(list(g.neighbors(nid))) == 0:
                continue
            self.nodes[str(nid)] = Node(
                nid=nid, neighbors=[], descendants=[], queue=deque()
            )

        for node in self.nodes.values():
            node.neighbors = self.get_node_neighbors(node)
            node.descendants = self.get_node_descendants(node)

    def generate_vehicles(self, node: "Node", r: "float") -> "int":
        """
        generates a number of cars selected with poisson distribution
        for a given node.
        """
        num_vehicles = poisson(lam=r)
        for _ in range(num_vehicles):
            node.queue.append(
                Vehicle(
                    start_nid=node.nid,
                    current_shortest_path_idx=-1,
                    shortest_path=self.generate_shortest_path(node),
                )
            )
        return num_vehicles

    def generate_shortest_path(self, start_node: "Node") -> "list[int]":
        """
        picks a random end_node in the graph and generates the
        shortest path starting from the given start Node.
        """
        end_node: "Node" = random.choice(start_node.descendants)
        # make sure that end_node differs from start node
        while end_node.nid == start_node.nid:
            end_node: "Node" = random.choice(start_node.descendants)

        # random walk does not need shortest path
        if self.strategy == VehicleRouterStrategy.RANDOM_WALK:
            return [end_node]

        return nx.shortest_path(self.g, source=start_node.nid, target=end_node.nid)

    def reset(self) -> "None":
        """
        reset all nodes
        """
        for node in self.nodes.values():
            node.queue = deque()

    def timestep(self, r: "float") -> "float":
        """
        runs a single timestep for the network
        """
        timestep_updates: "int" = 0
        for node in self.nodes.values():
            # don't process nodes with no neighbors
            if not node.has_neighbors:
                continue

            routes_started = self.generate_vehicles(node, r)
            routes_ended = self.process(node)
            timestep_updates += routes_started - routes_ended

        return timestep_updates

    def get_node_descendants(self, node: "Node") -> "list[Node]":
        """
        returns the Node classes which are descendants according
        to the nx graph
        """
        descendants = nx.descendants(self.g, node.nid)
        return [n for n in self.nodes.values() if n.nid in descendants]

    def get_node_neighbors(self, node: "Node") -> "list[Node]":
        """
        returns the Node classes which are neighbors according
        to the nx graph
        """
        return [n for n in self.nodes.values() if n.nid in self.g.neighbors(node.nid)]

    def get_next_node(self, vehicle: "Vehicle") -> "Node":
        """
        returns the next Node for the given shortest path according
        to the given (current) nid
        """
        vehicle.current_shortest_path_idx += 1
        return self.nodes[str(vehicle.shortest_path[vehicle.current_shortest_path_idx])]

    def process(self, node: "Node") -> "int":
        """
        processes all vehicles for a given node based on the
        router's strategy
        """
        routes_ended = 0
        for _ in range(T):
            if len(node.queue) == 0:
                continue
            vehicle = node.queue.popleft()
            if vehicle.shortest_path[-1] == node.nid:
                routes_ended += 1
            else:
                if self.strategy == VehicleRouterStrategy.RANDOM_WALK:
                    random_neighbor = random.choice(node.neighbors)
                    random_neighbor.queue.append(vehicle)
                else:  # VehicleRouterStrategy.SHORTEST_PATH
                    next_node = self.get_next_node(vehicle)
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


def create_scatter_plot(
    simulation: "Simulation", betweenness_list: "list[float]", betweeness_type: "str"
) -> "None":
    plt.figure(figsize=(10, 6))
    plt.scatter(betweenness_list, simulation.occupation_rates)
    plt.xlabel(f"{betweeness_type.capitalize()} Betweenness Centrality")
    plt.ylabel("Node Occupation Rate (vehicles/timestep)")
    plt.yscale("log")
    plt.title("Traffic Load vs. Betweenness Centrality")
    plt.savefig(f"fig-{betweeness_type}-betweeness-r{simulation.R}.png")


def create_eta_vs_rho_plot(
    simulations: "list[Simulation]",
    max_betweenness: "float",
    num_nodes: "int",
) -> "None":
    """
    creates a plot showing eta (congestion) versus rho (R) for different simulations
    """
    r_values = [sim.R for sim in simulations]
    eta_values = [sim.congestion for sim in simulations]

    # theoretical critical point according to paper
    rho_c = T * (num_nodes - 1) / (max_betweenness + 2 * (num_nodes - 1))

    plt.legend()

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, eta_values, "o-", markersize=6)
    plt.axvline(x=rho_c, color="g", linestyle="--", label=f"ρc ≈ {rho_c:.3f}")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    plt.xlabel("ρ (Vehicle Generation Rate)")
    plt.ylabel("η (Congestion)")
    plt.title("Phase Diagram: η vs ρ")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"eta_vs_rho_phase_diagram_{ROUTING_STRATEGY}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    simulations: "list[Simulation]" = []
    g = load_map_graph()
    node_betweenness = nx.betweenness_centrality(g, normalized=False, weight=None)

    # get the number of random r values
    router = VehicleRouter(g=g)

    for r in R_VALUES:
        # initialize metrics per repetition
        rep_congestion: "list[float]" = []

        for rep in range(R_REPETITIONS):
            router.reset()
            logger.info(
                f"settings:\n\tREP: {rep}\n\tRHO:{r}\n\tTAU:{T}\n\t"
                + f"NUM_TIMESTEPS: {NUM_TIMESTEPS}\n\tROUTING_STRATEGY: {ROUTING_STRATEGY}",
            )
            increments: "list[int]" = []

            for step in range(NUM_TIMESTEPS):
                timestep_increment = router.timestep(r)
                if step >= STATIONARY_THRESHOLD:
                    logger.info(
                        f"main:: Running timestep: {step} | timestep increment: {timestep_increment}"
                    )
                    increments.append(timestep_increment)
                else:
                    logger.info(
                        f"main:: Running timestep: {step} | not in stationary state yet. skipping..."
                    )
            congestion = np.mean(timestep_increment) / (r * len(router.nodes.keys()))
            logger.info(f"main:: average congestion {congestion}")

            occupation_rates: "list[int]" = []
            for nid, node in router.nodes.items():
                occupation_rates.append(len(node.queue))

            # append values for this repetition
            rep_congestion.append(congestion)

        simulations.append(
            Simulation(
                R=r,
                congestion=float(np.mean(rep_congestion)),
                occupation_rates=occupation_rates,
            )
        )

    node_betweenness_list: "list[float]" = []
    for nid, node in router.nodes.items():
        node_betweenness_list.append(node_betweenness[int(nid)])

    for simulation in simulations:
        create_scatter_plot(simulation, node_betweenness_list, "node")

    create_eta_vs_rho_plot(
        simulations, max(node_betweenness.values()), len(router.nodes)
    )
