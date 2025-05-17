#!/usr/bin/python3

import os
import random
from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox
from numpy.random import poisson

# CITY: The chosen city that the graph represents
CITY = "San Cristóbal de La Laguna, Canary Islands, Spain"

# NUM_NODES: The number of nodes of the network
NUM_NODES = int(os.getenv("NUM_NODES", 50))

# NUM_TIMESTEPS: The number of total timesteps
NUM_TIMESTEPS = int(os.getenv("NUM_TOTAL_TIMESTEPS", 1000))

# R: The average entrance of vehicles in the network
# per time step per node.
R = int(os.getenv("RHO", 20))

# STATIONARY_RATIO: The ratio of the timesteps that have to run first
# without taking into account the congestion.
STATIONARY_RATIO = 0.20

# STATIONARY_THRESHOLD: After that threshold we take into account
# the congestion values
STATIONARY_THRESHOLD = int(NUM_TIMESTEPS * STATIONARY_RATIO)

# T: The number of vehicles a node can process per
# timestep.
T = int(os.getenv("TAU", 5))


class Vehicle:
    """
    vehicle represents a item inside the network.
    """

    def __init__(self, start_node_idx: "int") -> "None":
        self.start_node_idx = start_node_idx
        self.end_node_idx = self._generate_end_node_idx()
        self.current_node_idx = start_node_idx

    def _generate_end_node_idx(self) -> "int":
        end_node_idx = random.choice(range(NUM_NODES))
        while end_node_idx == self.start_node_idx:
            end_node_idx = random.choice(range(NUM_NODES))
        return end_node_idx


@dataclass
class Node:
    """
    extends the graph's node with a queue.
    """

    nid: "int"
    neighbors: "list[Node]"
    queue: "deque[Vehicle]" = deque()

    def generate_vehicles(self) -> "int":
        num_vehicles = poisson(lam=R)
        for _ in range(num_vehicles):
            self.queue.append(
                Vehicle(
                    start_node_idx=self.nid,
                )
            )
        return num_vehicles

    @property
    def has_neighbors(self) -> "bool":
        return len(self.neighbors) > 0


class VehicleRouter:
    def __init__(self, g: "nx.Graph") -> "None":
        self.nodes: "list[Node]" = []
        self.g = g
        for nid in g.nodes():
            self.nodes.append(Node(nid=nid, neighbors=[]))

        for node in self.nodes:
            node.neighbors = self.get_node_neighbors(node)

    def congestion(self, timestep_updates: "list[int]") -> "float":
        return float(np.mean(timestep_updates) / (R * NUM_NODES))

    def timestep(self) -> "float":
        timestep_updates: "list[int]" = []
        for node in self.nodes:
            if not node.has_neighbors:
                continue
            routes_started = node.generate_vehicles()
            routes_ended = self.process(node)
            timestep_updates.append(routes_started - routes_ended)
        return self.congestion(timestep_updates)

    def get_node_neighbors(self, node: "Node") -> "list[Node]":
        return [
            self.nodes[i]
            for i, n in enumerate(self.g.nodes())
            if n in self.g.neighbors(node.nid)
        ]

    def process(self, node: "Node") -> "int":
        routes_ended = 0
        for _ in range(T):
            vehicle = node.queue.popleft()
            if vehicle.end_node_idx == node.nid:
                routes_ended += 1
            else:
                random_neighbor = random.choice(node.neighbors)
                random_neighbor.queue.append(vehicle)
        return routes_ended


def load_map_graph(city=CITY, network_type="drive") -> "nx.Graph":
    ox.config(use_cache=True, log_console=True)  # type: ignore
    return ox.graph_from_place(city, network_type=network_type)


if __name__ == "__main__":
    G = load_map_graph()

    router = VehicleRouter(g=G)

    for step in range(NUM_TIMESTEPS):
        current_congestion = router.timestep()
        # if step >= STATIONARY_THRESHOLD:
        print(current_congestion)
