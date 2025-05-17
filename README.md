# Model Routing with Limited Capacity Nodes

An example model of routing in complex networks with nodes that have limited capacity, showing congestion.

## Background

This project is an example implementation of a traffic network that has nodes with limited capacity. In each timestep we generate a number of vehicles in all nodes. The number of vehicles generated per node is selected with the help of [Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution).

Each new vehicle gets assigned to a random end node and according to the selected routing strategy it reaches its destination timestep by timestep. The available routing strategies are:

- _Shortest Path_: That means after we have randomly chosen the end node, we get the shortest path for the start node and the end node.
- _Random Walk_: That means that each vehicle randomly chooses a neighbor node to move in until it reaches its destination.

## Network Congestion

After every timestep we calculate the [Network Congestion](https://en.wikipedia.org/wiki/Network_congestion). The formula we use is:

```python
    def congestion(self, timestep_updates: "list[int]") -> "float":
        """
        the congestion metric eta / R * N
        """
        return float(np.mean(timestep_updates) / (R * len(self.nodes)))
```

That said, congestion of the system is `H / (Ρ * N)`, where `H` is the average increment in the number of packages in the network (i.e., the average difference between vehicles generated and vehicles removed), and `P * N` is the average number of packages generated per time step.

## Installation

Installing all the requirements of the project is easy as we provide a [Makefile](./Makefile):

```bash
make install
```

## Usage

Again the usage of the script is fairly easy:

```bash
make run
```

However the script is fairly configurable from many perspectives. That said we have a list of variables that one could adjust according to their needs:

| Name               | Description                                                                                                                                  | Type    | Default                      |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------------------------- |
| `CITY`             | The chosen city that the graph represents                                                                                                    | `str`   | "San Cristóbal de La Laguna" |
| `NUM_TIMESTEPS`    | The number of total timesteps that the network will run                                                                                      | `int`   | 1000                         |
| `OX_LOG_CONSOLE`   | An `osmnx` setting to make its output verbose or not [0/1]                                                                                   | `int`   | 0                            |
| `OX_USE_CACHE`     | An `osmnx` setting to use cache or not [0/1]                                                                                                 | `int`   | 1                            |
| `RHO`              | The average entrance of vehicles in the network per time step per node                                                                       | `int`   | 20                           |
| `ROUTING_STRATEGY` | ["shortest"/"random"] selects the routing strategy for every vehicle. Defaults to "random" if not set or value not in ["shortest", "random"] | `str`   | "random"                     |
| `STATIONARY_RATIO` | The ratio of the timesteps that have to run first without taking into account the congestion.                                                | `float` | 0.20                         |
| `TAU`              | The number of vehicles a node can process per timestep                                                                                       | `int`   | 5                            |

## Contributions

Contributions are welcomed in the repo, feel free to open an issue if you have spotted anything or even a PR if you already know a workaround for the issue.
