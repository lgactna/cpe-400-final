"""
"""

import networkx as nx
import numpy as np
import random

# Simulation function (Set Display flag for individual paths taken)
def simulate(graph: nx.DiGraph, display: bool):
    """
    Simulate random transmissions within a battery-constrained sensor node network.

    `graph`'s nodes must have the attributes `energy`. `energy` is
    the "battery" of the sensor node.

    :param graph: A networkx digraph representing the network with the attributes
        mentioned above. Edge attributes and all other node attributes are ignored.
    :param display: Enables the display of various debug messages.
    :returns: A tuple of:
        - int: the number of transmissions successfully done before a node died
        - int: the overall energy efficiency of the simulation
    """
    # Initialize the random number generator, which dictates which nodes
    # need to make transmissions.
    random.seed()

    # Create a copy of the input graph, so it isn't edited in-place. This allows
    # the same graph object to be used across multiple runs while still getting
    # different results each time.
    G = graph.copy()

    # Counters/overhead
    transmissions = 0
    energy_used = 0

    while True:
        # Sample a random source and destination
        u, v = random.sample(range(len(G.nodes)), 2)
        path = nx.dijkstra_path(G, u, v)
        transmissions += 1
        if display:
            print(f"T{transmissions}: {path}")

        # Account for all nodes except the last for transmission costs and burden
        for node in path[:-1]:
            G.nodes[node]["energy"] -= 1
            energy_used += 1

            # Communicate to all neighbors that energy was used and make path less favorable
            for adj in G.adj[node]:
                G.adj[node][adj]["weight"] += 1

            # If any node runs out of energy, finalize attributes and return values
            if G.nodes[node]["energy"] <= 0:
                if display:
                    print(f"Node {node} is out of energy!")
                    for n in range(len(G.nodes)):
                        print(f"Node {n} | Battery Left: {G.nodes[n]['energy']} hops")
                    print(f"Completed: {transmissions} transmissions")

                return transmissions, energy_used / len(G.nodes)


if __name__ == "__main__":
    # Adjacency matrix for six nodes. View the report for a visual representation
    # of the network this represents.
    matrix = np.array(
        [
            [0, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 0],
        ]
    )

    # The "battery" of each node. This is equivalent to the number of transmissions
    # each node can perform before dying, at which point the simulation should stop.
    energy = 50

    # Initiate graph network and node batteries
    sensor_network = nx.from_numpy_array(matrix, create_using=nx.DiGraph())
    nx.set_node_attributes(sensor_network, energy, "energy")

    # Execute 100 simulations.
    avg_tr = []
    avg_eg = []
    for _ in range(100):
        t, e = simulate(sensor_network, False)
        avg_tr.append(t)
        avg_eg.append(e)

    # Print results.
    print(
        "Average Transmission Completion [100 runs]:"
        f" {round(sum(avg_tr)/len(avg_tr), 2)} transmissions"
    )
    print(
        "Overall Energy Efficiency [100 runs]:"
        f" {round(sum(avg_eg)/len(avg_eg), 2)/energy*100} % used"
    )
