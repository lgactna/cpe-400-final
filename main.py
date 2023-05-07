"""
"""

import networkx as nx
import numpy as np
import random

# Simulation function (Set Display flag for individual paths taken)
def simulate(graph: nx.DiGraph, display: bool):
    """
    Simulate random transmissions within a battery-constrained sensor node network.

    `graph`'s nodes must have the attributes `energy` and `burden`. `energy` is
    the "battery" of the sensor node; `burden` should be set to 0.

    :param graph: A networkx digraph representing the network with the attributes
        mentioned above. Edge attributes and all other node attributes are ignored.
    :param display: Enables the display of various debug messages.
    :returns: A tuple of:
        - int: the number of transmissions successfully done before a node died
        - int: the average burden across nodes
        - int: the average remaining energy across nodes
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
    burden = 0
    remaining = 0

    while True:
        # Sample a random source and destination
        u, v = random.sample(range(0, 6), 2)
        path = nx.dijkstra_path(G, u, v)
        transmissions += 1
        if display:
            print(f"T{transmissions}: {path}")

        # Account for all nodes except the last for transmission costs and burden
        for node in path[:-1]:
            G.nodes[node]["energy"] -= 1
            G.nodes[node]["burden"] += 1

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

                for n in range(len(G.nodes)):
                    burden += G.nodes[n]["burden"]
                    remaining += G.nodes[n]["energy"]

                return transmissions, burden / len(G.nodes), remaining / len(G.nodes)


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
    nx.set_node_attributes(sensor_network, 0, "burden")

    # Execute 100 simulations.
    avg_tr = []
    avg_bd = []
    avg_rm = []
    for _ in range(100):
        t, b, r = simulate(sensor_network, False)
        avg_tr.append(t)
        avg_bd.append(b)
        avg_rm.append(r)

    # Print results.
    print(
        "Average Transmission Completion [100 runs]:"
        f" {round(sum(avg_tr)/len(avg_tr), 2)} transmissions"
    )
    print(
        "Average Burden Across Nodes [100 runs]:"
        f" {round(sum(avg_bd)/len(avg_bd), 2)} hops"
    )
    print(
        "Average Remaining Energy Across Nodes [100 runs]:"
        f" {round(sum(avg_rm) / len(avg_rm), 2)} hops"
    )
