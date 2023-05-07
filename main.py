"""
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from typing import Union
from itertools import combinations, groupby

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
        # TODO: The paper currently mentions that all nodes go to a C2, which I've
        # assumed to be node 0. This is currently hardcoded (and naturally, if
        # we decide to go this route, random.sample should be random.randint).
        #
        # Still thinking on the easiest thing to write about - should the focus
        # of the report be on testing a bunch of different networks with the
        # same algorithm, or on testing the same network with different algorithms?

        # Sample a random source and destination
        u, v = random.sample(range(len(G.nodes)), 2) 
        path = nx.dijkstra_path(G, u, 0)

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

                return transmissions, energy_used / len(G.nodes), G

def gnp_random_connected_graph(n, p) -> nx.Graph:
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    # From https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

def baseline_graph() -> nx.DiGraph:
    """
    The "baseline" directed graph.
    """
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
    sensor_network = nx.from_numpy_array(matrix, create_using=nx.DiGraph())

def initialize_graph(
    graph: Union[nx.Graph, nx.DiGraph],
    initial_energy: int = 50
) -> nx.DiGraph:
    """
    Return an initialized graph for simulation.

    If the graph is undirected, the graph is converted to a digraph.
    The original graph is not modified.

    :param graph: The graph to initialize.
    :param initial_energy: The inital energy
    :returns: A digraph with the necessary attributes initialized.
    """
    # Create copy of incoming graph
    G = graph.copy()

    # If the graph is undirected, make it a directed graph with edges
    # in both directions (i.e. the graph is functionally the same)
    if isinstance(G, nx.Graph):
        G = G.to_directed()

    # Initialize node batteries and weight "hints"
    nx.set_node_attributes(G, initial_energy, "energy")
    nx.set_edge_attributes(G, 0, "weight")

    return G

def draw_graph(graph: nx.DiGraph) -> None:
    """
    Display the digraph using the report's conventions for graph-making.

    Note that `graph` should have already been run through the simulator.

    :param graph: The graph to visualize using matplotlib.
    """
    # Create "base" plot
    plt.figure(figsize=(10,6))

    # Calculate positions of nodes according to the spring layout,
    # which has been the most visually "appealing" in our tests
    positions = nx.spring_layout(graph)

    # Draw the graph using the report's conventions. Note that the weight
    # attribute of edges is added as well. Formally, the edges are in both directions
    # (because nodes should always be able to transmit to each other), but
    # the "advertised" edge weight is the only thing that matters, since it indicates
    # how much or how little that edge should be used in trying to get to the C2.
    node_labels = nx.get_node_attributes(graph, 'energy')
    nx.draw(
        graph,
        pos=positions,
        node_color='lightblue', 
        with_labels=True, 
        node_size=500,
    )
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(
        graph, 
        pos=positions, 
        edge_labels=labels,
    )

    plt.show()

if __name__ == "__main__":
    # The "battery" of each node. This is equivalent to the number of transmissions
    # each node can perform before dying, at which point the simulation should stop.
    energy = 50

    # Using random graphs for now. The idea is that a suitable graph for simulation
    # can easily be swapped out for another valid graph/diagraph by just calling
    # initialze_graph.
    sensor_network = gnp_random_connected_graph(12, 0.01)
    sensor_network = initialize_graph(sensor_network, initial_energy=energy)

    # Execute 100 simulations on the same network, 'reinitialized" each time.
    avg_tr = []
    avg_eg = []
    for _ in range(100):
        t, e, final_graph = simulate(sensor_network, True)
        avg_tr.append(t)
        avg_eg.append(e)

        draw_graph(graph)

    # Print results.
    print(
        "Average Transmission Completion [100 runs]:"
        f" {round(sum(avg_tr)/len(avg_tr), 2)} transmissions"
    )
    print(
        "Overall Energy Efficiency [100 runs]:"
        f" {round(sum(avg_eg)/len(avg_eg), 2)/energy*100} % used"
    )