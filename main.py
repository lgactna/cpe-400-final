"""
NOTE: The generation of a powerlaw tree is non-deterministic and may not succeed
within the default number of attempts.

By default, the script will attempt to re-generate the powerlaw tree indefinitely
until it succeeds, which is guaranteed to occur. However, you can also fix this
by removing the powerlaw tree tuple entry from `gen_tops()`. This is a constraint
of NetworkX's implementation of powerlaw tree generation.

Please ensure that you have installed the dependencies in requirements.txt
with pip install -r requirements.txt before running this script, and that you
have Python 3.8+ installed.
"""
import random
import sys
from typing import Callable, Tuple, Union
from itertools import combinations, groupby

if sys.version_info < (3, 8):
    sys.exit("Python 3.8+ is required for this script to operate.")

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
except ImportError:
    sys.exit(
        "One or more non-standard library dependencies are missing. Run `pip install -r"
        " requirements.txt` (or install matplotlib, networkx, and numpy yourself)."
    )

# Simulation function (Set Display flag for individual paths taken)
def simulate(graph: nx.DiGraph, central_node: int, display: bool) -> Tuple:
    """
    Simulate random transmissions within a battery-constrained sensor node network.

    `graph`'s nodes must have the attributes `energy`. `energy` is
    the "battery" of the sensor node.

    :param graph: A networkx digraph representing the network with the attributes
        mentioned above. Edge attributes and all other node attributes are ignored.
    :param central_node: The index of the node to treat as the C2 to which all
        other nodes send their messages to. The central node itself may not
        be the source of a transmission.
    :param display: Enables the display of various debug messages.
    :returns: A tuple of:
        - int: the number of transmissions successfully done before a node died
        - int: the overall energy efficiency of the simulation
        - nx.DiGraph: The final graph generated from the simulation.
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

    sample_set = list(range(len(G.nodes)))
    sample_set.remove(central_node)

    while True:
        # Sample a random source, excluding the central node itself
        u = random.choice(sample_set)
        path = nx.dijkstra_path(G, u, central_node)

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

                return transmissions, energy_used, G


def gnp_random_connected_graph(n, p) -> nx.Graph:
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected.

    From https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx

    :param n: The number of nodes in the graph.
    :param p: The probability of edge formation. A probability of 0 creates a
        fully disconnected graph; a probability of 1 creates a fully connected
        graph.
    :returns: The randomized graph.
    """
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
    return nx.from_numpy_array(matrix, create_using=nx.DiGraph())


def initialize_graph(
    graph: Union[nx.Graph, nx.DiGraph], initial_energy: int = 50
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


def draw_graph(graph: nx.DiGraph, layout: Callable = nx.shell_layout) -> None:
    """
    Display the digraph using the report's conventions for graph-making.

    Note that `graph` should have already been run through the simulator.

    :param graph: The graph to visualize using matplotlib.
    :param layout: The layout function to use. Shell layout is used by default.
    """
    # Create "base" plot
    plt.figure(figsize=(10, 6))

    # Calculate positions of nodes according to the spring layout,
    # which has been the most visually "appealing" in our tests
    positions = layout(graph)

    # Draw the graph using the report's conventions. Note that the weight
    # attribute of edges is added as well. Formally, the edges are in both directions
    # (because nodes should always be able to transmit to each other), but
    # the "advertised" edge weight is the only thing that matters, since it indicates
    # how much or how little that edge should be used in trying to get to the C2.
    node_labels = nx.get_node_attributes(graph, "energy")
    nx.draw(
        graph,
        pos=positions,
        node_color="lightblue",
        with_labels=True,
        node_size=500,
    )
    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(
        graph,
        pos=positions,
        edge_labels=labels,
    )

    plt.show()


def gen_tops(nodes: int) -> Tuple:
    """
    Generate the topology tuples.

    Each tuple represents the data required to simulate and graph one topology,
    where:
    - the first element is the "common" name of the topology
    - the second element is the constant graph to use for the topology
    - the third element is the layout to use when visualizing is enabled

    Note that the powerlaw tree is a nondeterministic algorithm that may take
    multiple attempts to generate. This function returns None if the powerlaw
    tree generation fails within the default number of tries, in which case the
    topology tuples should be regenerated.

    :param nodes: The number of nodes for each topology.
    :returns: The tuple of topologies described above, or `None` if powerlaw
        tree generation has failed.
    """
    try:
        # fmt: off
        tops = (
            ("Mesh", nx.complete_graph(nodes), nx.circular_layout),
            ("Star", nx.star_graph(nodes), nx.circular_layout),
            ("Wheel", nx.wheel_graph(nodes), nx.circular_layout),
            ("Ring", nx.cycle_graph(nodes), nx.circular_layout),
            ("Barbell", nx.barbell_graph(int(nodes / 2 - 1), 2), nx.spring_layout),
            ("Connected Erdős-Rényi", gnp_random_connected_graph(nodes, 0.01), nx.spring_layout),
            ("Connected Caveman", nx.connected_caveman_graph(3, 4), nx.spring_layout),
            ("Powerlaw Tree", nx.random_powerlaw_tree(nodes, 2), nx.spring_layout),
            ("Watts-Strogatz", nx.watts_strogatz_graph(nodes, 4, 0.25), nx.spring_layout),
            ("Barabasi-Albert", nx.barabasi_albert_graph(nodes, 3), nx.spring_layout),
        )
        # fmt: on
        return tops
    except nx.NetworkXError:
        # Powerlaw may fail - that's ok, just return None to signal something
        # else to try again.

        # technically this message is misplaced, but it works for our purposes
        print(
            "Powerlaw tree generation has failed - this is expected and you may see"
            " this message multiple times as the script retries it"
        )
        return None


if __name__ == "__main__":
    # The "battery" of each node. This is equivalent to the number of transmissions
    # each node can perform before dying, at which point the simulation should stop.
    INTIAL_ENERGY = 50

    # The number of nodes in each topology.
    NODES = 12

    DRAW_INITIAL_GRAPH = False
    DRAW_INTERMEDIATE_GRAPHS = False

    # Try generating the topology set; if it fails, try again.
    topologies = None
    while topologies is None:
        topologies = gen_tops(NODES)

    for top in topologies:
        tr_data = []
        eg_data = []
        eff_data = []
        for n in range(NODES):
            print(
                f"Executing 100 simulations on {top[0]} topology w/ Central Node {n}:"
            )
            # Execute 100 simulations on the same network, 'reinitialized" each time.
            avg_tr = []
            avg_eg = []
            for _ in range(100):
                t, e, final_graph = simulate(
                    initialize_graph(top[1], INTIAL_ENERGY), n, False
                )
                avg_tr.append(t)
                avg_eg.append(e)

            if DRAW_INTERMEDIATE_GRAPHS:
                draw_graph(final_graph, top[2])

            overall_tr = sum(avg_tr) / len(avg_tr)
            overall_eg = sum(avg_eg) / len(avg_eg)
            overall_eff = overall_eg / overall_tr
            tr_data.append(overall_tr)
            eg_data.append(overall_eg / NODES / INTIAL_ENERGY * 100)
            eff_data.append(overall_eff)

            # Print results.
            print(
                "-Average Transmission Completion:"
                f" {round(overall_tr, 2)} transmissions"
            )
            print(
                "-Overall Energy Used:"
                f" {round(overall_eg/NODES/INTIAL_ENERGY*100, 2)} % used"
            )
            print(
                f"-Overall Energy Efficiency: {round(overall_eff, 2)} hops/transmission"
            )
            print()

        if DRAW_INITIAL_GRAPH:
            draw_graph(top[1], top[2])

        plt.figure(dpi=100)
        plt.subplot(2, 2, 1)
        plt.bar(range(NODES), tr_data)
        plt.title("Transmission vs. Central Node")
        plt.subplot(2, 2, 2)
        plt.bar(range(NODES), eg_data)
        plt.title("Energy Used vs. Central Node")
        plt.subplot(2, 2, 3)
        plt.bar(range(NODES), eff_data)
        plt.title("Energy Efficiency vs. Central Node")
        plt.tight_layout(pad=2.0)
        plt.show()
