import networkx as nx
import random

random.seed()
G = nx.DiGraph()
G.add_node(1, energy=40)
G.add_node(2, energy=40)
G.add_node(3, energy=40)
G.add_node(4, energy=40)
G.add_weighted_edges_from([(1, 2, G.nodes[1]['energy']),
                            (1, 3, G.nodes[1]['energy']),
                            (2, 1, G.nodes[2]['energy']),
                            (2, 3, G.nodes[2]['energy']),
                            (2, 4, G.nodes[2]['energy']),
                            (3, 1, G.nodes[3]['energy']),
                            (3, 2, G.nodes[3]['energy']),
                            (3, 4, G.nodes[3]['energy']),
                            (4, 1, G.nodes[4]['energy']),
                            (4, 3, G.nodes[4]['energy'])])


def simulate(graph):
    transmissions = 0
    network_cond = 1
    while network_cond:
        u, v = random.sample(range(1, 5), 2)
        path = nx.dijkstra_path(graph, u, v)
        length = nx.dijkstra_path_length(graph, u, v)
        edges = list(zip(path[0:], path[1:]))
        transmissions += 1
        print(f'T{transmissions}: {path} {length}')
        for node in path[:-1]:
            graph.nodes[node]['energy'] -= 1
        for edge in edges:
            graph.edges[edge]['weight'] = graph.nodes[edge[0]]['energy']
            if graph.edges[edge]['weight'] <= 0:
                graph.remove_edge(*edge)
                print(f"Node {edge[0]} is out of energy")
                print(dict(graph.nodes))
                print(f'Completed: {transmissions} transmissions')
                network_cond = 0


simulate(G)


