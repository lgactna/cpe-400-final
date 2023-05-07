import networkx as nx
import numpy as np
import random

matrix = np.array([[0, 1, 1, 0],
         [1, 0, 1, 1],
         [1, 1, 0, 1],
         [0, 1, 1, 0]])

energy = 40

random.seed()
G = nx.from_numpy_array(matrix, create_using=nx.DiGraph())
nx.set_node_attributes(G, energy, 'energy')


def simulate(graph):
    transmissions = 0
    network_cond = 1
    while network_cond:
        u, v = random.sample(range(0, 4), 2)
        path = nx.dijkstra_path(graph, u, v)
        transmissions += 1
        print(f'T{transmissions}: {path}')
        for node in path[:-1]:
            graph.nodes[node]['energy'] -= 1
            for adj in graph.adj[node]:
                graph.adj[node][adj]['weight'] += 1
            if graph.nodes[node]['energy'] <= 0:
                print(f"Node {node} is out of energy")
                print(dict(graph.nodes))
                print(f'Completed: {transmissions} transmissions')
                network_cond = 0


simulate(G)