import networkx as nx
import numpy as np
import random

# 6 node adj matrix
matrix = np.array([[0, 1, 0, 0, 1, 1],
                   [1, 0, 1, 0, 0, 1],
                   [0, 1, 0, 1, 0, 1],
                   [0, 0, 1, 0, 1, 1],
                   [1, 0, 0, 1, 0, 1],
                   [1, 1, 1, 1, 1, 0]])

# Hop capacity of each node
energy = 50

# Initiate graph network and node batteries
sensor_network = nx.from_numpy_array(matrix, create_using=nx.DiGraph())
nx.set_node_attributes(sensor_network, energy, 'energy')


def simulate(graph, display):
    random.seed()
    G = graph.copy()
    transmissions = 0
    while True:
        u, v = random.sample(range(0, 6), 2)
        path = nx.dijkstra_path(G, u, v)
        transmissions += 1
        if display:
            print(f'T{transmissions}: {path}')
        for node in path[:-1]:
            G.nodes[node]['energy'] -= 1
            for adj in G.adj[node]:
                G.adj[node][adj]['weight'] += 1
            if G.nodes[node]['energy'] <= 0:
                if display:
                    print(f"Node {node} is out of energy!")
                    for n in range(len(G.nodes)):
                        print(f"Node {n} | Battery Left: {G.nodes[n]['energy']} hops")
                    print(f'Completed: {transmissions} transmissions')
                return transmissions


runs = []
for _ in range(100):
    runs.append(simulate(sensor_network, False))
print(f'Average Transmission Longevity: {sum(runs)/len(runs)}')
