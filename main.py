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
nx.set_node_attributes(sensor_network, 0, 'burden')


# Simulation function (Set Display flag for individual paths taken)
def simulate(graph, display):
    random.seed()
    G = graph.copy()
    transmissions = 0
    burden = 0
    while True:
        # Sample a random source and destination
        u, v = random.sample(range(0, 6), 2)
        path = nx.dijkstra_path(G, u, v)
        transmissions += 1
        if display:
            print(f'T{transmissions}: {path}')
        # Account for all nodes except the last for transmission costs and burden
        for node in path[:-1]:
            G.nodes[node]['energy'] -= 1
            G.nodes[node]['burden'] += 1
            # Communicate to all neighbors that energy was used and make path less favorable
            for adj in G.adj[node]:
                G.adj[node][adj]['weight'] += 1
            # If any node runs out of energy, finalize attributes and return values
            if G.nodes[node]['energy'] <= 0:
                if display:
                    print(f"Node {node} is out of energy!")
                    for n in range(len(G.nodes)):
                        print(f"Node {n} | Battery Left: {G.nodes[n]['energy']} hops")
                    print(f'Completed: {transmissions} transmissions')
                for n in range(len(G.nodes)):
                    burden += G.nodes[n]['burden']
                return transmissions, burden/len(G.nodes)


# Summary of 100 simulations
avg_tr = []
avg_bd = []
for _ in range(100):
    t, b = simulate(sensor_network, False)
    avg_tr.append(t)
    avg_bd.append(b)
print(f'Average Transmission Completion [100 runs]: {round(sum(avg_tr)/len(avg_tr), 2)} transmissions')
print(f'Average Burden Across Nodes [100 runs]: {round(sum(avg_bd)/len(avg_bd), 2)} hops')
