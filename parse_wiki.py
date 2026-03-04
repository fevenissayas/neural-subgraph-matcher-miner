import networkx as nx
import pickle

G = nx.DiGraph()

with open("data/wiki-Vote.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            src, dst = int(parts[0]), int(parts[1])
            if not G.has_node(src):
                G.add_node(src, label="Node", id=str(src))
            if not G.has_node(dst):
                G.add_node(dst, label="Node", id=str(dst))
            
            G.add_edge(src, dst, weight=1.0, type="VOTED_FOR")

data_to_save = {
    'nodes': list(G.nodes(data=True)),
    'edges': list(G.edges(data=True))
}

with open("data/wiki-vote.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

print(f"Graph saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
