import networkx as nx
import matplotlib.pyplot as plt

def draw_motif(G, title, filename):
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=12, font_weight='bold', 
            arrows=True, arrowsize=20, edge_color='gray')
    plt.title(title, fontsize=14)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# 1. Feed-Forward Loop (Transitive Support)
G_ffl = nx.DiGraph()
G_ffl.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
draw_motif(G_ffl, "Feed-Forward Loop (Transitive Support)", "motif_feed_forward.png")

# 2. Reciprocal Voting (Mutual Support)
G_rec = nx.DiGraph()
G_rec.add_edges_from([('A', 'B'), ('B', 'A')])
draw_motif(G_rec, "Reciprocal Voting (Mutual Support)", "motif_reciprocal.png")

# 3. High Out-Degree Star (Gatekeeper)
G_out = nx.DiGraph()
G_out.add_edges_from([('Gatekeeper', 'Cand1'), ('Gatekeeper', 'Cand2'), ('Gatekeeper', 'Cand3')])
draw_motif(G_out, "Star Motif (Gatekeeper)", "motif_gatekeeper.png")

# 4. High In-Degree Star (Popular Candidate)
G_in = nx.DiGraph()
G_in.add_edges_from([('Voter1', 'Candidate'), ('Voter2', 'Candidate'), ('Voter3', 'Candidate')])
draw_motif(G_in, "Star Motif (Popular Candidate)", "motif_popular_hub.png")

print("Visualizations created and saved to .png files!")
