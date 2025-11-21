import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gml('power.gml', label='id')

print('Number of nodes:', len(G.nodes()))
print('Number of edges:', len(G.edges()))
print('Average degree:', sum(dict(G.degree()).values()) / len(G.nodes()))

plt.figure(figsize=(12, 10))

pos = nx.spring_layout(G, seed=42)

degrees = dict(G.degree())
node_colors = [degrees[n] for n in G.nodes()]

nx.draw_networkx(
    G,
    pos,
    node_size=80,
    node_color=node_colors,
    cmap='viridis',
    with_labels=True,
    font_size=4,
    font_color='red',
    edge_color='gray',
    linewidths=0.2
)

plt.title('Infrastructure networks: Power grid ')
plt.axis('off')
plt.show()
