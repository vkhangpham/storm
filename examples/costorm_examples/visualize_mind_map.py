import json
import networkx as nx
import matplotlib.pyplot as plt

def build_graph(data, G=None, parent=None):
    if G is None:
        G = nx.Graph()
    
    if isinstance(data, dict):
        for key, value in data.items():
            G.add_node(key)
            if parent:
                G.add_edge(parent, key)
            build_graph(value, G, key)
    elif isinstance(data, list):
        for item in data:
            build_graph(item, G, parent)
    
    return G

# Load the JSON file
with open('results/computer_science/app/mind_map.json', 'r') as f:
    mind_map = json.load(f)

# Build the graph
G = build_graph(mind_map)

# Create a layout for the graph
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(20, 10))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=8, font_weight='bold')

# Add labels to edges
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Mind Map Visualization")
plt.axis('off')
plt.tight_layout()
plt.show()