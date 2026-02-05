import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_network(graph, trust_model=None, filename="network_topology.png", return_fig=False, paths=None, show_drops=None):
    """
    Visualizes the network topology.
    - Nodes are colored based on reliability/trust if provided.
    - Bad/Low Trust nodes -> Red
    - Good/High Trust nodes -> Green
    
    Args:
        paths: Dict of {'Label': [node_list]} to highlight specific paths.
        show_drops: List of [node_id] where packet drops should be annotated.
    """
    fig = plt.figure(figsize=(12, 10))
    
    pos = nx.spring_layout(graph, seed=42)
    
    # 1. Node Coloring
    node_colors = []
    node_sizes = []
    
    for node in graph.nodes():
        is_malicious = False
        
        # Determine based on Trust Model or Ground Truth
        if trust_model:
            trust = trust_model.get_trust(node)
            if trust < 0.5:
                is_malicious = True
        else:
            rel = graph.nodes[node].get('reliability', 1.0)
            if rel < 0.8:
                is_malicious = True
        
        if is_malicious:
            node_colors.append('#FF4444') # Red
            node_sizes.append(700)
        else:
            node_colors.append('#44FF44') # Green
            node_sizes.append(500)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    
    # Draw base edges
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.2, edge_color='gray', style='dashed')
    
    # 2. Path Highlighting
    # Define colors for specific known keys
    path_colors = {
        'RIP': 'blue',
        'OSPF': 'purple', 
        'Trust-Aware': 'green',
        'Trust': 'green'
    }
    
    legend_patches = []
    legend_patches.append(mpatches.Patch(color='#44FF44', label='Trusted Node'))
    legend_patches.append(mpatches.Patch(color='#FF4444', label='Malicious/Low Trust'))
    
    if paths:
        offset = 0.05 # To separate overlapping paths slightly (visual hack)
        
        for label, path in paths.items():
            if not path or len(path) < 2:
                continue
                
            color = path_colors.get(label, 'orange')
            
            # Create edge list for this path
            path_edges = list(zip(path, path[1:]))
            
            # Draw edges with increased width
            # We can't easily offset edges in networkx without complex transforms or curved edges.
            # For now, we rely on transparency or different widths/styles if needed.
            # Or simplified: specific styles per Algo.
            
            style = 'solid'
            width = 3
            if label == 'RIP': 
                style = 'dotted'
                width = 4
            elif label == 'OSPF': 
                style = 'dashed'
                width = 4
            
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=path_edges,
                edge_color=color,
                width=width,
                style=style,
                arrows=True,
                arrowsize=20,
                alpha=0.8
            )
            
            legend_patches.append(mpatches.Patch(color=color, label=f"{label} Path"))
            
    # 3. Packet Drop Annotations
    if show_drops:
        for node in show_drops:
            if node in pos:
                x, y = pos[node]
                plt.text(x, y + 0.08, "Packet Dropped\n(Blackhole)", 
                         fontsize=10, color='red', weight='bold', 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'),
                         horizontalalignment='center')

    # Labels
    nx.draw_networkx_labels(graph, pos, font_weight='bold')
    
    # Edge Labels (Weights)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Network Topology & Routing Paths Analysis")
    plt.axis('off')
    plt.tight_layout()
    
    if return_fig:
        plt.close(fig)
        return fig
        
    try:
        plt.savefig(filename)
        print(f"Network visualization saved to {filename}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close()
