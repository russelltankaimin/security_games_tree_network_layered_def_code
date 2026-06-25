import networkx as nx
import matplotlib.pyplot as plt
import os

def save_tree_topology_figure(root, seed, file_format='pdf'):
    """
    Draws the directed out-tree using NetworkX and saves it to the seed directory.
    embeds key node information directly into the visual nodes.
    """
    dir_name = f"outtree/results/images/seed_{seed}"
    os.makedirs(dir_name, exist_ok=True)
    
    G = nx.DiGraph()
    pos = {}
    
    def traverse_and_build(node, depth=0, x_offset=0.5, width=1.0):
        # Extract the initial Gittins index (k=0) safely
        gittins_k0 = node.gittins.get(0, 0.0) if hasattr(node, 'gittins') else 0.0
        
        # Construct a rich, multi-line label for the node bubble
        label = f"{node.node_id}\ng={gittins_k0:.3f}\nl={node.l:.1f}, q={node.q}"
        
        G.add_node(node.node_id, label=label)
        pos[node.node_id] = (x_offset, -depth)
        
        if not node.children: 
            return
            
        child_width = width / len(node.children)
        start_x = x_offset - (width / 2) + (child_width / 2)
        
        for i, child in enumerate(node.children):
            G.add_edge(node.node_id, child.node_id)
            traverse_and_build(child, depth + 1, start_x + (i * child_width), child_width)

    # 1. Build the graph layout
    traverse_and_build(root)
    labels = nx.get_node_attributes(G, 'label')
    
    # 2. Configure a large, academic-styled figure
    plt.figure(figsize=(24, 20))
    plt.title(f"Out-Tree Topology and Initial Indices (Seed: {seed})", fontsize=16, fontweight='bold')
    
    # 3. Draw the graph with enhanced aesthetics
    nx.draw(G, pos, 
            with_labels=True, 
            labels=labels, 
            node_color='#e6f2ff',      # Light blue fill
            edgecolors='#999999',      # Grey node borders
            node_size=4500,            # Large enough to fit the text
            font_size=9, 
            font_weight='bold', 
            arrows=True, 
            arrowsize=20,
            width=1.5,
            edge_color='#555555')
    
    plt.tight_layout()
    
    # 4. Save to the specific seed directory
    filename = os.path.join(dir_name, f"Figure_0_{seed}_Tree_Topology.{file_format}")
    plt.savefig(filename, format=file_format, bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved Tree Topology: {filename}")
    
    # Free memory
    plt.close()