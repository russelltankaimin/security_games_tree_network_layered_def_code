def export_tree_to_dot(root, filename="attack_tree.dot"):
    """
    Traverses the out-tree and generates a Graphviz DOT text file.
    Each node displays its ID and a table of its success probabilities.
    """
    with open(filename, 'w') as f:
        # Initialize the directed graph (digraph)
        f.write("digraph LayeredSecurityTree {\n")
        f.write('  node [shape=plaintext, fontname="Helvetica"];\n')
        f.write('  edge [color="#555555"];\n\n')
        
        # Use a queue for a simple Breadth-First Search traversal
        queue = [root]
        
        while queue:
            current = queue.pop(0)
            
            # 1. Construct the HTML-like table label for the node
            # The header contains the Node ID and its time cost l(e)
            label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
            
            # Color target leaf nodes differently for clarity
            bg_color = '"#e6f2ff"' if not current.is_leaf else '"#e6ffe6"'
            
            label += f'<TR><TD COLSPAN="2" BGCOLOR={bg_color}><B>{current.node_id}</B><BR/>Cost l: {current.l:.2f}</TD></TR>'
            label += '<TR><TD><I>Attempt (k)</I></TD><TD><I>Prob p(k)</I></TD></TR>'
            
            # List the probabilities for each attempt state
            for k in range(current.q):
                label += f'<TR><TD>{k}</TD><TD>{current.p[k]:.3f}</TD></TR>'
            
            # Add the final lockout state
            label += f'<TR><TD BGCOLOR="#ffeeee">{current.q} (Lockout)</TD><TD BGCOLOR="#ffeeee">0.000</TD></TR>'
            label += '</TABLE>>'
            
            # Write the node definition to the file
            f.write(f'  {current.node_id} [label={label}];\n')
            
            # 2. Write the directed edges connecting to children
            for child in current.children:
                f.write(f'  {current.node_id} -> {child.node_id};\n')
                queue.append(child)
                
        f.write("}\n")
    print(f"Tree structure and probabilities successfully written to {filename}")

# --- Execution ---
# Assuming 'network_root' is the tree generated from the previous script:
# export_tree_to_dot(network_root, "attack_tree.dot")