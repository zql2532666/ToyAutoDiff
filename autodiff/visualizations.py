from graphviz import Digraph


def draw_dag_forward(root, rankdir='LR'):
    assert rankdir in ['LR', 'TB']
    # Compute the nodes and edges using the `trace` function we just
    # described in the `Variable` class.
    nodes, edges = root.build_graph()

    # A diagraph is the kind of visualization we're after.
    # Format is the output type. You can change this to png, jpeg, etc.
    # Rankdir must be either LR (left-to-right) or TB (top-to-bottom)
    dot = Digraph(format='svg', graph_attr={'rankdir': rankdir})
    
    for n in nodes: # For each node
        # Add a node with the machine name of this object
        # There are many kinds of shapes available; https://graphviz.org/doc/info/shapes.html
        # We will use the 'record' shape containing the primal and tangent value rounded
        # to 3 digits.
        dot.node(name=str(id(n)), label = "{%s = %s}" % (n.name, n.value), shape='record')
        if n.op: # If this is an op (as in the name op is not empty)
            # Add a node and name it the machine name + the op
            dot.node(name=str(id(n)) + n.op, label=n.op)
            # Connect an edge between this newly created op and the parent node
            dot.edge(str(id(n)) + n.op, str(id(n)))
    
    for n1, n2 in edges: # For the nodes in each edge
        # Add an edge
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)
    
    # Return the graph
    return dot

def draw_dag_backward(root, rankdir='RL'):
    assert rankdir in ['LR', 'RL', 'TB']
    # Compute the nodes and edges using the `trace` function we just
    # described in the `Variable` class.
    nodes, edges = root.build_graph()

    # A diagraph is the kind of visualization we're after.
    # Format is the output type. You can change this to png, jpeg, etc.
    # Rankdir must be either LR (left-to-right) or TB (top-to-bottom)
    dot = Digraph(format='png', graph_attr={'rankdir': rankdir})
    
    for n in nodes: # For each node
        # Add a node with the machine name of this object
        # There are many kinds of shapes available; https://graphviz.org/doc/info/shapes.html
        # We will use the 'record' shape containing the primal and tangent value rounded
        # to 3 digits.
        dot.node(name=str(id(n)), label = "{d%s/d%s = %s} | {%s = %s}" % (root.name, n.name, n.grad, n.name, n.value), shape='record')
        if n.op: # If this is an op (as in the name op is not empty)
            # Add a node and name it the machine name + the op
            dot.node(name=str(id(n)) + n.op, label=n.op)
            # Connect an edge between this newly created op and the parent node
            dot.edge(str(id(n)),str(id(n)) + n.op)
    
    for n1, n2 in edges: # For the nodes in each edge
        # Add an edge
        dot.edge(str(id(n2)) + n2.op,str(id(n1)))
    
    # Return the graph
    return dot