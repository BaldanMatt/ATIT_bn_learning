import networkx as nx
import matplotlib.pyplot as plt
import random

def draw_pgm(ax, model, pos: dict = None, title: str = None):
    """
    Draw a DAG model with matplotlib and networkx
    """
    #Init network instance
    g = nx.DiGraph()

    #Add nodes and edges
    g.add_nodes_from(model.nodes())
    g.add_edges_from(model.edges())

    for layer, nodes in enumerate(nx.topological_generations(g)):
        for node in nodes:
            g.nodes[node]["layer"] = layer
    
    if pos is None:
        pos = nx.multipartite_layout(g, subset_key="layer")
    elif pos == "circular":
        pos = nx.circular_layout(g)

    nx.draw(g, pos,
        with_labels=True,
        node_size = 1000, node_color = "skyblue",
        font_size = 10, font_weight = "bold",
        arrowsize=30, ax=ax)
    ax.set_title(title)

def random_arc_change(graph : nx.DiGraph):
    """
    Picks a random edge in the graph and moves it to a new place
    """
    edge_removed = random.choice(list(graph.edges))
    graph.remove_edge(*edge_removed)
    #print("removed", edge_removed)
    nodes = list(nx.topological_sort(graph))
    edge_added = [(u,v) for u,v in graph.edges][0]
    while edge_added in graph.edges:
        #print("retrying, edge ", edge_added, " already in graph")
        edge_added = random.sample(list(graph.nodes), k=2)
        if nodes.index(edge_added[0]) > nodes.index(edge_added[1]):
            edge_added = (edge_added[1], edge_added[0])

    graph.add_edge(*edge_added)
    #print("added", edge_added)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Tua madre mucca non sai come funzionano i grafi")

    return (edge_removed[1], edge_added[1])

def due_opt(graph : nx.DiGraph):
    """
    Picks two pairs of nodes and swaps the children.
    """
    nodes = list(nx.topological_sort(graph))
    children = nodes[0:2]
    parents = nodes[-2:]
    while (max(nodes.index(parents[0]), nodes.index(parents[1])) >= min(nodes.index(children[0]), nodes.index(children[1]))):
        #print("retrying")
        new_children = random.sample(nodes, 2)
        predecessors1 = [ i for i in graph.predecessors(new_children[0]) if i not in graph.predecessors(new_children[1])]
        if len(predecessors1) == 0: continue
        predecessors2 = [ i for i in graph.predecessors(new_children[1]) if i not in graph.predecessors(new_children[0])]
        if len(predecessors2) == 0: continue
        children = new_children
        parents = [
            *random.sample(predecessors1, k=1),
            *random.sample(predecessors2, k=1)
        ]
    graph.remove_edge(parents[0], children[0])
    graph.remove_edge(parents[1], children[1])
    graph.add_edge(parents[0], children[1])
    graph.add_edge(parents[1], children[0])
    return tuple(children)



