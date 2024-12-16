import networkx as nx
import matplotlib.pyplot as plt
import random

def draw_pgm(ax, model, pos: dict = None, title: str = None):
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

    return (edge_removed, edge_added)

