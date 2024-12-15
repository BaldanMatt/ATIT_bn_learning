import networkx as nx
import matplotlib.pyplot as plt
import random

def draw_pgm(ax, model, pos: dict = None, title: str = None):
    #Init network instance
    g = nx.DiGraph()

    #Add nodes and edges
    g.add_nodes_from(model.nodes())
    g.add_edges_from(model.edges())

    if pos is None:
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
    edge = random.choice(list(graph.edges))
    print("removing", edge)
    graph.remove_edge(*edge)
    edge = random.choices(list(graph.nodes), k=2)
    graph.add_edge(*edge)
    while not nx.is_directed_acyclic_graph(graph):
        graph.remove_edge(*edge)
        edge = random.choices(list(graph.nodes), k=2)
        graph.add_edge(*edge)

    print("adding", edge)
    return edge

