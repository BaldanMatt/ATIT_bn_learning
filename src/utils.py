import networkx as nx
import matplotlib.pyplot as plt

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
