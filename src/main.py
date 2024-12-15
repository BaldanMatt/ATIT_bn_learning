#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from utils import draw_pgm, random_arc_change
from core import estimate_parameters, kl_bn

# this is sample data copied from the tutorial paper
data = np.array([
    [3.626, 2.811, 17.683, 12.626],
    [1.895, 3.571, 15.320, 10.459],
    [2.725, 0.724, 10.270,  5.990],
    [2.966, 1.584, 13.572, 10.042],
    [2.762, 1.697, 15.469, 10.742],
    [2.305, 2.293, 11.471,  9.167],
    [3.752, 1.580, 12.955,  9.666],
    [2.315, -0.258, 2.677,  1.154],
    [4.205, -0.090, 7.916,  6.756],
    [2.344, 2.823, 14.594, 10.383]
])
coulumns_names=['X1', 'X2', 'X3', 'X4']
df = pd.DataFrame(data, columns=coulumns_names)
map_nodes_to_indexes = {node: i for i, node in enumerate(coulumns_names)}


edges_example_1 = [("X1","X4"),("X2","X4"),("X4","X3")]
edges_example_2 = [("X1","X2"),("X2","X4"),("X2","X3")]



structure_pair = (edges_example_1, edges_example_2)
associated_data_list = []
fit_results_list = []
bn_entoropy_old = np.inf
bn_conditional_entropies = np.ones(len(coulumns_names)) * np.inf

G = nx.DiGraph()
G.add_nodes_from(coulumns_names)
G.add_edges_from(edges_example_2)
winning_graph = G.copy()

MAX_ITERATIONS=10
EARLY_STOP_COUNTER=0
EARLY_STOP_TH=3
# the first time we have to compute all nodes' entropies
changed_nodes=list(G.nodes)
associated_data_old = np.zeros(data.shape)
for istruct in range(MAX_ITERATIONS):
    print("STRUCTURE ", list(G.edges))
    print("CHANGED NODES ", changed_nodes)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    draw_pgm(axs, G)
    if plt.isinteractive():
        plt.show()
    else:
        plt.savefig(f"image_{istruct}.png")

    ## INIT MODEL STRUCTURE BASED ON BAYESIAN NETWORK G STRUCTURE
    model_structure = {var: list(G.predecessors(var)) for var in G.nodes()}
    print("\n\nMODEL STRUCTURE: \n\t", model_structure)
    new_struct = dict()
    for node in model_structure.keys():
        if node in changed_nodes:
            new_struct[node] = model_structure[node]
    model_structure = new_struct
    print("CHANGED STRUCTURE: \n\t", model_structure)

    #### ESTIMATE PARAMETERS ####
    fit_results = estimate_parameters(model_structure, data, map_nodes_to_indexes)
    associated_data = associated_data_old.copy()
    for var_name in model_structure.keys():
        ivar = map_nodes_to_indexes[var_name]
        dependent_vars = model_structure.get(var_name, [])
        if len(dependent_vars)==0:
            associated_data[:, ivar] = fit_results[ivar]["intercept"]
        else:
            associated_data[:,ivar] = fit_results[ivar]["intercept"] \
                    + np.dot(data[:, [map_nodes_to_indexes[var] for var in dependent_vars]], fit_results[ivar]["coefficients"])

    # print("ASSOCIATED DATA: \n", associated_data)
    associated_data_list.append(associated_data)
    fit_results_list.append(fit_results)

    #### COMPUTE SCORE ####
    bn_entropy = 0
    for var, results in fit_results.items():
        bn_conditional_entropies[var] = results["residual_variance"]
        # bn_entropy += 0.5 + 0.5*np.log(2*np.pi*results["residual_variance"])

    bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
            + len(bn_conditional_entropies)/2

    if bn_entropy < bn_entoropy_old :
        bn_entoropy_old = bn_entropy
        winning_graph = G.copy()
    else:
        # reset to last best
        G = winning_graph.copy()
        EARLY_STOP_COUNTER += 1

    #### PRINT RESULTS ####
    print("FIT RESULTS: \n", pd.DataFrame(fit_results))
    print("BAYESIAN NETWORK ENTROPY: ", bn_entropy)

    if EARLY_STOP_COUNTER > EARLY_STOP_TH:
        break

    #### CHAGNE GRAPH BEFORE REPEATING ####
    edge_removed, edge_added = random_arc_change(G)
    changed_nodes=[edge_removed[1], edge_added[1]]

print("\n\nWINNING_STRUCTURE: ",{var: list(winning_graph.predecessors(var)) \
        for var in winning_graph.nodes()})


kl = kl_bn(associated_data_list[0], associated_data_list[1],
           fit_results_list[0], fit_results_list[1])


#### UPDATE STRUCTURE BASED ON SCORE ####
##### COMPUTE ALL LEGAL MOVES ######

print("KL BETWEEN EXAMPLES: ", kl)
