#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
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

# INIT DATA
N_OBS = 1000
N_VARS = 6
N_EDGES = 6
## true model parameters
mu_vars = np.random.randint(1, 50, N_VARS)
sigma_vars = np.random.randint(1, 10, N_VARS)

## generate true model structure
nodes = list(range(N_VARS))
edges = set()
while len(edges) < N_EDGES:
    u, v = random.sample(nodes, 2)
    if u < v:
        edges.add((u, v))
true_model_structure = {f"X{i}": [
    f"X{v}" for u, v in edges if u == i
    ] for i in nodes}
map_nodes_to_indexes = {f"X{i}": i for i in nodes}  
## assert that the true model is DAG
G = nx.DiGraph()
G.add_nodes_from(true_model_structure.keys())
for node, parents in true_model_structure.items():
    for parent in parents:
        G.add_edge(parent, node)
assert nx.is_directed_acyclic_graph(G), "True model is not a DAG"

## generate causal data
data = np.zeros((N_OBS, N_VARS))
topological_order = list(nx.topological_sort(G)) # ensures to process parents before children
for node in topological_order:
    inode = map_nodes_to_indexes[node]
    parents = true_model_structure[node]
    if len(parents) == 0:
        data[:, inode] = np.random.normal(mu_vars[inode], sigma_vars[inode], N_OBS)
    else:
        data[:, inode] = np.random.normal(
            mu_vars[inode] + np.dot(data[:, [map_nodes_to_indexes[jparent] for jparent in parents]], np.random.uniform(-1, 1, len(parents))),
            sigma_vars[inode]
        )
## Compute entropy score of the true model
fit_results = estimate_parameters(true_model_structure, data, map_nodes_to_indexes)
bn_conditional_entropies = np.ones(N_VARS) * np.inf
for var, results in fit_results.items():
    ivar = map_nodes_to_indexes[var]
    bn_conditional_entropies[ivar] = results["residual_variance"]
true_bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
    + len(bn_conditional_entropies)/2

## store data in a pandas DataFrame
df = pd.DataFrame(data, columns=[f"X{i}" for i in nodes])
## inspect the data distributions
fig, axs = plt.subplots(ncols = 2, figsize=(10, 5))
for node in nodes:
    axs[1].hist(data[:, node], bins=30, alpha=0.5, label=f"X{node}")
draw_pgm(axs[0], G)
plt.show()

## INIT STRUCTURE
g = nx.DiGraph()
### start from a random configuration of a DAG model with N_VARS nodes
init_edges = set()
while len(init_edges) < N_EDGES:
    u, v = random.sample(nodes, 2)
    if u < v:
        init_edges.add((u, v))
g.add_nodes_from([f"X{i}" for i in nodes])
g.add_edges_from([(f"X{u}", f"X{v}") for u, v in init_edges])
assert nx.is_directed_acyclic_graph(g), "Initial model is not a DAG"

## INIT LOOP
MAX_ITERATIONS = 100
EARLY_STOP_COUNTER = 0
EARLY_STOP_TH = 10
bn_entropy_old = np.inf
bn_conditional_entropies = np.ones(N_VARS) * np.inf
winning_graph = g.copy()

## RUN LOOP
for istruct in range(MAX_ITERATIONS):
    ## LOG
    print("ITERATION ", istruct)
    print("\tSTRUCTURE ", list(g.edges))
    #fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    #draw_pgm(axs, g)
    #if plt.isinteractive():
    #    plt.show()
    #else:
    #    plt.savefig(f"image_{istruct}.png")

    ## ESTIMATE PARAMETERS
    model_structure = {var: list(g.predecessors(var)) for var in g.nodes()}
    print("\tMODEL STRUCTURE iter ", istruct, " : \n\t\t", model_structure)
    fit_results = estimate_parameters(model_structure, data, map_nodes_to_indexes)

    ## COMPUTE SCORE
    bn_entropy = 0
    ### update the conditional entropies of the changed variables
    for var, results in fit_results.items():
        ivar = map_nodes_to_indexes[var]
        bn_conditional_entropies[ivar] = results["residual_variance"]
    ### compute the formula for the entropy
    bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
        + len(bn_conditional_entropies)/2
       
    ## UPDATE STRUCTURE BASED ON SCORE     
    if bn_entropy < bn_entropy_old:
        bn_entropy_old = bn_entropy
        winning_graph = g.copy()
        for var, results in fit_results.items():
            winning_graph.nodes[var]["intercept"] = results["intercept"]
            winning_graph.nodes[var]["residual_variance"] = results["residual_variance"]
            ## add coefficients to all edges with predecessors
            for parent in model_structure[var]:
                winning_graph.edges[parent, var]["coefficients"] = results["coefficients"][model_structure[var].index(parent)]
    else:
        EARLY_STOP_COUNTER += 1
        # reset to last best
        g = winning_graph.copy()
    
    if EARLY_STOP_COUNTER > EARLY_STOP_TH:
        break
    random_arc_change(g)

print("\n\nWINNING_STRUCTURE: ", winning_graph)
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
draw_pgm(axs[0], G)
axs[0].set_title("True model | entropy: {:.2f}".format(true_bn_entropy))

draw_pgm(axs[1], winning_graph)
axs[1].set_title("Winning model | entropy: {:.2f}".format(bn_entropy_old))
plt.show()
#
    #print("STRUCTURE ", list(g.edges))
    #
    #fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    #draw_pgm(axs, g)
    #if plt.isinteractive():
        #plt.show()
    #else:
        #plt.savefig(f"image_{istruct}.png")
#
    # INIT MODEL STRUCTURE BASED ON BAYESIAN NETWORK G STRUCTURE
    #model_structure = {var: list(g.predecessors(var)) for var in g.nodes()}
    #print("\n\nMODEL STRUCTURE iter ", istruct, " : \n\t", model_structure)
#
    ### ESTIMATE PARAMETERS ####
    #fit_results = estimate_parameters(model_structure, data, map_nodes_to_indexes)
    #associated_data = np.zeros(data.shape)
    #for var_name in model_structure.keys():
        #ivar = map_nodes_to_indexes[var_name]
        #dependent_vars = model_structure.get(var_name, [])
        #if len(dependent_vars) == 0:
            #associated_data[:, ivar] = fit_results[ivar]["intercept"]
        #else:
            #associated_data[:, ivar] = fit_results[ivar]["intercept"] \
                #+ np.dot(data[:, [map_nodes_to_indexes[var] for var in dependent_vars]], fit_results[ivar]["coefficients"])
#
    ### COMPUTE SCORE ####
    #bn_entropy = 0
    #for var, results in fit_results.items():
        #bn_conditional_entropies[var] = results["residual_variance"]
    #bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
        #+ len(bn_conditional_entropies)/2
#
    #if bn_entropy < bn_entropy_old:
        #bn_entropy_old = bn_entropy
        #winning_graph = g.copy()
        #for var_name in model_structure.keys():
            #ivar = map_nodes_to_indexes[var_name]
#             Add to all winning graph nodes the intercept and residual variance of the nodes i have changed
            #winning_graph.nodes[var_name]["intercept"] = fit_results[ivar]["intercept"]
            #winning_graph.nodes[var_name]["residual_variance"] = fit_results[ivar]["residual_variance"]
 #            Add to all the edges incoming to var_name the coefficients with the related parent
    #else:
  #       reset to last best
        #g = winning_graph.copy()
        #EARLY_STOP_COUNTER += 1
#
    ### PRINT RESULTS ####
    #print("FIT RESULTS: \n", pd.DataFrame(fit_results))
    #print("BAYESIAN NETWORK ENTROPY: ", bn_entropy)
#
    #if EARLY_STOP_COUNTER > EARLY_STOP_TH:
        #break
#
    ### CHANGE GRAPH BEFORE REPEATING ####
    #edge_removed, edge_added = random_arc_change(g)
# PRINT BEST MODEL
#print("\n\nWINNING_STRUCTURE: ", winning_graph) \

#edges_example_1 = [("X1","X4"),("X2","X4"),("X4","X3")]
#edges_example_2 = [("X1","X2"),("X2","X4"),("X2","X3")]
#
#structure_pair = (edges_example_1, edges_example_2)
#associated_data_list = []
#fit_results_list = []
#bn_entoropy_old = np.inf
#bn_conditional_entropies = np.ones(len(coulumns_names)) * np.inf
#
#G = nx.DiGraph()
#G.add_nodes_from(coulumns_names)
#G.add_edges_from(edges_example_2)
#winning_graph = G.copy()
#
#MAX_ITERATIONS=10
#EARLY_STOP_COUNTER=0
#EARLY_STOP_TH=3
#the first time we have to compute all nodes' entropies
#changed_nodes=list(G.nodes)
#associated_data_old = np.zeros(data.shape)
#for istruct in range(MAX_ITERATIONS):
    #print("STRUCTURE ", list(G.edges))
    #print("CHANGED NODES ", changed_nodes)
#
    #fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    #draw_pgm(axs, G)
    #if plt.isinteractive():
        #plt.show()
    #else:
        #plt.savefig(f"image_{istruct}.png")
#
    # INIT MODEL STRUCTURE BASED ON BAYESIAN NETWORK G STRUCTURE
    #model_structure = {var: list(G.predecessors(var)) for var in G.nodes()}
    #print("\n\nMODEL STRUCTURE: \n\t", model_structure)
    #new_struct = dict()
    #for node in model_structure.keys():
        #if node in changed_nodes:
            #new_struct[node] = model_structure[node]
    #model_structure = new_struct
    #print("CHANGED STRUCTURE: \n\t", model_structure)
#
    ### ESTIMATE PARAMETERS ####
    #fit_results = estimate_parameters(model_structure, data, map_nodes_to_indexes)
    #associated_data = associated_data_old.copy()
    #for var_name in model_structure.keys():
        #ivar = map_nodes_to_indexes[var_name]
        #dependent_vars = model_structure.get(var_name, [])
        #if len(dependent_vars)==0:
            #associated_data[:, ivar] = fit_results[ivar]["intercept"]
        #else:
            #associated_data[:,ivar] = fit_results[ivar]["intercept"] \
                    #+ np.dot(data[:, [map_nodes_to_indexes[var] for var in dependent_vars]], fit_results[ivar]["coefficients"])
#
 #    print("ASSOCIATED DATA: \n", associated_data)
    #associated_data_list.append(associated_data)
    #fit_results_list.append(fit_results)
#
    ### COMPUTE SCORE ####
    #bn_entropy = 0
    #for var, results in fit_results.items():
        #bn_conditional_entropies[var] = results["residual_variance"]
  #       bn_entropy += 0.5 + 0.5*np.log(2*np.pi*results["residual_variance"])
#
    #bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
            #+ len(bn_conditional_entropies)/2
#
    #if bn_entropy < bn_entoropy_old :
        #bn_entoropy_old = bn_entropy
        #winning_graph = G.copy()
    #else:
       #  reset to last best
        #G = winning_graph.copy()
        #EARLY_STOP_COUNTER += 1
#
    ### PRINT RESULTS ####
    #print("FIT RESULTS: \n", pd.DataFrame(fit_results))
    #print("BAYESIAN NETWORK ENTROPY: ", bn_entropy)
#
    #if EARLY_STOP_COUNTER > EARLY_STOP_TH:
        #break
#
    ### CHAGNE GRAPH BEFORE REPEATING ####
    #edge_removed, edge_added = random_arc_change(G)
    #changed_nodes=[edge_removed[1], edge_added[1]]
#
#print("\n\nWINNING_STRUCTURE: ",{var: list(winning_graph.predecessors(var)) \
        #for var in winning_graph.nodes()})
#
#
#kl = kl_bn(associated_data_list[0], associated_data_list[1],
           #fit_results_list[0], fit_results_list[1])
#
#
### UPDATE STRUCTURE BASED ON SCORE ####
#### COMPUTE ALL LEGAL MOVES ######
#
#print("KL BETWEEN EXAMPLES: ", kl)
