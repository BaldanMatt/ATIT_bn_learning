#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.interactive(True)
import matplotlib.pyplot as plt

import networkx as nx
import random
from utils import draw_pgm, random_arc_change
from core import estimate_parameters, kl_bn
from pathlib import Path
import os

# this is sample data copied from the tutorial paper
#data = np.array([
    #[3.626, 2.811, 17.683, 12.626],
    #[1.895, 3.571, 15.320, 10.459],
    #[2.725, 0.724, 10.270,  5.990],
    #[2.966, 1.584, 13.572, 10.042],
    #[2.762, 1.697, 15.469, 10.742],
    #[2.305, 2.293, 11.471,  9.167],
    #[3.752, 1.580, 12.955,  9.666],
    #[2.315, -0.258, 2.677,  1.154],
    #[4.205, -0.090, 7.916,  6.756],
    #[2.344, 2.823, 14.594, 10.383]
#])
#coulumns_names=['X1', 'X2', 'X3', 'X4']
#df = pd.DataFrame(data, columns=coulumns_names)
#map_nodes_to_indexes = {node: i for i, node in enumerate(coulumns_names)}
#
import argparse

parser = argparse.ArgumentParser(description='Bayesian Network Structure Learning')
# Add a command line argument that simulates some data or uses an example
parser.add_argument('--simulate', action='store_true', help='Simulate data')
parser.add_argument('--example', action='store_true', help='Use example data')
parser.add_argument('--dataset', type=str, help='Use a dataset from the data folder')
args = parser.parse_args()

if args.simulate:
# INIT DATA
    N_OBS = 100000
    N_VARS =6 
    N_EDGES = N_VARS - 1
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
            data[:, inode] = np.sum(data[:, [map_nodes_to_indexes[parent] for parent in parents]], axis = 1) + np.random.normal(mu_vars[inode], sigma_vars[inode]/len(parents), N_OBS) 
## store data in a pandas DataFrame
    df = pd.DataFrame(data, columns=[f"X{i}" for i in nodes])
## inspect the data distributions
    fig, axs = plt.subplots(ncols = 2, figsize=(10, 5))
    for node in nodes:
        axs[1].hist(data[:, node], bins=30, alpha=0.5, label=f"X{node}")
    axs[1].legend()
    draw_pgm(axs[0], G)
    fig.savefig("simulated_data.png")
    del fig, axs
else:
    # MAJOR TODO
    ## NEED to fix the map nodes to indexes to generalize name of the columns
    ## NEED to fix the idea that toy dataset are not multivariate gaussian distribution
        # and titanic example has both continuous and discrete variables
    
    parent_dir = Path(__file__).resolve().parents[1]
    data_path = parent_dir / "data" / args.dataset
    df = pd.read_csv(data_path / "train.csv")
    if args.dataset == "titanic":
        true_model_structure = {
            "Pclass": ["Survived"],
            "Survived": ["Age"],
            "Age": [],
            "Parch": ["SibSp"],
            "SibSp": [],
        } # thisis found with hill climb and default scoring scheme with root node Survived crom bnlearn
        #https://www.kaggle.com/code/vbmokin/titanic-predict-using-a-simple-bayesian-network
    df = df.dropna()
    data = df.to_numpy()
    map_nodes_to_indexes = {node: i for i, node in enumerate(df.columns)}
    N_OBS, N_VARS = data.shape
    N_EDGES = 3
    nodes = list(range(N_VARS))
    
## Compute entropy score of the true model
print("TRUE MODEL STRUCTURE: ", true_model_structure)
print("DATA: ", data)
fit_results = estimate_parameters(true_model_structure, data, map_nodes_to_indexes)
bn_conditional_entropies = np.ones(N_VARS) * np.inf
for var, results in fit_results.items():
    ivar = map_nodes_to_indexes[var]
    bn_conditional_entropies[ivar] = results["residual_variance"]
true_bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
    + len(bn_conditional_entropies)/2

# INIT BOOTSTRAP #
N_BOOTSTRAPS = 5 
MAX_ITERATIONS = 20
EARLY_STOP_TH = 20
EARLY_STOP_BOOT_COUNTER = 0
debug = False 
# define a winning graph to store the best model
winning_boot_graph = nx.DiGraph()
winning_boot_bn_entropy = np.inf

parent_dir = Path(__file__).resolve().parents[1]
# RUN BOOTSTRAP LOOP
for jboot in range(N_BOOTSTRAPS): # Start bootstrap loop
    if not os.path.exists(parent_dir / "bootstrap" / f"bootstrap_{jboot}"):
        os.makedirs(parent_dir / "bootstrap" / f"bootstrap_{jboot}")
    print("BOOTSTRAP ", jboot)
    # init structure
    g = nx.DiGraph()
    ### start from a random configuration of a DAG model with N_VARS nodes
    init_edges = set()
    while len(init_edges) < N_EDGES:
        u, v = random.sample(nodes, 2)
        if u < v:
            init_edges.add((u, v))
    g.add_nodes_from([f"X{i}" for i in nodes])
    g.add_edges_from([(f"X{u}", f"X{v}") for u, v in init_edges])
    init_g = g.copy()
    assert nx.is_directed_acyclic_graph(g), "Initial model is not a DAG"

    ## init iteration loop
    bn_entropy_old = np.inf
    bn_conditional_entropies = np.ones(N_VARS) * np.inf
    winning_graph = g.copy()

    # run iteration loop
    EALRY_STOP_COUNTER = 0
    changed_nodes = list(g.nodes)
    for istruct in range(MAX_ITERATIONS):
        ## LOG
        print("\tITERATION ", istruct)
#
        ## ESTIMATE PARAMETERS
        model_structure = {var: list(g.predecessors(var)) for var in g.nodes()}
        new_struct = dict()
        for node in model_structure.keys():
            if node in changed_nodes:
                new_struct[node] = model_structure[node]

        print("\t\tMODEL to estimate STRUCTURE iter ", istruct, " : \n\t\t", new_struct) 
        fit_results = estimate_parameters(new_struct, data, map_nodes_to_indexes)
        for var, results in fit_results.items():
            g.nodes[var]["intercept"] = results["intercept"]
            g.nodes[var]["residual_variance"] = results["residual_variance"]
            ## add coefficients to all edges with predecessors
            for parent in model_structure[var]:
                g.edges[parent, var]["weight"] = results["coefficients"][model_structure[var].index(parent)]

        ## COMPUTE SCORE
        bn_entropy = 0
        ### update the conditional entropies of the changed variables
        for var, results in fit_results.items():
            ivar = map_nodes_to_indexes[var]
            bn_conditional_entropies[ivar] = results["residual_variance"]
        ### compute the formula for the entropy
        bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
            + len(bn_conditional_entropies)/2
           
        if debug:
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            draw_pgm(axs, g)
            #if plt.isinteractive():
            #    plt.show()
            #else:
            axs.set_title(f"Structure iter {istruct} | entropy : {bn_entropy}")    
            fig.savefig(parent_dir / "bootstrap" / f"bootstrap_{jboot}" / f"image_{istruct}.png")
            del fig, axs

        ## UPDATE STRUCTURE BASED ON SCORE     
        if bn_entropy < bn_entropy_old:
            EARLY_STOP_COUNTER = 0
            bn_entropy_old = bn_entropy
            winning_graph = g.copy()
        else:
            EARLY_STOP_COUNTER += 1
            # reset to last best
            #g = winning_graph.copy()
        
        if EARLY_STOP_COUNTER > EARLY_STOP_TH:
            print("EARLY STOP..")
            break
        edge_removed, edge_added = random_arc_change(g)
        changed_nodes=[edge_removed[1], edge_added[1]]

    print("\n\nWINNING_STRUCTURE: ", winning_graph)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[:,0])
    draw_pgm(ax1, G)
    ax1.set_title("True model | entropy: {:.2f}".format(true_bn_entropy))

    ax2 = fig.add_subplot(gs[0,1])
    draw_pgm(ax2, winning_graph)
    ax2.set_title("Winning model | entropy: {:.2f}".format(bn_entropy_old))

    ax3 = fig.add_subplot(gs[1,1])
    draw_pgm(ax3, init_g)
    ax3.set_title("Initial model")
    fig.savefig(f"comparison_boot_{jboot}.png")
    del fig, gs, ax1, ax2, ax3

    # compare with winning best boot models
    if bn_entropy_old < winning_boot_bn_entropy:
        winning_boot_bn_entropy = bn_entropy_old
        winning_boot_graph = winning_graph.copy()
    else:
        EARLY_STOP_BOOT_COUNTER += 1
    
    if EARLY_STOP_BOOT_COUNTER > EARLY_STOP_TH:
        print("EARLY STOP BOOTSTRAP..")
        break

print("\n\nWINNING_BOOT_STRUCTURE: ", winning_boot_graph)
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1,2)
ax1 = fig.add_subplot(gs[0])
draw_pgm(ax1, G)
ax1.set_title("True model | entropy: {:.2f}".format(true_bn_entropy))
ax2 = fig.add_subplot(gs[1])
draw_pgm(ax2, winning_boot_graph)
ax2.set_title("Winning boot model | entropy: {:.2f}".format(winning_boot_bn_entropy))
fig.savefig("comparison.png")
del fig, gs, ax1, ax2
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
