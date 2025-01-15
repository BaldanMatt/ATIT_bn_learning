#!/usr/bin/env python3
# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.interactive(True)
import bnlearn as bn
import matplotlib.pyplot as plt

import networkx as nx
import random
from utils import draw_pgm, random_arc_change, due_opt
from core import estimate_parameters, kl_bn
from pathlib import Path
import os
from tqdm import tqdm
import argparse

# GLOBAL VARIABLES
N_OBS = 10000
N_VARS = 8 
N_EDGES = 2*N_VARS - 1
DO_BNLEARN = False
DO_OPT = True
# INIT BOOTSTRAP #
N_BOOTSTRAPS = 50
MAX_ITERATIONS = 100
EARLY_STOP_TH = 50
EARLY_STOP_BOOT_COUNTER = 0

# SET COMMAND LINE PARSER FUNCTIONALITIES
parser = argparse.ArgumentParser(description='Bayesian Network Structure Learning')
# Add a command line argument that simulates some data or uses an example
parser.add_argument('--simulate', action='store_true', help='Simulate data')
parser.add_argument('--example', action='store_true', help='Use example data')
parser.add_argument('--dataset', type=str, help='Use a dataset from the data folder')
args = parser.parse_args()

# MAIN FUNCTION
if args.simulate:
    """
    If the simulate flag is set, we simulate a dataset with a DAG structure and
    accordingly to the edges of the DAG we create an instance of a multivariate
    gaussian distribution. The parameters of the distribution are randomly generated
    from some ranges. 

    The nodes connected byy edges share a linear dependency with the parent nodes
    """
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
    map_nodes_to_indexes = {f"X{i}": i for i in nodes}
    map_indexes_to_nodes = {i: f"X{i}" for i in nodes}
    ## assert that the true model is DAG
    G = nx.DiGraph()
    G.add_nodes_from([f"X{i}" for i in nodes])
    G.add_edges_from([(f"X{u}", f"X{v}") for u, v in edges])
    true_model_structure = {f"X{i}": [j for j in G.predecessors(f"X{i}")] for i in nodes}
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
else: # Not args.simulate
    """
    If the simulate flag is not set, we load a dataset from the data folder and
    use that to learn its bayesian network structure. We also use the bnlearn library
    """
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
    map_indexes_to_nodes = {i: node for i, node in enumerate(df.columns)}
    N_OBS, N_VARS = data.shape
    N_EDGES = 3
    nodes = list(range(N_VARS))
    
## Compute entropy score of the true model
true_fit_results = estimate_parameters(true_model_structure, data, map_nodes_to_indexes)
bn_conditional_entropies = np.ones(N_VARS) * np.inf

for var, results in true_fit_results.items():
    ivar = map_nodes_to_indexes[var]
    bn_conditional_entropies[ivar] = results["residual_variance"]
true_bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
    + len(bn_conditional_entropies)/2
true_associated_data = np.zeros(data.shape)

for var_name in true_model_structure.keys():
    ivar = map_nodes_to_indexes[var_name]
    dependent_vars = true_model_structure.get(var_name, [])
    if len(dependent_vars) == 0:
        true_associated_data[:, ivar] = true_fit_results[var_name]["intercept"]
    else:
        true_associated_data[:, ivar] = true_fit_results[var_name]["intercept"] \
            + np.dot(data[:, [map_nodes_to_indexes[var] for var in dependent_vars]],
                     true_fit_results[var_name]["coefficients"]
                     )

## Compute library estimated model (bnlearn)
df_data = pd.DataFrame(data, columns=G.nodes) # convert to pandas DataFrame
est_G = nx.DiGraph()
if N_VARS <= 10 and DO_BNLEARN:
    est_G.add_nodes_from(G.nodes)
    bn_model = bn.structure_learning.fit(df_data, methodtype="hc", scoretype="k2")
    est_G.add_edges_from(bn_model["model_edges"])
    est_model_structure = {var: list(est_G.predecessors(var)) for var in est_G.nodes()}
    est_fit_results = estimate_parameters(est_model_structure, data, map_nodes_to_indexes)
    est_associated_data = np.zeros(data.shape)
    for var_name in est_model_structure.keys():
        ivar = map_nodes_to_indexes[var_name]
        dependent_vars = est_model_structure.get(var_name, [])
        if len(dependent_vars) == 0:
            est_associated_data[:, ivar] = est_fit_results[var_name]["intercept"]
        else:
            est_associated_data[:, ivar] = est_fit_results[var_name]["intercept"] \
                + np.dot(data[:, [map_nodes_to_indexes[var] for var in dependent_vars]],
                         est_fit_results[var_name]["coefficients"]
                         )
    kl_est = kl_bn(est_associated_data, true_associated_data,
                   est_fit_results, true_fit_results, map_indexes_to_nodes)
    print("KL DIVERGENCE WITH TRUE MODEL: ", kl_est)
    est_bn_entropy = 0
    for var, results in est_fit_results.items():
        bn_conditional_entropies[map_nodes_to_indexes[var]] = results["residual_variance"]
    est_bn_entropy = np.sum(1/2*np.log(2*np.pi*bn_conditional_entropies)) \
        + len(bn_conditional_entropies)/2

else:
    kl_est = np.inf
    est_bn_entropy = np.inf

"""
BOOTSTRAP LOOP to learn multiple bayesian network structures and search for the ones
that minimizes the entropy score. The loop is stopped when the entropy score does not
update. The structure is updated by changing the arcs of the graph.
"""
debug = False # define a winning graph to store the best model
winning_boot_graph = nx.DiGraph()
winning_boot_bn_entropy = np.inf

# INIT RESULTS #
bn_best_entropies = {jboot: [] for jboot in range(N_BOOTSTRAPS)}
bn_best_kl_with_true = {jboot: [] for jboot in range(N_BOOTSTRAPS)}

# define the parent directory
parent_dir = Path(__file__).resolve().parents[1]

# RUN BOOTSTRAP LOOP
for jboot in tqdm(range(N_BOOTSTRAPS)): # Start bootstrap loop
    if not os.path.exists(parent_dir / "bootstrap" / f"bootstrap_{jboot}"):
        os.makedirs(parent_dir / "bootstrap" / f"bootstrap_{jboot}")
    #print("BOOTSTRAP ", jboot)
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
    associated_data_old = np.zeros(data.shape)
    fit_results_old = estimate_parameters({var: list(g.predecessors(var)) for var in g.nodes()}, data, map_nodes_to_indexes)
    # run iteration loop
    EALRY_STOP_COUNTER = 0
    changed_nodes = list(g.nodes)

    for istruct in range(MAX_ITERATIONS):
        ## LOG
        #print("\tITERATION ", istruct)

        ## ESTIMATE PARAMETERS
        model_structure = {var: list(g.predecessors(var)) for var in g.nodes()}
        new_struct = dict()
        for node in model_structure.keys():
            if node in changed_nodes:
                new_struct[node] = model_structure[node]

        #print("\t\tMODEL to estimate STRUCTURE iter ", istruct, " : \n\t\t", new_struct) 
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
        if istruct ==0:
            init_bn_entropy = bn_entropy

        ## Compute KL divergence with winning model
        # first estimate the associated data
        associated_data = associated_data_old.copy()
        for var_name in new_struct.keys():
            ivar = map_nodes_to_indexes[var_name]
            dependent_vars = new_struct.get(var_name, [])
            if len(dependent_vars) == 0:
                associated_data[:, ivar] = fit_results[var_name]["intercept"]
            else:
                associated_data[:, ivar] = fit_results[var_name]["intercept"] \
                    + np.dot(data[:, [map_nodes_to_indexes[var] for var in dependent_vars]],
                             fit_results[var_name]["coefficients"]
                             )
        
        fit_results_kl = {**fit_results_old, **fit_results}
        kl = kl_bn(associated_data, associated_data_old,
                   fit_results_kl, fit_results_old, map_indexes_to_nodes)
        kl_with_true = kl_bn(associated_data_old, true_associated_data,
                             fit_results_old, true_fit_results, map_indexes_to_nodes)
    
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
            fit_results_old = fit_results_kl.copy()
            associated_data_old = associated_data.copy()
            winning_graph = g.copy()
        else:
            EARLY_STOP_COUNTER += 1
            # reset to last best
            if DO_OPT:
                g = winning_graph.copy()
        
        if EARLY_STOP_COUNTER > EARLY_STOP_TH:
            #print("EARLY STOP..")
            break
        changed_nodes = list()
        for i in range(1):
            if not DO_OPT:
                child1, child2 = random_arc_change(g)
            else:
                child1, child2 = due_opt(g)
            changed_nodes.append(child1)
            changed_nodes.append(child2)

        bn_best_entropies[jboot].append(bn_entropy_old)
        bn_best_kl_with_true[jboot].append(kl_with_true)

    #print("\n\nWINNING_STRUCTURE: ", winning_graph)
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
    ax3.set_title("Initial model | entropy: {:.2f}".format(init_bn_entropy))
    fig.savefig(f"comparison_boot_{jboot}.png")
    del fig, gs, ax1, ax2, ax3

    # compare with winning best boot models
    if bn_entropy_old < winning_boot_bn_entropy:
        winning_boot_bn_entropy = bn_entropy_old
        winning_boot_graph = winning_graph.copy()
    else:
        EARLY_STOP_BOOT_COUNTER += 1
    
    if EARLY_STOP_BOOT_COUNTER > EARLY_STOP_TH:
        #print("EARLY STOP BOOTSTRAP..")
        break

#print("\n\nWINNING_BOOT_STRUCTURE: ", winning_boot_graph)
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2,2)
ax1 = fig.add_subplot(gs[0,0])
draw_pgm(ax1, G)
ax1.set_title("True model | entropy: {:.2f}".format(true_bn_entropy))
ax_est = fig.add_subplot(gs[1,0])
draw_pgm(ax_est, est_G, pos="circular")
ax_est.set_title("Estimated model with bnlearn K2 score\nentropy: {:.2f}".format(est_bn_entropy))
ax2 = fig.add_subplot(gs[:,1])
draw_pgm(ax2, winning_boot_graph)
ax2.set_title("Winning boot model | entropy: {:.2f}".format(winning_boot_bn_entropy))
fig.savefig("comparison.png")
del fig, gs, ax1, ax2
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Set a presentation-ready theme
plt.style.use('tableau-colorblind10')  # Choose a suitable style (e.g., vibrant colors)
# First subplot (left column)
for jboot in range(N_BOOTSTRAPS):
    axs[0].plot(bn_best_entropies[jboot], label=f'Boot {jboot  + 1}')
axs[0].set_title('Entropy over iterations')
axs[0].set_xlabel('# iteration')
axs[0].set_ylabel('H(B)')
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[0].legend() if N_BOOTSTRAPS < 10 else None  # Show legend only if less than 10 bootstraps

# Second subplot (right column)
for jboot in range(N_BOOTSTRAPS):
    axs[1].plot(bn_best_kl_with_true[jboot], label=f'Boot {jboot + 1}')
axs[1].set_title('KL divergence with true model over iterations')
axs[1].set_xlabel('# iteration')
axs[1].set_ylabel("KL(B,B') (Log Scale)")
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1].set_yscale('log')  # Log scale for y-axis
axs[1].legend() if N_BOOTSTRAPS < 10 else None  # Show legend only if less than 10 bootstraps

# Adjust layout for better appearance
plt.tight_layout()
fig.savefig("bootstraps.png")


# Step 1: Find the maximum length of the lists
max_length = max(len(v) for v in bn_best_kl_with_true.values())

# Step 2: Pad shorter lists with NaN
bn_best_kl_with_true_df = {k: v + [None] * (max_length - len(v)) for k, v in bn_best_kl_with_true.items()}

# Step 3: Convert to DataFrame
bn_best_kl_with_true_df= pd.DataFrame(bn_best_kl_with_true_df)
if not os.path.exists(parent_dir / "media" / f"nobs{N_OBS}_nvars{N_VARS}_nboots{N_BOOTSTRAPS}_niter{MAX_ITERATIONS}_nedges{N_EDGES}"):
    os.makedirs(parent_dir / "media" / f"nobs{N_OBS}_nvars{N_VARS}_nboots{N_BOOTSTRAPS}_niter{MAX_ITERATIONS}_nedges{N_EDGES}")
bn_best_kl_with_true_df.to_csv(parent_dir / "media" / f"nobs{N_OBS}_nvars{N_VARS}_nboots{N_BOOTSTRAPS}_niter{MAX_ITERATIONS}_nedges{N_EDGES}" / "kl_divergence.csv", index=False)

max_length = max(len(v) for v in bn_best_entropies.values())
bn_best_entropies_df = {k: v + [None] * (max_length - len(v)) for k, v in bn_best_entropies.items()}
bn_best_entropies_df = pd.DataFrame(bn_best_entropies_df)
bn_best_entropies_df.to_csv(parent_dir / "media" / f"nobs{N_OBS}_nvars{N_VARS}_nboots{N_BOOTSTRAPS}_niter{MAX_ITERATIONS}_nedges{N_EDGES}" / "entropy.csv", index=False)

