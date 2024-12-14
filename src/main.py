import numpy as np

import pandas as pd

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

def estimate_parameters(structure, data) -> dict:
    n_obs, n_vars = data.shape
    fit_results = {}
    for ivar, var_name in enumerate(structure.keys()):
        dependent_vars = structure.get(var_name, [])
        if len(dependent_vars)==0:
            # then var is independent
            intercept = np.mean(data[:, ivar])
            residuals = data[:, ivar] - intercept
            #print("RESIDUALS: ", np.var(residuals))
            #print("RESIDUALS SQUARED: ", np.var(residuals)**2)
            #print("SQRT RESIDUALS VARIANCE: ", np.sqrt(np.var(residuals)))
            fit_results[ivar] = {
                "intercept": intercept,
                "coefficients": np.zeros(n_vars),
                "residual_variance": np.var(residuals, ddof=1)
            }
        else:
            mask = [map_nodes_to_indexes[var] for var in dependent_vars]
            X = data[:, mask] 
            y = data[:, ivar]
            reg = LinearRegression(fit_intercept=True).fit(X, y)
            intercept = reg.intercept_
            coefficients = reg.coef_
            residuals = y - reg.predict(X)
            #print("RESIDUALS: ", np.var(residuals))
            #print("RESIDUALS SQUARED: ", np.var(residuals)**2)
            #print("SQRT RESIDUALS VARIANCE: ", np.sqrt(np.var(residuals)))
            fit_results[ivar] = {
                "intercept": intercept,
                "coefficients": coefficients,
                "residual_variance": np.var(residuals, ddof = X.shape[1])
            }
    return fit_results

from pgmpy.base import DAG
import matplotlib.pyplot as plt
import networkx as nx
def draw_pgm(ax, model, pos: dict = None, title: str = None):
    #Init network instance
    g = nx.DiGraph()

    #Add nodes and edges
    g.add_nodes_from(model.nodes())
    g.add_edges_from(model.edges())

    if pos is None:
        pos = nx.circular_layout(G)

    nx.draw(g, pos,
        with_labels=True,
        node_size = 1000, node_color = "skyblue",
        font_size = 10, font_weight = "bold",
        arrowsize=30, ax=ax)
    ax.set_title(title)

df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4'])

map_nodes_to_indexes = {node: i for i, node in enumerate(df.columns)}

from sklearn.linear_model import LinearRegression

edges_example_1 = [("X1","X4"),("X2","X4"),("X4","X3")]
edges_example_2 = [("X1","X2"),("X2","X4"),("X2","X3")]



structure_pair = (edges_example_1, edges_example_2)
associated_data_list = []
fit_results_list = []
for istruct, edges_example in enumerate(structure_pair):
    print("STRUCTURE ", edges_example)
    G = nx.DiGraph()
    G.add_nodes_from(['X1', 'X2', 'X3', 'X4'])
    G.add_edges_from(edges_example)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    draw_pgm(axs, G)
    plt.show()

## INIT MODEL STRUCTURE BASED ON BAYESIAN NETWORK G STRUCTURE
    model_structure = {var: list(G.predecessors(var)) for var in G.nodes()}
    print("MODEL STRUCTURE: \n\t", model_structure)

#### ESTIMATE PARAMETERS ####
    fit_results = estimate_parameters(model_structure, data)
    associated_data = np.zeros(data.shape)
    for ivar, var_name in enumerate(model_structure.keys()):
        dependent_vars = model_structure.get(var_name, [])
        if len(dependent_vars)==0:
            associated_data[:, ivar] = fit_results[ivar]["intercept"]
        else:
            associated_data[:,ivar] = fit_results[ivar]["intercept"] + np.dot(data[:, [map_nodes_to_indexes[var] for var in dependent_vars]], fit_results[ivar]["coefficients"])

    #print("ASSOCIATED DATA: \n", associated_data)
    associated_data_list.append(associated_data)
    fit_results_list.append(fit_results)
            
    #### COMPUTE SCORE ####
    bn_entropy = 0
    for var, results in fit_results.items():
        bn_entropy += 0.5 + 0.5*np.log(2*np.pi*results["residual_variance"])

    #### PRINT RESULTS ####
    print("FIT RESULTS: \n", pd.DataFrame(fit_results))
    print("BAYESIAN NETWORK ENTROPY: ", bn_entropy)

norm_l2_list = []
kl = 0
for ivar in range(data.shape[1]):
    norm_l2 = np.linalg.norm(associated_data_list[0][:, ivar] - associated_data_list[1][:, ivar], ord=2)
    norm_l2_list.append(norm_l2)
    kl_ivar = 0.5 * (
                    np.log(fit_results_list[1][ivar]["residual_variance"] / fit_results_list[0][ivar]["residual_variance"]) \
                    + fit_results_list[0][ivar]["residual_variance"] / fit_results_list[1][ivar]["residual_variance"] \
                    - 1 \
                    ) \
                    + 1/(2*data.shape[0]) * ( norm_l2 / fit_results_list[1][ivar]["residual_variance"] )
    print("KL FOR VAR ", ivar, " : ", kl_ivar)
    kl += kl_ivar


#### UPDATE STRUCTURE BASED ON SCORE ####
##### COMPUTE ALL LEGAL MOVES ######

print("KL BETWEEN EXAMPLES: ", kl)
