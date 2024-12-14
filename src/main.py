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

from pgmpy.base import DAG
import matplotlib.pyplot as plt
import networkx as nx
def draw_pgm(ax, model, pos: dict = None, title: str = None):
    #Init network instance
    G = nx.DiGraph()

    #Add nodes and edges
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    if pos is None:
        pos = nx.circular_layout(G)

    nx.draw(G, pos,
        with_labels=True,
        node_size = 1000, node_color = "skyblue",
        font_size = 10, font_weight = "bold",
        arrowsize=30, ax=ax)
    ax.set_title(title)

df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4'])
G = DAG()

map_nodes_to_indexes = {node: i for i, node in enumerate(df.columns)}
G.add_nodes_from(nodes=['X1', 'X2', 'X3', 'X4'])

#edges_example = [("X1","X4"),("X2","X4"),("X4","X3")]
edges_example = [("X1","X2"),("X2","X4"),("X2","X3")]
G.add_edges_from(ebunch=edges_example)

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
draw_pgm(axs, G)
plt.show()

from sklearn.linear_model import LinearRegression

model_structure = {var: list(G.predecessors(var)) for var in G.nodes()}
print(model_structure)

n_obs, n_vars = data.shape

fit_results = {}
for ivar, var_name in enumerate(model_structure.keys()):
    print("Fitting variable...", var_name)
    dependent_vars = model_structure.get(var_name, [])
    print("\tDependent variables: ", dependent_vars)
    if len(dependent_vars)==0:
        # then var is independent
        intercept = np.mean(data[:, ivar])
        residuals = data[:, ivar] - intercept
        fit_results[ivar] = {
            "intercept": intercept,
            "coefficients": np.zeros(n_vars),
            "residual_variance": np.var(residuals)
        }
    else:
        mask = [map_nodes_to_indexes[var] for var in dependent_vars]
        X = data[:, mask] 
        print("\t\t train data to fit the model:\n", X)
        y = data[:, ivar]
        print("\t\t target data to fit the model:\n", y)
        reg = LinearRegression(fit_intercept=True).fit(X, y)
        intercept = reg.intercept_
        coefficients = reg.coef_
        print("\t\t coefficients:\n", coefficients)
        residuals = y - reg.predict(X)
        fit_results[ivar] = {
            "intercept": intercept,
            "coefficients": coefficients,
            "residual_variance": np.var(residuals)
        }

print(pd.DataFrame(fit_results))
bn_entropy = 0
for var, results in fit_results.items():
    bn_entropy += 0.5 + 0.5*np.log(2*np.pi*results["residual_variance"])
print(bn_entropy)


#
#print("shape of the data:\n",data.shape)
#
#sd_list = np.std(data, axis=0)
#print(sd_list)
#
#
#cov_matrix = np.cov(data.T)
#print("cov matrix:\n",cov_matrix)
#det_cov_matrix = np.linalg.det(cov_matrix)
#print("det cov matrix:\n",det_cov_matrix)
#joint_entropy_bn = 0.5*cov_matrix.shape[0]*np.log(2*np.pi*np.e) + 0.5*np.log(det_cov_matrix)
#
#print(joint_entropy_bn)
#
#print(data.shape)
#print(np.std(data,axis=0))
#print(np.std(data[:,3])**2-(1.5**2)*0.8+(2.6**2)*0.6)
#
#
#print("test:\n", np.std(data[:,3])**2-cov_matrix[3,0]/cov_matrix[0,0] - cov_matrix[3,1]/cov_matrix[1,1])
#
#n_random_vars = data.shape[1]
#
#for i in range(n_random_var
#
#from pgmpy.estimators import HillClimbSearch, PC, ExhaustiveSearch
#from pgmpy.base import DAG
#
#G = DAG()
#G.add_nodes_from(nodes=['X1', 'X2', 'X3', 'X4'])
#
#G.add_edges_from(ebunch=[("X1","X2"),("X2","X4"),("X2","X3")])
#
#from pgmpy.utils import get_example_model
#model = get_example_model('alarm')
#
#print(type(model))  
#
#from pgmpy.models import BayesianNetwork
#
#df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4'])
#from pgmpy.estimators import MaximumLikelihoodEstimator
#from pgmpy.estimators import BicScore
#
#bn = BayesianNetwork
#
#hill_climb = HillClimbSearch(df)
#
#best_model = hill_climb.estimate(scoring_method="bdeuscore")
#
#import networkx as nx
#import matplotlib.pyplot as plt
#
#def draw_pgm(ax, model, pos: dict = None, title: str = None):
   #Init network instance
  #G = nx.DiGraph()
#
   #Add nodes and edges
  #G.add_nodes_from(model.nodes())
  #G.add_edges_from(model.edges())
#
  #if pos is None:
    #pos = nx.circular_layout(G)
#
  #nx.draw(G, pos,
        #with_labels=True,
        #node_size = 1000, node_color = "skyblue",
        #font_size = 10, font_weight = "bold",
        #arrowsize=30, ax=ax)
  #ax.set_title(title)
#
#SCORES = ["k2score","bdeuscore","bicscore","aicscore"]
#best_models = []
#
#ncols = len(SCORES)//2
#fig, axs = plt.subplots(ncols, ncols, figsize=(10, 10))
#axs = axs.flatten()
#for iplot, score in enumerate(SCORES):
    #best_model = hill_climb.estimate(scoring_method=score)
    #best_models.append(best_model)
    #print(f"Best model with {score}: {best_model.edges()}")
    #draw_pgm(axs[iplot], best_model, title=score)
#plt.show()
#


