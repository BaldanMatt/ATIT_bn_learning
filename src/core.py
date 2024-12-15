import numpy as np
from sklearn.linear_model import LinearRegression

def estimate_parameters(structure : dict, data : np.array, map_nodes_to_indexes : dict) -> dict:
    """
    Given the sampled data, and the DAG skeleton, estimate the parameters of
    the conditional probabilities
    The structure must be a dict that maps each variable node to the list of
    parents
    The map_nodes_to_indexes is needed to define which coulumn corresponds to which node
    TODO: maybe switch to dataframe for data
    """
    n_obs, n_vars = data.shape
    fit_results = {}
    for ivar, var_name in enumerate(structure.keys()):
        dependent_vars = structure.get(var_name, [])
        if len(dependent_vars)==0:
            # then var is independent
            intercept = np.mean(data[:, ivar])
            residuals = data[:, ivar] - intercept
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
            fit_results[ivar] = {
                "intercept": intercept,
                "coefficients": coefficients,
                "residual_variance": np.var(residuals, ddof = X.shape[1])
            }
    return fit_results

