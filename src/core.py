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
    for var_name in structure.keys():
        ivar = map_nodes_to_indexes[var_name]
        dependent_vars = structure.get(var_name, [])
        if len(dependent_vars)==0:
            # then var is independent
            intercept = np.mean(data[:, ivar])
            residuals = data[:, ivar] - intercept
            fit_results[var_name] = {
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
            fit_results[var_name] = {
                "intercept": intercept,
                "coefficients": coefficients,
                "residual_variance": np.var(residuals, ddof = X.shape[1])
            }
    return fit_results

def kl_bn(associated_data_old, associated_data_new, fit_results_old, fit_results_new, map_indexes_to_nodes):
    """
    Returns the KL divergence between two bayesian networks
    """
    kl = 0
    norm_l2_list = np.linalg.norm(associated_data_old - associated_data_new, ord=2, axis=0)
    for ivar in range(associated_data_old.shape[1]):
        var = map_indexes_to_nodes[ivar]
        norm_l2 = norm_l2_list[ivar]
        kl_ivar = 0.5 * (
                        np.log(fit_results_new[var]["residual_variance"] / fit_results_old[var]["residual_variance"]) \
                        + fit_results_old[var]["residual_variance"] / fit_results_new[var]["residual_variance"] \
                        - 1 \
                        ) \
                        + 1/(2*associated_data_old.shape[0]) * ( norm_l2 / fit_results_new[var]["residual_variance"] )
        print("KL FOR VAR ", ivar, " : ", kl_ivar)
        kl += kl_ivar
    return kl

