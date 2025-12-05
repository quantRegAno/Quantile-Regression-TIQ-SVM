import numpy as np
from scipy.stats import beta

def simulate_censored_data(param_weights, n_samples=1000, d=3, theta=0.7, random_state=None, verbose=False):
    """
    Simulates censored survival data with Beta-distributed T and C conditional on covariates X.
    
    Parameters:
        n_samples (int): Number of samples to generate.
        d (int): Number of covariate dimensions.
        theta (float): Parameter controlling the censoring level (0 = high censorship, 1 = low censorship).
        random_state (int or None): Seed for reproducibility.
    
    Returns:
        dict: Dictionary containing:
            - X (ndarray): Covariate matrix (n_samples, d).
            - T (ndarray): Event times.
            - C (ndarray): Censoring times.
            - delta (ndarray): Censoring indicator (1 = uncensored, 0 = censored).
            - T_tilde (ndarray): Observed times (min(T, C)).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate covariates X
    X = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n_samples)

    # Define weights for Beta distribution parameters
    weights_T_alpha = param_weights["T_alpha"]
    weights_T_beta = param_weights["T_beta"]
    weights_C_alpha = param_weights["C_alpha"]
    weights_C_beta = param_weights["C_beta"]

    # Function to compute Beta parameters based on X and theta
    def compute_alpha_beta(X, weights_alpha, weights_beta, theta):
        alpha = None
        beta = None
        if theta is not None:    
            alpha = np.exp(np.dot(X, weights_alpha)) * theta
            beta = np.exp(np.dot(X, weights_beta)) / theta
        else:
            alpha = np.exp(np.dot(X, weights_alpha))
            beta = np.exp(np.dot(X, weights_beta))
        return alpha, beta

    # Compute parameters for T and C
    alpha_T, beta_T = compute_alpha_beta(X, weights_T_alpha, weights_T_beta, theta=None)  # T is independent of theta
    alpha_C, beta_C = compute_alpha_beta(X, weights_C_alpha, weights_C_beta, theta)

    # Generate T and C
    T = beta.rvs(alpha_T, beta_T)
    C = beta.rvs(alpha_C, beta_C)

    # Compute observed times and censoring status
    T_tilde = np.minimum(T, C)
    delta = (T <= C).astype(int)
    
    if verbose:
        # Compute and print censoring rate
        censorship_rate = 1 - np.mean(delta)
        print(f"Censorship rate: {censorship_rate:.2%}")

    return {"X": X, "T": T, "C": C, "delta": delta, "T_tilde": T_tilde}