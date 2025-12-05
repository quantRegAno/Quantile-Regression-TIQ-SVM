#Import
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

import osqp
from scipy.sparse import csc_matrix

import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys

trickMachineEpislon = 1/sys.float_info.epsilon

#Kernel examples
class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def kernel(self,X,Y):
        N, d = X.shape
        M, _ = Y.shape
        X = X.reshape(N,1,d,1)
        Y = Y.reshape(1,M,d,1)
        G = ( (X-Y).transpose((0,1,3,2)) @(X-Y) ).reshape(N,M)
        return np.exp(-G/(2*self.sigma**2))
    
class Linear:
    def __init__(self):
        self = self
    def kernel(self,X,Y):
        return X @ Y.T
    
    
def pinball_loss(u, tau):
    """
    Fonction de perte quantile : rho_tau(u) = (tau - 1{u < 0}) * u
    """
    return (tau - (u < 0)) * u

def solve_qp(P, q, A, l, u, solver="osqp", max_attempts=1):
    if solver == "osqp":
        prob = osqp.OSQP()
        prob.setup(
            P=P, q=q, A=A, l=l, u=u,
            verbose=False,
            max_iter=5000,
            rho=1.0,
            scaling=True,
            eps_abs=1e-6,
            eps_rel=1e-6,
        )
        result = prob.solve()

        # Check solver status
        if result.info.status != "solved":
            return None

        # Check for numerical issues
        x = result.x
        if x is None or not np.all(np.isfinite(x)):
            return None

        return x

    else:
        raise ValueError(f"Unsupported solver: {solver}")

        
def standard_conditional_quantiles(X, T, tau, eps=1e-6):
    """
    Estimate conditional quantiles of a positive survival time using
    standard linear quantile regression applied to log-transformed times.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Input features.
    T : array of shape (n_samples,)
        Observed survival times (must be >= 0).
    tau : float in (0, 1)
        Quantile level to estimate.
    eps : float, optional (default=1e-6)
        Small positive constant added before log-transform to avoid log(0).

    Returns
    -------
    q_hat : ndarray of shape (n_samples,)
        Estimated conditional tau-quantiles of T(x), guaranteed to be >= 0.
    
    Notes
    -----
    - The model performs quantile regression on Z = log(T + eps).
    - Predictions are mapped back to the original time scale using:
        T_hat = exp(Z_hat) - eps
    - This ensures positivity and stabilizes estimation for highly skewed
      or heavy-tailed time distributions, which is typical in survival data.
    """

    #Log-transform survival times to stabilize estimation
    T_log = np.log(T + eps)

    #Fit standard quantile regression on the log scale
    X_with_const = sm.add_constant(X)
    quant_reg = sm.QuantReg(T_log, X_with_const)
    res = quant_reg.fit(q=tau)

    #Predict log-quantiles
    T_log_hat = res.predict(X_with_const)

    #Map predictions back to the time scale (ensure positivity)
    T_hat = np.exp(T_log_hat) - eps
    T_hat = np.clip(T_hat, 0.0, None)

    return T_hat




def cond_survival(T_c, delta, X):
    # Convertir les données de survie en format Surv (qui est requis pour sksurv)
    y = Surv.from_arrays(delta, T_c)
    
    # Entraîner un modèle RandomSurvivalForest
    estimator = RandomSurvivalForest().fit(X, y)
    
    # Liste pour stocker les résultats de la fonction de survie
    res = []
    
    # Traiter chaque observation dans X
    for i in range(len(delta)):
        # Calculer la fonction de survie pour l'observation i
        temp = estimator.predict_survival_function(X[i].reshape(1, -1), return_array=True)
        
        # Obtenir l'indice de la prédiction au temps T_c[i]
        # Remarque: ici, temp[0] est la fonction de survie pour X[i]
        temp_ind = np.searchsorted(temp[0, :], T_c[i])  # Trouver l'indice où la survie change
        
        # Si temp_ind est supérieur à la taille de temp, on le corrige
        if temp_ind >= len(temp[0]):
            temp_ind = len(temp[0]) - 1
        
        # Ajouter la valeur de la fonction de survie pour ce temps spécifique
        res.append(temp[0, temp_ind])
    
    # Retourner les valeurs de la fonction de survie pour chaque observation
    return np.array(res)
    
#Functions
class TIQ_SVM(BaseEstimator):
    
    def __init__(self, kernel, lambda_, tau, epsilon=1e-10):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.tau = tau
        self.epsilon = epsilon
        self.alpha = None
        self.support_Indices = None
        self.support_vec = None
        self.offset = None
        
    def fit(self, X, delta, T_c, func_cond_surv, truncation_func=standard_conditional_quantiles,max_attempts=5):
        quantiles_T_c = standard_conditional_quantiles(X, T_c.flatten(), (3*self.tau+1)/4)
        truncated_T_c = np.minimum(T_c, quantiles_T_c)

        ## Variable needed
        K = self.kernel(X, X)
        N = len(X)
        C = 1 / (2 * N * self.lambda_)
        arr_cond_surv = func_cond_surv(truncated_T_c, 1 - delta, X)
        w = delta / arr_cond_surv
        w = np.where((np.isnan(w)) | (arr_cond_surv == 0), trickMachineEpislon, w)
        
        P = csc_matrix(K)
        q = -truncated_T_c
        A = csc_matrix(np.vstack((np.eye(N), -np.eye(N))))
        l = -np.inf * np.ones(2*N) 
        u = C * np.vstack((self.tau * w.reshape(-1, 1), (1 - self.tau) * w.reshape(-1, 1))).flatten()
        
        self.alpha = solve_qp(P, q, A, l, u, solver="osqp", max_attempts=5)
        if self.alpha is None:
            raise ValueError("Alpha is None. The optimization might have failed.")
        
        # Find support indices
        self.support_Indices = np.where(self.epsilon < np.abs(self.alpha))[0]
        if len(self.support_Indices) == 0:
            self.support_Indices = np.linspace(0, N - 1, N).astype(int)
        
        self.support_vec = X[self.support_Indices]
        self.offset = (T_c - K @ self.alpha)[self.support_Indices[0]]
        
        return self
    
    def predict(self, x):
        return self.kernel(x, self.support_vec) @ self.alpha[self.support_Indices] + self.offset
    
class CSVM_biased(BaseEstimator):
    
    def __init__(self, kernel, lambda_, tau, arr_cond_surv, epsilon=1e-10):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.tau = tau
        self.epsilon = epsilon
        self.alpha = None
        self.support_Indices = None
        self.support_vec = None
        self.offset = None
        self.arr_cond_surv = arr_cond_surv
        
    def fit(self, X, delta, T_c, trick, max_attempts=5):
        ## Variable needed
        K = self.kernel(X, X)
        N = len(X)
        C = 1 / (2 * N * self.lambda_)
        
        arr_cond_surv = self.arr_cond_surv
        w = delta / arr_cond_surv 
        w = np.where((np.isnan(w)) | (arr_cond_surv < 0.1), trick, w)
        
        P = csc_matrix(K)
        q = -T_c
        A = csc_matrix(np.vstack((np.eye(N), -np.eye(N))))
        l = -np.inf * np.ones(2*N) 
        u = C * np.vstack((self.tau * w.reshape(-1, 1), (1 - self.tau) * w.reshape(-1, 1))).flatten()
        
        self.alpha = solve_qp(P, q, A, l, u, solver="osqp", max_attempts=5)
        if self.alpha is None:
            raise ValueError("Alpha is None. The optimization might have failed.")
        
        # Find support indices
        self.support_Indices = np.where(self.epsilon < np.abs(self.alpha))[0]
        if len(self.support_Indices) == 0:
            self.support_Indices = np.linspace(0, N - 1, N).astype(int)
        
        self.support_vec = X[self.support_Indices]
        self.offset = (T_c - K @ self.alpha)[self.support_Indices[0]]
        
        return self
    
    def predict(self, x):
        return self.kernel(x, self.support_vec) @ self.alpha[self.support_Indices] + self.offset
