# LLM-assisted
# Tool: Claude, Sonnet 5.5 (Anthropic, October 2026)
# Level: 4 - Substantial
# Role: Suggested and substantially assisted with an additional 
# condition-number analysis for gradient based methods.
# The suggestion was reviewed and retained by the project authors to provide
# quantitative support for the discussion of optimisation dynamics and 
# convergence rates.
# Verification: Reviewed, executed and interpreted by the project authors.

import numpy as np
from src.data import generate_data, design_matrix, scale_matrix, split_data

degree, lam, beta = 5, 0.01, 0.9    

x, y, _ = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
X = scale_matrix(design_matrix(x_train, degree=degree))   

n = X.shape[0]                        
s = np.linalg.svd(X, compute_uv=False)


ev_ols = 2 * s**2 / n
lmax, lmin = ev_ols.max(), ev_ols.min()
print("kappa(X)       =", s.max() / s.min())
print("kappa(H_OLS)   =", lmax / lmin)             
print("eta_max (OLS)  =", 2 / lmax)
print("momentum bound =", 2 * (1 + beta) / lmax)   

ev_r = ev_ols + 2 * lam
print("kappa(H_Ridge) =", ev_r.max() / ev_r.min())
print("eta_max (Ridge)=", 2 / ev_r.max())