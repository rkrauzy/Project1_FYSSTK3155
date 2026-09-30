import numpy as np
from sklearn.linear_model import Lasso


def ols(X, y):
    return np.linalg.pinv(X) @ y


def ridge(X, y, lmbda):
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lmbda * np.eye(p), X.T @ y)

def lasso_fit(X, y, lmbda):
    """Minimiser of ||y - X theta||^2 / n + lmbda ||theta||_1, Eq. (3.lassoproblem).
    scikit-learn minimises ||y - X theta||^2 / (2n) + alpha ||theta||_1: alpha = lmbda/2.
    
    got this from lecture notes chapter 3. 
    """
    return Lasso(alpha=lmbda / 2.0, fit_intercept=False,
                 max_iter=100000, tol=1e-10).fit(X, y).coef_
