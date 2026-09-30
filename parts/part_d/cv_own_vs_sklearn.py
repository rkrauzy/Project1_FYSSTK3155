"""
Own k-fold cross-validation loop for Ridge regression, compared with
Scikit-Learn's cross_val_score using the same fold assignment, at degree 12.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with adapting the own k-fold loop from week36tuesday (Case 2,
    Steps 1 and 2) to the repository structure using design_matrix, scale_matrix,
    center_y and our own ridge function, and with matching the lambda convention
    by using alpha = n_train * lambda in Scikit-Learn.

Verification: 
    Reviewed and executed by the project authors. The own loop and
    cross_val_score agree to a relative difference of order 1e-10.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

from src.data import generate_data, design_matrix, scale_matrix, center_y
from src.models import ridge
from src.plotting import save_fig

BLUE, RED, YELLOW = "#004488", "#BB5566", "#DDAA33"

n = 100
x, y, y_true = generate_data(n)

degree = 12
nlambdas = 50
lambdas = np.logspace(-10, 2, nlambdas)
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=2026)

scores_KFold = np.zeros((nlambdas, k))
for i, lmb in enumerate(lambdas):
    for j, (train_inds, test_inds) in enumerate(kfold.split(x)):
        xtrain, ytrain = x[train_inds], y[train_inds]
        xtest, ytest = x[test_inds], y[test_inds]

        Xtrain_s, Xtest_s = scale_matrix(design_matrix(xtrain, degree), design_matrix(xtest, degree))
        ytrain_c, y_mean = center_y(ytrain)

        theta = ridge(Xtrain_s, ytrain_c, lmb)
        ypred = Xtest_s @ theta + y_mean
        scores_KFold[i, j] = np.mean((ypred - ytest)**2)
mse_KFold = np.mean(scores_KFold, axis=1)

n_train = n * (k - 1) // k
mse_sklearn = np.zeros(nlambdas)
for i, lmb in enumerate(lambdas):
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=n_train * lmb))
    scores = cross_val_score(pipe, design_matrix(x, degree), y, cv=kfold,
                             scoring="neg_mean_squared_error")
    mse_sklearn[i] = np.mean(-scores)

best = np.argmin(mse_sklearn)
print(f"best lambda {lambdas[best]:.4g}, cross-validated MSE {mse_sklearn[best]:.4f}")
print(f"largest relative difference: {np.max(np.abs(mse_KFold - mse_sklearn) / mse_sklearn):.2e}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.log10(lambdas), mse_KFold, color=BLUE, lw=2, label="own KFold loop")
ax.plot(np.log10(lambdas), mse_sklearn, "--", color=RED, lw=2, label="cross_val_score")
ax.plot(np.log10(lambdas[best]), mse_sklearn[best], "o", color=YELLOW, ms=9,
        label=rf"minimum: $\lambda\approx{lambdas[best]:.3g}$")
ax.set_yscale("log")
ax.set_xlabel(r"$\log_{10}\lambda$")
ax.set_ylabel("Cross-validated MSE")
ax.set_title(f"Ridge, degree {degree}: own k-fold loop and cross_val_score")
ax.legend(frameon=False)
plt.tight_layout()
save_fig("cv_own_vs_sklearn")