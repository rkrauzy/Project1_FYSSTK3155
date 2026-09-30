"""
Plain SGD for OLS at degree 5: dependence on the minibatch size, the number of
epochs and the learning-rate schedule, measured by the distance to the
closed-form OLS solution after every epoch.

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with assembling the script from the week 38 Tuesday notebook
    (week38tuesday.ipynb) and our week 38 exercises (Exercise 4), run with
    our own sgd from src/optimiser.py: the batch sizes and the stability limit
    for M = 1 from Case 2, Step 3, the schedules from Case 2, Step 4, the sum
    of gamma_t from Exercise 4(c), and np.seterr and the axis limits from the
    notebook. Claude suggested splitting panel (b) of fig_sgd into one panel
    for the batch size and one for the schedule, with epochs on the x-axis.

Verification:
    Reviewed and executed by the project authors. M = 1 diverges at
    gamma = 0.1, above the single-point limit of 0.026, and the schedule
    (1, 10) freezes, the same behaviour as on the degree-5 exercise data in
    Case 2, Steps 3 and 4 of the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols
from src.optimiser import sgd, step_length
from src.plotting import save_fig

n = 100
x, y, y_true = generate_data(n)
x_train, x_test, y_train, y_test = split_data(x, y)

degree = 5
X_train_s = scale_matrix(design_matrix(x_train, degree))
y_train_c, y_mean = center_y(y_train)
n_train = len(y_train_c)
theta_ols = ols(X_train_s, y_train_c)

n_epochs = 1000
epochs = np.arange(n_epochs + 1)
np.seterr(over="ignore", invalid="ignore")   # diverging runs overflow; that is the answer, not an error

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for M in (1, 5, 20, n_train):
    hist = sgd(X_train_s, y_train_c, n_epochs=n_epochs, batch_size=M, gamma=0.1)
    d = np.linalg.norm(hist - theta_ols, axis=1)
    print(f"M = {M:3d}, gamma = 0.1: distance after 10 / 100 / 1000 epochs "
          f"{d[10]:.1e} / {d[100]:.1e} / {d[1000]:.1e}")
    axes[0].semilogy(epochs, d, label=f"M = {M}")
xmax = np.max(np.linalg.norm(X_train_s, axis=1))
print(f"largest single-point Hessian eigenvalue 2|x_i|^2 = {2 * xmax**2:.1f}, "
      f"so gamma must be below {2 / (2 * xmax**2):.3f} for M = 1\n")

T = n_epochs * int(np.ceil(n_train / 5))
for label, gamma, schedule in ((r"constant $\gamma = 0.1$", 0.1, None), (r"constant $\gamma = 0.02$", 0.02, None),
                               (r"$(t_0, t_1) = (1, 10)$", None, (1.0, 10.0)), (r"$(t_0, t_1) = (20, 200)$", None, (20.0, 200.0)),
                               (r"$(t_0, t_1) = (100, 1000)$", None, (100.0, 1000.0))):
    hist = sgd(X_train_s, y_train_c, n_epochs=n_epochs, batch_size=5, gamma=gamma, schedule=schedule)
    d = np.linalg.norm(hist - theta_ols, axis=1)
    gamma_t = np.full(T, gamma) if schedule is None else step_length(np.arange(1, T + 1), *schedule)
    print(f"{label:21s}: gamma after {n_epochs} epochs {gamma_t[-1]:.4f}, sum of gamma_t {gamma_t.sum():7.1f}, "
          f"distance after 10 / 100 / 1000 epochs {d[10]:.1e} / {d[100]:.1e} / {d[1000]:.1e}")
    axes[1].semilogy(epochs, d, label=label)

axes[0].set_title(r"Batch size, constant $\gamma = 0.1$")
axes[1].set_title(r"Learning-rate schedule, $M = 5$")
axes[0].set_ylabel(r"$\|\theta - \hat{\theta}_{\mathrm{OLS}}\|_2$")
axes[0].set_ylim(1e-3, 10)
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
plt.tight_layout()
save_fig("sgd_study")
