"""
Plain gradient descent, momentum, AdaGrad, RMSprop and Adam with and without
stochastic gradient descent, for OLS and Ridge at degree 5, compared with the
closed-form solutions. After the same number of epochs, full batch and SGD
have used the same number of single-point gradient evaluations (epochs x n).

LLM-assisted
------------
Tool: Claude, Opus 5.5 (Anthropic, September 2026)
Level: 3 - Skeleton
Role: Assisted with assembling the script from the week 38 Tuesday notebook
    (week38tuesday.ipynb): the full-batch learning rates from Case 1, Step 3,
    and the SGD learning rates and distance printout from Case 2, Step 5, run
    with our own sgd from src/optimiser.py. Claude suggested running full batch
    and SGD in the same loop, using 1000 epochs so that full batch has time to
    converge, and timing each run with time.perf_counter.

Verification:
    Reviewed and executed by the project authors. Full-batch momentum reaches
    the closed-form OLS and Ridge solutions to 1e-15. Momentum with M = 5 at
    gamma = 0.05 diverges, as it did on the degree-5 exercise data in
    Case 2, Step 5.
"""

import time
import numpy as np

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols, ridge
from src.optimiser import sgd

n = 100
x, y, y_true = generate_data(n)
x_train, x_test, y_train, y_test = split_data(x, y)

degree = 5
X_train_s = scale_matrix(design_matrix(x_train, degree))
y_train_c, y_mean = center_y(y_train)
n_train = len(y_train_c)

n_epochs = 1000
batch_size = 5
summary = {}                              # (name, method) -> {"full"/"sgd": (gamma, d10, d1000, time)}

for name, lmbda, theta_exact in (("OLS", 0.0, ols(X_train_s, y_train_c)),
                                 ("Ridge", 0.01, ridge(X_train_s, y_train_c, 0.01))):
    eigs = np.linalg.eigvalsh((2 / n_train) * X_train_s.T @ X_train_s + 2 * lmbda * np.eye(degree))
    gammas_full = {"plain": 0.9 * 2 / eigs.max(), "momentum": 0.9 * 2 / eigs.max(),
                   "adagrad": 0.5, "rmsprop": 0.01, "adam": 0.05}
    gammas_sgd = {"plain": 0.1, "momentum": 0.05, "adagrad": 0.5, "rmsprop": 0.01, "adam": 0.05}

    print(f"\n{name}")
    for method in gammas_full:
        for label, M, gamma in (("full batch", n_train, gammas_full[method]),
                                (f"SGD M = {batch_size}", batch_size, gammas_sgd[method])):
            start = time.perf_counter()
            hist = sgd(X_train_s, y_train_c, method=method, n_epochs=n_epochs, batch_size=M,
                       gamma=gamma, lmbda=lmbda)
            elapsed = time.perf_counter() - start
            d = np.linalg.norm(hist - theta_exact, axis=1)
            print(f"{method:8s} {label:11s} gamma = {gamma:.3f}: distance after 10 / 100 / 1000 epochs "
                  f"{d[10]:.1e} / {d[100]:.1e} / {d[1000]:.1e}, "
                  f"updates {n_epochs * int(np.ceil(n_train / M)):5d}, time {elapsed:.2f} s")
            key = "full" if M == n_train else "sgd"
            summary.setdefault((name, method), {})[key] = (gamma, d[10], d[1000], elapsed)

# One line per method, in the layout of tables/sgd_methods.tex
for name in ("OLS", "Ridge"):
    print(f"\nSummary {name}: gamma full / SGD, distance after 10 epochs full / SGD, "
          f"after 1000 epochs full / SGD, time full / SGD")
    for method in ("plain", "momentum", "adagrad", "rmsprop", "adam"):
        (gf, f10, f1000, tf), (gs, s10, s1000, ts) = (summary[(name, method)][k] for k in ("full", "sgd"))
        print(f"{method:8s}  {gf:.2f} / {gs:.2f}   {f10:.1e} / {s10:.1e}   "
              f"{f1000:.1e} / {s1000:.1e}   {tf:.2f} / {ts:.2f} s")
