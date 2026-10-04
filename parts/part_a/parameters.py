# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 3 - Skeleton
# Role: Assisted with the overall structure of the coefficient analysis,
# including storing and plotting OLS coefficients across polynomial degrees.
# Verification: Reviewed, executed and interpreted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ols
from src.plotting import save_fig


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
y_train_c, y_mean = center_y(y_train)

max_degree = 15
degrees = range(1, max_degree + 1)

# coeff_matrix[d-1, k] = theta_k for polynomial of degree d (NaN if k >= d)
coeff_matrix = np.full((max_degree, max_degree), np.nan)

for d in degrees:
    X_train = design_matrix(x_train, d)
    X_train_s = scale_matrix(X_train)
    theta = ols(X_train_s, y_train_c)
    coeff_matrix[d - 1, :d] = theta

plt.figure(figsize=(9, 7))
colors = plt.cm.viridis(np.linspace(0, 1, max_degree))   # one colour per power, light to dark

for k in range(max_degree):
    mask = ~np.isnan(coeff_matrix[:, k])
    plt.plot(
        np.array(list(degrees))[mask],
        coeff_matrix[mask, k],
        "o-",
        linewidth=2,
        color=colors[k],
        label=rf"$\theta_{{{k + 1}}}$",
    )

plt.xlabel("Polynomial degree")
plt.ylabel("Coefficient value")
plt.title("OLS: Coefficients as a function of polynomial degree")
plt.xticks(degrees)
plt.yscale("symlog", linthresh=1)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, max_degree))
plt.colorbar(sm, ax=plt.gca(), label=r"Index $j$ of $\theta_j$")   # replaces a legend with one entry per coefficient
plt.tight_layout()
save_fig("ols_parameters")
