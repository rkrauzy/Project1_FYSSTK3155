
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix
from src.regression import ridge


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degree = 15
lambdas = np.logspace(-8, 2, 11)

X_train = design_matrix(x_train, degree)

mean_X = np.mean(X_train[:, 1:], axis=0)
X_train[:, 1:] = X_train[:, 1:] - mean_X

theta_values = np.zeros((len(lambdas), X_train.shape[1]))

for i, lmbda in enumerate(lambdas):
    theta = ridge(X_train, y_train, lmbda)
    theta_values[i, :] = theta

plt.figure(figsize=(10, 6))

for j in range(theta_values.shape[1]):
    plt.plot(
        lambdas,
        theta_values[:, j],
        marker="o",
        linewidth=2,
        label=f"theta_{j}"
    )

plt.xscale("log")
plt.yscale("symlog", linthresh=1)
plt.xlabel("Lambda")
plt.ylabel("Coefficient value")
plt.title("Ridge: Coefficients as a function of lambda")
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()
