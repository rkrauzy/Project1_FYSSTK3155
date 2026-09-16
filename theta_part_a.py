import numpy as np
import matplotlib.pyplot as plt

data = np.load("results_part_a.npz")

degrees = data["degrees"]
theta_values = data["theta_values"]

plt.figure(figsize=(9, 6))

for j in range(16):
    plt.plot(degrees, theta_values[:, j], marker="o", label=fr"$\theta_{j}$")

plt.yscale("symlog", linthresh=1e-2)

plt.xlabel("Polynomial degree")
plt.ylabel("Parameter value")
plt.legend(ncol=4, fontsize=8)
plt.grid(alpha=0.3)
plt.show()
