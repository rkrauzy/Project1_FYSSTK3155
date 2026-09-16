import numpy as np
import matplotlib.pyplot as plt

data = np.load("results_part_a.npz")

degrees = data["degrees"]
r2_train = data["r2_train"]
r2_test = data["r2_test"]

plt.figure(figsize=(8, 5))

plt.plot(degrees, r2_train, marker="o", label="Training $R^2$")
plt.plot(degrees, r2_test, marker="o", label="Test $R^2$")

plt.xlabel("Polynomial degree")
plt.ylabel("$R^2$")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
