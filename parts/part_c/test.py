
import numpy as np
import matplotlib.pyplot as plt
from src.plotting import save_fig

x = np.linspace(-2, 2, 100)

def f(x):
    return x**3


plt.plot(x, f(x))
plt.title("This is just a test")
save_fig("test")
plt.show()
plt.close()



