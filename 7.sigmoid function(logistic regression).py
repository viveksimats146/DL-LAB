import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
y = 1/(1+np.exp(-x))

plt.plot(x,y)
plt.title("Sigmoid Function")
plt.show()
