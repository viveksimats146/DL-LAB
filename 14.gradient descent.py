import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def gradient_descent(x, y, lr=0.01, iters=1000):
    w, b = 0, 0
    n = len(x)

    for _ in range(iters):
        y_pred = w*x + b
        w -= lr * (-2/n) * np.sum(x*(y - y_pred))
        b -= lr * (-2/n) * np.sum(y - y_pred)

    return w, b

X = np.array([32.5,53.4,61.5,47.4,59.8,55.1,52.2,39.2,48.1,52.5,
              45.4,54.3,44.1,58.1,56.7,48.9,44.6,60.2,45.6,38.8])
Y = np.array([31.7,68.7,62.5,71.5,87.2,78.2,79.6,59.1,75.3,71.3,
              55.1,82.4,62.0,75.3,81.4,60.7,82.8,97.3,48.8,56.8])

X = StandardScaler().fit_transform(X.reshape(-1,1)).flatten()

w, b = gradient_descent(X, Y)

print("Weight:", w, "\nBias:", b)

plt.scatter(X, Y)
plt.plot(X, w*X + b)
plt.show()
    