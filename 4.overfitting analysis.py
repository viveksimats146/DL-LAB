import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.sort(np.random.rand(30))
y = np.cos(1.5 * np.pi * X) + np.random.randn(30)*0.1

for d in [1,4,15]:
    model = Pipeline([
        ("poly", PolynomialFeatures(d)),
        ("lin", LinearRegression())
    ])
    model.fit(X.reshape(-1,1), y)

    X_test = np.linspace(0,1,100)
    plt.plot(X_test, model.predict(X_test.reshape(-1,1)), label=f"Degree {d}")

plt.scatter(X, y)
plt.legend()
plt.show()
