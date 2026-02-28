import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print("Accuracy:", round(accuracy, 2))

# Plot data points
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
            edgecolor='k', s=60)

# Plot decision boundary
coef = model.coef_[0]
intercept = model.intercept_[0]

x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(coef[0] * x_vals + intercept) / coef[1]

plt.plot(x_vals, y_vals, 'k--')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear Separability using Logistic Regression')
plt.show()
