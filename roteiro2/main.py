import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Parameters
mu0, mu1 = [1.5, 1.5], [5, 5]
cov = [[0.5, 0], [0, 0.5]]

X0 = rng.multivariate_normal(mu0, cov, size=1000)
X1 = rng.multivariate_normal(mu1, cov, size=1000)

X = np.vstack((X0, X1))
y = np.hstack((np.ones(1000)*-1, np.ones(1000)))  # -1 and +1 labels

# Visualization
plt.scatter(X0[:,0], X0[:,1], c='blue', alpha=0.6, label='Class 0')
plt.scatter(X1[:,0], X1[:,1], c='red', alpha=0.6, label='Class 1')
plt.title("Exercise 1 — Linearly Separable Data")
plt.xlabel("x₁"); plt.ylabel("x₂")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("ex1_data.png", dpi=150)
plt.close()

class Perceptron:
    def __init__(self, lr=0.01, max_epochs=100, seed=42):
        self.lr = lr
        self.max_epochs = max_epochs
        self.rng = np.random.default_rng(seed)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = self.rng.standard_normal(n_features)
        self.b = 0.0
        
        self.accuracy_ = []
        for epoch in range(self.max_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = target * (np.dot(self.w, xi) + self.b)
                if update <= 0:  # Misclassified
                    self.w += self.lr * target * xi
                    self.b += self.lr * target
                    errors += 1
            y_pred = self.predict(X)
            acc = np.mean(y_pred == y)
            self.accuracy_.append(acc)
            if errors == 0:
                break
        return self
    
    def predict(self, X):
        linear_output = X @ self.w + self.b
        return np.where(linear_output >= 0, 1, -1)

perceptron = Perceptron(lr=0.01, max_epochs=100)
perceptron.fit(X, y)

# Decision boundary
x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2_boundary = -(perceptron.w[0]*x1_range + perceptron.b) / perceptron.w[1]

plt.scatter(X[y==-1,0], X[y==-1,1], color='blue', label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], color='red', label='Class 1')
plt.plot(x1_range, x2_boundary, 'k--', label='Decision Boundary')
plt.legend(); plt.grid(True)
plt.title("Perceptron Decision Boundary — Exercise 1")
plt.savefig("ex1_boundary.png", dpi=150)
plt.close()

plt.plot(perceptron.accuracy_)
plt.title("Exercise 1 — Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("ex1_accuracy.png", dpi=150)
plt.close()

mu0, mu1 = [3, 3], [4, 4]
cov = [[1.5, 0], [0, 1.5]]

X0 = rng.multivariate_normal(mu0, cov, size=1000)
X1 = rng.multivariate_normal(mu1, cov, size=1000)

X = np.vstack((X0, X1))
y = np.hstack((np.ones(1000)*-1, np.ones(1000)))

plt.scatter(X0[:,0], X0[:,1], c='blue', alpha=0.6, label='Class 0')
plt.scatter(X1[:,0], X1[:,1], c='red', alpha=0.6, label='Class 1')
plt.title("Exercise 2 — Overlapping Data")
plt.xlabel("x₁"); plt.ylabel("x₂")
plt.legend(); plt.grid(True)
plt.savefig("ex2_data.png", dpi=150)
plt.close()

p2 = Perceptron(lr=0.01, max_epochs=100)
p2.fit(X, y)

x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2_boundary = -(p2.w[0]*x1_range + p2.b) / p2.w[1]

plt.scatter(X[y==-1,0], X[y==-1,1], color='blue', label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], color='red', label='Class 1')
plt.plot(x1_range, x2_boundary, 'k--', label='Decision Boundary')
plt.legend(); plt.grid(True)
plt.title("Perceptron Decision Boundary — Exercise 2")
plt.savefig("ex2_boundary.png", dpi=150)
plt.close()