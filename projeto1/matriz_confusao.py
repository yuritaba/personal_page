import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Carregar dados
y_test = np.load("processed/y_test.npy")
y_pred = np.load("processed/final_yhat_test.npy")  # ou gere a predição no seu script final

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ["Não Inadimplente", "Inadimplente"]

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão — Conjunto de Teste")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png")  # ajuste o caminho conforme o GitHub Pages
plt.show()