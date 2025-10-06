import os
import pandas as pd
import matplotlib.pyplot as plt

# Garantir que o diretório de saída existe
os.makedirs("assets", exist_ok=True)

# Carregar o CSV com as curvas
df = pd.read_csv("processed/training_curves.csv")

# Plotar curvas de perda
plt.figure(figsize=(8, 5))
plt.plot(df["loss_tr"], label="Treino", linewidth=2)
plt.plot(df["loss_vl"], label="Validação", linewidth=2)
plt.xlabel("Época")
plt.ylabel("Loss (Cross-Entropy)")
plt.title("Curva de perda (loss) por época")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Salvar imagem
plt.savefig("assets/loss_curve.png")
plt.show()