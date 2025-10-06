import numpy as np
import json
from pathlib import Path
from projeto1 import MLP, one_hot  # importa sua classe e função

# Paths
outdir = Path("processed")
X_test = np.load(outdir / "X_test.npy")
y_test = np.load(outdir / "y_test.npy")
X_train = np.load(outdir / "X_train.npy")
y_train = np.load(outdir / "y_train.npy")
X_val = np.load(outdir / "X_val.npy")
y_val = np.load(outdir / "y_val.npy")

# Hiperparâmetros (você pode pegar isso do final_metrics.json também)
with open(outdir / "final_metrics.json") as f:
    final_metrics = json.load(f)

layers = tuple(final_metrics["layers"])
l2 = final_metrics["l2"]
momentum = final_metrics["momentum"]
threshold = final_metrics["threshold_val_f1"]
batch_size = final_metrics["batch_size"]
epochs = final_metrics["epochs"]
lr_init = final_metrics["lr_init"]
patience = final_metrics["patience"]
class_weights = np.array(final_metrics["class_weights"], dtype=np.float32)

# One-hot
Y_train = one_hot(y_train, 2)
Y_val = one_hot(y_val, 2)

# Treinar novamente o MLP com os mesmos dados (replicando treino para recuperar pesos)
mlp = MLP(
    d_in=X_train.shape[1],
    layers=layers,
    d_out=2,
    seed=42,
    l2=l2,
    momentum=momentum
)
mlp.set_class_weights(class_weights)
mlp.init_output_bias_with_prior(float((y_train == 1).mean()))

# Repetir treino
rng = np.random.default_rng(42)
best_params = {k: v.copy() for k, v in mlp.params.items()}
best_vl = float("inf")
wait = 0
lr = lr_init

def iterate_minibatches(X, Y, batch, rng):
    idx = rng.permutation(len(X))
    for i in range(0, len(X), batch):
        ib = idx[i:i+batch]
        yield X[ib], Y[ib]

for ep in range(1, epochs+1):
    for xb, yb in iterate_minibatches(X_train, Y_train, batch_size, rng):
        P, cache = mlp.forward(xb)
        loss = mlp.loss(P, yb)
        grads = mlp.backward(cache, yb)
        mlp.step_sgd(grads, lr)
    # Validação
    P_vl, _ = mlp.forward(X_val)
    loss_vl = mlp.loss(P_vl, Y_val)
    if loss_vl < best_vl - 1e-4:
        best_vl = loss_vl
        wait = 0
        best_params = {k: v.copy() for k, v in mlp.params.items()}
    else:
        wait += 1
        if wait >= patience:
            break
    if ep % 5 == 0:
        lr *= 0.9

# Restaurar melhores pesos
mlp.params = best_params

# Predição no teste com threshold ótimo
yhat_test, _ = mlp.predict(X_test, threshold=threshold)
np.save(outdir / "final_yhat_test.npy", yhat_test)
print("✅ Predições salvas em processed/final_yhat_test.npy")