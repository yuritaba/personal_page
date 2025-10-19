import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import json
import time
from sklearn.preprocessing import StandardScaler
import warnings
import yfinance as yf

# ======================
# 1) Download dos dados (ajustes importantes)
# ======================
# - auto_adjust=False para manter "Adj Close"
# - group_by='column' para evitar MultiIndex nas colunas
# - actions=False porque não precisamos de dividendos/splits neste passo
df = yf.download(
    "^GSPC",
    start="1990-01-01",
    progress=False,
    auto_adjust=False,
    group_by="column",
    actions=False,
)

if df is None or df.empty:
    raise SystemExit(
        "Sem dados baixados. Verifique conexão/ambiente ou ajuste o intervalo de datas."
    )

# Garante colunas simples (não-MultiIndex), por segurança extra
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Renomeia o que existir
rename_map = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
df = df.rename(columns=existing_map)
df.index.name = "date"

# Se por algum motivo "adj_close" não existir, use "close"
if "adj_close" not in df.columns and "close" in df.columns:
    df["adj_close"] = df["close"]

# ======================
# 2) Garantir tipos numéricos (apenas no que existir)
# ======================
numeric_candidates = ["open", "high", "low", "close", "adj_close", "volume"]
numeric_cols = [c for c in numeric_candidates if c in df.columns]
for c in numeric_cols:
    # to_numeric espera Series/array; df[c] é Series aqui
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ======================
# 3) Engenharia de features
# ======================
# Retornos (t): simples e log
df["ret_1d"] = df["adj_close"].pct_change()
df["logret_1d"] = np.log(df["adj_close"]).diff()

# Lags (t-1) de preço ajustado e volume
if "adj_close" in df.columns:
    df["adj_close_lag1"] = df["adj_close"].shift(1)
if "volume" in df.columns:
    df["volume_lag1"] = df["volume"].shift(1)

# Médias móveis e volatilidade (janelas)
for w in (5, 20, 60):
    df[f"ma_{w}"] = df["adj_close"].rolling(window=w, min_periods=w).mean()
df["vol_20"] = df["logret_1d"].rolling(window=20, min_periods=20).std()

# Indicadores de amplitude e posição do fechamento
# Usa denominador com proteção para zero
rng = df["high"] - df["low"]
df["hl_range"] = (df["high"] - df["low"]) / df["low"].where(df["low"] != 0, np.nan)
df["close_pos_range"] = (df["close"] - df["low"]) / rng.where(rng != 0, np.nan)

# Remove linhas incompletas introduzidas por rolling/shift
df = df.dropna().copy()

# ======================
# 4) Target e features
# ======================
df["target_adj_close_tplus1"] = df["adj_close"].shift(-1)
df = df.dropna(subset=["target_adj_close_tplus1"]).copy()

feature_cols = [
    c
    for c in [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ret_1d",
        "logret_1d",
        "adj_close_lag1",
        "volume_lag1",
        "ma_5",
        "ma_20",
        "ma_60",
        "vol_20",
        "hl_range",
        "close_pos_range",
    ]
    if c in df.columns
]

X = df[feature_cols].copy()
y = df["target_adj_close_tplus1"].copy()

# ======================
# 5) Diagnósticos: faltantes e outliers
# ======================
missing = X.isna().sum().sort_values(ascending=False)
missing_pct = (X.isna().mean() * 100).round(3)

# Outliers por z-score |z|>4 (diagnóstico)
zscores = X.apply(zscore, nan_policy="omit")
outlier_mask = np.abs(zscores) > 4
outliers_por_col = outlier_mask.sum().sort_values(ascending=False)

# ======================
# 6) Estatísticas e salvamento
# ======================
desc_X = X.describe().T
desc_y = y.describe()

os.makedirs("eda_outputs", exist_ok=True)
desc_X.to_csv("eda_outputs/summary_stats_features.csv")
pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).to_csv(
    "eda_outputs/missing_values_report.csv"
)
outliers_por_col.to_csv("eda_outputs/outliers_by_feature_z4.csv")
df.to_csv("eda_outputs/sp500_features_target.csv")

# ======================
# 7) Gráficos (Matplotlib — sem seaborn)
# ======================
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["adj_close"])
plt.title("S&P 500 - Adj Close (Histórico)")
plt.xlabel("Data")
plt.ylabel("Adj Close")
plt.tight_layout()
plt.savefig("eda_outputs/timeseries_adj_close.png", dpi=150)
plt.close()


def save_hist(series, fname, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna().values, bins=bins)
    plt.title(f"Histograma - {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


for col in [
    c
    for c in ["ret_1d", "logret_1d", "volume", "vol_20", "hl_range", "close_pos_range"]
    if c in X.columns
]:
    save_hist(X[col], f"eda_outputs/hist_{col}.png")

# Matriz de correlação
corr = X.corr(numeric_only=True)
plt.figure(figsize=(9, 8))
im = plt.imshow(corr.values, aspect="auto", interpolation="nearest")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
plt.title("Matriz de Correlação — Features (numéricas)")
plt.tight_layout()
plt.savefig("eda_outputs/corr_matrix_features.png", dpi=150)
plt.close()


def save_scatter(x, y, fname, xlabel, ylabel):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=3, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs. {xlabel}")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


if "adj_close" in X.columns:
    save_scatter(
        X["adj_close"],
        y,
        "eda_outputs/scatter_target_vs_adj_close.png",
        "adj_close (t)",
        "target_adj_close (t+1)",
    )
if "vol_20" in X.columns:
    save_scatter(
        X["vol_20"],
        y,
        "eda_outputs/scatter_target_vs_vol_20.png",
        "vol_20 (t)",
        "target_adj_close (t+1)",
    )

print("✅ EDA concluída. Saídas salvas na pasta: eda_outputs/")
for f in sorted(os.listdir("eda_outputs")):
    print("-", f)

# ================================================================
# PARTE 3 — Data Cleaning and Normalization
# ================================================================
print("\n=== PARTE 3 — Data Cleaning and Normalization ===")

warnings.filterwarnings("ignore")

# -------------------------
# 1. Checar duplicatas e faltantes
# -------------------------
print("\nChecando duplicatas e faltantes...")

df = df[~df.index.duplicated(keep="first")].copy()
missing_report = df.isna().sum().sort_values(ascending=False)
print("Valores faltantes após EDA:\n", missing_report.head())

missing_report.to_csv("eda_outputs/missing_report_after_cleaning.csv")

# -------------------------
# 2. Split temporal: 70/15/15
# -------------------------
print("\nRealizando split temporal...")

dates = df.index.sort_values()
n = len(dates)
train_end = dates[int(0.70 * n)]
val_end = dates[int(0.85 * n)]

train_df = df.loc[:train_end].copy()
val_df = df.loc[train_end:].loc[:val_end].copy()
test_df = df.loc[val_end:].copy()

print(f"Tamanhos → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# -------------------------
# 3. Winsorização (limite 1% e 99%) para suavizar outliers
# -------------------------
print("\nAplicando winsorização (1–99%) em colunas assimétricas...")


def winsorize(df_in, cols, lower=0.01, upper=0.99, ref=None):
    df = df_in.copy()
    ref = ref if ref is not None else df
    for c in cols:
        lo, hi = ref[c].quantile([lower, upper])
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


winsor_cols = [c for c in ["volume", "vol_20", "hl_range"] if c in df.columns]

train_df = winsorize(train_df, winsor_cols)
val_df = winsorize(val_df, winsor_cols, ref=train_df)
test_df = winsorize(test_df, winsor_cols, ref=train_df)

# -------------------------
# 4. Normalização (StandardScaler — média 0, std 1)
# -------------------------
print("\nNormalizando features numéricas (StandardScaler)...")

feature_cols = [
    c
    for c in [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ret_1d",
        "logret_1d",
        "adj_close_lag1",
        "volume_lag1",
        "ma_5",
        "ma_20",
        "ma_60",
        "vol_20",
        "hl_range",
        "close_pos_range",
    ]
    if c in df.columns
]

target_col = "target_adj_close_tplus1"

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_val, y_val = val_df[feature_cols], val_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train), index=X_train.index, columns=X_train.columns
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val), index=X_val.index, columns=X_val.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), index=X_test.index, columns=X_test.columns
)

# -------------------------
# 5. Salvar conjuntos processados
# -------------------------
os.makedirs("preproc_outputs", exist_ok=True)
X_train_scaled.to_csv("preproc_outputs/X_train.csv")
y_train.to_csv("preproc_outputs/y_train.csv")
X_val_scaled.to_csv("preproc_outputs/X_val.csv")
y_val.to_csv("preproc_outputs/y_val.csv")
X_test_scaled.to_csv("preproc_outputs/X_test.csv")
y_test.to_csv("preproc_outputs/y_test.csv")

print("\nArquivos escalonados salvos em 'preproc_outputs/'")

# -------------------------
# 6. Comparação Before/After
# -------------------------
before_after = pd.concat(
    [
        X_train[winsor_cols].describe().T.add_prefix("raw_"),
        pd.DataFrame(X_train_scaled[winsor_cols]).describe().T.add_prefix("scaled_"),
    ],
    axis=1,
)
before_after.to_csv("preproc_outputs/before_after_summary.csv")

# -------------------------
# 7. Visualizações Before/After
# -------------------------
plt.figure(figsize=(6, 4))
plt.hist(X_train["volume"], bins=60)
plt.title("Volume - Antes da Normalização")
plt.tight_layout()
plt.savefig("preproc_outputs/hist_volume_before.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(X_train_scaled["volume"], bins=60)
plt.title("Volume - Após Normalização (StandardScaler)")
plt.tight_layout()
plt.savefig("preproc_outputs/hist_volume_after.png")
plt.close()

print("✅ Parte 3 finalizada com sucesso — dados limpos, winsorizados e normalizados.")

print("\n=== PARTE 4 — MLP Implementation (NumPy) ===")


# -------------------------
# 0) Carregar dados pré-processados (Parte 3)
# -------------------------
def _load_xy(prefix_dir="preproc_outputs"):
    X_train = pd.read_csv(f"{prefix_dir}/X_train.csv", index_col=0)
    y_train = pd.read_csv(f"{prefix_dir}/y_train.csv", index_col=0).values.reshape(
        -1, 1
    )
    X_val = pd.read_csv(f"{prefix_dir}/X_val.csv", index_col=0)
    y_val = pd.read_csv(f"{prefix_dir}/y_val.csv", index_col=0).values.reshape(-1, 1)
    X_test = pd.read_csv(f"{prefix_dir}/X_test.csv", index_col=0)
    y_test = pd.read_csv(f"{prefix_dir}/y_test.csv", index_col=0).values.reshape(-1, 1)
    return (X_train, y_train, X_val, y_val, X_test, y_test)


Xtr_df, y_train, Xva_df, y_val, Xte_df, y_test = _load_xy()

# -------------------------
# 1) Sanity checks e saneamento (evita NaN/Inf/colunas com std=0)
# -------------------------
# remover colunas com desvio 0 (podem causar NaN no scaler anterior)
zero_std_cols = Xtr_df.columns[
    (Xtr_df.std(axis=0, ddof=0) == 0) | (~np.isfinite(Xtr_df.std(axis=0, ddof=0)))
]
if len(zero_std_cols) > 0:
    print("⚠️ Removendo colunas com std=0:", list(zero_std_cols))
    Xtr_df = Xtr_df.drop(columns=zero_std_cols)
    Xva_df = Xva_df.drop(columns=zero_std_cols, errors="ignore")
    Xte_df = Xte_df.drop(columns=zero_std_cols, errors="ignore")


def sanitize(df, name):
    n_nan = np.isnan(df.values).sum()
    n_inf = np.isinf(df.values).sum()
    if n_nan or n_inf:
        print(f"⚠️ {name}: substituindo {n_nan} NaN e {n_inf} Inf por 0.0")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


Xtr_df = sanitize(Xtr_df, "X_train")
Xva_df = sanitize(Xva_df, "X_val")
Xte_df = sanitize(Xte_df, "X_test")

X_train = Xtr_df.values
X_val = Xva_df.values
X_test = Xte_df.values

assert (
    np.isfinite(X_train).all()
    and np.isfinite(X_val).all()
    and np.isfinite(X_test).all()
), "Ainda há valores não finitos nas features!"

# Padronizar o alvo (z-score) para estabilizar gradientes
y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train))
if y_std == 0 or not np.isfinite(y_std):
    y_std = 1.0
y_train_z = (y_train - y_mean) / y_std
y_val_z = (y_val - y_mean) / y_std
y_test_z = (y_test - y_mean) / y_std


# -------------------------
# 2) Métricas de regressão
# -------------------------
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


# -------------------------
# 3) Ativações e derivadas
# -------------------------
def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def dtanh(z):
    return 1 - np.tanh(z) ** 2


ACTS = {"relu": (relu, drelu), "tanh": (tanh, dtanh), "sigmoid": (sigmoid, dsigmoid)}


# -------------------------
# 4) Inicialização (He para ReLU; Xavier aproximado para demais)
#    compatível com default_rng (standard_normal)
# -------------------------
def init_weights(in_dim, out_dim, act_name, rng):
    if act_name == "relu":
        std = np.sqrt(2.0 / in_dim)
    else:
        std = np.sqrt(1.0 / in_dim)
    if hasattr(rng, "standard_normal"):
        W = rng.standard_normal((in_dim, out_dim)) * std
    else:
        W = rng.randn(in_dim, out_dim) * std
    b = np.zeros((1, out_dim))
    return W, b


# -------------------------
# 5) MLP from scratch (mini-batch SGD + L2 + dropout + early stopping)
# -------------------------
class MLPRegressorScratch:
    def __init__(
        self,
        input_dim,
        hidden_dims=(64, 32),
        activation="relu",
        lr=5e-4,
        batch_size=128,
        epochs=300,
        l2=1e-4,
        dropout=0.0,
        seed=42,
        early_stopping=True,
        patience=25,
        verbose=True,
    ):
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2 = l2
        self.dropout = dropout
        self.seed = seed
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose

        assert activation in ACTS, f"Ativação inválida: {activation}"
        self.act, self.dact = ACTS[activation]
        self.rng = np.random.default_rng(seed)
        self._init_params()

    def _init_params(self):
        dims = [self.input_dim] + self.hidden_dims + [1]  # saída escalar
        self.W, self.b = [], []
        # ocultas
        for i in range(len(dims) - 2):
            Wi, bi = init_weights(dims[i], dims[i + 1], self.activation, self.rng)
            self.W.append(Wi)
            self.b.append(bi)
        # saída linear (Xavier ok)
        Wi, bi = init_weights(dims[-2], dims[-1], "tanh", self.rng)
        self.W.append(Wi)
        self.b.append(bi)

    def _forward(self, X, train_mode=False):
        a = X
        caches = {"A0": a, "Z": [], "A": [], "drop_masks": []}
        # ocultas
        for L in range(len(self.hidden_dims)):
            z = a @ self.W[L] + self.b[L]
            a = self.act(z)
            mask = None
            if train_mode and self.dropout > 0:
                mask = (self.rng.random(a.shape) > self.dropout).astype(a.dtype)
                a = a * mask / (1.0 - self.dropout)  # inverted dropout
            caches["Z"].append(z)
            caches["A"].append(a)
            caches["drop_masks"].append(mask)
        # saída linear
        z_out = a @ self.W[-1] + self.b[-1]
        y_hat = z_out
        caches["Z_out"] = z_out
        caches["A_out"] = y_hat
        return y_hat, caches

    def _backward(self, y_hat, y_true, caches):
        N = y_true.shape[0]
        grad_W = [None] * len(self.W)
        grad_b = [None] * len(self.b)

        # saída
        dL_dy = 2.0 * (y_hat - y_true) / N
        a_prev = caches["A"][-1] if len(self.hidden_dims) > 0 else caches["A0"]
        grad_W[-1] = a_prev.T @ dL_dy + self.l2 * self.W[-1]
        grad_b[-1] = np.sum(dL_dy, axis=0, keepdims=True)

        da = dL_dy @ self.W[-1].T
        for L in reversed(range(len(self.hidden_dims))):
            zL = caches["Z"][L]
            a_prev = caches["A"][L - 1] if L > 0 else caches["A0"]
            mask = caches["drop_masks"][L]
            if mask is not None:
                da = da * mask / (1.0 - self.dropout)
            dz = da * self.dact(zL)
            grad_W[L] = a_prev.T @ dz + self.l2 * self.W[L]
            grad_b[L] = np.sum(dz, axis=0, keepdims=True)
            if L > 0:
                da = dz @ self.W[L].T
        return grad_W, grad_b

    def _update(self, grad_W, grad_b):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grad_W[i]
            self.b[i] -= self.lr * grad_b[i]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        rng = self.rng
        N = X_train.shape[0]
        bsz = self.batch_size
        self.history_ = {"train_rmse": [], "val_rmse": []}
        best_val = np.inf
        best_state = None
        patience_left = self.patience

        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            idx = rng.permutation(N)
            X_train, y_train = X_train[idx], y_train[idx]
            for s in range(0, N, bsz):
                e = min(s + bsz, N)
                xb, yb = X_train[s:e], y_train[s:e]
                y_hat, cache = self._forward(xb, train_mode=True)
                gW, gB = self._backward(y_hat, yb, cache)
                self._update(gW, gB)

            # métricas por época (no espaço z)
            tr_pred, _ = self._forward(X_train, train_mode=False)
            tr_rmse = rmse(y_train, tr_pred)
            self.history_["train_rmse"].append(tr_rmse)

            if X_val is not None:
                v_pred, _ = self._forward(X_val, train_mode=False)
                v_rmse = rmse(y_val, v_pred)
                self.history_["val_rmse"].append(v_rmse)
            else:
                v_rmse = np.nan

            if self.verbose and (epoch == 1 or epoch % 10 == 0):
                print(
                    f"[Epoch {epoch:4d}] train_RMSE={tr_rmse:.4f}  val_RMSE={v_rmse:.4f}"
                )

            if self.early_stopping and X_val is not None:
                if v_rmse + 1e-9 < best_val:
                    best_val = v_rmse
                    patience_left = self.patience
                    best_state = {
                        "W": [w.copy() for w in self.W],
                        "b": [b.copy() for b in self.b],
                        "epoch": epoch,
                    }
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if self.verbose:
                            print(
                                f"Early stopping em epoch {epoch} (melhor val_RMSE={best_val:.4f})"
                            )
                        if best_state is not None:
                            self.W = [w.copy() for w in best_state["W"]]
                            self.b = [b.copy() for b in best_state["b"]]
                        break

        self.train_time_ = time.time() - t0
        return self

    def predict(self, X):
        y_hat, _ = self._forward(X, train_mode=False)
        return y_hat


# -------------------------
# 6) Configuração, treino e avaliação
# -------------------------
INPUT_DIM = X_train.shape[1]
mlp = MLPRegressorScratch(
    input_dim=INPUT_DIM,
    hidden_dims=(64, 32),  # experimente (128, 64, 32)
    activation="relu",  # "relu" | "tanh" | "sigmoid"
    lr=5e-4,  # taxa de aprendizado (estável)
    batch_size=128,
    epochs=300,
    l2=1e-4,  # weight decay
    dropout=0.10,  # 10% na(s) oculta(s) — pode começar com 0.0
    seed=42,
    early_stopping=True,
    patience=25,
    verbose=True,
)

# treino no espaço z do alvo
mlp.fit(X_train, y_train_z, X_val, y_val_z)

# predições no espaço z
yhat_tr_z = mlp.predict(X_train)
yhat_va_z = mlp.predict(X_val)
yhat_te_z = mlp.predict(X_test)

# retornar à escala original do preço
yhat_tr = yhat_tr_z * y_std + y_mean
yhat_va = yhat_va_z * y_std + y_mean
yhat_te = yhat_te_z * y_std + y_mean


def eval_and_print(tag, y_true, y_pred):
    res = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    print(f"{tag}: RMSE={res['RMSE']:.4f}  MAE={res['MAE']:.4f}  R2={res['R2']:.4f}")
    return res


os.makedirs("model_outputs", exist_ok=True)

metrics = {
    "train": eval_and_print("TRAIN", y_train, yhat_tr),
    "val": eval_and_print("VAL  ", y_val, yhat_va),
    "test": eval_and_print("TEST ", y_test, yhat_te),
    "train_time_sec": mlp.train_time_,
}
with open("model_outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# curvas de RMSE (treino/val) — ainda no espaço z
plt.figure(figsize=(7, 4))
plt.plot(mlp.history_.get("train_rmse", []), label="Train RMSE (z)")
if len(mlp.history_.get("val_rmse", [])) > 0:
    plt.plot(mlp.history_["val_rmse"], label="Val RMSE (z)")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Curvas de RMSE — MLP (NumPy)")
plt.legend()
plt.tight_layout()
plt.savefig("model_outputs/loss_curves_rmse.png", dpi=150)
plt.close()

# scatter no teste (escala real)
plt.figure(figsize=(5, 5))
plt.scatter(y_test, yhat_te, s=6, alpha=0.4)
plt.xlabel("y_true (test)")
plt.ylabel("y_pred (test)")
plt.title("Dispersão — Test (y_true vs y_pred)")
plt.tight_layout()
plt.savefig("model_outputs/scatter_test_true_vs_pred.png", dpi=150)
plt.close()

print("✅ Treino do MLP finalizado. Resultados salvos em 'model_outputs/'.")
