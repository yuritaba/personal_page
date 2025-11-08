"""
Standalone script to train and evaluate a Variational Autoencoder (VAE) on MNIST
or Fashion-MNIST. The implementation relies only on PyTorch and torchvision and
is intended to be easy to run on CPU-only machines.

Example usage:
    python docs/roteiro4/vae_training.py \
        --dataset fashion \
        --latent-dim 2 \
        --epochs 30 \
        --batch-size 256 \
        --output-dir docs/roteiro4/assets

Running the script will create (or overwrite) the following artifacts inside
``output-dir``:
    - training_history.csv
    - reconstructions.png
    - samples.png
    - latent_space.png  (only for latent_dim <= 3)
    - metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, utils as tv_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple VAE on MNIST/Fashion-MNIST.")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=(512, 256))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE).")
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/roteiro4/assets"),
        help="Directory where artifacts will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


class VAE(nn.Module):
    """Fully-connected VAE tailored to 28x28 grayscale images."""

    def __init__(self, latent_dim: int, hidden_dims: Tuple[int, int]) -> None:
        super().__init__()
        input_dim = 28 * 28
        h1, h2 = hidden_dims

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),  # constrains reconstruction to [0, 1]
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.encoder(x)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        recon = self.decoder(z)
        return recon.view(-1, 1, 28, 28)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def reconstruction_loss(recon: Tensor, x: Tensor) -> Tensor:
    return F.binary_cross_entropy(recon, x, reduction="sum")


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def get_dataset(name: str, train: bool, transform: transforms.Compose) -> Dataset:
    root = Path("data")
    if name == "mnist":
        return datasets.MNIST(root=str(root), train=train, download=True, transform=transform)
    if name == "fashion":
        return datasets.FashionMNIST(root=str(root), train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def prepare_loaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = get_dataset(args.dataset, train=True, transform=transform)
    test_dataset = get_dataset(args.dataset, train=False, transform=transform)

    val_size = int(len(full_train) * args.validation_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    return train_loader, val_loader, test_loader


def train_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
) -> float:
    model.train()
    epoch_loss = 0.0

    for batch, _ in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        recon_loss = reconstruction_loss(recon, batch)
        kl = kl_divergence(mu, logvar)
        loss = (recon_loss + beta * kl) / batch.size(0)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)

    return epoch_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> float:
    model.eval()
    epoch_loss = 0.0

    for batch, _ in loader:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        recon_loss = reconstruction_loss(recon, batch)
        kl = kl_divergence(mu, logvar)
        loss = (recon_loss + beta * kl) / batch.size(0)
        epoch_loss += loss.item() * batch.size(0)

    return epoch_loss / len(loader.dataset)


@torch.no_grad()
def save_reconstructions(model: VAE, loader: DataLoader, device: torch.device, path: Path) -> None:
    model.eval()
    samples, _ = next(iter(loader))
    samples = samples.to(device)[:16]
    recon, _, _ = model(samples)
    comparison = torch.cat([samples.cpu(), recon.cpu()], dim=0)
    grid = tv_utils.make_grid(comparison, nrow=16, pad_value=1.0)
    tv_utils.save_image(grid, path)


@torch.no_grad()
def save_latent_samples(model: VAE, device: torch.device, latent_dim: int, path: Path) -> None:
    model.eval()
    z = torch.randn(64, latent_dim, device=device)
    generated = model.decode(z).cpu()
    grid = tv_utils.make_grid(generated, nrow=8, pad_value=1.0)
    tv_utils.save_image(grid, path)


@torch.no_grad()
def save_latent_space(model: VAE, loader: DataLoader, device: torch.device, path: Path) -> None:
    latent_dim = model.fc_mu.out_features
    if latent_dim > 3:
        print("Latent dimension greater than 3. Skipping latent space plot.")
        return

    import matplotlib.pyplot as plt  # Imported lazily to avoid hard dependency during static analysis

    model.eval()
    zs, labels = [], []
    for batch, y in loader:
        batch = batch.to(device)
        mu, _ = model.encode(batch)
        zs.append(mu.cpu())
        labels.append(y)

    latent = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()

    fig = plt.figure(figsize=(6, 6))
    if latent_dim == 2:
        scatter = plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
        plt.xlabel("z1")
        plt.ylabel("z2")
    else:
        ax = fig.add_subplot(111, projection="3d")  # type: ignore
        scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=labels, cmap="tab10", s=5, alpha=0.7)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")

    plt.colorbar(scatter, ticks=range(10))
    plt.title("Latent space (μ)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = prepare_loaders(args)
    model = VAE(latent_dim=args.latent_dim, hidden_dims=tuple(args.hidden_dims)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    best_state: Dict[str, Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, beta=args.beta)
        val_loss = evaluate(model, val_loader, device, beta=args.beta)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[Epoch {epoch:03d}] train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, device, beta=args.beta)
    print(f"Test loss: {test_loss:.4f}")

    # Save history
    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']:.6f},{row['val_loss']:.6f}\n")

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "latent_dim": args.latent_dim,
                "beta": args.beta,
                "epochs": args.epochs,
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
            },
            f,
            indent=2,
        )

    save_reconstructions(model, val_loader, device, args.output_dir / "reconstructions.png")
    save_latent_samples(model, device, args.latent_dim, args.output_dir / "samples.png")
    save_latent_space(model, val_loader, device, args.output_dir / "latent_space.png")
    print(f"Artifacts saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

