

# ─── Imports ──────────────────────────────────────────────────────────────────
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – no display required
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



class PrunableLinear(nn.Module):
   

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same shapes and init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores: same shape as weight; registered as a trainable parameter.
        # Initialised to 0 so sigmoid(0)=0.5, giving balanced starting gates.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming-uniform init for weights (mirrors nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 – squash gate_scores into (0, 1)
        gates = torch.sigmoid(self.gate_scores)           # shape (out, in)

        # Step 2 – element-wise mask: zero gate => pruned connection
        pruned_weights = self.weight * gates

        # Step 3 – standard affine transform  y = x @ W^T + b
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return current gate values, detached to CPU (for analysis)."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of gate values for this layer.
        L1 = sum |gate| = sum gate   (all gates >= 0 after sigmoid)
        """
        return torch.sigmoid(self.gate_scores).sum()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}")




class SelfPruningNet(nn.Module):
    

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                       # flatten
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def prunable_layers(self) -> list:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def network_sparsity_loss(self) -> torch.Tensor:
        """Aggregate L1 sparsity loss over all PrunableLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    @torch.no_grad()
    def compute_sparsity(self, threshold: float = 0.05) -> float:
        """Percentage of gates whose value is below `threshold`."""
        all_gates = torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        )
        return 100.0 * (all_gates < threshold).float().mean().item()

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        return torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        ).numpy()




def get_cifar10_loaders(batch_size: int = 128, data_root: str = "./data"):
    """Download CIFAR-10 and return (train_loader, test_loader)."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_root, train=True,  download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader




def train_one_epoch(model, loader, optimizer, lam, device):
   
    model.train()
    total_loss = cls_loss_sum = sp_loss_sum = 0.0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        log_probs = model(images)

        cls_loss = F.nll_loss(log_probs, labels)
        sp_loss  = model.network_sparsity_loss()
        loss     = cls_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        cls_loss_sum += cls_loss.item()
        sp_loss_sum  += sp_loss.item()
        correct      += (log_probs.argmax(1) == labels).sum().item()

    n = len(loader)
    return (total_loss/n, cls_loss_sum/n, sp_loss_sum/n,
            100.0 * correct / len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Compute test accuracy (%)."""
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return 100.0 * correct / len(loader.dataset)


def run_experiment(lam, train_loader, test_loader,
                   epochs=30, lr=1e-3, device=None, prune_threshold=0.05):
    """
    Full training run for a single lambda value.
    Returns a results dict: lam, test_acc, sparsity, gates, history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  lambda={lam:.0e}   device={device}   epochs={epochs}")
    print(f"{'='*65}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"total_loss": [], "cls_loss": [], "train_acc": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tot_l, cls_l, sp_l, tr_acc = train_one_epoch(
            model, train_loader, optimizer, lam, device
        )
        scheduler.step()

        history["total_loss"].append(tot_l)
        history["cls_loss"].append(cls_l)
        history["train_acc"].append(tr_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}/{epochs}  loss={tot_l:.4f} "
                  f"(cls={cls_l:.4f} sp={lam*sp_l:.4f})  acc={tr_acc:.2f}%")

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.compute_sparsity(threshold=prune_threshold)
    gates    = model.all_gate_values()

    print(f"\n  Test accuracy  : {test_acc:.2f}%")
    print(f"  Sparsity level : {sparsity:.2f}%  (threshold={prune_threshold})")
    print(f"  Time elapsed   : {time.time()-t0:.1f}s")

    return {"lam": lam, "model": model, "test_acc": test_acc,
            "sparsity": sparsity, "gates": gates, "history": history}




def plot_gate_distribution(results_list, save_path="gate_distribution.png",
                           prune_threshold=0.05):
    """
    Histogram of final gate values for each lambda.
    A successful result shows a large spike at 0 (pruned) and a smaller
    cluster of non-zero gates (important surviving connections).
    """
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    palette = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12"]

    for ax, res, color in zip(axes, results_list, palette):
        ax.hist(res["gates"], bins=100, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(prune_threshold, color="black", ls="--", lw=1.5,
                   label=f"Prune threshold ({prune_threshold})")
        ax.set_title(
            f"lambda = {res['lam']:.0e}\n"
            f"Test acc: {res['test_acc']:.2f}%   Sparsity: {res['sparsity']:.2f}%",
            fontsize=11)
        ax.set_xlabel("Gate value  sigmoid(gate_score)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle("Distribution of Final Gate Values — Self-Pruning Network",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {save_path}")


def plot_training_curves(results_list, save_path="training_curves.png"):
    """Plot total training loss and accuracy for all lambda values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    palette = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12"]

    for res, color in zip(results_list, palette):
        lbl = f"lambda={res['lam']:.0e}"
        ax1.plot(res["history"]["total_loss"], label=lbl, color=color, lw=2)
        ax2.plot(res["history"]["train_acc"],  label=lbl, color=color, lw=2)

    ax1.set_title("Total Training Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_title("Training Accuracy (%)", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {save_path}")







def parse_args():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lambdas",    type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--threshold",  type=float, default=0.05)
    parser.add_argument("--data_root",  type=str,   default="./data")
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | Lambdas: {args.lambdas}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.data_root)

    results_list = []
    for lam in args.lambdas:
        res = run_experiment(lam, train_loader, test_loader,
                             args.epochs, args.lr, device, args.threshold)
        results_list.append(res)

    plot_gate_distribution(results_list, "gate_distribution.png", args.threshold)
    plot_training_curves(results_list, "training_curves.png")
    generate_report(results_list, "report.md")

    print("\n" + "="*55)
    print(f"  {'Lambda':>10}  {'Test Acc':>12}  {'Sparsity':>12}")
    print("  " + "-"*40)
    for r in results_list:
        print(f"  {r['lam']:>10.0e}  {r['test_acc']:>11.2f}%  {r['sparsity']:>11.2f}%")
    print("="*55)


if __name__ == "__main__":
    main()
