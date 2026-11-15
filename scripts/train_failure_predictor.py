
import torch
import torch.nn as nn
import torch.nn.functional as F
from farc.prediction import NUM_FEATURES, FeatureConfig, build_datasets
from torch.utils.data import DataLoader

from farc.paths import data_dir


class FailurePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def compute_loss(logits, labels, timesteps, weighted):
    losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )

    if not weighted:
        return losses.mean()

    weights = torch.where(
        (timesteps >= 5) & (timesteps <= 35),
        2.0,
        1.0,
    ).to(device=logits.device, dtype=logits.dtype)

    return (losses * weights).sum() / weights.sum()


def run_epoch(model, loader, optimizer=None, weighted=False):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_weight = 0.0

    for batch in loader:
        x = batch["features"]
        y = batch["label"]
        t = batch["t"]

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = compute_loss(logits, y, t, weighted)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if weighted:
            batch_weight = torch.where(
                (t >= 5) & (t <= 35), 2.0, 1.0
            ).sum().item()
        else:
            batch_weight = len(y)

        total_loss += loss.item() * batch_weight
        total_weight += batch_weight

    return total_loss / total_weight


def train_model(train_loader, val_loader, weighted, output_dir):
    torch.manual_seed(42)
    model = FailurePredictor(NUM_FEATURES)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    name = "weighted" if weighted else "unweighted"
    best_loss = float("inf")

    for epoch in range(20):
        train_loss = run_epoch(
            model, train_loader, optimizer, weighted
        )

        val_loss = run_epoch(
            model, val_loader, weighted=weighted
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                output_dir / f"{name}.pt",
            )

        print(
            f"{name} | epoch {epoch + 1:02d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f}"
        )


def main():
    ds = build_datasets(
        data_dir() / "experiments" / "phase1_extended",
        validation_fraction=0.2,
        split_seed=0,
        config=FeatureConfig(
            progress_window=5,
            reset_approach_on_goal_change=True,
        ),
    )

    train_loader = DataLoader(
        ds.train, batch_size=512, shuffle=True
    )
    val_loader = DataLoader(
        ds.validation, batch_size=512, shuffle=False
    )

    output_dir = data_dir() / "models" / "failure_predictor"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_model(train_loader, val_loader, False, output_dir)
    train_model(train_loader, val_loader, True, output_dir)


if __name__ == "__main__":
    main()
