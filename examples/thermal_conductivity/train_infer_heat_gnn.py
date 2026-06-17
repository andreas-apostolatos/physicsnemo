"""Train and run a graph neural network for transient heat transfer.

The script expects three CSV files in this directory:

* heat_nodes.csv: node_id,x,y
* heat_edges.csv: src,dst
* heat_transient_samples.csv:
  observation_id,time,node_id,initial_temperature,dirichlet_boundary,
  neumann_boundary,target_temperature

Examples
--------
CPU smoke test:
    python train_infer_heat_gnn.py train --epochs 1 --limit-observations 8

GPU training:
    python train_infer_heat_gnn.py train --device cuda --epochs 20000

Inference:
    python train_infer_heat_gnn.py infer --checkpoint outputs/heat_gnn.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = (
    "x",
    "y",
    "time",
    "initial_temperature",
    "dirichlet_boundary",
    "neumann_boundary",
)
CONDITION_COLUMNS = (
    "initial_temperature",
    "dirichlet_boundary",
    "neumann_boundary",
)


@dataclass
class NormalizationStats:
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: float
    target_std: float


@dataclass
class ModelConfig:
    num_features: int = len(FEATURE_COLUMNS)
    latent_channels: int = 64
    graph_layers: int = 2


class HeatGraphDataset(Dataset):
    """One graph sample is one observation at one saved time."""

    def __init__(
        self,
        coordinates: torch.Tensor,
        temperatures: torch.Tensor,
        conditions: torch.Tensor,
        times: torch.Tensor,
        graph_indices: list[tuple[int, int]],
        stats: NormalizationStats,
    ) -> None:
        self.coordinates = coordinates
        self.temperatures = temperatures
        self.conditions = conditions
        self.times = times
        self.graph_indices = graph_indices
        self.feature_mean = torch.tensor(stats.feature_mean, dtype=torch.float32)
        self.feature_std = torch.tensor(stats.feature_std, dtype=torch.float32)
        self.target_mean = torch.tensor(stats.target_mean, dtype=torch.float32)
        self.target_std = torch.tensor(stats.target_std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.graph_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_idx, time_idx = self.graph_indices[index]
        num_nodes = self.coordinates.shape[0]
        time_column = self.times[time_idx].expand(num_nodes, 1)
        condition_columns = self.conditions[obs_idx].expand(num_nodes, -1)
        features = torch.cat((self.coordinates, time_column, condition_columns), dim=1)
        features = (features - self.feature_mean) / self.feature_std

        target = self.temperatures[obs_idx, time_idx].unsqueeze(-1)
        target = (target - self.target_mean) / self.target_std

        meta = torch.tensor([obs_idx, time_idx], dtype=torch.long)
        return features, target, meta


class NormalizedGraphConvolution(nn.Module):
    """Linear projection followed by symmetric normalized adjacency aggregation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        values = self.linear(node_features)
        source, destination = edge_index
        messages = values[:, source, :] * edge_weight.view(1, -1, 1)
        aggregated = values.new_zeros(values.shape)
        aggregated.index_add_(1, destination, messages)
        return torch.tanh(aggregated)


class HeatGNN(nn.Module):
    """MATLAB-style encoder, graph-convolution stack, and decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.num_features, config.latent_channels)
        self.graph_convolutions = nn.ModuleList(
            NormalizedGraphConvolution(config.latent_channels)
            for _ in range(config.graph_layers)
        )
        self.decoder = nn.Linear(config.latent_channels, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.tanh(self.encoder(node_features))
        for graph_convolution in self.graph_convolutions:
            values = graph_convolution(values, edge_index, edge_weight)
        return self.decoder(values)


def read_graph(
    data_dir: Path,
    add_self_loops: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    nodes = pd.read_csv(data_dir / "heat_nodes.csv").sort_values("node_id")
    expected_ids = list(range(len(nodes)))
    actual_ids = nodes["node_id"].astype(int).tolist()
    if actual_ids != expected_ids:
        raise ValueError("heat_nodes.csv must contain zero-based contiguous node_id values")

    edges = pd.read_csv(data_dir / "heat_edges.csv")
    edge_pairs = set(zip(edges["src"].astype(int), edges["dst"].astype(int)))
    if add_self_loops:
        edge_pairs.update((node_id, node_id) for node_id in expected_ids)

    edge_index = torch.tensor(sorted(edge_pairs), dtype=torch.long).t().contiguous()
    num_nodes = len(nodes)
    degree = torch.bincount(edge_index[1], minlength=num_nodes).float()
    source, destination = edge_index
    edge_weight = 1.0 / torch.sqrt(degree[source] * degree[destination])
    coordinates = torch.tensor(nodes[["x", "y"]].to_numpy(), dtype=torch.float32)
    return coordinates, edge_index, edge_weight.float(), nodes


def load_transient_data(
    data_dir: Path,
    coordinates: torch.Tensor,
    limit_observations: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[float]]:
    samples = pd.read_csv(data_dir / "heat_transient_samples.csv")
    required_columns = {
        "observation_id",
        "time",
        "node_id",
        *CONDITION_COLUMNS,
        "target_temperature",
    }
    missing_columns = required_columns.difference(samples.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in heat_transient_samples.csv: {missing_columns}")

    samples["observation_id"] = samples["observation_id"].astype(int)
    samples["node_id"] = samples["node_id"].astype(int)

    observation_ids = sorted(samples["observation_id"].unique().tolist())
    if limit_observations is not None:
        observation_ids = observation_ids[:limit_observations]
        samples = samples[samples["observation_id"].isin(observation_ids)]

    times = sorted(samples["time"].unique().tolist())
    num_observations = len(observation_ids)
    num_times = len(times)
    num_nodes = coordinates.shape[0]
    expected_rows = num_observations * num_times * num_nodes
    if len(samples) != expected_rows:
        raise ValueError(
            "Unexpected sample count. Expected "
            f"{expected_rows}, found {len(samples)}. Check observation/time/node coverage."
        )

    obs_to_idx = {obs_id: idx for idx, obs_id in enumerate(observation_ids)}
    time_to_idx = {time: idx for idx, time in enumerate(times)}
    samples["obs_idx"] = samples["observation_id"].map(obs_to_idx)
    samples["time_idx"] = samples["time"].map(time_to_idx)
    samples = samples.sort_values(["obs_idx", "time_idx", "node_id"])

    temperatures = torch.tensor(
        samples["target_temperature"].to_numpy().reshape(
            num_observations, num_times, num_nodes
        ),
        dtype=torch.float32,
    )

    conditions_df = (
        samples.sort_values(["obs_idx", "time_idx", "node_id"])
        .drop_duplicates("obs_idx")
        .sort_values("obs_idx")
    )
    conditions = torch.tensor(
        conditions_df[list(CONDITION_COLUMNS)].to_numpy(),
        dtype=torch.float32,
    )
    times_tensor = torch.tensor(times, dtype=torch.float32)
    return temperatures, conditions, times_tensor, observation_ids, times


def make_stats(
    coordinates: torch.Tensor,
    conditions: torch.Tensor,
    times: torch.Tensor,
    temperatures: torch.Tensor,
    train_observation_indices: list[int],
) -> NormalizationStats:
    num_nodes = coordinates.shape[0]
    feature_chunks = []
    target_chunks = []
    for obs_idx in train_observation_indices:
        for time in times:
            time_column = time.expand(num_nodes, 1)
            condition_columns = conditions[obs_idx].expand(num_nodes, -1)
            feature_chunks.append(torch.cat((coordinates, time_column, condition_columns), dim=1))
        target_chunks.append(temperatures[obs_idx])

    features = torch.cat(feature_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0).reshape(-1)
    feature_std = features.std(dim=0).clamp_min(1.0e-8)
    target_std = targets.std().clamp_min(1.0e-8)
    return NormalizationStats(
        feature_mean=features.mean(dim=0).tolist(),
        feature_std=feature_std.tolist(),
        target_mean=float(targets.mean()),
        target_std=float(target_std),
    )


def split_observations(
    num_observations: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_observations, generator=generator).tolist()
    num_val = max(1, int(round(num_observations * val_fraction)))
    val_indices = sorted(permutation[:num_val])
    train_indices = sorted(permutation[num_val:])
    if not train_indices:
        raise ValueError("Training split is empty. Reduce --val-fraction.")
    return train_indices, val_indices


def make_graph_indices(observation_indices: list[int], num_times: int) -> list[tuple[int, int]]:
    return [(obs_idx, time_idx) for obs_idx in observation_indices for time_idx in range(num_times)]


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def train_one_epoch(
    model: HeatGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    loss_fn = nn.MSELoss()
    for features, target, _meta in loader:
        features = features.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, edge_index, edge_weight)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()
        batch_size = features.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_graphs += batch_size
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(
    model: HeatGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    stats: NormalizationStats,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_values = 0
    target_mean = torch.tensor(stats.target_mean, dtype=torch.float32, device=device)
    target_std = torch.tensor(stats.target_std, dtype=torch.float32, device=device)
    for features, target, _meta in loader:
        features = features.to(device)
        target = target.to(device)
        prediction = model(features, edge_index, edge_weight)
        prediction = prediction * target_std + target_mean
        target = target * target_std + target_mean
        diff = prediction - target
        total_mse += float((diff * diff).sum().cpu())
        total_mae += float(diff.abs().sum().cpu())
        total_values += diff.numel()
    return total_mse / max(total_values, 1), total_mae / max(total_values, 1)


def save_checkpoint(
    path: Path,
    model: HeatGNN,
    stats: NormalizationStats,
    model_config: ModelConfig,
    train_observation_ids: list[int],
    val_observation_ids: list[int],
    times: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "normalization": asdict(stats),
            "model_config": asdict(model_config),
            "train_observation_ids": train_observation_ids,
            "val_observation_ids": val_observation_ids,
            "times": times,
            "feature_columns": FEATURE_COLUMNS,
        },
        path,
    )


def write_loss_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_mse_standardized",
                "val_mse",
                "val_mae",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_loss_plot(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 520
    margin_left = 80
    margin_right = 30
    margin_top = 30
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    epochs = [float(row["epoch"]) for row in rows]
    train_values = [float(row["train_mse_standardized"]) for row in rows]
    val_values = [float(row["val_mse"]) for row in rows]
    all_values = [value for value in train_values + val_values if value > 0.0]
    if not all_values:
        return

    min_epoch = min(epochs)
    max_epoch = max(epochs)
    min_log = math.floor(math.log10(min(all_values)))
    max_log = math.ceil(math.log10(max(all_values)))
    if min_log == max_log:
        max_log += 1

    def x_position(epoch: float) -> float:
        if max_epoch == min_epoch:
            return margin_left + plot_width / 2.0
        return margin_left + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_width

    def y_position(value: float) -> float:
        log_value = math.log10(max(value, 10.0**min_log))
        fraction = (log_value - min_log) / (max_log - min_log)
        return margin_top + (1.0 - fraction) * plot_height

    def polyline(values: list[float]) -> str:
        return " ".join(
            f"{x_position(epoch):.2f},{y_position(value):.2f}"
            for epoch, value in zip(epochs, values)
        )

    y_ticks = []
    for power in range(min_log, max_log + 1):
        y = y_position(10.0**power)
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" '
            f'x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" />'
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" '
            f'text-anchor="end" font-size="12" fill="#374151">1e{power}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2:.0f}" y="22" text-anchor="middle" font-size="18" font-family="Arial" fill="#111827">Heat GNN Loss History</text>
  {''.join(y_ticks)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" />
  <polyline points="{polyline(train_values)}" fill="none" stroke="#2563eb" stroke-width="2.5" />
  <polyline points="{polyline(val_values)}" fill="none" stroke="#dc2626" stroke-width="2.5" />
  <text x="{width / 2:.0f}" y="{height - 24}" text-anchor="middle" font-size="14" font-family="Arial" fill="#111827">Epoch</text>
  <text x="18" y="{height / 2:.0f}" transform="rotate(-90 18 {height / 2:.0f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#111827">Loss, log scale</text>
  <line x1="{width - 220}" y1="48" x2="{width - 180}" y2="48" stroke="#2563eb" stroke-width="2.5" />
  <text x="{width - 170}" y="52" font-size="13" font-family="Arial" fill="#111827">train standardized MSE</text>
  <line x1="{width - 220}" y1="70" x2="{width - 180}" y2="70" stroke="#dc2626" stroke-width="2.5" />
  <text x="{width - 170}" y="74" font-size="13" font-family="Arial" fill="#111827">validation MSE</text>
  <text x="{margin_left}" y="{height - 48}" font-size="12" font-family="Arial" fill="#374151">{int(min_epoch)}</text>
  <text x="{width - margin_right}" y="{height - 48}" text-anchor="end" font-size="12" font-family="Arial" fill="#374151">{int(max_epoch)}</text>
</svg>
"""
    path.write_text(svg)


def command_train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    coordinates, edge_index, edge_weight, _nodes = read_graph(data_dir, args.add_self_loops)
    temperatures, conditions, times, observation_ids, time_values = load_transient_data(
        data_dir, coordinates, args.limit_observations
    )

    train_indices, val_indices = split_observations(
        len(observation_ids), args.val_fraction, args.seed
    )
    stats = make_stats(coordinates, conditions, times, temperatures, train_indices)

    train_dataset = HeatGraphDataset(
        coordinates,
        temperatures,
        conditions,
        times,
        make_graph_indices(train_indices, len(time_values)),
        stats,
    )
    val_dataset = HeatGraphDataset(
        coordinates,
        temperatures,
        conditions,
        times,
        make_graph_indices(val_indices, len(time_values)),
        stats,
    )
    if args.training_mode == "full-batch":
        train_batch_size = len(train_dataset)
        val_batch_size = len(val_dataset)
        shuffle_train = False
    else:
        train_batch_size = args.batch_size
        val_batch_size = args.batch_size
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    model_config = ModelConfig(
        latent_channels=args.latent_channels,
        graph_layers=args.graph_layers,
    )
    model = HeatGNN(model_config).to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(
        json.dumps(
            {
                "device": str(device),
                "num_nodes": int(coordinates.shape[0]),
                "num_edges": int(edge_index.shape[1]),
                "num_observations": len(observation_ids),
                "num_times": len(time_values),
                "train_graphs": len(train_dataset),
                "val_graphs": len(val_dataset),
                "training_mode": args.training_mode,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
            },
            indent=2,
        )
    )

    best_val_mse = math.inf
    loss_rows: list[dict[str, float]] = []
    loss_history_path = Path(args.loss_history)
    loss_plot_path = Path(args.loss_plot)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, edge_index, edge_weight, optimizer, device
        )
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_mse, val_mae = evaluate(
                model, val_loader, edge_index, edge_weight, stats, device
            )
            print(
                f"epoch={epoch:05d} train_mse_standardized={train_loss:.6e} "
                f"val_mse={val_mse:.6e} val_mae={val_mae:.6e}"
            )
            loss_rows.append(
                {
                    "epoch": epoch,
                    "train_mse_standardized": train_loss,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                }
            )
            write_loss_history(loss_history_path, loss_rows)
            write_loss_plot(loss_plot_path, loss_rows)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                save_checkpoint(
                    Path(args.checkpoint),
                    model,
                    stats,
                    model_config,
                    [observation_ids[i] for i in train_indices],
                    [observation_ids[i] for i in val_indices],
                    time_values,
                )

    print(f"Saved best checkpoint to {args.checkpoint}")
    print(f"Wrote loss history to {loss_history_path}")
    print(f"Wrote loss plot to {loss_plot_path}")


@torch.no_grad()
def command_infer(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = NormalizationStats(**checkpoint["normalization"])
    model_config = ModelConfig(**checkpoint["model_config"])

    coordinates, edge_index, edge_weight, nodes = read_graph(data_dir, args.add_self_loops)
    temperatures, conditions, times, observation_ids, time_values = load_transient_data(
        data_dir, coordinates, args.limit_observations
    )

    requested_observations = (
        observation_ids
        if args.observation_ids is None
        else [int(value) for value in args.observation_ids.split(",")]
    )
    obs_to_idx = {obs_id: idx for idx, obs_id in enumerate(observation_ids)}
    selected_indices = [obs_to_idx[obs_id] for obs_id in requested_observations]
    dataset = HeatGraphDataset(
        coordinates,
        temperatures,
        conditions,
        times,
        make_graph_indices(selected_indices, len(time_values)),
        stats,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = HeatGNN(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    target_mean = torch.tensor(stats.target_mean, dtype=torch.float32, device=device)
    target_std = torch.tensor(stats.target_std, dtype=torch.float32, device=device)

    rows = []
    node_ids = nodes["node_id"].astype(int).tolist()
    x_values = nodes["x"].tolist()
    y_values = nodes["y"].tolist()
    for features, target, meta in loader:
        features = features.to(device)
        prediction = model(features, edge_index, edge_weight)
        prediction = (prediction * target_std + target_mean).cpu().squeeze(-1)
        target = (target * target_std.cpu() + target_mean.cpu()).squeeze(-1)
        for batch_idx in range(prediction.shape[0]):
            obs_idx, time_idx = meta[batch_idx].tolist()
            observation_id = observation_ids[obs_idx]
            time_value = time_values[time_idx]
            for local_node_idx, node_id in enumerate(node_ids):
                rows.append(
                    {
                        "observation_id": observation_id,
                        "time": time_value,
                        "node_id": node_id,
                        "x": x_values[local_node_idx],
                        "y": y_values[local_node_idx],
                        "prediction": float(prediction[batch_idx, local_node_idx]),
                        "target_temperature": float(target[batch_idx, local_node_idx]),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the transient heat GNN")
    train.add_argument("--data-dir", default=str(DATA_DIR))
    train.add_argument("--checkpoint", default=str(DATA_DIR / "outputs" / "heat_gnn.pt"))
    train.add_argument(
        "--loss-history",
        default=str(DATA_DIR / "outputs" / "loss_history.csv"),
    )
    train.add_argument(
        "--loss-plot",
        default=str(DATA_DIR / "outputs" / "loss_history.svg"),
    )
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    train.add_argument("--epochs", type=int, default=20000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument(
        "--training-mode",
        choices=("minibatch", "full-batch"),
        default="minibatch",
        help="Use minibatch training or MATLAB-style full-batch training.",
    )
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--latent-channels", type=int, default=64)
    train.add_argument("--graph-layers", type=int, default=2)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--limit-observations", type=int, default=None)
    train.add_argument("--add-self-loops", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(func=command_train)

    infer = subparsers.add_parser("infer", help="Run inference with a saved checkpoint")
    infer.add_argument("--data-dir", default=str(DATA_DIR))
    infer.add_argument("--checkpoint", default=str(DATA_DIR / "outputs" / "heat_gnn.pt"))
    infer.add_argument("--output", default=str(DATA_DIR / "outputs" / "predictions.csv"))
    infer.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    infer.add_argument("--batch-size", type=int, default=16)
    infer.add_argument("--limit-observations", type=int, default=None)
    infer.add_argument("--observation-ids", default=None, help="Comma-separated observation ids")
    infer.add_argument("--add-self-loops", action=argparse.BooleanOptionalAction, default=True)
    infer.set_defaults(func=command_infer)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
