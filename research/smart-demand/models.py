"""共享商品预测模型；所有标准化只使用当前训练窗口。"""
from __future__ import annotations

import hashlib
import math
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _arrays(train_x, train_y, pred_x):
    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.float32)
    px = np.asarray(pred_x, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("Training features and targets must be aligned nonempty matrices.")
    if not x.shape[1] or not y.shape[1] or px.ndim != 2 or px.shape[1] != x.shape[1]:
        raise ValueError("Prediction and training feature dimensions must match.")
    if not np.isfinite(x).all() or not np.isfinite(px).all():
        raise ValueError("Features must be finite; encode missing inputs before fitting.")
    if np.isinf(y).any() or (y[np.isfinite(y)] < 0).any():
        raise ValueError("Targets must be nonnegative observations or NaN for unknown values.")
    keep = np.isfinite(y).any(axis=1)
    if not keep.any():
        raise ValueError("At least one observed target is required.")
    return x[keep], y[keep], px, keep


def _standardize(x, px):
    # Float64 accumulation avoids unstable scales when many items share a feature.
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.).astype(np.float32)
    return (np.clip((x - mean) / scale, -8., 8.).astype(np.float32),
            np.clip((px - mean) / scale, -8., 8.).astype(np.float32), mean, scale)


def _bases(values, length, label):
    base = np.asarray(values, dtype=np.float32)
    if base.shape != (length,) or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError(f"{label} must have one strictly positive finite scale per row.")
    return base


class SharedDemandMLP(nn.Module):
    def __init__(self, features, horizon, hidden=(64, 32)):
        super().__init__()
        widths = [int(features), *[int(width) for width in hidden]]
        if len(widths) != 3 or any(width <= 0 for width in widths):
            raise ValueError("Use two positive hidden widths.")
        self.body = nn.Sequential(nn.Linear(widths[0], widths[1]), nn.ReLU(),
                                  nn.Linear(widths[1], widths[2]), nn.ReLU())
        self.head = nn.Linear(widths[2], int(horizon))
        # Start every comparison at the same historical level rather than a random scale.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, math.log(math.expm1(1.)))

    def forward(self, x, base):
        return F.softplus(self.head(self.body(x))) * base[:, None]


def masked_objective(prediction, target, kind, scale):
    """Unknown future cells contribute neither a zero answer nor a loss term."""
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        raise ValueError("A batch must contain an observed target.")
    predicted, observed = prediction[mask], target[mask]
    if kind == "mae":
        values = torch.abs(predicted - observed)
    elif kind == "poisson":
        values = predicted - observed * torch.log(predicted + 1e-6)
    else:
        raise ValueError("loss must be 'mae' or 'poisson'.")
    return values.mean() / scale


def fit_predict(train_x, train_y, train_base, pred_x, pred_base, *, seed,
                loss="mae", hidden=(64, 32), epochs=25, batch_size=512,
                lr=.002, weight_decay=.001):
    """Fit a fixed-epoch MLP and predict the full horizon without test early stopping."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    if loss not in ("mae", "poisson") or int(epochs) != epochs or epochs < 0:
        raise ValueError("Select a supported loss and a nonnegative integer epoch count.")
    if int(batch_size) != batch_size or batch_size < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Batch size, learning rate, and weight decay are invalid.")
    x, y, px, keep = _arrays(train_x, train_y, pred_x)
    base = _bases(train_base, len(keep), "train_base")[keep]
    pbase = _bases(pred_base, len(px), "pred_base")
    x, px, mean, std = _standardize(x, px)
    tx, ty, tb = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(base)
    scale = max(float(np.mean(y[np.isfinite(y)], dtype=np.float64)), 1.)
    model = SharedDemandMLP(x.shape[1], y.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=float(weight_decay))
    generator = torch.Generator().manual_seed(int(seed))
    log = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        order = torch.randperm(len(tx), generator=generator)
        for indices in order.split(int(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(tx[indices], tb[indices])
            objective = masked_objective(predicted, ty[indices], loss, scale)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Training objective became nonfinite.")
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if epoch % 5 == 0 or epoch == int(epochs):
            with torch.no_grad():
                objective = masked_objective(model(tx, tb), ty, loss, scale)
            log.append({"epoch": epoch, "objective": float(objective)})
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(px), torch.from_numpy(pbase)).numpy()
        final_objective = float(masked_objective(model(tx, tb), ty, loss, scale))
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise FloatingPointError("Nonfinite or negative model prediction.")
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().numpy().tobytes())
    trace = {
        "seed": int(seed), "loss": loss, "epochs": int(epochs),
        "batch_size": int(batch_size), "learning_rate": float(lr),
        "weight_decay": float(weight_decay), "gradient_clip": 5.,
        "hidden": list(hidden), "parameters": sum(p.numel() for p in model.parameters()),
        "training_rows": len(x), "dropped_unobserved_rows": int((~keep).sum()),
        "observed_target_cells": int(np.isfinite(y).sum()), "prediction_rows": len(px),
        "features": x.shape[1], "horizon": y.shape[1], "loss_scale": scale,
        "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "normalized_feature_cap": 8., "initial_prediction": "train/pred_base",
        "final_training_objective": final_objective, "training_log": log,
        "weights_sha256": digest.hexdigest(), "torch": torch.__version__,
        "numpy": np.__version__, "python": platform.python_version(),
        "device": "cpu", "threads": 1,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return predicted, trace


def ridge_predict(train_x, train_y, pred_x, alpha=100.):
    """A linear control uses exactly the same inputs and observed target cells."""
    from sklearn.linear_model import Ridge

    started = time.perf_counter()
    if not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("alpha must be positive and finite.")
    x, y, px, keep = _arrays(train_x, train_y, pred_x)
    x, px, mean, std = _standardize(x, px)
    prediction = np.empty((len(px), y.shape[1]), dtype=np.float32)
    rows = []
    for column in range(y.shape[1]):
        mask = np.isfinite(y[:, column])
        if not mask.any():
            raise ValueError(f"Horizon {column} has no observed training target.")
        model = Ridge(alpha=float(alpha), solver="cholesky")
        model.fit(x[mask], y[mask, column])
        if len(px):
            prediction[:, column] = np.maximum(model.predict(px), 0.)
        rows.append(int(mask.sum()))
    trace = {"alpha": float(alpha), "training_rows": len(x),
             "dropped_unobserved_rows": int((~keep).sum()),
             "observed_rows_by_horizon": rows, "features": x.shape[1],
             "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
             "normalized_feature_cap": 8.,
             "elapsed_seconds": time.perf_counter() - started}
    return prediction, trace
