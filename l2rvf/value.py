"""Value network V_theta(x; psi) and regression utilities (CPU torch)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ValueNet(nn.Module):
  """MLP on [features(x), psi] -> non-negative cost-to-go.

  Inputs are standardised with statistics stored as buffers, so the network
  can be saved/loaded as a single state dict.
  """

  def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3):
    super().__init__()
    layers, d = [], in_dim
    for _ in range(depth):
      layers += [nn.Linear(d, hidden), nn.SiLU()]
      d = hidden
    layers += [nn.Linear(d, 1)]
    self.net = nn.Sequential(*layers)
    self.register_buffer("mu", torch.zeros(in_dim))
    self.register_buffer("sd", torch.ones(in_dim))
    self.register_buffer("y_scale", torch.ones(()))

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    h = (z - self.mu) / self.sd
    return nn.functional.softplus(self.net(h)).squeeze(-1) * self.y_scale

  @torch.no_grad()
  def predict(self, z: np.ndarray, chunk: int = 262144) -> np.ndarray:
    """numpy in/out; z has shape (..., in_dim)."""
    shp = z.shape[:-1]
    flat = torch.as_tensor(z.reshape(-1, z.shape[-1]), dtype=torch.float32)
    out = [self(flat[i : i + chunk]) for i in range(0, len(flat), chunk)]
    return torch.cat(out).numpy().reshape(shp).astype(np.float64)


def fit_value(
    net: ValueNet,
    z: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_frac: float = 0.1,
    seed: int = 0,
    reinit_stats: bool = True,
    verbose: bool = False,
) -> dict:
  """Fits ``net`` by MSE on (z, y). Returns train/val metrics."""
  g = torch.Generator().manual_seed(seed)
  Z = torch.as_tensor(z, dtype=torch.float32)
  Y = torch.as_tensor(y, dtype=torch.float32)
  n = len(Z)
  perm = torch.randperm(n, generator=g)
  n_val = int(val_frac * n)
  vi, ti = perm[:n_val], perm[n_val:]
  if reinit_stats:
    net.mu.copy_(Z[ti].mean(0))
    net.sd.copy_(Z[ti].std(0).clamp_min(1e-3))
    net.y_scale.copy_(Y[ti].abs().mean().clamp_min(1e-3))
  opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
  scale = float(net.y_scale)
  for ep in range(epochs):
    net.train()
    order = ti[torch.randperm(len(ti), generator=g)]
    for i in range(0, len(order), batch_size):
      b = order[i : i + batch_size]
      loss = ((net(Z[b]) - Y[b]) / scale).pow(2).mean()
      opt.zero_grad()
      loss.backward()
      opt.step()
    sched.step()
    if verbose and (ep % 50 == 0 or ep == epochs - 1):
      print(f"  epoch {ep:4d} loss {loss.item():.4f}")
  net.eval()
  with torch.no_grad():
    def _err(idx):
      if len(idx) == 0:
        return float("nan"), float("nan")
      p = net(Z[idx])
      rmse = (p - Y[idx]).pow(2).mean().sqrt().item()
      rel = ((p - Y[idx]).abs().mean() / Y[idx].abs().mean().clamp_min(1e-6)).item()
      return rmse, rel
    tr, trr = _err(ti)
    va, var = _err(vi)
  return {"train_rmse": tr, "train_rel": trr, "val_rmse": va, "val_rel": var}


class ValueEnsemble:
  """Pessimistic ensemble: terminal cost = mean + kappa * std over members.

  The std term raises the cost of states the members disagree on (typically
  states outside the training distribution), so MPPI cannot exploit the
  errors of a single network there.
  """

  def __init__(self, nets, kappa: float = 1.0):
    self.nets = list(nets)
    self.kappa = kappa

  def predict_all(self, z: np.ndarray) -> np.ndarray:
    return np.stack([n.predict(z) for n in self.nets])

  def predict(self, z: np.ndarray) -> np.ndarray:
    p = self.predict_all(z)
    return p.mean(0) + self.kappa * p.std(0)

  def mean(self, z: np.ndarray) -> np.ndarray:
    return self.predict_all(z).mean(0)

  def state_dicts(self):
    return {"kappa": self.kappa, "members": [n.state_dict() for n in self.nets]}

  @classmethod
  def load(cls, path, in_dim, hidden=256, kappa=None):
    blob = torch.load(path, weights_only=True)
    nets = []
    for sd in blob["members"]:
      n = ValueNet(in_dim, hidden=hidden)
      n.load_state_dict(sd)
      n.eval()
      nets.append(n)
    return cls(nets, blob["kappa"] if kappa is None else kappa)
