import copy

import casadi as ca
import l4casadi as l4c
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nlotrajectories.core.sdf.casadi import IObstacle
from nlotrajectories.core.metrics import surface_loss


def sample_points(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_samples: int,
    random: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if random:
        xs = np.random.uniform(*x_range, size=n_samples)
        ys = np.random.uniform(*y_range, size=n_samples)
    else:
        side = int(np.sqrt(n_samples))
        xs = np.linspace(*x_range, side)
        ys = np.linspace(*y_range, side)
        xs, ys = np.meshgrid(xs, ys)
        xs, ys = xs.ravel(), ys.ravel()
    return xs, ys


class NNObstacleTrainer:
    def __init__(
        self,
        obstacle: IObstacle,
        model: nn.Module,
        device: None | str = None,
        epochs: int = 100,
        eikonal_weight: float = 0,
        n_samples: int = 20000,
        random: bool = True,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        self.obstacle = obstacle
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)

        self.epochs = epochs
        self.eikonal_weight = eikonal_weight
        self.n_samples = n_samples
        self.random = random
        self.batch_size = batch_size
        self.lr = lr

    def generate_data(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        n_samples: int,
        random: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = sample_points(x_range, y_range, n_samples, random)
        sdf_vals = np.array([self.obstacle.sdf(x, y) for x, y in zip(xs, ys)])

        inputs = torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.float32)
        targets = torch.tensor(sdf_vals, dtype=torch.float32).unsqueeze(1)
        return inputs, targets

    def train(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        early_stop: bool = True,
        patience: int = 10,
        min_delta: float = 1e-4,
        surface_loss_weight: float = 0,
        surface_loss_eps: float = 1e-2,
    ):
        X, Y = self.generate_data(x_range, y_range, self.n_samples, self.random)

        indices = torch.randperm(len(X))
        X, Y = X[indices], Y[indices]
        n_val = int(0.1 * len(X))
        X_val, Y_val = X[:n_val], Y[:n_val]
        X_train, Y_train = X[n_val:], Y[n_val:]

        train_ds = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=self.batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.model.train()

        best_val_loss = float("inf")
        best_model_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        for ep in range(self.epochs):
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)

                # Compute the MSE loss
                mse_loss = loss_fn(pred, yb)
                # Compute the surface loss
                if surface_loss_weight > 0:
                    surface_loss_value = surface_loss(
                        yb.detach().cpu().numpy(),
                        pred.detach().cpu().numpy(),
                        xb[:, 0].detach().cpu().numpy(),
                        xb[:, 1].detach().cpu().numpy(),
                        eps=surface_loss_eps,
                    )
                    surface_loss_value = torch.tensor(surface_loss_value, dtype=torch.float32, device=self.device)
                    #if surface_loss_value is NaN because not common surface points, set it to 0.0 (no surface loss)
                    if torch.isnan(surface_loss_value):
                        surface_loss_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    loss = mse_loss + surface_loss_weight * surface_loss_value
                else:
                    loss = mse_loss

                # Eikonal loss
                if self.eikonal_weight > 0:
                    grad = torch.autograd.grad(
                        outputs=pred,
                        inputs=xb,
                        grad_outputs=torch.ones_like(pred),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grad_norm = torch.linalg.norm(grad, dim=1)
                    eikonal_loss = ((grad_norm - 1.0) ** 2).mean()
                    loss += self.eikonal_weight * eikonal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            with torch.no_grad():
                self.model.eval()
                val_loss = 0.0
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    if surface_loss_weight > 0:
                        surface_loss_value = surface_loss(
                            yb.detach().cpu().numpy(),
                            pred.detach().cpu().numpy(),
                            xb[:, 0].detach().cpu().numpy(),
                            xb[:, 1].detach().cpu().numpy(),
                            eps=surface_loss_eps,
                        )
                        surface_loss_value = torch.tensor(surface_loss_value, dtype=torch.float32, device=self.device)
                        #if surface_loss_value is NaN because not common surface points, set it to 0.0 (no surface loss)
                        if torch.isnan(surface_loss_value):
                            surface_loss_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        loss = mse_loss + surface_loss_weight * surface_loss_value
                    else:
                        loss = mse_loss
                    val_loss += loss.item() * xb.size(0)
                val_loss /= len(val_loader.dataset)
                self.model.train()

            if ep % 10 == 0 or ep == 0:
                print(f"Epoch {ep:3d} - Train loss: {total_loss/len(train_ds):.6f} - Val loss: {val_loss:.6f}")

            if early_stop:
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {ep:3d} (no improvement for {patience} epochs).")
                        break

        self.model.load_state_dict(best_model_state)
        self.model.eval()


class NNObstacle(IObstacle):
    def __init__(self, obstacle: IObstacle, model: l4c.L4CasADi):
        self.obstacle = obstacle
        self.model = model.to("cpu")

    def sdf(self, x: ca.MX, y: ca.MX) -> float:
        return self.obstacle.sdf(x, y)

    def approximated_sdf(self, x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # NumPy case
            coords = np.stack([x.ravel(), y.ravel()], axis=1)  # (N, 2)
            result = self.model(torch.tensor(coords, dtype=torch.float32)).detach().cpu().numpy()
            return result.reshape(x.shape)
        elif isinstance(x, ca.MX) and isinstance(y, ca.MX):
            # CasADi case
            rows = x.size1()
            cols = x.size2()
            x_flat = ca.reshape(x, rows * cols, 1)
            y_flat = ca.reshape(y, rows * cols, 1)
            coords = ca.hcat([x_flat, y_flat])  # (N, 2)
            result = self.model(coords)  # Must be CasADi-compatible NN (e.g., via l4casadi)
            return ca.reshape(result, rows, cols)
        else:
            raise TypeError("Inputs must be both NumPy arrays or both CasADi MX types.")

    def draw(self, ax, **kwargs):
        self.obstacle.draw(ax, **kwargs)
