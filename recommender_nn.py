import zipfile

import hydra
import numpy as np
import os 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from humancompatible.train.dual_optim import ALM, MoreauEnvelope
from torch.nn import MSELoss
import logging
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge

from data_helper import load_all_data


DATA_PATH = './...'
W_PATH = DATA_PATH + '...'

# ------------------------------------------------------------
# Simple MLP Regressor
# ------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        for i in range(len(hidden_dim)-1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def predict(self, X):
        return self.net(torch.tensor(X, dtype=torch.float32)).squeeze(-1)


# ------------------------------------------------------------
# Augmented Lagrangian Training
# ------------------------------------------------------------

def fit_aug_lagrangian_nn_constraint(
    X, y, W, cfg, verbose=True, device="cpu", 
):
    torch.set_default_dtype(torch.float32)

    # Convert to tensors
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
    W = torch.tensor(np.asarray(W), dtype=torch.float32, device=device)

    n, d = X.shape
    print(f"Data shape: {X.shape}")
    if W.shape != (d + 1, d + 1):
        raise ValueError(f"W must be (d+1)x(d+1) = {(d+1)}x{(d+1)}; got {W.shape}")

    # Build model and optimizers
    model = MLPRegressor(
        input_dim=d,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = MoreauEnvelope(
        optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    )
    dual_opt = ALM(
        m=d+1,
        lr=cfg.lambda_update_rate,
        penalty=cfg.rho0,
        init_duals=cfg.lambda0
    )

    # Precompute constraint components
    M = W - torch.eye(d + 1, device=device)
    muX = X.mean(dim=0)
    # g0 = M[:, :-1] @ muX
    # v = M[:, -1]
    
    zbar_const = torch.concatenate([muX, torch.tensor([0.0])])           # [muX; 0]
    g0 = M @ zbar_const                               # constant part when mean y = 0
    v = M[:, -1].clone()

    if torch.allclose(v, torch.zeros_like(v)):
        raise ValueError("Constraint does not depend on predictions.")
    logging.info(f"Sanity check: GT constraint = {W_constraint(v, g0, y)}")

    loss = MSELoss()

    for outer in range(cfg.n_outer):
        for _ in range(cfg.n_inner):
            optimizer.zero_grad()
            yhat = model(X)
            mse = loss(yhat, y)

            g = W_constraint(v, g0, yhat)
            if cfg.constrained:
                aug_loss = dual_opt.forward(loss=mse, constraints=g)
                aug_loss.backward()
            else:
                mse.backward()
            optimizer.step()

        if cfg.constrained:
            with torch.no_grad():
                dual_opt.update(g)
            lam = dual_opt.duals.detach().numpy()

        for param_group in optimizer.param_groups:
            param_group['lr'] *= cfg.lr_decay

        lam = dual_opt.duals.detach().numpy()
        g = g.detach().numpy()
        with torch.no_grad():
            if verbose:
                print(
                    f"outer={outer:02d}  "
                    f"mean(yhat)={yhat.mean().item():+.6e}  "
                    f"MSE={mse.item():+.6e}  "
                    f"||g||={np.max(g).item():.6e}  "
                    f"lambda_norm={np.linalg.norm(lam).item():.6e}"
                )

    return model, lam

def W_constraint(v, g0, y):
    if y.ndim < 2:
        y = y.unsqueeze(1)
    g = g0 + y.mean(axis=0) * v
    return g



def fit_lr(X, y, W, cfg, verbose=True):
    model = Ridge()
    model.fit(X, y)
    return model



@torch.no_grad
def compute_predictor_errors_and_cs_scikit(model, X, y, W, _y_mean=None, _y_std=None, scaler=None):
    y_pred = np.array(model.predict(X))
    test_mse = mean_squared_error(y, y_pred)
    # if _y_mean is not None:
        # renormalize Y after denormalizing it in estimator.predict
        # y_pred_norm = (y_pred - _y_mean) / _y_std
    # X = scaler.transform(X)
    M = W - np.eye(X.shape[1] + 1)
    muX = X.mean(axis=0)
    zbar_const = np.concatenate([muX, [0.0]])
    g0 = M @ zbar_const
    v = M[:, -1].copy()
    test_c = np.linalg.norm(g0 + y_pred.mean(axis=0) * v)
    
    return test_mse, test_c


@hydra.main(version_base=None,  config_path="./experiments_conf", config_name="config")
def main(cfg: DictConfig):

    n_folds = 5

    torch.manual_seed(42)
    if cfg.problem.name == "codiet":
        food_feats, non_food_feats, X_full = load_all_data()
        with zipfile.ZipFile("./data/W_est.csv.zip") as z:
            with z.open("W_est.csv") as f:
                w_est = np.loadtxt(f, delimiter=",")
                print(w_est.shape)
    elif cfg.problem.name == "cds":
        import cds_utils
        X_full = cds_utils.load_data(cfg.problem.n, cfg.problem.granularity, cfg.problem.p, cfg.problem.data_path)
    elif cfg.problem.name == 'Sachs':
            import sachs_utils
            X_full = sachs_utils.load_data(cfg.problem.variant, cfg.problem.normalize, cfg.problem.data_path)

    if cfg.problem.name in ["cds", "Sachs"]:
            with zipfile.ZipFile(os.path.join(cfg.problem.data_path, "W_est.csv.zip")) as z:
                with z.open(f"W_est_{cfg.problem.name}.csv") as f:
                    w_est = np.loadtxt(f, delimiter=",")
            row_and_col_names = X_full.columns

    X_full.dropna(axis=1, inplace=True)
    target_col = cfg.problem.target
    y = X_full[target_col]
    X_full.drop(target_col, axis=1, inplace=True)
    X_full = X_full.to_numpy()
    y = y.to_numpy()

    scaler = StandardScaler()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    mse_train = []
    c_train = []

    mse_val = []
    c_val = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        X_train, y_train = X_full[train_idx], y[train_idx]
        X_val, y_val = X_full[val_idx], y[val_idx]

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean)/y_std
        y_val = (y_val - y_mean)/y_std

        if cfg.solver.name == 'hc_predictor':
            model, lam = fit_aug_lagrangian_nn_constraint(X_train, y_train, w_est, cfg.solver)
        else:
            model = fit_lr(X_train, y_train, w_est, cfg.solver)

        stats = compute_predictor_errors_and_cs_scikit(model, X_train, y_train, w_est)
        mse_train.append(stats[0])
        c_train.append(stats[1])

        stats = compute_predictor_errors_and_cs_scikit(model, X_val, y_val, w_est)
        mse_val.append(stats[0])
        c_val.append(stats[1])

    # stats = pd.DataFrame(stats)
    # print(stats)
    print('############### TRAIN ################')
    print('MSE:')
    print(np.mean(mse_train))
    print(mse_train)
    print("C")
    print(np.mean(c_train))
    print(c_train)

    print('################ VAL ##################')
    print('MSE:')
    print(np.mean(mse_val))
    print(mse_val)
    print("C")
    print(np.mean(c_val))
    print(c_val)

if __name__ == "__main__":
    main()
