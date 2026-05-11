import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_STATE, TARGET
from src.data_loading import get_feature_columns, load_modeling_dataset
from src.evaluation import save_model_outputs, save_not_run_metrics


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_train_val_test():
    df = load_modeling_dataset()
    features = get_feature_columns(df)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    return df, features, train, val, test


def train_random_forest() -> dict:
    _df, features, train, val, test = get_train_val_test()
    model_name = "random_forest"
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(train[features], train[TARGET])
    joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")
    return save_model_outputs(
        model_name,
        train,
        val,
        test,
        model.predict(val[features]),
        model.predict(test[features]),
        features,
    )


def train_xgboost() -> dict:
    model_name = "xgboost"
    if not dependency_available("xgboost"):
        return save_not_run_metrics(
            model_name,
            "not_run_dependency_missing",
            "Install xgboost with: pip install xgboost",
        )
    from xgboost import XGBRegressor

    _df, features, train, val, test = get_train_val_test()
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=350,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train[features], train[TARGET])
    joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")
    return save_model_outputs(
        model_name,
        train,
        val,
        test,
        model.predict(val[features]),
        model.predict(test[features]),
        features,
    )


def build_sequences(df: pd.DataFrame, features: list[str], sequence_length: int = 8):
    rows = []
    seqs = []
    ys = []
    for _, group in df.sort_values(["nombre_parcela", "fecha"]).groupby("nombre_parcela", sort=False):
        values = group[features].to_numpy(dtype=np.float32)
        target = group[TARGET].to_numpy(dtype=np.float32)
        for i in range(sequence_length - 1, len(group)):
            seqs.append(values[i - sequence_length + 1 : i + 1])
            ys.append(target[i])
            rows.append(group.iloc[i])
    if not seqs:
        return np.empty((0, sequence_length, len(features))), np.empty(0), pd.DataFrame(rows)
    return np.stack(seqs), np.asarray(ys), pd.DataFrame(rows)


def train_torch_sequence_model(model_name: str, architecture: str) -> dict:
    if not dependency_available("torch"):
        return save_not_run_metrics(
            model_name,
            "not_run_dependency_missing",
            "Install PyTorch with: pip install torch",
        )

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    df, features, _train, _val, _test = get_train_val_test()
    train_df = df[df["split"] == "train"].copy()
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy="median")
    train_df[features] = scaler.fit_transform(imputer.fit_transform(train_df[features]))
    df[features] = scaler.transform(imputer.transform(df[features]))

    x, y, seq_rows = build_sequences(df, features, sequence_length=8)
    if len(seq_rows) < 10:
        return save_not_run_metrics(model_name, "not_run_insufficient_data", "Not enough sequences.")
    train_mask = seq_rows["split"].eq("train").to_numpy()
    val_mask = seq_rows["split"].eq("val").to_numpy()
    test_mask = seq_rows["split"].eq("test").to_numpy()

    class SequenceRegressor(nn.Module):
        def __init__(self, arch: str, n_features: int):
            super().__init__()
            self.arch = arch
            hidden = 48
            if arch == "lstm":
                self.recurrent = nn.LSTM(n_features, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)
            elif arch == "gru":
                self.recurrent = nn.GRU(n_features, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)
            elif arch == "tcn":
                self.net = nn.Sequential(
                    nn.Conv1d(n_features, 48, kernel_size=3, padding=2, dilation=1),
                    nn.ReLU(),
                    nn.Conv1d(48, 48, kernel_size=3, padding=4, dilation=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Linear(48, 1)
            elif arch == "cnn_lstm":
                self.conv = nn.Sequential(nn.Conv1d(n_features, 48, kernel_size=3, padding=1), nn.ReLU())
                self.recurrent = nn.LSTM(48, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)
            else:
                raise ValueError(f"Unknown architecture: {arch}")

        def forward(self, batch):
            if self.arch in {"lstm", "gru"}:
                out, _ = self.recurrent(batch)
                return self.head(out[:, -1, :]).squeeze(-1)
            if self.arch == "cnn_lstm":
                conv = self.conv(batch.transpose(1, 2)).transpose(1, 2)
                out, _ = self.recurrent(conv)
                return self.head(out[:, -1, :]).squeeze(-1)
            out = self.net(batch.transpose(1, 2)).squeeze(-1)
            return self.head(out).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceRegressor(architecture, x.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.tensor(x[train_mask]), torch.tensor(y[train_mask]))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    for _epoch in range(18):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    def predict(mask):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(x[mask]).to(device)).cpu().numpy()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": features,
            "architecture": architecture,
            "sequence_length": 8,
        },
        MODELS_DIR / f"{model_name}.pt",
    )
    joblib.dump({"imputer": imputer, "scaler": scaler}, MODELS_DIR / f"{model_name}_preprocess.joblib")
    return save_model_outputs(
        model_name,
        seq_rows[train_mask],
        seq_rows[val_mask],
        seq_rows[test_mask],
        predict(val_mask),
        predict(test_mask),
        features,
        notes="PyTorch sequence model using 8 previous observations per parcel.",
    )


def train_convlstm_placeholder() -> dict:
    return save_not_run_metrics(
        "convlstm",
        "not_applicable_with_current_dataset",
        "ConvLSTM needs spatial tensors or image sequences. The current CSV is tabular by parcel/date.",
    )


def train_chronos_placeholder() -> dict:
    if not dependency_available("chronos"):
        return save_not_run_metrics(
            "chronos_bolt",
            "not_run_dependency_missing",
            "Install a Chronos-compatible package, for example: pip install chronos-forecasting",
        )
    return save_not_run_metrics(
        "chronos_bolt",
        "not_implemented_optional",
        "Chronos dependency was detected, but this MVP keeps the model as an optional extension.",
    )


def train_tft_placeholder() -> dict:
    if not dependency_available("pytorch_forecasting"):
        return save_not_run_metrics(
            "tft",
            "not_run_dependency_missing",
            "Install optional dependencies with: pip install pytorch-forecasting torch",
        )
    return save_not_run_metrics(
        "tft",
        "not_implemented_optional",
        "pytorch-forecasting was detected, but this MVP keeps TFT as an optional extension.",
    )
