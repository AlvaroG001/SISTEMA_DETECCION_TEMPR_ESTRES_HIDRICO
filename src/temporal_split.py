import pandas as pd


def add_temporal_split(
    df: pd.DataFrame,
    date_col: str = "fecha",
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> pd.DataFrame:
    out = df.sort_values(date_col).copy()
    dates = pd.Series(out[date_col].dropna().sort_values().unique())
    if len(dates) < 3:
        raise ValueError("Need at least three dates to create train/val/test temporal split.")
    train_cut = dates.iloc[max(int(len(dates) * train_size) - 1, 0)]
    val_cut = dates.iloc[max(int(len(dates) * (train_size + val_size)) - 1, 1)]
    out["split"] = "test"
    out.loc[out[date_col] <= train_cut, "split"] = "train"
    out.loc[(out[date_col] > train_cut) & (out[date_col] <= val_cut), "split"] = "val"
    return out


def split_xy(df: pd.DataFrame, features: list[str], target: str):
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    return (
        train[features],
        train[target],
        val[features],
        val[target],
        test[features],
        test[target],
    )
