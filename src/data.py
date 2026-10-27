"""Télécharger les fichiers traçables et préparer des partitions sans fuite."""

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 42

# Les URL sont figées sur un commit, pas sur une branche qui peut changer.
# Ce sont des copies publiques des jeux cités dans le sujet, pas des données simulées.
SOURCES = {
    "prices-split-adjusted.csv": {
        "kaggle": "https://www.kaggle.com/datasets/dgawlik/nyse",
        "url": "https://raw.githubusercontent.com/griffinschwab/NYSE-ML-Project/dfd0e9cf577d6fa2271957ab470c2c81c654bc6b/data/raw/prices-split-adjusted.csv",
        "sha256": "0a8e0505591c8ae8c545d6d9e7bb69735c18568745ef848bf4d01e1afd871ec5",
    },
    "predictive_maintenance.csv": {
        "kaggle": "https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification",
        "url": "https://raw.githubusercontent.com/beratcmn/machine-maintenance-prediction/9683ce9ccd58e29fb4ef39d116b16416606a2190/data/predictive_maintenance.csv",
        "sha256": "ba1eb264c69e69258acb2df6991055afc93092b1be436a29d2e6037b0ebd80cc",
    },
}

NUMERIC = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
REG_FEATURES = ["open", "close", "low", "high", "volume",
                "return_1", "mean_5", "std_5"]


def download_data():
    """Réutiliser les fichiers valides ; vérifier SHA-256 avant de les exploiter."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        path = RAW / name
        if not path.exists():
            temporary = path.with_suffix(".download")
            try:
                with urllib.request.urlopen(source["url"], timeout=120) as response:
                    temporary.write_bytes(response.read())
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                if digest != source["sha256"]:
                    raise ValueError(f"Empreinte incorrecte pour {name} : {digest}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"{path} diffère du fichier de référence. Ne pas l'écraser.")
    return SOURCES


def load_regression(symbol="AAPL"):
    """À la clôture du jour t, prédire la clôture de la prochaine séance.

    Toutes les entrées existent au jour t. La cible seule utilise shift(-1).
    Une seule action évite de mélanger les échelles de prix de sociétés différentes.
    """
    raw = pd.read_csv(RAW / "prices-split-adjusted.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    stock = raw.loc[raw.symbol.eq(symbol)].copy()
    if stock.empty:
        raise ValueError(f"Action absente : {symbol}")
    stock = stock.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    stock["return_1"] = stock.close.pct_change(fill_method=None)
    stock["mean_5"] = stock.close.rolling(5).mean()
    stock["std_5"] = stock.close.rolling(5).std()
    stock["next_close"] = stock.close.shift(-1)
    stock["target_date"] = stock.date.shift(-1)
    stock = stock.replace([np.inf, -np.inf], np.nan)
    stock = stock.dropna(subset=REG_FEATURES + ["next_close", "target_date"])
    n = len(stock)
    first, second = int(n * .70), int(n * .85)
    # Purger une ligne à chaque frontière : une cible du train ne doit pas
    # correspondre à une date utilisée comme entrée de validation ou de test.
    parts = {"train": stock.iloc[:first - 1].copy(),
             "validation": stock.iloc[first:second - 1].copy(),
             "test": stock.iloc[second:].copy()}
    assert parts["train"].target_date.max() < parts["validation"].date.min()
    assert parts["validation"].target_date.max() < parts["test"].date.min()
    return raw, stock, parts


def load_classification():
    """Prédire Failure Type, sans Target ni identifiants dans les entrées."""
    raw = pd.read_csv(RAW / "predictive_maintenance.csv", encoding="utf-8-sig")
    needed = NUMERIC + ["Type", "Failure Type", "Target", "UDI", "Product ID"]
    if not set(needed).issubset(raw):
        raise ValueError("Colonnes du jeu de maintenance non reconnues.")
    clean = raw.drop_duplicates().dropna(subset=["Failure Type"]).copy()
    # Ne pas supprimer une panne rare seulement parce qu'elle semble atypique.
    # Les valeurs numériques manquantes seront imputées dans chaque fold du train.
    clean[NUMERIC] = clean[NUMERIC].replace([np.inf, -np.inf], np.nan)
    train, other = train_test_split(clean, test_size=.30, random_state=SEED,
                                    stratify=clean["Failure Type"])
    valid, test = train_test_split(other, test_size=.50, random_state=SEED,
                                  stratify=other["Failure Type"])
    return raw, clean, {"train": train, "validation": valid, "test": test}


def write_json(path, value):
    """JSON lisible, sans NaN non standard."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    download_data()
    print("Deux fichiers téléchargés et empreintes SHA-256 vérifiées.")
