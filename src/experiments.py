"""Déroulement des deux parties, figures et résultats reproductibles.

Exécution : python -m src.experiments --task all
Les notebooks appellent ces mêmes fonctions pour éviter deux versions du code.
"""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import platform

# Les caches sont locaux au processus ; rien à écrire dans un HOME protégé.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atelier-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/atelier-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as BalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import (NUMERIC, REG_FEATURES, ROOT, SEED, SOURCES, download_data,
                   load_classification, load_regression, write_json)
from .models import TorchClassifier, TorchRegressor, seed_everything

OUT = ROOT / "results"
sns.set_theme(style="whitegrid")


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def eda(raw, data, task, output):
    """Les graphes détaillés portent sur le train, pas sur le futur test."""
    raw.dtypes.astype(str).rename("dtype").to_csv(output / "columns.csv")
    raw.isna().sum().rename("missing").to_csv(output / "missing_values.csv")
    data.describe(include="all").to_csv(output / "train_statistics.csv")
    summary = {"raw_rows": len(raw), "raw_columns": raw.shape[1],
               "exact_duplicates": int(raw.duplicated().sum()),
               "missing_values": int(raw.isna().sum().sum())}
    if task == "regression":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(data.date, data.close)
        axes[0].set(title="Apple : clôtures ajustées (train)", ylabel="Prix (USD)")
        axes[1].hist(data.return_1, bins=40)
        axes[1].set(title="Rendements journaliers (train)", xlabel="Rendement")
        save_figure(output / "eda.png")
        columns = REG_FEATURES + ["next_close"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        data["Failure Type"].value_counts().plot.bar(ax=axes[0])
        axes[0].set(title="Types de panne (train)", ylabel="Nombre")
        sns.boxplot(data=data, x="Failure Type", y="Torque [Nm]", ax=axes[1])
        axes[1].tick_params(axis="x", rotation=90)
        axes[1].set_title("Couple selon le type de panne (train)")
        save_figure(output / "eda.png")
        data[NUMERIC].hist(figsize=(11, 7), bins=30)
        save_figure(output / "distributions.png")
        columns = NUMERIC
        summary["class_counts_train"] = data["Failure Type"].value_counts().to_dict()
        # Quelques lignes Target/Failure Type peuvent être incohérentes dans
        # ce jeu synthétique. On les signale sans inventer de nouvelles étiquettes.
        summary["target_failure_type_disagreements"] = int(
            (raw.Target.ne(0) != raw["Failure Type"].ne("No Failure")).sum())
    plt.figure(figsize=(9, 7))
    sns.heatmap(data[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Corrélations sur les données d'entraînement")
    save_figure(output / "correlations.png")
    write_json(output / "eda_summary.json", summary)
    return summary


def make_pipeline(task):
    if task == "regression":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("net", TorchRegressor()),
        ])
    # SMOTENC distingue le type de produit des mesures continues : on ne crée
    # pas un type fictif « moitié L, moitié M » en interpolant des one-hot.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    category = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ordinal", OrdinalEncoder(categories=[["L", "M", "H"]]))])
    prepare = ColumnTransformer([("numeric", numeric, NUMERIC),
                                 ("category", category, ["Type"])])
    encode = ColumnTransformer([
        ("type", OneHotEncoder(categories=[[0., 1., 2.]],
                               sparse_output=False, handle_unknown="ignore"), [5]),
        ("numeric", "passthrough", list(range(5))),
    ])
    return BalancedPipeline([
        ("prepare", prepare),
        ("balance", SMOTENC(categorical_features=[5], k_neighbors=3,
                            random_state=SEED)),
        ("encode", encode),
        ("net", TorchClassifier()),
    ])


def parameter_grid(quick=False):
    # Recherche volontairement compacte, pas une promesse d'optimum global.
    epochs = [8, 16] if quick else [40, 80]
    return [
        {"net__lr": [.001, .01], "net__optimizer": ["adam"],
         "net__hidden": [(32,), (64, 32)], "net__epochs": epochs},
        {"net__lr": [.01], "net__optimizer": ["sgd"],
         "net__hidden": [(64, 32)], "net__epochs": [epochs[-1]]},
    ]


def metrics(task, truth, prediction):
    if task == "regression":
        return {"mae": float(mean_absolute_error(truth, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
                "r2": float(r2_score(truth, prediction))}
    return {"accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(truth, prediction, average="weighted", zero_division=0))}


def plot_history(history, task, output, name):
    frame = pd.DataFrame(history)
    score = "rmse" if task == "regression" else "accuracy"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for partition in ["train", "validation", "test"]:
        for ax, metric in zip(axes, ["loss", score]):
            ax.plot(frame.epoch, frame[f"{partition}_{metric}"], label=partition)
            ax.set_xlabel("Époque")
            ax.legend()
    axes[0].set_title("MSE (USD²)" if task == "regression" else "Entropie croisée")
    axes[1].set_title("RMSE (USD)" if task == "regression" else "Accuracy")
    if task == "regression":
        # L'échelle log laisse visibles les erreurs après convergence.
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        axes[0].set_title("MSE (USD², échelle logarithmique)")
        axes[1].set_title("RMSE (USD, échelle logarithmique)")
    fig.suptitle(name + " — test évalué après sélection, sans effet sur l'apprentissage")
    save_figure(output / f"curves_{name}.png")
    frame.to_csv(output / f"history_{name}.csv", index=False)


def run_experiment(task, quick=False, output_root=OUT):
    """Entraîner, choisir sur validation, puis ouvrir le test une fois tout figé."""
    seed_everything()
    download_data()
    output = Path(output_root) / task
    output.mkdir(parents=True, exist_ok=True)
    if task == "regression":
        raw, clean, parts = load_regression()
        features, target = REG_FEATURES, "next_close"
        # gap=1 purge également les cibles à la frontière de chaque fold CV.
        cv = TimeSeriesSplit(n_splits=3, gap=1)
        scoring = "neg_root_mean_squared_error"
    else:
        raw, clean, parts = load_classification()
        features, target = NUMERIC + ["Type"], "Failure Type"
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scoring = "f1_macro"
    eda(raw, parts["train"], task, output)
    X = {name: frame[features] for name, frame in parts.items()}
    y = {name: frame[target].to_numpy() for name, frame in parts.items()}
    pipe = make_pipeline(task)
    # Le pipeline est refait dans CHAQUE fold : imputation, normalisation et
    # SMOTENC n'accèdent donc jamais aux observations de validation du fold.
    search = GridSearchCV(pipe, parameter_grid(quick), cv=cv, scoring=scoring,
                          n_jobs=1, refit=False, error_score="raise", verbose=1)
    print(f"\n{task} : recherche des hyperparamètres", flush=True)
    search.fit(X["train"], y["train"])
    pd.DataFrame(search.cv_results_).to_csv(output / "grid_search.csv", index=False)
    write_json(output / "best_parameters.json", search.best_params_)
    selected = make_pipeline(task).set_params(**search.best_params_)
    # Préparer les tableaux finaux sans refaire les transformations sur le test.
    if task == "regression":
        pre = selected[:-1]
        fit_X = pre.fit_transform(X["train"], y["train"])
        fit_y = y["train"]
        transformed = {key: pre.transform(value) for key, value in X.items()}
    else:
        pre = selected.named_steps["prepare"]
        prepared = pre.fit_transform(X["train"], y["train"])
        fit_X, fit_y = selected.named_steps["balance"].fit_resample(prepared, y["train"])
        encoder = selected.named_steps["encode"]
        fit_X = encoder.fit_transform(fit_X)
        transformed = {key: encoder.transform(pre.transform(value)) for key, value in X.items()}
        counts = pd.DataFrame({"avant": pd.Series(y["train"]).value_counts(),
                               "apres": pd.Series(fit_y).value_counts()})
        counts.to_csv(output / "class_balance.csv")
        counts.plot.bar(figsize=(10, 4), title="SMOTENC : entraînement seulement")
        save_figure(output / "class_balance.png")
    base_params = selected.named_steps["net"].get_params()
    # Même architecture, même graine et même budget pour isoler les effets.
    variants = {"initial": {}, "dropout": {"dropout": .2},
                "l2": {"weight_decay": .001},
                "early_stopping": {"patience": 10},
                "combined": {"dropout": .2, "weight_decay": .001, "patience": 10}}
    estimator = TorchRegressor if task == "regression" else TorchClassifier
    models, rows = {}, []
    for name, changes in variants.items():
        print(f"{task} : modèle {name}", flush=True)
        model = estimator(**(base_params | changes))
        model.fit(fit_X, fit_y,
                  validation_data=(transformed["validation"], y["validation"]),
                  curve_sets={"train": (transformed["train"], y["train"])},
                  record_states=True)
        models[name] = model
        row = {"model": name, "epochs_run": model.epochs_run_,
               "selected_epoch": model.selected_epoch_}
        for partition in ["train", "validation"]:
            row.update({f"{partition}_{k}": v for k, v in
                        metrics(task, y[partition], model.predict(transformed[partition])).items()})
        rows.append(row)
    # Le modèle retenu est fixé AVANT de calculer la moindre métrique de test.
    selection_metric = "validation_rmse" if task == "regression" else "validation_f1_macro"
    winner = (min if task == "regression" else max)(rows, key=lambda r: r[selection_metric])["model"]
    for row in rows:
        name, model = row["model"], models[row["model"]]
        prediction = model.predict(transformed["test"])
        row.update({f"test_{k}": v for k, v in metrics(task, y["test"], prediction).items()})
        # Évaluation rétrospective des poids sauvegardés : les courbes de test
        # demandées par le sujet ne pilotent ni l'arrêt ni le choix du modèle.
        final_state = deepcopy(model.model_.state_dict())
        for history, state in zip(model.history_, model.epoch_states_):
            model.model_.load_state_dict(state)
            history.update({f"test_{k}": v for k, v in model._measure(transformed["test"], y["test"]).items()})
        model.model_.load_state_dict(final_state)
        plot_history(model.history_, task, output, name)
        if task == "classification":
            for partition in ["train", "test"]:
                pred = model.predict(transformed[partition])
                report = classification_report(y[partition], pred, output_dict=True, zero_division=0)
                pd.DataFrame(report).T.to_csv(output / f"report_{name}_{partition}.csv")
        model.epoch_states_.clear()
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    if task == "regression":
        baseline = {key: metrics(task, y[key], parts[key]["close"].to_numpy()) for key in parts}
        prediction = models[winner].predict(transformed["test"])
        pd.DataFrame({"date": parts["test"].target_date.to_numpy(), "real": y["test"],
                      "mlp": prediction, "persistence": parts["test"].close.to_numpy()}).to_csv(
                          output / "predictions.csv", index=False)
        plt.figure(figsize=(12, 4))
        plt.plot(parts["test"].target_date, y["test"], label="Réel")
        plt.plot(parts["test"].target_date, prediction, label="MLP retenu")
        plt.plot(parts["test"].target_date, parts["test"].close, label="Persistance", alpha=.6)
        plt.ylabel("Clôture ajustée (USD)")
        plt.legend()
        save_figure(output / "predictions.png")
    else:
        dummy = DummyClassifier(strategy="most_frequent").fit(X["train"], y["train"])
        baseline = {key: metrics(task, y[key], dummy.predict(X[key])) for key in parts}
        labels = models[winner].classes_
        prediction = models[winner].predict(transformed["test"])
        matrix = confusion_matrix(y["test"], prediction, labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / "confusion_matrix.csv")
        plt.figure(figsize=(9, 7))
        sns.heatmap(matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Prédiction")
        plt.ylabel("Vérité")
        plt.title("Matrice de confusion — modèle choisi sur validation")
        save_figure(output / "confusion_matrix.png")
    summary = {"task": task, "mode": "quick" if quick else "full",
               "seed": SEED, "rows_raw": len(raw), "rows_prepared": len(clean),
               "split_sizes": {key: len(value) for key, value in parts.items()},
               "best_parameters_cv": search.best_params_,
               "best_score_cv": float(search.best_score_), "cv_scoring": scoring,
               "cv_folds": 3, "grid_candidates": len(search.cv_results_["params"]),
               "selected_on_validation": winner, "reference": baseline,
               "sources": SOURCES, "python": platform.python_version(),
               "torch": torch.__version__, "sklearn": sklearn.__version__}
    write_json(output / "summary.json", summary)
    print(comparison.to_string(index=False), flush=True)
    print(f"Choix sur validation : {winner}. Résultats : {output}", flush=True)
    return comparison, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atelier 1 : expériences PyTorch")
    parser.add_argument("--task", choices=["regression", "classification", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="8/16 époques, contrôle rapide")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    for task in (["regression", "classification"] if args.task == "all" else [args.task]):
        run_experiment(task, args.quick, args.output)
