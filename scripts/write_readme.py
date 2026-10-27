"""Actualiser le bref rapport à partir des mesures, sans recopier de scores."""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(task):
    folder = ROOT / "results" / task
    summary = json.loads((folder / "summary.json").read_text())
    rows = pd.read_csv(folder / "comparison.csv")
    chosen = rows.set_index("model").loc[summary["selected_on_validation"]]
    return summary, rows, chosen


def table(rows, columns, labels):
    text = "| " + " | ".join(labels) + " |\n"
    text += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    for _, row in rows.iterrows():
        values = [f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]) for col in columns]
        text += "| " + " | ".join(values) + " |\n"
    return text


if __name__ == "__main__":
    reg, reg_rows, r = load("regression")
    cls, cls_rows, c = load("classification")
    if reg["mode"] != "full" or cls["mode"] != "full":
        raise ValueError("Le rapport final nécessite deux expériences en mode complet.")
    report = pd.read_csv(ROOT / "results" / "classification" /
                         f"report_{cls['selected_on_validation']}_test.csv", index_col=0)
    tool = report.loc["Tool Wear Failure"]
    random_failure = report.loc["Random Failures"]
    rp, cp = reg["best_parameters_cv"], cls["best_parameters_cv"]
    majority = cls["reference"]["test"]
    persistence = reg["reference"]["test"]
    text = f"""# Atelier 1 — Deep Learning avec PyTorch

Ce projet réalise les deux parties de l'atelier : **prédire un prix boursier**
avec un réseau de régression et **reconnaître un type de panne** avec un réseau
de classification multiclasse. Le code et les notebooks contiennent des commentaires
en français pour expliquer chaque étape.

## 1. Par où commencer ?

1. Lire le [notebook de régression](notebooks/01_regression.ipynb).
2. Lire le [notebook de classification](notebooks/02_classification.ipynb).
3. Consulter les résultats ci-dessous, puis refaire les expériences si nécessaire.

Les notebooks ont des sorties enregistrées : ils se lisent directement sur GitHub.
Un [rapport PDF détaillé](reports/Rapport_Atelier1_DeepLearning.pdf) explique aussi
le projet de bout en bout, avec les graphiques, les métriques et leur interprétation.
Les fonctions de `src/` contiennent l'implémentation partagée ; les notebooks en
expliquent l'utilisation sans maintenir une deuxième version du code.

## 2. Installer et exécuter

Environnement vérifié : **Python {reg['python']}, PyTorch {reg['torch']}, CPU**.
Python 3.12 est recommandé pour reproduire les versions de `requirements.txt`.
Un GPU et une clé Kaggle ne sont pas nécessaires.

```bash
git clone https://github.com/KarmasHammam/Atelier1_DeepLearning.git
cd Atelier1_DeepLearning
bash scripts/setup.sh
source .venv/bin/activate

# Refaire la recherche et les cinq modèles de chaque partie.
python -m src.experiments --task all

# Vérifier les frontières temporelles, les partitions et les pipelines réels.
python -m unittest discover -s tests -v

# Actualiser les sorties pédagogiques et le rapport après un nouvel entraînement.
python scripts/execute_notebooks.py
python scripts/write_readme.py
```

Pour un contrôle plus court :

```bash
python -m src.experiments --task all --quick --output results_quick
```

Ce mode limite l'apprentissage à 8/16 époques ; ses scores ne remplacent pas les résultats complets ci-dessous.
Les fichiers bruts sont téléchargés dans `data/raw/`, qui est exclu de Git.

Pour ouvrir les notebooks dans VS Code, sélectionner `.venv/bin/python` comme noyau.
Dans **Google Colab**, exécuter d'abord dans une cellule :

```python
!git clone https://github.com/KarmasHammam/Atelier1_DeepLearning.git
%cd Atelier1_DeepLearning
%pip install -r requirements.txt
```

Puis exécuter les cellules du notebook depuis ce dossier. Si Colab remplace des
paquets déjà chargés, redémarrer sa session avant les imports. L'exécution Colab
n'a pas été vérifiée ici ; les expériences ont été exécutées dans l'environnement CPU.

## 3. D'où viennent les données ?

Le sujet cite [NYSE, par dgawlik](https://www.kaggle.com/datasets/dgawlik/nyse)
et [Machine Predictive Maintenance, par shivamb](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).

Kaggle étant inaccessible depuis cet environnement, le téléchargement utilise des
copies publiques : [NYSE](https://github.com/griffinschwab/NYSE-ML-Project/tree/dfd0e9cf577d6fa2271957ab470c2c81c654bc6b/data/raw)
et [maintenance](https://github.com/beratcmn/machine-maintenance-prediction/tree/9683ce9ccd58e29fb4ef39d116b16416606a2190/data).
Les URL sont figées sur des commits et les fichiers sont vérifiés par **SHA-256**.
Ces empreintes garantissent la stabilité des copies utilisées ; leur identité octet
par octet avec les archives Kaggle n'a pas été vérifiée. Sources et empreintes
figurent dans [src/data.py](src/data.py) et dans les `summary.json` des résultats.
Les données restent soumises aux conditions de leurs sources ; elles ne sont pas
redistribuées dans ce dépôt.

## 4. Partie 1 : régression boursière

### La question et la préparation

À la clôture du jour **t**, prévoir le cours de clôture ajusté d'**Apple (AAPL)**
à la prochaine séance. Ce choix précise une cible que le sujet laisse ouverte.
Le jeu complet contient **{reg['rows_raw']:,} lignes**. Cette première étude porte
sur **{reg['rows_prepared']:,} observations préparées d'Apple**, et non sur toutes les sociétés.

Les huit entrées sont les prix d'ouverture, clôture, minimum et maximum du jour t,
le volume, le rendement journalier, la moyenne et l'écart-type des cinq dernières
clôtures. Elles sont toutes disponibles au moment de la prédiction.
Les lignes sans historique ou sans cible sont retirées.

Le partage est chronologique : **{reg['split_sizes']['train']} train /
{reg['split_sizes']['validation']} validation / {reg['split_sizes']['test']} test**.
Une ligne est purgée à chaque frontière pour éviter qu'une date cible du train
soit une date d'entrée de validation ou de test. Les statistiques de standardisation,
y compris celles de la cible, sont apprises uniquement sur le train de chaque fold.

L'EDA comprend les valeurs manquantes, les statistiques descriptives, les cours,
la distribution des rendements et les corrélations. Les graphiques détaillés
utilisent le train. Les prix proches dans le temps sont fortement corrélés,
d'où l'intérêt d'une référence simple.

![Exploration des cours et rendements](results/regression/eda.png)

### Le modèle et la recherche

MLP avec ReLU, sortie linéaire et loss MSE. `GridSearchCV` compare **9 configurations
sur 3 folds chronologiques**, avec une purge d'une ligne. La grille fait varier
le taux d'apprentissage, l'architecture, Adam/SGD et 40/80 époques.
Meilleure configuration CV : **couches {rp['net__hidden']}, {rp['net__optimizer']},
taux {rp['net__lr']}, {rp['net__epochs']} époques**.

Le sujet demande une courbe d'accuracy. Pour un prix continu, cette mesure ne
convient pas : nous traçons **MSE / époques et RMSE / époques**. Nous calculons
aussi la MAE et le R². Les erreurs sont exprimées en dollars pour MAE/RMSE.

### Comparer les régularisations

La même architecture est testée sans régularisation, avec Dropout (0,2), L2
(`weight_decay=0.001`), arrêt anticipé (patience 10) et combinaison des trois.
La graine et le budget maximal d'époques restent les mêmes.

{table(reg_rows, ['model', 'selected_epoch', 'validation_rmse', 'test_mae', 'test_rmse', 'test_r2'], ['Modèle', 'Époque retenue', 'RMSE validation', 'MAE test', 'RMSE test', 'R² test'])}

Le modèle **`{reg['selected_on_validation']}` est choisi sur validation**, avant
d'évaluer le test. Sa RMSE test vaut **{r['test_rmse']:.4f} USD**, contre
**{persistence['rmse']:.4f} USD** pour la persistance : « demain = aujourd'hui ».
Le gain est très faible. La persistance fait même mieux sur la validation
({reg['reference']['validation']['rmse']:.4f} USD). Un R² de **{r['test_r2']:.4f}**
ne suffit donc pas à montrer une prévision utile des variations boursières.
Le Dropout seul dégrade ici les erreurs : régulariser davantage n'est pas toujours mieux.
Les erreurs baissent rapidement durant les premières époques, puis se stabilisent.
L'écart train/validation invite à surveiller la généralisation. Les courbes de
régression utilisent une échelle logarithmique pour garder visibles les erreurs
après convergence malgré les grandes erreurs des premières époques.

![Courbes du modèle initial](results/regression/curves_initial.png)
![Cours prédits sur le test](results/regression/predictions.png)

## 5. Partie 2 : classification des pannes

### La question et la préparation

Prévoir **Failure Type**, avec six classes : absence de panne, dissipation thermique,
puissance, usure d'outil, surcharge et panne aléatoire. Le jeu contient
**{cls['rows_raw']:,} lignes**. Le partage stratifié conserve les proportions :
**{cls['split_sizes']['train']} train / {cls['split_sizes']['validation']} validation /
{cls['split_sizes']['test']} test**.

Les entrées sont les températures de l'air et du procédé, la vitesse de rotation,
le couple, l'usure de l'outil et le type de produit. `UDI` et `Product ID` sont exclus.
**`Target` est aussi exclue**, car elle révèle déjà si une panne est présente.
L'EDA vérifie les doublons, les valeurs manquantes, les distributions, les
corrélations et les effectifs des classes. Les incohérences éventuelles entre
`Target` et `Failure Type` sont comptées dans `eda_summary.json`, sans réétiquetage arbitraire.

Les valeurs manquantes sont imputées (médiane pour les mesures, mode pour le type).
Les mesures sont standardisées. **SMOTENC** augmente les classes minoritaires
en respectant la variable catégorielle ; le type est ensuite encodé en one-hot.
Ces opérations sont ajustées **dans chaque fold d'entraînement**. La validation
et le test ne sont jamais augmentés et conservent leurs proportions naturelles.

![Rééquilibrage du train avec SMOTENC](results/classification/class_balance.png)

### Le modèle et les résultats

MLP avec ReLU, six logits en sortie et loss d'entropie croisée.
La même grille de neuf configurations utilise trois folds stratifiés et le **F1 macro**.
Meilleure configuration CV : **couches {cp['net__hidden']}, {cp['net__optimizer']},
taux {cp['net__lr']}, {cp['net__epochs']} époques**.

{table(cls_rows, ['model', 'selected_epoch', 'validation_f1_macro', 'test_accuracy', 'test_balanced_accuracy', 'test_f1_macro'], ['Modèle', 'Époque retenue', 'F1 macro validation', 'Accuracy test', 'Balanced accuracy test', 'F1 macro test'])}

Le modèle retenu sur validation est **`{cls['selected_on_validation']}`**.
Sur le test, son accuracy vaut **{c['test_accuracy']:.2%}** et son F1 macro
**{c['test_f1_macro']:.4f}**. La référence qui prédit toujours la classe majoritaire
atteint **{majority['accuracy']:.2%} d'accuracy**, mais seulement **{majority['f1_macro']:.4f}
de F1 macro**. Cela montre pourquoi une accuracy élevée ne suffit pas pour détecter les pannes.

La **sensibilité est le rappel** : parmi les vrais cas d'une classe, combien sont
détectés ? La précision indique la proportion de prédictions de cette classe qui
sont correctes. Le F1 combine précision et rappel. Les rapports détaillent ces
mesures par classe, ainsi que leur support, sur train et test.

Les rappels des pannes thermiques, de puissance et de surcharge valent respectivement
**{report.loc['Heat Dissipation Failure', 'recall']:.1%}**,
**{report.loc['Power Failure', 'recall']:.1%}** et
**{report.loc['Overstrain Failure', 'recall']:.1%}**, mais cela repose sur de petits
effectifs et n'exclut pas des fausses alertes. Le rappel pour l'usure des outils
est de **{tool['recall']:.1%}** sur **{int(tool['support'])} cas**, et celui des pannes
aléatoires est de **{random_failure['recall']:.1%}** sur **{int(random_failure['support'])} cas**.
Le modèle n'est donc pas une
solution de maintenance prête à l'emploi. Le Dropout réduit l'écart de F1 macro
entre train et validation par rapport au modèle initial, avec une amélioration
modeste du F1 macro de validation ; les autres régularisations ne font pas mieux ici.

![Courbes du modèle initial](results/classification/curves_initial.png)
![Matrice de confusion du modèle retenu](results/classification/confusion_matrix.png)

## 6. Interprétation et limites

- **Train / validation / test** : les folds du train règlent les hyperparamètres ;
  la validation choisit la régularisation et pilote l'arrêt anticipé ; le test
  sert à l'évaluation finale. Le modèle n'est pas choisi selon son résultat de test.
- **Courbes de test** : pour respecter le sujet sans utiliser le test pendant
  l'apprentissage, elles sont calculées après sélection à partir des poids
  sauvegardés à chaque époque. Leurs mesures ne changent aucun paramètre.
- **Régression** : une seule action, un seul horizon et une seule période test.
  Les résultats ne prouvent ni rentabilité ni généralisation à d'autres marchés.
- **Classification** : le jeu est synthétique et certaines pannes sont très rares.
  Une poignée d'exemples test donne des rappels/F1 instables ; SMOTENC ne crée
  pas de nouvelles informations physiques, et les pannes aléatoires restent difficiles.
- **Régularisation** : les tableaux montrent l'effet réellement mesuré. Un modèle
  régularisé peut moins bien apprendre, et le meilleur sur validation n'est pas
  nécessairement le meilleur a posteriori sur test.
- **Recherche limitée** : neuf configurations et une graine ne constituent pas
  une optimisation exhaustive ni une estimation de l'incertitude.

## 7. Synthèse des apprentissages

Cet atelier m'a permis de mieux comprendre comment construire et évaluer un
réseau de neurones avec PyTorch. J'ai appris à distinguer la régression, qui
prédit une valeur numérique, de la classification multiclasse, qui prédit une
catégorie. J'ai aussi mieux compris les étapes de l'apprentissage : calculer
les prédictions, mesurer l'erreur, calculer les gradients et mettre à jour les poids.

J'ai compris que la préparation des données est aussi importante que le choix du
réseau. Pour les cours boursiers, il faut respecter l'ordre chronologique pour ne
pas utiliser le futur pendant l'apprentissage. Pour les pannes, il faut tenir
compte du déséquilibre des classes et appliquer le rééquilibrage uniquement sur
les données d'entraînement.

Les résultats m'ont montré qu'un bon score ne suffit pas à conclure qu'un modèle
est performant. En régression, le réseau fait presque aussi bien que la méthode
simple qui reprend le dernier cours connu. En classification, une accuracy élevée
peut masquer une mauvaise détection des pannes rares. C'est pour cela que je
regarde aussi le F1 macro, le rappel par classe et la matrice de confusion.

Enfin, la comparaison des modèles m'a permis de comprendre que le Dropout, la
régularisation L2 et l'arrêt anticipé n'améliorent pas automatiquement les résultats.
Il faut vérifier leur effet sur la validation et garder le test pour l'évaluation
finale. Ce que je retiens surtout de cet atelier, c'est qu'il faut expliquer les
résultats et leurs limites, au lieu de chercher seulement le meilleur score.

## 8. Organisation et vérification

```text
notebooks/             Deux notebooks pédagogiques avec sorties
src/data.py            Sources, empreintes, nettoyage et partitions
src/models.py          MLP, boucle PyTorch et adaptateurs scikit-learn
src/experiments.py     EDA, GridSearchCV, régularisation, courbes et métriques
scripts/               Installation, exécution des notebooks et mise à jour du rapport
reports/               Rapport PDF détaillé
tests/                 Contrôles des partitions et des pipelines
results/regression/    Résultats complets de régression
results/classification/ Résultats complets de classification
data/raw/              Données locales téléchargées, exclues de Git
```

Les `grid_search.csv` conservent tous les scores CV (`mean_test_score` désigne
le score des folds de validation croisée, pas le test final). Les `comparison.csv` regroupent
les cinq modèles. Les `history_*.csv` contiennent les courbes ; les `report_*.csv`
contiennent les métriques de classification par classe sur train et test.
Les `summary.json` enregistrent la graine, les versions et les sources.
Les modèles binaires et les données brutes ne sont pas committés.

Pour régénérer le PDF à partir des résultats enregistrés :

```bash
python -m pip install -r requirements-report.txt
python scripts/generate_report.py
```

Validation réalisée : **54 ajustements de validation croisée**, dix modèles finaux,
quatre tests automatiques et exécution des deux notebooks. Les scores ci-dessus
proviennent du mode complet, avec la graine 42. Pour une reproduction comparable,
utiliser les versions épinglées ; du matériel ou des versions différents peuvent
produire de petites variations numériques.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")
