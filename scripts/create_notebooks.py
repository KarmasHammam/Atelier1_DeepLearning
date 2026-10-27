"""Générer les deux supports pédagogiques sans dupliquer l'implémentation."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def notebook(task):
    regression = task == "regression"
    title = "Régression : prévoir la prochaine clôture d'Apple" if regression else "Classification : identifier le type de panne"
    cells = [markdown(f"""
# Atelier 1 — {title}

Ce notebook suit les questions du sujet. Le code partagé dans `src/` est commenté
en français. Les résultats enregistrés proviennent d'une exécution complète sur CPU,
avec une graine de 42. La cellule d'entraînement permet de refaire l'expérience.

**Avant de commencer :** suivre l'installation du README et sélectionner le Python
de l'environnement virtuel comme noyau. Dans Colab, cloner le dépôt, entrer dans
son dossier et installer les dépendances ; les commandes sont dans le README.
"""), code("""
from pathlib import Path
import os
import sys

# Accepter un lancement depuis la racine du dépôt ou depuis notebooks/.
ROOT = Path.cwd()
if not (ROOT / 'src').is_dir() and (ROOT.parent / 'src').is_dir():
    ROOT = ROOT.parent
if not (ROOT / 'src').is_dir():
    raise RuntimeError('Cloner le dépôt et lancer ce notebook depuis son dossier.')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from IPython.display import display, Image
from src.data import download_data, load_regression, load_classification, NUMERIC, REG_FEATURES
from src.experiments import run_experiment, make_pipeline, parameter_grid
from src.models import MLP

# Le téléchargement est réutilisable et son empreinte est vérifiée à chaque appel.
download_data()
""")]
    if regression:
        cells += [markdown("""
## 1. Comprendre les données et définir la question

Le jeu NYSE contient plusieurs actions. Nous choisissons **AAPL (Apple)** pour
construire une première expérience claire sur une série homogène.
À la clôture du jour t, nous prédisons la clôture de la prochaine séance.
Les prix du jour t et les statistiques sur les cinq dernières séances sont connus.
Le prix de demain est uniquement la cible : il n'est jamais une entrée.
"""), code("""
raw, prepared, parts = load_regression('AAPL')
print(f'Jeu NYSE : {raw.shape[0]:,} lignes, {raw.symbol.nunique()} actions')
display(raw.head())
display(raw.isna().sum().to_frame('Valeurs manquantes'))
display(parts['train'][REG_FEATURES + ['next_close']].describe())
"""), markdown("""
## 2. Préparer les données sans utiliser le futur

Nous créons un rendement journalier, une moyenne et un écart-type glissants sur
cinq séances. Les premières lignes sans historique et la dernière ligne sans cible
sont retirées. Le partage est chronologique : environ **70 % / 15 % / 15 %**.
Une ligne est purgée aux frontières pour empêcher le chevauchement des dates cibles.
L'imputation et la standardisation sont apprises sur le train, dans chaque fold.
La cible est aussi standardisée dans `fit`, puis remise en dollars dans `predict`.
"""), code("""
display(pd.DataFrame({name: {'lignes': len(frame), 'début': frame.date.min(),
                            'fin': frame.date.max(), 'dernière cible': frame.target_date.max()}
                      for name, frame in parts.items()}).T)
display(prepared[['date', 'close', 'mean_5', 'target_date', 'next_close']].head())
display(make_pipeline('regression'))
""")]
    else:
        cells += [markdown("""
## 1. Nettoyer et comprendre les données

La cible **Failure Type** contient six classes, y compris `No Failure`.
`Target` est retirée des entrées car elle révèle déjà la présence d'une panne.
`UDI` et `Product ID` sont des identifiants : ils ne sont pas des mesures physiques.
Nous conservons cinq mesures numériques et la catégorie de produit `Type`.
Les mesures manquantes sont imputées par la médiane du train et standardisées.
La catégorie est imputée puis encodée. Les valeurs extrêmes ne sont pas supprimées
automatiquement : elles peuvent justement décrire une panne.
"""), code("""
raw, prepared, parts = load_classification()
display(raw.head())
display(raw.isna().sum().to_frame('Valeurs manquantes'))
print('Doublons exacts :', raw.duplicated().sum())
display(parts['train'][NUMERIC].describe())
display(pd.DataFrame({name: frame['Failure Type'].value_counts()
                      for name, frame in parts.items()}))
"""), markdown("""
## 2. Séparer puis équilibrer les classes

Le partage 70 % / 15 % / 15 % est stratifié : chaque partition garde les six classes.
Nous appliquons **SMOTENC**, une extension de SMOTE pour les variables mixtes.
Elle crée des exemples minoritaires sans interpoler la catégorie du produit.
L'encodage one-hot vient après cette augmentation.

SMOTENC agit seulement sur les données d'entraînement de chaque fold. La validation
et le test conservent leur distribution naturelle. Cela évite d'évaluer le modèle
sur des exemples synthétiques trop proches du train.
"""), code("""
display(make_pipeline('classification'))
print('Entrées :', NUMERIC + ['Type'])
print('Cible : Failure Type ; Target et identifiants exclus.')
""")]
    cells += [markdown("""
## 3. Construire un MLP et rechercher les hyperparamètres

Un **MLP** est un réseau de couches entièrement connectées. Les couches cachées
utilisent ReLU. En régression, la sortie est une valeur et la loss est MSE.
En classification, la sortie contient six logits et la loss est l'entropie croisée.

`GridSearchCV` compare neuf configurations : taux d'apprentissage, architecture,
nombre d'époques et optimiseur. Il utilise trois folds, chronologiques avec purge
pour la régression, stratifiés pour la classification. Le score est la RMSE négative
en régression et le F1 macro en classification. Le test n'intervient pas dans ce choix.
"""), code("""
# Afficher la grille réellement utilisée, pas seulement la configuration gagnante.
display(parameter_grid(quick=False))
"""), markdown("""
## 4. Exécuter l'expérience et comparer les régularisations

Le modèle initial est comparé à Dropout, L2, arrêt anticipé et leur combinaison.
L'arrêt anticipé surveille la loss de validation, avec une patience de dix époques,
et restaure les meilleurs poids. Le modèle final est choisi sur la validation
(RMSE ou F1 macro). Les mesures et courbes de test sont calculées ensuite,
à partir des poids enregistrés à chaque époque ; elles ne guident pas l'entraînement.

Par défaut, nous lisons les résultats déjà produits. Passer `REENTRAINER` à `True`
pour refaire toute la recherche et les entraînements ; cela remplace les résultats
de cette partie dans `results/`. Le mode complet peut prendre quelques minutes.
"""), code(f"""
TASK = '{task}'
REENTRAINER = False
output = ROOT / 'results' / TASK
if REENTRAINER or not (output / 'summary.json').exists():
    run_experiment(TASK, quick=False)

summary = json.loads((output / 'summary.json').read_text())
print('Mode enregistré :', summary['mode'])
print('Configuration CV :', summary['best_parameters_cv'])
print('Modèle choisi sur validation :', summary['selected_on_validation'])
print('Partitions :', summary['split_sizes'])
display(pd.read_csv(output / 'comparison.csv'))
"""), code("""
# Afficher le réseau retenu : 8 entrées après préparation pour les deux tâches.
params = summary['best_parameters_cv']
display(MLP(inputs=8, outputs=1 if TASK == 'regression' else 6,
            hidden=tuple(params['net__hidden']),
            dropout=.2 if summary['selected_on_validation'] in ['dropout', 'combined'] else 0.))

# Les neuf configurations et leur dispersion entre folds sont consultables.
grid = pd.read_csv(output / 'grid_search.csv')
display(grid[['rank_test_score', 'mean_test_score', 'std_test_score', 'params']]
        .sort_values('rank_test_score'))
"""), markdown("""
## 5. Visualiser et interpréter

Une loss qui diminue sur le train mais augmente sur validation signale un possible
surapprentissage. Une régularisation n'est pas automatiquement bénéfique : elle
peut aussi empêcher le réseau d'apprendre suffisamment.
"""), code("""
display(Image(filename=str(output / 'eda.png')))
display(Image(filename=str(output / 'correlations.png')))
display(Image(filename=str(output / 'curves_initial.png')))
display(Image(filename=str(output / ('curves_' + summary['selected_on_validation'] + '.png'))))
""")]
    if regression:
        cells += [markdown("""
### Pourquoi la RMSE remplace l'accuracy

L'accuracy compte des catégories correctement prédites ; elle n'a pas de sens
pour un prix continu. Nous traçons donc **MSE / époques et RMSE / époques**, et
rapportons aussi MAE et R². Une petite RMSE/MAE est préférable ; un grand R²
ne prouve pas à lui seul une capacité à prévoir les variations de cours.

Nous comparons au modèle de **persistance** : « demain aura le même cours qu'aujourd'hui ».
Il constitue une référence exigeante. Consulter le README pour l'interprétation
des mesures de l'exécution enregistrée, sans modifier le choix sur la base du test.
"""), code("""
display(pd.DataFrame(summary['reference']).T)
display(Image(filename=str(output / 'predictions.png')))
display(pd.read_csv(output / 'predictions.csv').head())
""")]
    else:
        cells += [markdown("""
### Pourquoi l'accuracy seule peut tromper

La majorité des machines ne présentent pas de panne. Prédire toujours `No Failure`
peut donc donner une bonne accuracy tout en ratant toutes les pannes.
Le **rappel / sensibilité** mesure la part de cas d'une classe détectés ; la précision
mesure la fiabilité des alertes ; le F1 les combine. Le **F1 macro** traite chaque
classe à égalité. Le support indique combien d'exemples soutiennent chaque mesure.
Les classes très rares ont des résultats particulièrement incertains.
"""), code("""
display(pd.read_csv(output / 'class_balance.csv', index_col=0))
display(Image(filename=str(output / 'class_balance.png')))
chosen = summary['selected_on_validation']
for partition in ['train', 'test']:
    print('Rapport :', partition)
    display(pd.read_csv(output / f'report_{chosen}_{partition}.csv', index_col=0))
display(Image(filename=str(output / 'confusion_matrix.png')))
print('Référence : toujours la classe majoritaire')
display(pd.DataFrame(summary['reference']).T)
""")]
    cells += [markdown("""
## 6. Bilan à discuter

- Distinguer données d'entraînement, validation et test pour éviter une évaluation optimiste.
- Ajuster les prétraitements dans chaque fold de la recherche d'hyperparamètres.
- Choisir une métrique adaptée et comparer à une référence simple.
- Vérifier l'effet réel des régularisations, sans supposer qu'elles améliorent tout.
- Relier les conclusions aux effectifs et aux limites du jeu de données.

Le README rassemble les résultats mesurés et leurs limites. Ma synthèse
personnelle se trouve dans la section 7 : « Synthèse des apprentissages ».
""")]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.14"}}
    return nb


if __name__ == "__main__":
    target = ROOT / "notebooks"
    target.mkdir(exist_ok=True)
    for number, task in enumerate(["regression", "classification"], start=1):
        nbf.write(notebook(task), target / f"{number:02d}_{task}.ipynb")
