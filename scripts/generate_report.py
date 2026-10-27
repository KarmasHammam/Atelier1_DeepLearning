"""Générer le rapport PDF à partir des résultats effectivement enregistrés.

Prérequis : dépendances du projet + requirements-report.txt.
Commande depuis la racine : python scripts/generate_report.py
Aucun entraînement n'est relancé et aucun score n'est inventé.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import ast
import json
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                               PageBreak, PageTemplate, Paragraph, Spacer,
                               Table, TableStyle, Preformatted)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reports' / 'Rapport_Atelier1_DeepLearning.pdf'
W, H = A4
MARGIN = 48
WIDTH = W - 2 * MARGIN
NAVY = colors.HexColor('#17324D')
TEAL = colors.HexColor('#087E8B')
PALE = colors.HexColor('#EFF5F8')
GRAY = colors.HexColor('#536372')


def register_fonts():
    directory = Path('/usr/share/fonts/truetype/dejavu')
    # Un repli sur les polices PDF standard permet la génération hors Linux.
    if (directory / 'DejaVuSans.ttf').exists():
        for name, file in [('Body', 'DejaVuSans.ttf'), ('Body-Bold', 'DejaVuSans-Bold.ttf'),
                           ('Body-Italic', 'DejaVuSans-Oblique.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(directory / file)))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-Bold')
        return 'Body', 'Body-Bold', 'Body-Italic', 'Mono'
    return 'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique', 'Courier'


BODY, BOLD, ITALIC, MONO = register_fonts()
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Text', fontName=BODY, fontSize=9.6, leading=14.6,
                          textColor=NAVY, spaceAfter=9, allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle(name='Chapter', fontName=BOLD, fontSize=18, leading=24,
                          textColor=TEAL, spaceAfter=17, keepWithNext=True))
styles.add(ParagraphStyle(name='Sub', fontName=BOLD, fontSize=11.4, leading=16,
                          textColor=NAVY, spaceBefore=8, spaceAfter=7, keepWithNext=True))
styles.add(ParagraphStyle(name='SmallText', fontName=BODY, fontSize=8, leading=11,
                          textColor=GRAY, spaceAfter=7))
styles.add(ParagraphStyle(name='TableText', fontName=BODY, fontSize=7.7, leading=10.5,
                          textColor=NAVY))
styles.add(ParagraphStyle(name='TableHead', fontName=BOLD, fontSize=7.7, leading=10.5,
                          textColor=colors.white))
styles.add(ParagraphStyle(name='Caption', fontName=ITALIC, fontSize=8, leading=11,
                          textColor=GRAY, alignment=TA_CENTER, spaceBefore=5, spaceAfter=12))
styles.add(ParagraphStyle(name='CoverTitle', fontName=BOLD, fontSize=27, leading=36,
                          textColor=NAVY, alignment=TA_CENTER, spaceAfter=18))
styles.add(ParagraphStyle(name='CoverText', fontName=BODY, fontSize=11, leading=18,
                          textColor=NAVY, alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle(name='CodeText', fontName=MONO, fontSize=7.4, leading=11,
                          backColor=PALE, borderPadding=8, spaceBefore=5, spaceAfter=12))


class Report(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=47, bottomMargin=47, title='Atelier 1 — Deep Learning avec PyTorch',
                         author='KARMAS Hammam', subject='Régression NYSE et classification multiclasse de maintenance')
        frame = Frame(MARGIN, 47, WIDTH, H - 94, id='normal', leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='report', frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(.5)
        canvas.line(MARGIN, H - 30, W - MARGIN, H - 30)
        canvas.setFont(BODY, 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(MARGIN, H - 22, 'ATELIER 1 · DEEP LEARNING · MASTER MBD')
        canvas.drawString(MARGIN, 28, 'KARMAS Hammam · Rapport des expériences PyTorch')
        canvas.drawRightString(W - MARGIN, 28, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'Chapter':
            text = flowable.getPlainText()
            # Les clés doivent rester identiques entre les passes du sommaire.
            key = 'section-' + re.sub(r'[^a-z0-9]+', '-', text.lower())
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, False)
            if text != 'Sommaire':
                self.notify('TOCEntry', (0, text, self.page, key))


def P(text, style='Text'):
    return Paragraph(text, styles[style])


def table(headers, rows, widths=None):
    data = [[P(escape(str(x)), 'TableHead') for x in headers]]
    for row in rows:
        data.append([P(escape(str(x)), 'TableText') for x in row])
    widths = widths or [WIDTH / len(headers)] * len(headers)
    result = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#CCDAE2')),
    ]))
    return result


def load(task):
    path = ROOT / 'results' / task
    return (json.loads((path / 'summary.json').read_text()),
            pd.read_csv(path / 'comparison.csv'),
            json.loads((path / 'eda_summary.json').read_text()))


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    reg, rt, re_eda = load('regression')
    cls, ct, ce_eda = load('classification')
    if reg['mode'] != 'full' or cls['mode'] != 'full':
        raise ValueError('Le rapport nécessite les résultats complets des deux parties.')
    chosen_r = rt.set_index('model').loc[reg['selected_on_validation']]
    chosen_c = ct.set_index('model').loc[cls['selected_on_validation']]
    report_c = pd.read_csv(ROOT / 'results/classification' /
                           f"report_{cls['selected_on_validation']}_test.csv", index_col=0)
    report_train = pd.read_csv(ROOT / 'results/classification' /
                               f"report_{cls['selected_on_validation']}_train.csv", index_col=0)
    balance = pd.read_csv(ROOT / 'results/classification/class_balance.csv', index_col=0)
    story = []

    def para(text):
        story.append(P(text))

    def sub(title):
        story.append(P(title, 'Sub'))

    def chapter(title):
        story.extend([PageBreak(), P(title, 'Chapter')])

    def figure(task, filename, caption, height=205):
        path = ROOT / 'results' / task / filename
        img = Image(str(path))
        scale = min(WIDTH / img.imageWidth, height / img.imageHeight)
        img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
        story.append(KeepTogether([img, P(caption, 'Caption')]))

    def code(text):
        story.append(Preformatted(text.strip(), styles['CodeText']))

    def metrics_table(frame, cols, heads):
        rows = []
        for _, row in frame.iterrows():
            rows.append([str(row[c]) if c == 'model' else
                         str(int(row[c])) if c in ['epochs_run', 'selected_epoch'] else
                         f'{row[c]:.4f}' for c in cols])
        story.extend([table(heads, rows), Spacer(1, 10)])

    # Couverture.
    story.extend([Spacer(1, 35), P('Université Abdelmalek Essaâdi', 'CoverText'),
                  P('Faculté des Sciences et Techniques de Tanger<br/>Département Génie Informatique<br/>Master MBD', 'CoverText'),
                  Spacer(1, 42), P('Atelier 1<br/>Deep Learning avec PyTorch', 'CoverTitle'),
                  P('Régression boursière et classification<br/>multiclasse des pannes', 'CoverText'),
                  Spacer(1, 35), P('Réalisé par <b>KARMAS Hammam</b><br/>Enseignant : Pr. Lotfi Elaachak', 'CoverText'),
                  Spacer(1, 30), P('Rapport détaillé : données, méthode, implémentation,<br/>résultats, interprétation et reproduction', 'CoverText'),
                  Spacer(1, 22), P('Les résultats présentés proviennent des expériences complètes enregistrées dans le dépôt. Aucun résultat supplémentaire n’est supposé.', 'Caption')])

    chapter('Résumé du travail')
    para("L'objectif de cet atelier est de comprendre le fonctionnement d'un réseau de neurones multicouche avec PyTorch, puis de l'appliquer à deux problèmes différents. La première partie prédit une valeur continue : le prochain cours de clôture d'Apple. La seconde prédit une catégorie : le type de panne d'une machine.")
    para("Le travail suit une même logique : comprendre les données, définir la cible, séparer les observations, préparer les entrées, rechercher des hyperparamètres, comparer plusieurs régularisations et évaluer le modèle sur des données conservées à part.")
    story.append(table(['Élément', 'Régression', 'Classification'], [
        ['Données', 'NYSE : prix ajustés, action AAPL', 'Maintenance prédictive : 10 000 observations'],
        ['Objectif', 'Clôture de la prochaine séance', 'Six classes de Failure Type'],
        ['Score de sélection', 'RMSE minimale', 'F1 macro maximal'],
        ['Modèle choisi sur validation', reg['selected_on_validation'], cls['selected_on_validation']],
        ['Résultat test principal', f"RMSE = {chosen_r.test_rmse:.4f} USD", f"F1 macro = {chosen_c.test_f1_macro:.4f}"],
        ['Référence simple', f"Persistance : RMSE = {reg['reference']['test']['rmse']:.4f}",
         f"Classe majoritaire : F1 macro = {cls['reference']['test']['f1_macro']:.4f}"],
    ], [110, (WIDTH - 110) / 2, (WIDTH - 110) / 2]))
    sub('Ce que les résultats permettent de conclure')
    para("En régression, le réseau sélectionné est très proche d'une simple persistance du dernier prix. Le résultat ne démontre pas une capacité de prévision boursière forte. En classification, le F1 macro progresse par rapport à la prédiction systématique de la classe majoritaire, mais certaines pannes rares restent mal détectées. Le modèle ne constitue donc pas un système industriel validé.")
    para("La valeur de ce travail est aussi méthodologique : les prétraitements sont ajustés dans chaque fold, le test ne guide pas les choix et les résultats peu favorables sont conservés et expliqués.")

    chapter('Sommaire')
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('TOC0', fontName=BODY, fontSize=9.3, leading=17,
                                      textColor=NAVY, spaceBefore=6, leftIndent=0)]
    story.append(toc)

    chapter('1. Comprendre le sujet et les livrables')
    sub('1.1 Les objectifs pédagogiques')
    para("Le sujet demande de se familiariser avec PyTorch pour construire des architectures DNN/MLP. Un MLP est un réseau constitué de couches entièrement connectées : chaque neurone reçoit les sorties de la couche précédente. Les couches cachées apprennent des relations non linéaires entre les variables.")
    para("La régression et la classification partagent la préparation des données et la boucle d'apprentissage. Elles diffèrent surtout par la cible, la dernière couche, la fonction de perte et les métriques d'évaluation.")
    story.append(table(['Question du sujet', 'Réalisation dans le projet'], [
        ['Exploration et nettoyage', 'Statistiques, valeurs manquantes, doublons, distributions et corrélations'],
        ['DNN avec PyTorch', 'Module MLP, ReLU et boucle d’apprentissage explicite'],
        ['Recherche des hyperparamètres', 'GridSearchCV : neuf configurations, trois folds par tâche'],
        ['Courbes par époque', 'Loss et RMSE ou accuracy sur train, validation et test'],
        ['Régularisation', 'Dropout, L2, arrêt anticipé et combinaison'],
        ['Déséquilibre des classes', 'SMOTENC dans le pipeline de classification'],
        ['Évaluation et interprétation', 'Références simples, scores globaux et par classe'],
        ['Remise', 'Notebooks, code, README, synthèse et présent rapport PDF'],
    ], [160, WIDTH - 160]))
    sub('1.2 Deux précisions nécessaires')
    para("Le sujet ne fixe pas la variable à prédire pour NYSE. Le choix retenu est donc explicite : à la clôture du jour t, prévoir la clôture ajustée de la prochaine séance pour AAPL. Le sujet demande aussi une accuracy en régression ; cette métrique étant inadaptée à une valeur continue, elle est remplacée par la RMSE.")
    para("Pour la partie multiclasse, la cible choisie est Failure Type. La colonne Target, qui indique seulement panne ou absence de panne, ne répondrait pas à la demande multiclasse et révélerait une information liée à la cible si elle était utilisée comme entrée.")

    chapter('2. Méthode générale et prévention des fuites')
    sub('2.1 Le rôle des trois partitions')
    para("<b>Entraînement :</b> apprendre les transformations et les poids du réseau. La validation croisée y compare les hyperparamètres. <b>Validation indépendante :</b> comparer les régularisations et surveiller l'arrêt anticipé. <b>Test :</b> mesurer les performances finales après avoir fixé tous les choix.")
    para("Une fuite de données apparaît lorsqu'une information indisponible au moment de la prédiction ou provenant de la validation/test influence l'apprentissage. Cela produit des scores trop optimistes. Ici, imputation, standardisation et rééquilibrage sont intégrés dans les pipelines afin d'être ajustés seulement sur le train de chaque fold.")
    story.append(table(['Étape', 'Règle appliquée'], [
        ['Imputation et standardisation', 'Apprendre les paramètres sur le train du fold, transformer les autres partitions sans les réajuster'],
        ['Standardisation de la cible de régression', 'Moyenne et écart-type calculés dans fit sur la cible du train'],
        ['Augmentation SMOTENC', 'Uniquement dans les folds train ; aucune augmentation de validation/test'],
        ['Sélection des variantes', 'RMSE de validation en régression, F1 macro de validation en classification'],
        ['Arrêt anticipé', 'Loss de validation, patience de dix époques et restauration des meilleurs poids'],
        ['Courbes de test', 'Calcul rétrospectif à partir des poids sauvegardés, après sélection'],
    ], [160, WIDTH - 160]))
    sub('2.2 Lire correctement la validation croisée')
    para("Chaque tâche compare neuf configurations sur trois folds, soit 27 ajustements. Au total, les deux recherches effectuent 54 ajustements, auxquels s'ajoutent dix modèles finaux. Les colonnes mean_test_score des fichiers GridSearchCV désignent le score sur les folds de validation croisée, et non sur le test final du projet.")
    sub('2.3 Reproductibilité')
    para(f"La graine est fixée à 42 pour Python, NumPy et PyTorch. Le réseau utilise le CPU, deux threads PyTorch et des algorithmes déterministes. Les versions principales enregistrées sont Python {reg['python']}, PyTorch {reg['torch']} et scikit-learn {reg['sklearn']}. Une autre plateforme peut encore produire de petites différences numériques.")

    chapter('3. Régression : données, cible et variables')
    para(f"Le fichier prices-split-adjusted.csv contient {reg['rows_raw']:,} lignes et sept colonnes, pour plusieurs actions. Les prix ajustés réduisent les ruptures liées aux divisions d'actions. Pour éviter de mélanger les niveaux de prix de sociétés différentes, cette étude se limite à Apple, symbole AAPL.")
    para("Après tri chronologique et création des variables, 1 757 observations sont exploitables. Les premières lignes sans cinq séances d'historique et la dernière ligne sans clôture future sont retirées. Le contrôle du fichier brut ne détecte ni valeur manquante ni doublon exact.")
    story.append(table(['Variable', 'Signification et disponibilité'], [
        ['open, close, low, high', 'Prix d’ouverture, clôture, minimum et maximum de la séance t, connus à sa clôture'],
        ['volume', 'Volume échangé pendant la séance t'],
        ['return_1', 'Rendement close(t) / close(t-1) - 1'],
        ['mean_5', 'Moyenne des cinq dernières clôtures, y compris la séance t'],
        ['std_5', 'Écart-type des cinq dernières clôtures'],
        ['next_close — cible', 'Clôture de la prochaine séance, créée par shift(-1) ; absente des entrées'],
        ['target_date', 'Date associée à la cible, utilisée pour contrôler les frontières'],
    ], [150, WIDTH - 150]))
    sub('3.1 Séparation chronologique avec purge')
    story.append(table(['Partition', 'Lignes', 'Dates des entrées', 'Dernière cible'], [
        ['Train', '1 228', '08/01/2010 – 21/11/2014', '24/11/2014'],
        ['Validation', '263', '25/11/2014 – 10/12/2015', '11/12/2015'],
        ['Test', '264', '14/12/2015 – 29/12/2016', '30/12/2016'],
    ], [85, 55, 225, WIDTH - 365]))
    para("Le partage vise environ 70 % / 15 % / 15 %. Une ligne est purgée à chaque frontière afin que la date cible d'une partition soit strictement antérieure à la première date d'entrée de la suivante. La validation croisée utilise aussi TimeSeriesSplit(n_splits=3, gap=1). Un partage aléatoire serait inadapté à cette expérience temporelle.")

    chapter('4. Régression : exploration et préparation')
    figure('regression', 'eda.png', 'Figure 1 — Évolution du cours et distribution des rendements, uniquement sur le train.', 195)
    para("Les cours évoluent fortement sur la période d'entraînement : leur niveau n'est pas constant dans le temps. Les rendements sont davantage concentrés autour de zéro, avec quelques variations plus marquées. Cette différence explique pourquoi un réseau qui apprend le niveau du prix peut afficher un R² élevé sans prédire précisément ses variations.")
    figure('regression', 'correlations.png', 'Figure 2 — Corrélations entre les entrées et la cible sur le train.', 260)
    para("Les prix de la même séance et les moyennes récentes sont fortement liés. La clôture de la prochaine séance est également proche des clôtures récentes. Une corrélation forte n'est pas une preuve de causalité et justifie une comparaison à la persistance. Les variables sont standardisées pour éviter que le volume, beaucoup plus grand en échelle, domine numériquement les prix.")

    chapter('5. Régression : réseau et hyperparamètres')
    sub('5.1 Architecture et fonction de perte')
    para("Le modèle retenu par validation croisée comporte huit entrées, une couche cachée de 32 neurones avec activation ReLU et une sortie linéaire. Il contient 321 paramètres entraînables : 8 × 32 + 32, puis 32 × 1 + 1. La variante combinée ajoute un Dropout de 0,2 après l'activation, sans ajouter de paramètres entraînables.")
    code('8 variables → Linear(8, 32) → ReLU → [Dropout(0.2)]\n            → Linear(32, 1) → prix prédit')
    para("La cible est standardisée pendant fit : y_standard = (y - moyenne_train) / écart_type_train. Le réseau minimise la MSE sur cette cible. Lors de predict, la transformation inverse remet les prédictions en dollars. Les courbes de loss présentées sont recalculées en MSE exprimée en USD² pour être interprétables.")
    sub('5.2 Grille effectivement exécutée')
    para("La grille combine Adam avec les architectures [32] ou [64, 32], des taux d'apprentissage 0,001 ou 0,01 et 40 ou 80 époques : huit configurations. Une neuvième configuration teste SGD avec momentum 0,9, architecture [64, 32], taux 0,01 et 80 époques. La taille des mini-lots vaut 512 ; elle n'est pas optimisée.")
    para("Pour relier PyTorch à scikit-learn, TorchRegressor et TorchClassifier suivent l'interface BaseEstimator : fit apprend les poids, predict calcule les prédictions et les paramètres du constructeur peuvent être clonés. GridSearchCV crée ainsi un modèle neuf pour chaque configuration et chaque fold, sans réutiliser les poids d'un autre ajustement.")
    para(f"La meilleure configuration utilise Adam, le taux 0,01, [32] et 80 époques. La RMSE moyenne CV vaut {-reg['best_score_cv']:.4f} USD. Scikit-learn enregistre son opposé pour maximiser le score : plus la RMSE négative est proche de zéro, meilleur est le résultat.")
    sub('5.3 Comment les poids sont appris')
    code('pour chaque époque :\n    pour chaque mini-lot du train :\n        optimizer.zero_grad()\n        prediction = model(entrees)\n        loss = criterion(prediction, cible)\n        loss.backward()\n        optimizer.step()\n    mesurer train et validation sans gradient')
    para("Une époque correspond à un passage sur tout le train. Le taux d'apprentissage contrôle l'amplitude des mises à jour ; l'optimiseur détermine leur calcul. Un nombre trop faible d'époques peut laisser le modèle sous-entraîné, tandis qu'un apprentissage trop long peut favoriser le surapprentissage.")

    chapter('6. Régression : résultats et interprétation')
    metrics_table(rt, ['model', 'validation_rmse', 'test_mae', 'test_rmse', 'test_r2'],
                  ['Variante', 'RMSE validation', 'MAE test', 'RMSE test', 'R² test'])
    para("Le choix final est <b>combined</b>, car cette variante présente la plus faible RMSE sur validation parmi les cinq réseaux comparés. Le modèle initial a une RMSE test légèrement meilleure, mais le choisir après lecture du test rendrait l'évaluation biaisée. L'ordre des modèles sur validation n'est pas garanti sur une autre période.")
    figure('regression', 'curves_initial.png', 'Figure 3 — Modèle initial : MSE et RMSE par époque, échelles logarithmiques.', 195)
    figure('regression', 'curves_combined.png', 'Figure 4 — Variante combinée : entraînement arrêté après 32 époques, poids de l’époque 22 restaurés.', 195)
    para("Les erreurs baissent rapidement, puis se stabilisent. L'erreur de validation reste supérieure à celle du train, ce qui invite à surveiller la généralisation. Le Dropout seul dégrade ici la validation : la régularisation réduit la capacité effective du réseau et ne garantit pas une amélioration.")
    sub('6.1 Comparer à une référence exigeante')
    story.append(table(['Méthode', 'RMSE validation', 'MAE test', 'RMSE test', 'R² test'], [
        ['Persistance', f"{reg['reference']['validation']['rmse']:.4f}", f"{reg['reference']['test']['mae']:.4f}", f"{reg['reference']['test']['rmse']:.4f}", f"{reg['reference']['test']['r2']:.4f}"],
        ['MLP choisi', f'{chosen_r.validation_rmse:.4f}', f'{chosen_r.test_mae:.4f}', f'{chosen_r.test_rmse:.4f}', f'{chosen_r.test_r2:.4f}'],
    ]))
    para("La persistance prédit simplement : clôture prochaine = clôture actuelle. Elle obtient une meilleure RMSE sur validation que le réseau choisi. Sur le test, le gain du MLP reste minime et sa significativité n'a pas été évaluée. Les résultats ne démontrent ni un avantage économique ni une stratégie de trading rentable.")
    figure('regression', 'predictions.png', 'Figure 5 — Clôtures réelles, MLP retenu et persistance sur les 264 observations de test.', 225)
    sub('6.2 Comprendre les métriques')
    para("<b>MAE</b> = moyenne des erreurs absolues : elle donne une erreur typique en dollars. <b>RMSE</b> = racine de la moyenne des erreurs au carré : elle pénalise davantage les grosses erreurs. <b>R²</b> compare l'erreur du modèle à celle d'une prédiction constante égale à la moyenne des valeurs évaluées. Il peut être négatif et ne mesure pas la rentabilité.")
    para("Une accuracy classique n'est pas calculée : demander si deux prix continus sont exactement égaux n'est pas une mesure pertinente. MSE/époques et RMSE/époques remplacent donc la paire loss/accuracy demandée dans le sujet, avec justification explicite.")

    chapter('7. Classification : comprendre les données')
    para("Le fichier predictive_maintenance.csv contient 10 000 observations et dix colonnes. La cible Failure Type distingue six situations : No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure et Random Failures. Il s'agit d'un jeu synthétique, pas d'une validation sur un parc réel de machines.")
    story.append(table(['Colonne', 'Utilisation'], [
        ['Air temperature [K]', 'Température de l’air : variable numérique'],
        ['Process temperature [K]', 'Température du procédé : variable numérique'],
        ['Rotational speed [rpm]', 'Vitesse de rotation : variable numérique'],
        ['Torque [Nm]', 'Couple : variable numérique'],
        ['Tool wear [min]', 'Usure de l’outil : variable numérique'],
        ['Type', 'Catégorie de produit L, M ou H : variable catégorielle'],
        ['Failure Type', 'Cible multiclasse à prédire'],
        ['Target', 'Exclue : indique déjà la présence d’une panne'],
        ['UDI, Product ID', 'Exclus : identifiants, pas des mesures physiques'],
    ], [175, WIDTH - 175]))
    sub('7.1 Nettoyage et contrôles')
    para(f"Les contrôles du fichier brut trouvent {ce_eda['missing_values']} valeur manquante et {ce_eda['exact_duplicates']} doublon exact. Les pipelines conservent néanmoins une imputation pour rendre le traitement explicite et robuste. Les valeurs numériques non finies sont traitées comme manquantes. Les observations sans étiquette seraient retirées.")
    para(f"Le jeu présente {ce_eda['target_failure_type_disagreements']} désaccords entre l'indicateur Target et la présence d'un type de panne. Ces désaccords sont signalés et les étiquettes Failure Type sont conservées, sans correction arbitraire. Les valeurs extrêmes ne sont pas supprimées automatiquement : elles peuvent caractériser une panne réelle.")
    sub('7.2 Partage stratifié')
    para("Les partitions comprennent 7 000 observations de train, 1 500 de validation et 1 500 de test. Le tirage est stratifié sur Failure Type et la graine vaut 42. Chaque partition conserve les six classes et leurs proportions approximatives. Les identifiants UDI sont disjoints entre les partitions.")

    chapter('8. Classification : EDA et rééquilibrage')
    figure('classification', 'eda.png', 'Figure 6 — Déséquilibre des classes et distribution du couple par type de panne sur le train.', 190)
    para("La classe No Failure domine largement. Dans le train, elle compte 6 756 cas, contre seulement 13 pannes aléatoires. Un modèle qui minimise l'erreur globale sans précaution peut donc privilégier l'absence de panne. L'EDA des mesures et des corrélations aide aussi à repérer les échelles et les relations, sans supprimer automatiquement les cas rares.")
    story.append(table(['Classe du train', 'Avant SMOTENC', 'Après SMOTENC'],
                       [[label, int(row.avant), int(row.apres)] for label, row in balance.iterrows()],
                       [WIDTH - 160, 80, 80]))
    sub('8.1 Pourquoi SMOTENC et dans quel ordre ?')
    para("Les mesures sont imputées par leur médiane et standardisées. Le type est imputé par son mode, puis encodé temporairement par 0, 1 ou 2. SMOTENC est informé que la sixième colonne est catégorielle : il interpole les mesures tout en conservant une catégorie valide. Ensuite seulement, le type est encodé en trois indicateurs one-hot.")
    para("Cette méthode évite de créer par interpolation un type de produit fictif tel que « moitié L, moitié M ». Les six classes du train final sont portées à 6 756 observations chacune, soit 40 536 lignes. Ces exemples synthétiques ne sont pas des mesures physiques supplémentaires et ne garantissent pas une meilleure généralisation.")
    para("Le nombre de voisins de SMOTENC est fixé à trois pour rester compatible avec les faibles effectifs minoritaires dans les folds. La standardisation des mesures suit z = (x - moyenne_train) / écart_type_train. Elle évite que l'échelle d'une mesure domine le calcul des distances entre voisins.")
    para("Dans GridSearchCV, le même pipeline est réajusté dans chaque fold : l'imputation, la standardisation et SMOTENC ne voient pas les données du fold de validation. Ni la validation indépendante ni le test ne sont rééquilibrés.")
    story.append(PageBreak())
    sub('8.2 Lire les distributions et les corrélations')
    figure('classification', 'distributions.png', 'Figure 7 — Distributions des cinq mesures numériques sur le train.', 245)
    para("Ces histogrammes permettent de comparer les unités, les étendues et la forme des distributions. Une mesure n'a pas besoin de suivre une loi normale pour être standardisée. Une valeur éloignée du centre n'est pas automatiquement une erreur : la supprimer pourrait retirer précisément des situations de panne.")
    figure('classification', 'correlations.png', 'Figure 8 — Corrélations linéaires entre les cinq mesures du train.', 250)
    para("La température de l'air et celle du procédé sont liées ; vitesse et couple peuvent aussi présenter une relation. Une corrélation décrit une association linéaire, pas une causalité. Elle ne suffit pas à prévoir le type de panne et ne remplace pas l'évaluation du réseau. Les graphiques de rééquilibrage complémentaires sont conservés dans results/classification/ et le notebook.")

    chapter('9. Classification : MLP et recherche')
    sub('9.1 Architecture retenue')
    code('5 mesures + 3 indicateurs de Type = 8 entrées\nLinear(8, 64) → ReLU → Dropout(0.2)\nLinear(64, 32) → ReLU → Dropout(0.2)\nLinear(32, 6) → six logits')
    para("Le réseau possède 2 854 paramètres entraînables. Un logit est un score brut associé à une classe. Pendant l'apprentissage, CrossEntropyLoss reçoit directement ces logits : ajouter un softmax dans le modèle avant cette loss serait inutile et pourrait nuire au calcul numérique. Pour obtenir des probabilités à la prédiction, predict_proba applique softmax ; predict choisit le plus grand logit.")
    sub('9.2 Critère de recherche')
    para(f"La grille comporte les mêmes neuf configurations que la régression et utilise StratifiedKFold avec trois folds mélangés, graine 42. Le score maximisé est le F1 macro, qui donne autant de poids à chaque classe. La meilleure configuration est [64, 32], Adam, taux 0,01 et 80 époques. Son F1 macro CV moyen vaut {cls['best_score_cv']:.4f}.")
    para("Le F1 macro CV du modèle initial est distinct du F1 macro de la validation indépendante et de celui du test. Ils sont calculés sur des observations et des modèles ajustés différents ; ils n'ont donc aucune raison d'être identiques.")
    sub('9.3 Régularisation comparée')
    story.append(table(['Variante', 'Réglage'], [
        ['initial', 'Sans Dropout, sans pénalité L2, budget complet'],
        ['dropout', 'Probabilité de désactivation = 0,2 après chaque couche cachée'],
        ['l2', 'weight_decay = 0,001 dans l’optimiseur'],
        ['early_stopping', 'Patience = 10 ; restauration des poids à la meilleure loss validation'],
        ['combined', 'Dropout + L2 + arrêt anticipé'],
    ], [125, WIDTH - 125]))
    para("Le Dropout désactive aléatoirement des neurones pendant le train, mais est désactivé par model.eval() pour les mesures. L2 pénalise les poids importants. L'arrêt anticipé limite la durée d'apprentissage. Pour isoler leurs effets, les variantes gardent la même architecture, la même graine et le même budget maximal.")

    chapter('10. Classification : résultats globaux')
    metrics_table(ct, ['model', 'validation_f1_macro', 'test_accuracy', 'test_balanced_accuracy', 'test_f1_macro'],
                  ['Variante', 'F1 macro validation', 'Accuracy test', 'Balanced acc. test', 'F1 macro test'])
    para("La variante <b>dropout</b> est retenue sur la validation. Elle obtient un F1 macro validation de 0,5208, contre 0,5000 pour le modèle initial. La combinaison de toutes les régularisations ne donne pas le meilleur résultat : une contrainte trop forte peut empêcher le réseau de bien apprendre les classes.")
    figure('classification', 'curves_initial.png', 'Figure 9 — Entropie croisée et accuracy du modèle initial.', 190)
    figure('classification', 'curves_dropout.png', 'Figure 10 — Entropie croisée et accuracy avec Dropout, modèle retenu.', 190)
    para("Le modèle initial atteint un F1 macro train de 0,7959, contre 0,5000 sur validation : cet écart est compatible avec une généralisation limitée. Avec Dropout, les scores sont 0,6466 et 0,5208. Le train baisse et la validation progresse légèrement : la régularisation réduit ici l'écart, sans résoudre toutes les difficultés.")
    sub('10.1 Pourquoi le score de référence est trompeur')
    para(f"Prédire toujours No Failure donne {cls['reference']['test']['accuracy']:.2%} d'accuracy test, mais seulement {cls['reference']['test']['f1_macro']:.4f} de F1 macro et 16,67 % de balanced accuracy. Le réseau retenu donne {chosen_c.test_accuracy:.2%} d'accuracy et {chosen_c.test_f1_macro:.4f} de F1 macro. Il fait davantage d'erreurs globales, mais reconnaît une partie des classes de panne que la référence ignore entièrement.")
    para("Il n'est donc pas correct de conclure que le réseau est moins utile seulement parce que son accuracy est inférieure à celle de la classe majoritaire. Inversement, son F1 macro supérieur ne suffit pas à garantir son utilité industrielle : les coûts des fausses alertes et des pannes manquées ne sont pas modélisés.")

    chapter('11. Classification : métriques par classe')
    para("Pour une classe donnée, la <b>précision</b> répond à : « Parmi les alertes de cette classe, combien sont justes ? » Le <b>rappel</b>, ou sensibilité, répond à : « Parmi les vrais cas de cette classe, combien sont détectés ? » Le <b>F1</b> est la moyenne harmonique de précision et rappel. Le <b>support</b> est le nombre réel de cas de cette classe.")
    para("L'accuracy est la proportion totale de prédictions correctes. La balanced accuracy est la moyenne des rappels des classes. Le F1 macro moyenne leurs F1 à poids égaux ; le F1 pondéré les moyenne selon leurs effectifs, et peut donc rester élevé même si des classes rares sont ignorées.")
    para("Pour une classe, précision = TP / (TP + FP), rappel = TP / (TP + FN), et F1 = 2 × précision × rappel / (précision + rappel). TP signifie vrai positif, FP faux positif et FN faux négatif. Quand une métrique est indéfinie faute de prédictions ou de cas, le rapport utilise zero_division=0.")
    labels = list(balance.index)
    for title, frame in [('11.1 Modèle retenu — entraînement non augmenté', report_train),
                         ('11.2 Modèle retenu — test naturel', report_c)]:
        sub(title)
        rows = [[label, f"{frame.loc[label, 'precision']:.4f}", f"{frame.loc[label, 'recall']:.4f}",
                 f"{frame.loc[label, 'f1-score']:.4f}", str(int(frame.loc[label, 'support']))] for label in labels]
        story.append(table(['Classe', 'Précision', 'Rappel', 'F1', 'Support'], rows,
                           [WIDTH - 240, 60, 60, 60, 60]))
        story.append(Spacer(1, 12))
    para("Sur le test, les rappels des pannes thermiques, de puissance et de surcharge atteignent 100 %, mais sur 17, 14 et 12 observations seulement. Pour ces classes, la précision est inférieure à 100 %, donc des fausses alertes existent. Un rappel parfait sur un petit échantillon n'est pas une garantie pour des données futures.")
    para("L'usure de l'outil reste mal reconnue : un seul cas sur sept est détecté, avec une précision très faible. Les deux pannes aléatoires ne sont pas reconnues. Le déséquilibre initial, le faible nombre d'exemples originaux et le caractère aléatoire de certaines étiquettes limitent ce que l'augmentation peut apporter.")

    chapter('12. Matrice de confusion et diagnostic')
    figure('classification', 'confusion_matrix.png', 'Figure 11 — Modèle Dropout retenu : lignes = classes réelles ; colonnes = classes prédites.', 350)
    para("Les valeurs de la diagonale représentent les prédictions correctes. Les valeurs hors diagonale indiquent les confusions : par exemple, une machine sans panne prédite en usure d'outil correspond à une fausse alerte. Une panne réelle prédite No Failure correspond au contraire à une panne manquée.")
    matrix = pd.read_csv(ROOT / 'results/classification/confusion_matrix.csv', index_col=0)
    false_alarm = int(matrix.loc['No Failure'].sum() - matrix.loc['No Failure', 'No Failure'])
    missed = int(matrix['No Failure'].sum() - matrix.loc['No Failure', 'No Failure'])
    para(f"Dans ce test, {false_alarm} observations sans panne sont classées comme une panne, et {missed} observations présentant une panne sont classées No Failure. Ce regroupement binaire sert seulement à lire les erreurs ; le modèle reste bien multiclasse.")
    sub('12.1 Limites et améliorations possibles')
    para("Les résultats portent sur une seule graine, une seule partition de test et un jeu synthétique. Aucun intervalle de confiance ni test de significativité n'est calculé. Les scores par classe rare ont une forte incertitude ; les deux seules pannes aléatoires du test ne permettent pas une estimation stable.")
    para("Pour aller plus loin, on pourrait comparer plusieurs partitions, une pondération de la loss, d'autres stratégies de rééquilibrage ou des variables physiques dérivées. Ces pistes sont proposées, pas exécutées dans ce rapport. Si elles servent à choisir un nouveau modèle après lecture du test actuel, il faudra prévoir un nouveau test indépendant pour l'évaluation finale.")

    chapter('13. Organisation du code et reproduction')
    story.append(table(['Fichier ou dossier', 'Responsabilité'], [
        ['src/data.py', 'Sources figées, SHA-256, téléchargement, variables et partitions'],
        ['src/models.py', 'MLP, fit/predict, standardisation cible, états par époque'],
        ['src/experiments.py', 'EDA, pipelines, GridSearchCV, variantes et résultats'],
        ['notebooks/', 'Deux parcours pédagogiques commentés avec sorties enregistrées'],
        ['results/', 'Scores, historiques, figures, rapports et métadonnées'],
        ['tests/test_workflow.py', 'Quatre contrôles de données et de pipelines réels'],
        ['scripts/setup.sh', 'Environnement virtuel et dépendances épinglées'],
        ['scripts/generate_report.py', 'Génération du présent PDF sans réentraîner'],
    ], [160, WIDTH - 160]))
    sub('13.1 Installation et entraînement')
    code('git clone https://github.com/KarmasHammam/Atelier1_DeepLearning.git\ncd Atelier1_DeepLearning\nbash scripts/setup.sh\nsource .venv/bin/activate\npython -m src.experiments --task all\npython -m unittest discover -s tests -v\npython scripts/write_readme.py\npython scripts/execute_notebooks.py')
    para("L'installation CPU ne demande ni GPU ni clé Kaggle. Les fichiers bruts sont téléchargés dans data/raw/ et exclus de Git. Les scripts ne modifient pas le Python système. Le mode --quick réduit le budget à 8/16 époques ; il doit être écrit dans results_quick pour ne pas remplacer les résultats complets du rapport.")
    sub('13.2 Générer ce document')
    code('python -m pip install -r requirements-report.txt\npython scripts/generate_report.py')
    para("Le générateur lit les CSV/JSON et figures existants. Il ne relance aucun apprentissage. Les dépendances de rapport sont séparées de celles nécessaires aux expériences.")
    sub('13.3 Vérifications réalisées')
    para("Les deux recherches complètes et les dix modèles ont terminé. Les quatre tests automatiques passent : cible temporelle et frontières purgées ; partitions de classification et catégories valides après augmentation ; clonage du régresseur et standardisation de sa cible ; absence de modification du modèle lors de l'évaluation rétrospective du test. Les deux notebooks ont également été exécutés sans erreur, avec huit cellules de code chacun. Les tests ne prouvent pas une performance industrielle ; ils contrôlent des risques précis du workflow.")

    chapter('14. Synthèse personnelle et conclusion')
    readme = (ROOT / 'README.md').read_text()
    synthesis = readme.split('## 7. Synthèse des apprentissages\n', 1)[1].split('\n## 8.', 1)[0].strip()
    for paragraph in synthesis.split('\n\n'):
        para(escape(' '.join(paragraph.splitlines())))
    sub('Conclusion générale')
    para("Les deux tâches demandées ont été réalisées avec des MLP PyTorch, une recherche GridSearchCV, des courbes d'apprentissage et plusieurs régularisations. Le protocole conserve une distinction nette entre entraînement, validation et test. Les notebooks et le code permettent de retrouver chaque étape.")
    para("Le réseau boursier ne dépasse que très faiblement la persistance sur le test et reste moins bon qu'elle sur validation. Le réseau de classification améliore la prise en compte de plusieurs classes, mais échoue encore sur certaines pannes rares. Ces limites font partie du résultat : une expérimentation sérieuse ne consiste pas à présenter seulement des scores élevés, mais à expliquer ce qu'ils mesurent et ce qu'ils ne démontrent pas.")

    chapter('Annexe A — Détail de la recherche CV')
    para("Les tableaux ci-dessous reprennent les neuf configurations réellement évaluées. La valeur ± indique l'écart-type des scores sur les trois folds ; ce n'est pas un intervalle de confiance. Les architectures sont les tailles des couches cachées. Les scores de régression sont remis en RMSE positive pour faciliter la lecture.")
    for task, title in [('regression', 'A.1 Régression : RMSE moyenne CV'), ('classification', 'A.2 Classification : F1 macro moyen CV')]:
        sub(title)
        grid = pd.read_csv(ROOT / 'results' / task / 'grid_search.csv').sort_values('rank_test_score')
        rows = []
        for _, row in grid.iterrows():
            params = ast.literal_eval(row.params)
            score = -row.mean_test_score if task == 'regression' else row.mean_test_score
            rows.append([int(row.rank_test_score), str(params['net__hidden']), params['net__optimizer'],
                         str(params['net__lr']), params['net__epochs'], f'{score:.4f} ± {row.std_test_score:.4f}'])
        story.append(table(['Rang', 'Couches', 'Optim.', 'Taux', 'Époques', 'Moyenne ± écart-type'], rows,
                           [38, 68, 55, 45, 55, WIDTH - 261]))
        story.append(Spacer(1, 15))
    para("La recherche reste volontairement limitée. Elle explore surtout Adam et une seule configuration SGD. Elle ne permet donc pas de conclure qu'Adam est toujours supérieur à SGD, ni que l'architecture retenue est un optimum global.")

    chapter('Annexe B — Lire les points clés du code')
    sub('B.1 Créer une cible future sans la mettre dans les entrées')
    code("stock['return_1'] = stock.close.pct_change(fill_method=None)\nstock['mean_5'] = stock.close.rolling(5).mean()\nstock['std_5'] = stock.close.rolling(5).std()\nstock['next_close'] = stock.close.shift(-1)\nstock['target_date'] = stock.date.shift(-1)")
    para("rolling(5) regarde la séance courante et les quatre précédentes. shift(-1) est réservé à la cible et à sa date. REG_FEATURES n'inclut jamais next_close ni target_date.")
    sub('B.2 Encapsuler les transformations pendant la CV')
    code("search = GridSearchCV(\n    make_pipeline(task), parameter_grid(quick=False),\n    cv=cv, scoring=scoring, n_jobs=1,\n    refit=False, error_score='raise'\n)\nsearch.fit(X_train, y_train)")
    para("Le pipeline empêche d'ajuster StandardScaler ou SMOTENC avant la séparation en folds. refit=False signifie qu'aucun modèle final automatique n'est conservé : les variantes sont ensuite entraînées explicitement avec les paramètres sélectionnés.")
    sub('B.3 Mesurer sans poursuivre l’apprentissage')
    code('model.eval()\nwith torch.no_grad():\n    logits = model(features)')
    para("eval() désactive notamment le Dropout. no_grad() évite de construire le graphe de calcul des gradients. Les états de chaque époque sont conservés pendant les expériences finales ; les courbes de test sont obtenues après choix de la variante, puis les poids du modèle retenu sont restaurés. Aucun score de test ne pilote les gradients, la patience ou la sélection.")
    sub('B.4 Arrêt anticipé : époques exécutées et retenues')
    rows = []
    for task, frame in [('Régression', rt), ('Classification', ct)]:
        for _, row in frame[frame.model.isin(['early_stopping', 'combined'])].iterrows():
            rows.append([task, row.model, int(row.epochs_run), int(row.selected_epoch)])
    story.append(table(['Tâche', 'Variante', 'Époques exécutées', 'Époque des poids retenus'], rows))
    para("Les dernières époques peuvent ne pas améliorer la validation. Après dix époques sans amélioration suffisante, l'entraînement s'arrête et recharge les poids de la meilleure loss de validation, pas nécessairement ceux de la dernière époque exécutée.")

    chapter('Annexe C — Sources, traçabilité et glossaire')
    para('Dépôt du projet : <link href="https://github.com/KarmasHammam/Atelier1_DeepLearning" color="#087E8B">github.com/KarmasHammam/Atelier1_DeepLearning</link>. Les notebooks, métriques détaillées et scripts complètent ce rapport.')
    para("Les archives Kaggle n'étaient pas accessibles depuis l'environnement utilisé. Des copies publiques attribuées aux jeux du sujet ont donc été utilisées. Leurs URL sont figées sur des commits. Le contrôle SHA-256 garantit la stabilité de ces copies, mais leur identité octet par octet avec les archives Kaggle n'a pas été vérifiée. Les données brutes ne sont pas redistribuées dans le dépôt.")
    for name, source in reg['sources'].items():
        sub(name)
        para(f"Source du sujet : <link href=\"{source['kaggle']}\" color=\"#087E8B\">{escape(source['kaggle'])}</link>")
        # Une URL longue peut dépasser la largeur : le lien reste actif et le
        # libellé est court ; le chemin complet est disponible dans summary.json.
        para(f"Copie utilisée : <link href=\"{source['url']}\" color=\"#087E8B\">téléchargement du fichier figé sur GitHub</link>.")
        digest = source['sha256']
        code('SHA-256 :\n' + digest[:32] + '\n' + digest[32:])
    sub('Documentation des outils')
    for label, url in [('PyTorch : modules et apprentissage', 'https://pytorch.org/docs/stable/index.html'),
                       ('scikit-learn : GridSearchCV', 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html'),
                       ('imbalanced-learn : SMOTENC', 'https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html')]:
        para(f'<link href="{url}" color="#087E8B">{label}</link>')
    sub('Petit glossaire')
    story.append(table(['Terme', 'Sens dans le projet'], [
        ['DNN / MLP', 'Réseau profond / perceptron multicouche à couches entièrement connectées'],
        ['Loss', 'Fonction d’erreur utilisée pour apprendre les poids'],
        ['Hyperparamètre', 'Réglage choisi avant l’apprentissage, par exemple le taux ou l’architecture'],
        ['Fold', 'Sous-partition utilisée pendant la validation croisée'],
        ['Surapprentissage', 'Meilleure adaptation au train qu’à de nouvelles observations'],
        ['Logit', 'Score brut avant transformation en probabilité de classe'],
        ['Fuite de données', 'Information du futur ou de l’évaluation utilisée pour entraîner ou choisir'],
    ], [125, WIDTH - 125]))
    doc = Report(OUTPUT)
    doc.multiBuild(story)
    print(OUTPUT)


if __name__ == '__main__':
    main()
