# Détection d'anomalies dans des ordres de transaction — Pipeline RPA → ML → LLM

Projet d'entraînement personnel réalisé en préparation d'un entretien pour un
stage/alternance en IA appliquée à la gestion d'actifs, au sein d'une équipe
Risques & Conformité.

## Contexte

Dans une société de gestion de portefeuille, les ordres de transaction
passés par les gestionnaires laissent une trace numérique répartie sur
plusieurs systèmes (front office, back office, courtiers). Repérer
manuellement des opérations suspectes dans ce volume de données est long
et sujet à l'erreur humaine.

Ce projet simule un pipeline simple permettant de :
1. Consolider des données réparties sur plusieurs systèmes
2. Détecter automatiquement les ordres statistiquement anormaux
3. Produire une explication en langage naturel exploitable par un analyste
   conformité

Les données utilisées sont **entièrement simulées** (aucune donnée réelle
ou confidentielle), avec 6 anomalies injectées volontairement pour valider
le pipeline.

## Méthodologie

### 1. RPA — Consolidation des sources (`data/`)

Trois fichiers CSV simulent trois systèmes distincts d'une société de
gestion :
- `systeme_front_office.csv` — les ordres tels que passés par les gestionnaires
- `systeme_back_office.csv` — le suivi d'exécution (statut, montant, heure)
- `broker_export.csv` — les frais facturés par les courtiers exécutants

Ces trois sources sont fusionnées sur la clé commune `id_ordre`, comme le
ferait un robot RPA chaque matin en production.

### 2. ML — Détection d'anomalies (Isolation Forest)

En l'absence d'exemples déjà étiquetés "fraude confirmée" (situation
courante en démarrage de projet, en particulier dans une structure de
taille réduite), le choix se porte sur un algorithme de **détection
d'outliers non supervisé** : l'Isolation Forest (scikit-learn).

Variables utilisées (feature engineering) :
- `montant_execute` — montant de la transaction
- `heure_du_jour` — heure de passage de l'ordre (un ordre à 2h du matin est
  atypique)
- `delai_execution_min` — délai entre passage et exécution
- `taux_frais` — cohérence entre frais facturés et montant de l'ordre
- `quantite` — quantité de titres échangés

### 3. LLM — Interprétation et priorisation

Les anomalies détectées par le modèle ML sont des outliers statistiques :
le modèle ne sait pas *pourquoi* c'est suspect, ni comment le présenter à
un humain. Cette étape est réalisée en soumettant les anomalies détectées
à un LLM (Claude), avec le prompt suivant :

> *"Tu es analyste conformité dans une société de gestion de portefeuille.
> Voici une liste d'ordres de transaction signalés comme anormaux par un
> modèle de détection d'outliers (Isolation Forest). Pour chacun, explique
> en une phrase pourquoi il pourrait être suspect, et classe-les par niveau
> de priorité (élevé/moyen/faible) pour une revue humaine."*

## Résultat

Sur 226 ordres simulés, le modèle détecte 7 anomalies, dont les 6 injectées
volontairement (montant hors norme, heure atypique, délai d'exécution
anormal, quantité extrême, taux de frais incohérent) et **1 faux positif**.

## Limites identifiées

- Le modèle compare chaque ordre à l'ensemble des clients confondus : un
  client dont le comportement habituel diffère naturellement des autres
  peut être signalé à tort. Une amélioration possible est de comparer
  chaque ordre à l'historique propre de son client (calcul d'un z-score
  par client) plutôt qu'à la moyenne globale.
- Le modèle ML priorise les anomalies statistiques, mais ne remplace pas
  la décision humaine : dans un contexte réglementé (AMF), l'IA sert à
  **alerter et prioriser**, jamais à décider seule.
- La détection d'un faux positif dans ce test confirme la nécessité d'une
  revue humaine systématique avant toute conclusion.

## Utilisation

```bash
pip install pandas scikit-learn
python analyse_anomalies.py
```

## Structure du repo

```
.
├── README.md
├── analyse_anomalies.py
└── data/
    ├── systeme_front_office.csv
    ├── systeme_back_office.csv
    └── broker_export.csv
```
