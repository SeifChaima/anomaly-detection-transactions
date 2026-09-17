"""
Détection d'anomalies dans des ordres de transaction
Pipeline RPA -> ML -> LLM appliqué à un cas de gestion d'actifs / conformité.

Contexte :
Trois systèmes distincts (front office, back office, broker) exportent
chacun une partie de l'information sur les ordres de transaction passés
par des gestionnaires de portefeuille. L'objectif est de consolider ces
sources, détecter automatiquement les ordres statistiquement anormaux,
puis produire une explication exploitable par un analyste conformité.

Auteur : Chaima SEIF
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Étape 1 — RPA : consolidation des sources
# ---------------------------------------------------------------------------
# Simule ce qu'un robot RPA ferait chaque matin : récupérer des exports
# venant de systèmes différents et les assembler en une table unique.

def charger_et_consolider(chemin_data="data"):
    front = pd.read_csv(f"{chemin_data}/systeme_front_office.csv")
    back = pd.read_csv(f"{chemin_data}/systeme_back_office.csv")
    broker = pd.read_csv(f"{chemin_data}/broker_export.csv")

    data = front.merge(back, on="id_ordre").merge(broker, on="id_ordre")
    return data


# ---------------------------------------------------------------------------
# Étape 2 — ML : feature engineering + détection d'anomalies
# ---------------------------------------------------------------------------
# Isolation Forest : algorithme de détection d'outliers non supervisé,
# choisi car on ne dispose pas d'exemples déjà étiquetés "fraude confirmée".

def construire_variables(data):
    data["date_passage"] = pd.to_datetime(data["date_passage"])
    data["heure_execution"] = pd.to_datetime(data["heure_execution"])

    data["heure_du_jour"] = data["date_passage"].dt.hour
    data["delai_execution_min"] = (
        data["heure_execution"] - data["date_passage"]
    ).dt.total_seconds() / 60
    data["taux_frais"] = data["frais"] / data["montant_execute"]

    return data


def detecter_anomalies(data, contamination=0.03):
    features = ["montant_execute", "heure_du_jour", "delai_execution_min", "taux_frais", "quantite"]

    modele = IsolationForest(contamination=contamination, random_state=42)
    data["anomalie"] = modele.fit_predict(data[features])

    return data


def comparer_avec_historique_client(data):
    """Pour chaque anomalie détectée, affiche l'ordre suspect à côté de
    l'historique 'normal' du même client, pour une revue humaine rapide."""
    anomalies = data[data["anomalie"] == -1]

    for id_ordre in anomalies["id_ordre"]:
        client = data.loc[data["id_ordre"] == id_ordre, "client"].values[0]

        print(f"\n=== Anomalie {id_ordre} — client {client} ===")
        print("\nOrdre suspect :")
        print(data[data["id_ordre"] == id_ordre][
            ["id_ordre", "instrument", "sens", "montant_execute", "heure_du_jour"]
        ])

        print("\nHistorique habituel de ce client (hors anomalies) :")
        historique = data[(data["client"] == client) & (data["anomalie"] == 1)]
        print(historique[["id_ordre", "instrument", "sens", "montant_execute", "heure_du_jour"]])


# ---------------------------------------------------------------------------
# Étape 3 — LLM : préparation du texte pour analyse
# ---------------------------------------------------------------------------
# L'étape LLM elle-même n'est pas automatisée ici : les anomalies détectées
# sont formatées en texte, à soumettre à un LLM (ex: Claude) pour obtenir
# une explication en langage naturel et une priorisation pour la revue
# humaine. Voir README.md pour le prompt utilisé.

def exporter_anomalies_pour_llm(data):
    anomalies = data[data["anomalie"] == -1].sort_values("montant_execute", ascending=False)
    colonnes = ["id_ordre", "client", "gestionnaire", "instrument", "sens",
                "montant_execute", "heure_du_jour", "delai_execution_min", "taux_frais"]
    return anomalies[colonnes].to_string(index=False)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = charger_et_consolider()
    data = construire_variables(data)
    data = detecter_anomalies(data)

    print(f"{(data['anomalie'] == -1).sum()} anomalie(s) détectée(s) sur {len(data)} ordres.\n")

    comparer_avec_historique_client(data)

    print("\n\n--- Texte prêt à soumettre à un LLM pour analyse ---\n")
    print(exporter_anomalies_pour_llm(data))

