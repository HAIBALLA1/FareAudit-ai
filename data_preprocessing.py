from sklearn.ensemble import IsolationForest
import pandas as pd
import pyarrow.parquet as pq

# 1. Charger les données réelles
path = "data/raw/yellow_tripdata_2025-01.parquet"

columns = [
    "tpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "payment_type"
]

# Traduire les codes de paiement
payment_labels = {
    0: "Tarif flexible",
    1: "Carte bancaire",
    2: "Espèces",
    3: "Sans facturation",
    4: "Litige",
    5: "Inconnu",
    6: "Course annulée"
}

parquet = pq.ParquetFile(path)

batch = next(
    parquet.iter_batches(
        batch_size=100000,
        columns=columns
    )
)

df = batch.to_pandas()

print("Nombre de trajets :", len(df))

# 2. Vérifier les valeurs manquantes
print("\nValeurs manquantes :")
print(df.isna().sum())

# 3. Créer des indicateurs
df["negative_total"] = df["total_amount"] < 0

df["zero_distance"] = df["trip_distance"] == 0

df["high_amount"] = df["total_amount"] > 200

# 4. Extraire l'heure du trajet
df["pickup_hour"] = df[
    "tpep_pickup_datetime"
].dt.hour

# 5. Compter les cas à examiner
print("\nMontants négatifs :")
print(df["negative_total"].sum())

print("\nDistances nulles :")
print(df["zero_distance"].sum())

print("\nMontants supérieurs à 200 $ :")
print(df["high_amount"].sum())

# 6. Afficher les trajets signalés
flags = [
    "negative_total",
    "zero_distance",
    "high_amount"
]

anomalies = df[df[flags].any(axis=1)]

print("\nNombre de trajets signalés :", len(anomalies))

print("\nExemples :")
print(anomalies.head(10).to_string(index=False))

df["payment_label"] = df["payment_type"].map(payment_labels)

# Analyser les anomalies par type de paiement
stats = df.groupby("payment_label").agg(
    nombre_trajets=("total_amount", "size"),
    montants_negatifs=("negative_total", "sum"),
    distances_nulles=("zero_distance", "sum"),
    montants_eleves=("high_amount", "sum")
)

# Calculer le taux de montants négatifs
stats["taux_negatif_pct"] = (
    stats["montants_negatifs"]
    / stats["nombre_trajets"] * 100
).round(2)

print("\nAnalyse par type de paiement :")
print(stats.to_string())



# Variables destinées au Machine Learning
features = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "pickup_hour",
    "payment_type"
]

# Créer un DataFrame pour le ML
X = df[features].copy()

# Transformer le type de paiement en colonnes binaires
X = pd.get_dummies(
    X,
    columns=["payment_type"],
    dtype=int
)

print("\nFeatures du modèle :")
print(X.head())

print("\nDimensions :", X.shape)

print("\nColonnes :", X.columns.tolist())




# 1. Créer le modèle
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

# 2. Entraîner le modèle
model.fit(X)

# 3. Prédire les anomalies
predictions = model.predict(X)

# 4. Ajouter les résultats dans notre DataFrame
df["anomaly_ml"] = predictions == -1

# 5. Compter les anomalies détectées
print("\nAnomalies détectées par Isolation Forest :")
print(df["anomaly_ml"].sum())

# 6. Afficher les premières anomalies
anomalies_ml = df[df["anomaly_ml"]]

print("\nExemples d'anomalies détectées :")
print(
    anomalies_ml[
        [
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "total_amount",
            "payment_label"
        ]
    ].head(10).to_string(index=False)
)
