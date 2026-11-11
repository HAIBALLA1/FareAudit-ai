from pathlib import Path

# Racine du projet FareAudit-ai
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chemin du dataset Parquet
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2025-01.parquet"
)

# Chemin de sauvegarde du modèle
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest.joblib"
)

# Paramètres du pipeline
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Période d'analyse
START_DATE = "2025-01-01"
END_DATE = "2025-02-01"

# Colonnes à charger avec PyArrow
READ_COLUMNS = [
    "tpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "payment_type",
]

# Variables utilisées par Isolation Forest
MODEL_FEATURES = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "pickup_hour",
    "payment_type",
]

# Types de paiement
PAYMENT_LABELS = {
    0: "Tarif flexible",
    1: "Carte bancaire",
    2: "Espèces",
    3: "Sans facturation",
    4: "Litige",
    5: "Inconnu",
    6: "Course annulée",
}
