
from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Emplacements des fichiers
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2025-01.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "isolation_forest.joblib"
)

# Paramètres
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Colonnes à lire depuis le Parquet
READ_COLUMNS = [
    "tpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "payment_type",
]

# Variables pour le Machine Learning
MODEL_FEATURES = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "pickup_hour",
    "payment_type",
]

# Signification des types de paiement
PAYMENT_LABELS = {
    0: "Tarif flexible",
    1: "Carte bancaire",
    2: "Espèces",
    3: "Sans facturation",
    4: "Litige",
    5: "Inconnu",
    6: "Course annulée",
}
