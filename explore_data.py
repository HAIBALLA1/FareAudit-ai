
import pandas as pd
import pyarrow.parquet as pq

# 1. Emplacement du fichier
path = "data/raw/yellow_tripdata_2025-01.parquet"

# 2. Ouvrir le fichier Parquet
parquet = pq.ParquetFile(path)

# 3. Nombre total de lignes
print("Nombre total de trajets :", parquet.metadata.num_rows)

# 4. Afficher toutes les colonnes
print("\nColonnes disponibles :")
print(parquet.schema_arrow.names)

# 5. Sélectionner les colonnes financières
columns = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "payment_type"
]

# 6. Lire seulement 10 000 lignes
batch = next(
    parquet.iter_batches(
        batch_size=10000,
        columns=columns
    )
)

# 7. Convertir en DataFrame Pandas
df = batch.to_pandas()

print("\nDimensions :", df.shape)

print("\n5 premiers trajets :")
print(df.head())

print("\nStatistiques financières :")
print(df.describe())

print("\nValeurs manquantes :")
print(df.isna().sum())
