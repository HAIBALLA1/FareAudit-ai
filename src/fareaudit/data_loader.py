
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    READ_COLUMNS,
    RANDOM_STATE,
    SAMPLE_SIZE,
)


def load_sample(
    path,
    sample_size=SAMPLE_SIZE,
    batch_size=100_000,
    seed=RANDOM_STATE,
):
    # 1. Vérifier que le fichier existe
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    # 2. Ouvrir le fichier Parquet
    parquet = pq.ParquetFile(path)

    # 3. Nombre total de lignes
    total = parquet.metadata.num_rows

    if total == 0:
        raise ValueError("Le fichier Parquet est vide")

    # 4. Nombre de lignes à sélectionner
    n = min(sample_size, total)

    # 5. Générateur aléatoire
    rng = np.random.default_rng(seed)

    # 6. Sélectionner des positions aléatoires
    # dans tout le dataset
    selected_rows = np.sort(
        rng.choice(
            total,
            size=n,
            replace=False
        )
    )

    # 7. Lire le fichier par blocs
    parts = []
    offset = 0

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=READ_COLUMNS
    ):

        # Positions sélectionnées dans ce bloc
        first = np.searchsorted(
            selected_rows,
            offset
        )

        last = np.searchsorted(
            selected_rows,
            offset + batch.num_rows
        )

        # Le bloc contient des lignes sélectionnées
        if last > first:

            # Convertir les positions globales
            # en positions locales
            local_rows = (
                selected_rows[first:last] - offset
            )

            # Extraire uniquement ces lignes
            selected_batch = batch.take(
                pa.array(local_rows)
            )

            # Convertir en Pandas
            parts.append(
                selected_batch.to_pandas()
            )

        # Passer au bloc suivant
        offset += batch.num_rows

    # 8. Assembler les blocs sélectionnés
    df = pd.concat(
        parts,
        ignore_index=True
    )

    return df
