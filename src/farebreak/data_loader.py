import pyarrow.parquet as pq
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

    parquet = pq.ParquetFile(path)

    total_rows = parquet.metadata.num_rows
    n= min(sample_size, total_rows)

    rng=np.random.default_rng(seed)

    selected_row=np.sort(
        rng.choice(
            total_rows,
            size=n,
            replace=False
        )
    )

    parts=[]
    offset=0

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=READ_COLUMNS,
    ):
        # 8. Trouver les positions appartenant
        # au bloc courant
        first = np.searchsorted(
            selected_rows,
            offset
        )

        last = np.searchsorted(
            selected_rows,
            offset + batch.num_rows
        )

        # 9. Vérifier si le bloc contient
        # des lignes sélectionnées
        if last > first:
            # 10. Convertir les positions globales
            # en positions locales
            local_rows = (
                    selected_rows[first:last] - offset
            )

            # 11. Extraire les lignes du bloc
            selected_batch = batch.take(
                pa.array(local_rows)
            )

            # 12. Convertir en Pandas
            parts.append(
                selected_batch.to_pandas()
            )

        # 13. Passer au bloc suivant
        offset += batch.num_rows

        # 14. Regrouper tous les morceaux
    df = pd.concat(
        parts,
        ignore_index=True
    )

    return df