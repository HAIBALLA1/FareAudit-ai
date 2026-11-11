
def print_report(test_df):

    rules = test_df["anomaly_rules"]
    ml = test_df["anomaly_ml"]

    # Détectés par les deux méthodes
    both = rules & ml

    # Détectés uniquement par le ML
    only_ml = ml & ~rules

    # Détectés uniquement par les règles
    only_rules = rules & ~ml

    print("\n=== RESULTATS SUR LE TEST ===")

    print(
        "Nombre de trajets :",
        len(test_df)
    )

    print(
        "Anomalies règles :",
        rules.sum()
    )

    print(
        "Anomalies ML :",
        ml.sum()
    )

    print(
        "Règles ET ML :",
        both.sum()
    )

    print(
        "Seulement ML :",
        only_ml.sum()
    )

    print(
        "Seulement règles :",
        only_rules.sum()
    )

    # Colonnes à afficher
    columns = [
        "tpep_pickup_datetime",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "payment_label",
        "anomaly_score"
    ]

    print(
        "\n=== ANOMALIES UNIQUEMENT ML ==="
    )

    results = test_df.loc[
        only_ml,
        columns
    ]

    results = results.sort_values(
        "anomaly_score"
    )

    print(
        results.head(10).to_string(index=False)
    )
