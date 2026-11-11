from decimal import Decimal
from fastapi import FastAPI

# Import de notre API Machine Learning
from src.fareaudit.api import router as predict_router

# ==========================================
# 1. Création de l'application FastAPI
# ==========================================

app = FastAPI(
    title="FareAudit AI",
    description="API de détection d'anomalies financières sur les trajets de taxis",
    version="0.2.0"
)


# ==========================================
# 2. Intégration du modèle Isolation Forest
# ==========================================

# Ajoute automatiquement la route POST /predict
app.include_router(predict_router)


# ==========================================
# 3. Données fictives de démonstration
# ==========================================

transactions = [
    {
        "batch_id": "FB-001",
        "expected": "150.00",
        "received": "150.00"
    },
    {
        "batch_id": "FB-002",
        "expected": "200.00",
        "received": "185.00"
    },
    {
        "batch_id": "FB-003",
        "expected": "320.00",
        "received": "330.00"
    }
]


# ==========================================
# 4. Route de vérification de l'API
# ==========================================

@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "ok",
        "application": "FareAudit AI",
        "version": "0.2.0"
    }


# ==========================================
# 5. Route de rapprochement financier
# ==========================================

@app.get("/investigations", tags=["Investigations"])
def investigations():

    results = []

    for transaction in transactions:

        expected = Decimal(
            transaction["expected"]
        )

        received = Decimal(
            transaction["received"]
        )

        difference = received - expected

        results.append({
            "batch_id": transaction["batch_id"],
            "expected": str(expected),
            "received": str(received),
            "difference": str(difference),
            "anomaly": difference != 0
        })

    return results


# ==========================================
# 6. Démarrage du serveur
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
