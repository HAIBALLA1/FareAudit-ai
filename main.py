from decimal import Decimal
from fastapi import FastAPI

app = FastAPI(
    title="FareBreak AI",
    version="0.1.0"
)

# Données fictives pour le premier test
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/investigations")
def investigations():
    results = []

    for transaction in transactions:
        expected = Decimal(transaction["expected"])
        received = Decimal(transaction["received"])

        difference = received - expected

        results.append({
            "batch_id": transaction["batch_id"],
            "expected": str(expected),
            "received": str(received),
            "difference": str(difference),
            "anomaly": difference != 0
        })

    return results
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)