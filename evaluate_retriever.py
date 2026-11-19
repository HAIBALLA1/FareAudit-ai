
from src.fareaudit.retriever import retriever

"""
EVALUATION_DATA = [
    {
        "question": "Que signifie payment_type = 4 ?",
        "expected_source": "payment_types.md",
    },
    {
        "question": "Quel code correspond à un paiement en espèces ?",
        "expected_source": "payment_types.md",
    },
    {
        "question": "Que signifie payment_type = 6 ?",
        "expected_source": "payment_types.md",
    },
    {
        "question": "Une course sans facturation est-elle forcément frauduleuse ?",
        "expected_source": "payment_types.md",
    },
    {
        "question": "Que représente fare_amount ?",
        "expected_source": "financial_fields.md",
    },
    {
        "question": "Est-ce que tip_amount inclut les pourboires en espèces ?",
        "expected_source": "financial_fields.md",
    },
    {
        "question": "Dans quelle unité est exprimé trip_distance ?",
        "expected_source": "financial_fields.md",
    },
    {
        "question": "Le montant total inclut-il les pourboires en espèces ?",
        "expected_source": "financial_fields.md",
    },
    {
        "question": "Une course avec une distance nulle est-elle incorrecte ?",
        "expected_source": "investigation_rules.md",
    },
    {
        "question": "Que signifie un score decision_function négatif ?",
        "expected_source": "investigation_rules.md",
    },
    {
        "question": "Le score d'anomalie représente-t-il une probabilité de fraude ?",
        "expected_source": "investigation_rules.md",
    },
    {
        "question": "Un montant total supérieur à 200 dollars signifie-t-il une fraude ?",
        "expected_source": "investigation_rules.md",
    },
]


def source_in_top_k(documents, expected_source: str, k: int = 3) -> bool:
    return any(
        doc.metadata.get("source") == expected_source
        for doc in documents[:k]
    )


def evaluate_retriever():
    r = retriever()

    hits_1 = 0
    hits_3 = 0

    for case in EVALUATION_DATA:
        question = case["question"]
        expected_source = case["expected_source"]

        documents = r.invoke(question)

        hit_1 = source_in_top_k(documents, expected_source, k=1)
        hit_3 = source_in_top_k(documents, expected_source, k=3)

        hits_1 += int(hit_1)
        hits_3 += int(hit_3)

        print(f"\nQuestion : {question}")
        print(f"Hit@1 : {'PASS' if hit_1 else 'FAIL'}")
        print(f"Hit@3 : {'PASS' if hit_3 else 'FAIL'}")

    total = len(EVALUATION_DATA)

    print(f"\nHit@1 : {hits_1 / total:.2%}")
    print(f"Hit@3 : {hits_3 / total:.2%}")

if __name__ == "__main__":
    evaluate_retriever()



r = retriever()

question = "Que représente fare_amount ?"
documents = r.invoke(question)

for i, doc in enumerate(documents, start=1):
    print(f"\n===== CHUNK {i} =====")
    print("Source :", doc.metadata.get("source"))
    print("Contenu :", doc.page_content)

"""

questions = [
    "Une course sans facturation est-elle forcément frauduleuse ?",
    "Dans quelle unité est exprimé trip_distance ?",
    "Un montant total supérieur à 200 dollars signifie-t-il une fraude ?",
]

r = retriever()

for question in questions:
    print(f"\nQUESTION : {question}")

    documents = r.invoke(question)

    for i, doc in enumerate(documents, start=1):
        print(f"\nCHUNK {i}")
        print("Source :", doc.metadata.get("source"))
        print("Contenu :", doc.page_content)