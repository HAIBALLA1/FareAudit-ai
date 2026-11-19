
from pathlib import Path
import re
import unicodedata

from src.fareaudit.vector_store import get_vector_store


# ============================================================
# 1. JEU DE TEST : 24 QUESTIONS
# ============================================================

TESTS = [
    # Distance et localisation
    {
        "q": "Que représente trip_distance ?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "elapsed trip distance in miles",
    },
    {
        "q": "What does trip_distance measure?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "elapsed trip distance in miles",
    },
    {
        "q": "Que signifie PULocationID ?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "taximeter was engaged",
    },
    {
        "q": "What does DOLocationID represent?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "taximeter was disengaged",
    },

    # RatecodeID
    {
        "q": "Quel RatecodeID correspond au tarif standard ?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "1 = Standard rate",
    },
    {
        "q": "Quel code tarifaire correspond à JFK ?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "2 = JFK",
    },
    {
        "q": "Which RatecodeID corresponds to Newark?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "3 = Newark",
    },
    {
        "q": "Quel RatecodeID correspond à Nassau ou Westchester ?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "4 = Nassau or Westchester",
    },
    {
        "q": "What is the RatecodeID for a group ride?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "6 = Group ride",
    },
    {
        "q": "Quel RatecodeID signifie tarif inconnu ?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "99 = Null/unknown",
    },

    # Stockage des trajets
    {
        "q": "Que signifie store_and_fwd_flag = Y ?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "Y = store and forward trip",
    },
    {
        "q": "What does store_and_fwd_flag = N mean?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "N = not a store and forward trip",
    },

    # Types de paiement
    {
        "q": "Que signifie payment_type = 4 ?",
        "source": "payment_types.md",
        "answer": "4 = Dispute",
    },
    {
        "q": "Quel code correspond à un paiement en espèces ?",
        "source": "payment_types.md",
        "answer": "2 = Cash",
    },
    {
        "q": "Quel code signifie une course sans facturation ?",
        "source": "payment_types.md",
        "answer": "3 = No charge",
    },
    {
        "q": "What payment type indicates an unknown payment?",
        "source": "payment_types.md",
        "answer": "5 = Unknown",
    },
    {
        "q": "Quel code correspond à un trajet annulé ?",
        "source": "payment_types.md",
        "answer": "6 = Voided trip",
    },
    {
        "q": "What does payment_type = 0 mean?",
        "source": "payment_types.md",
        "answer": "0 = Flex Fare trip",
    },

    # Champs financiers
    {
        "q": "Comment est calculé fare_amount ?",
        "source": "financial_fields.md",
        "answer": "montant calculé par le taximètre",
    },
    {
        "q": "Que représente tip_amount ?",
        "source": "financial_fields.md",
        "answer": "pourboires enregistrés",
    },
    {
        "q": "Are cash tips included in tip_amount?",
        "source": "financial_fields.md",
        "answer": "pourboires payés en espèces ne sont pas inclus",
    },
    {
        "q": "Que signifie total_amount ?",
        "source": "financial_fields.md",
        "answer": "montant total facturé",
    },
    {
        "q": "What does total_amount exclude?",
        "source": "yellow_taxi_data_dictionary.pdf",
        "answer": "Does not include cash tips",
    },

    # Guide utilisateur
    {
        "q": "Quelles informations sont présentes dans les données de trajets ?",
        "source": "trip_record_user_guide.pdf",
        "answer": "trip distances, itemized fares",
    },
]


# ============================================================
# 2. NORMALISATION DU TEXTE
# ============================================================

def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def is_relevant(doc, test):
    source = Path(
        str(doc.metadata.get("source", ""))
    ).name

    return (
        source == test["source"]
        and normalize(test["answer"])
        in normalize(doc.page_content)
    )


# ============================================================
# 3. INITIALISATION CHROMADB
# ============================================================

db = get_vector_store()

CONFIGS = {
    "Similarity k=3": db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    ),

    "Similarity k=5": db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    ),

    "Similarity k=7": db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7},
    ),

    "MMR k=5": db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "lambda_mult": 0.8,
        },
    ),
}


# ============================================================
# 4. ÉVALUATION
# ============================================================

def evaluate(name, retriever):

    hits = {
        1: 0,
        3: 0,
        5: 0,
        7: 0,
    }

    reciprocal_ranks = []
    ranks = []
    failures = []

    total = len(TESTS)

    print(f"\n{'=' * 70}")
    print(f"CONFIGURATION : {name}")
    print(f"{'=' * 70}")

    for i, test in enumerate(TESTS, 1):

        question = test["q"]
        docs = retriever.invoke(question)

        # Identifier les positions des chunks pertinents
        relevant_ranks = [
            rank
            for rank, doc in enumerate(docs, 1)
            if is_relevant(doc, test)
        ]

        first_rank = (
            relevant_ranks[0]
            if relevant_ranks
            else None
        )

        ranks.append(first_rank)

        # Calcul Hit@K
        for k in hits:
            if first_rank is not None and first_rank <= k:
                hits[k] += 1

        # Calcul MRR
        rr = 1 / first_rank if first_rank is not None else 0
        reciprocal_ranks.append(rr)

        # Affichage des résultats
        print(f"\nQUESTION {i:02d} : {question}")

        for rank, doc in enumerate(docs, 1):

            relevant = is_relevant(doc, test)
            status = "OK" if relevant else "--"

            source = Path(
                str(doc.metadata.get("source", ""))
            ).name

            print(
                f"  {rank}. [{status}] {source}"
            )

        print(
            "  Premier rang pertinent :",
            first_rank if first_rank is not None else "Absent"
        )

        if first_rank is None:
            failures.append({
                "question": question,
                "source": test["source"],
                "answer": test["answer"],
                "docs": docs,
            })

    # ========================================================
    # 5. CALCUL DES MÉTRIQUES
    # ========================================================

    scores = {
        "Hit@1": hits[1] / total,
        "Hit@3": hits[3] / total,
        "Hit@5": hits[5] / total,
        "Hit@7": hits[7] / total,
        "MRR": sum(reciprocal_ranks) / total,
    }

    print(f"\n===== SCORES : {name} =====")

    print(f"Hit@1 : {scores['Hit@1']:.2%}")
    print(f"Hit@3 : {scores['Hit@3']:.2%}")
    print(f"Hit@5 : {scores['Hit@5']:.2%}")
    print(f"Hit@7 : {scores['Hit@7']:.2%}")
    print(f"MRR   : {scores['MRR']:.4f}")

    # ========================================================
    # 6. ANALYSE DES ÉCHECS
    # ========================================================

    print("\n===== QUESTIONS ÉCHOUÉES =====")

    if not failures:
        print("Aucun échec.")

    for failure in failures:

        print("\nQuestion :", failure["question"])
        print("Source attendue :", failure["source"])
        print("Réponse attendue :", failure["answer"])

        for rank, doc in enumerate(failure["docs"], 1):

            source = Path(
                str(doc.metadata.get("source", ""))
            ).name

            print(f"  {rank}. {source}")
            print(
                "     Contenu :",
                normalize(doc.page_content)[:200]
            )

    return scores, ranks


# ============================================================
# 7. COMPARAISON GLOBALE
# ============================================================

def main():

    results = {}
    all_ranks = {}

    for name, retriever in CONFIGS.items():

        scores, ranks = evaluate(
            name,
            retriever
        )

        results[name] = scores
        all_ranks[name] = ranks

    print("\n" + "=" * 85)
    print("COMPARAISON FINALE")
    print("=" * 85)

    print(
        f"{'Configuration':<20}"
        f"{'Hit@1':<12}"
        f"{'Hit@3':<12}"
        f"{'Hit@5':<12}"
        f"{'Hit@7':<12}"
        f"{'MRR':<10}"
    )

    for name, scores in results.items():

        print(
            f"{name:<20}"
            f"{scores['Hit@1']:<12.2%}"
            f"{scores['Hit@3']:<12.2%}"
            f"{scores['Hit@5']:<12.2%}"
            f"{scores['Hit@7']:<12.2%}"
            f"{scores['MRR']:<10.4f}"
        )

    # ========================================================
    # 8. COMPARAISON QUESTION PAR QUESTION
    # ========================================================

    print("\n===== COMPARAISON QUESTION PAR QUESTION =====")

    for i, test in enumerate(TESTS):

        print(f"\n{i+1:02d}. {test['q']}")

        for name in CONFIGS:

            rank = all_ranks[name][i]

            print(
                f"  {name:<20}: "
                f"{rank if rank is not None else 'Absent'}"
            )

    # ========================================================
    # 9. MEILLEURE CONFIGURATION
    # ========================================================

    # Comparaison équitable sur les 3 premiers résultats
    best = max(
        results,
        key=lambda name: (
            results[name]["Hit@3"],
            results[name]["MRR"],
        )
    )

    print("\n" + "=" * 70)
    print("MEILLEURE CONFIGURATION SUR HIT@3 ET MRR :", best)
    print("=" * 70)


if __name__ == "__main__":
    main()
