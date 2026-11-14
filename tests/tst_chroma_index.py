from fareaudit.vector_store import index_documents

retriever=index_documents()
docs = retriever.invoke(
    "Pourquoi un paiement est-il contesté ?"
)
print(type(retriever))

