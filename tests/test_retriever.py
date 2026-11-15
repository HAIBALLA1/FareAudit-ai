from src.fareaudit.retriever import retriever

retriever=retriever()

docs=retriever.invoke( "Pourquoi un paiement est-il contesté ?")

for doc in docs:
    print(doc.page_content)