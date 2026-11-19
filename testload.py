from src.fareaudit.document_loader import load_data, split_documents

documents = load_data()

for doc in documents:
    print("Source :", doc.metadata.get("source"))
    print("Page :", doc.metadata.get("page"))
    print("Taille :", len(doc.page_content))
    print("-" * 40)

chunks = split_documents(documents)

print("Documents LangChain :", len(documents))
print("Chunks générés :", len(chunks))