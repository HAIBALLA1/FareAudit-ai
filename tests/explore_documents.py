from src.fareaudit.document_loader import (
    load_data,
    split_documents
)


documents = load_data()

print("doc:", len(documents))

for document in documents:
    print("source:" , document.metadata["source"])


chunks = split_documents(documents)

print("chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3] , start=1):
    print(f"\n===== CHUNK {i} =====")

    print("Source :", chunk.metadata["source"])

    print("Taille :", len(chunk.page_content))

    print("Contenu :")
    print(chunk.page_content)

