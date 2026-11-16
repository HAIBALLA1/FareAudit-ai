from ..retriever import retriever

def get_retriever(query):

    documents=[]
    r=retriever()

    results= r.invoke(query)

    for doc in results:
        document={
            "content": doc.page_content,
            "source": doc.metadata.get("source")
        }
        documents.append(document)


    return documents
