from .vector_store import index_documents


def retriever():

    db=index_documents()

    retriever=db.as_retriever(search_kwargs={"k": 3})
    return retriever
