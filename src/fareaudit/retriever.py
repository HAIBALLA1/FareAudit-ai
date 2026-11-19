from .vector_store import get_vector_store


def retriever():

    db=get_vector_store()

    retriever=db.as_retriever( search_type="mmr",
                                search_kwargs={
                                    "k": 3,
                                    "fetch_k": 10,
                                    "lambda_mult": 0.8
                                })

    return retriever
