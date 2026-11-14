
from langchain_chroma import Chroma

from .config import PROJECT_ROOT
from .document_loader import load_data, split_documents
from .embeddings import get_embeddings_model

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "fareaudit_knowledge"


def get_vector_store():

    model =get_embeddings_model()

    db= Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=model,
        persist_directory=str(CHROMA_DIR)
    )

    return db

def index_documents():
    documents = load_data()

    chunks= split_documents(documents)

    db =get_vector_store()

    ids=[f"chunk-{i}"  for i in range(len(chunks))]
    db.add_documents(chunks , ids=ids)

    return db