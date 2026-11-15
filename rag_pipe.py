from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.fareaudit.embeddings import get_embeddings_model
from .config import PROJECT_ROOT


CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "fareaudit_knowledge"

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

def load_files():

    files=sorted(KNOWLEDGE_DIR.glob("*.md"))

    return files


#creer des docs langchain a l'aide du text brute et returner un tableua de tous les docs
def save_document():
    files=load_files()
    docs=[]

    for file in files:

        text=file.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=text
        ))

    return docs

#couper les docs
def splite_docs():

    docs=save_document()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300 ,
        chunk_overlap = 50
    )

    chunks=splitter.split_documents(docs)

    return chunks

#ouvrir une interface de connexion avec notre data base
def chroma_cnx():

    model= get_embeddings_model()
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=model,
        persist_directory=str(CHROMA_DIR)

    )

    return db

#indexation et le stockage des vecteur
def save_embeddings_vectors():

    db=chroma_cnx()
    chunks=splite_docs()

    ids=[f"chunk_{i}" for i in range(len(chunks))]

    db.add_documents(chunks, ids)

    return db

#responsable de quesionner note data base via db
def ivoke_retrievial(query):

    db=chroma_cnx()

    retrieve=db.as_retriever(search_kwargs={"k": 3})

    result=retrieve.invoke(query)

    return result
