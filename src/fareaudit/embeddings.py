from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from .document_loader import load_data , split_documents

@lru_cache(maxsize=1)
def get_embeddings_model():

    # 1. Charger le modèle pré-entraîné
    model = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return model

#cette fonction je l'utilise pas vu que j'utilise desormais CHromDM
#qui prends en charge l'embeding elle meme avec  la classe Chroma
@lru_cache(maxsize=1)
def doc_embddings():
    documents= load_data()
    chunks = split_documents(documents)

    texts=[chunk.page_content for chunk in chunks]

    model=get_embeddings_model()

    document_vectors= model.embed_documents(texts)

    return chunks , document_vectors


