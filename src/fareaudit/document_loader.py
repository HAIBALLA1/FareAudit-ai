from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import PROJECT_ROOT


KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def load_data():
    documents = []

    files=sorted(KNOWLEDGE_DIR.glob("*.md"))

    for file in files:

        text=file.read_text(encoding="utf-8")

        document=Document(
            page_content=text,
            metadata={
                "source":file.name
            }
        )
        documents.append(document)

    return documents

def split_documents(documents):

    # 4. Créer le découpeur
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    # 5. Découper tous les documents
    chunks = splitter.split_documents(
        documents
    )

    return chunks
