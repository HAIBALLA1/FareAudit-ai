from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from .config import PROJECT_ROOT


KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def load_data() -> list[Document]:
    documents = []

    # 1. Charger les fichiers Markdown
    md_files = sorted(KNOWLEDGE_DIR.rglob("*.md"))

    for file in md_files:
        text = file.read_text(encoding="utf-8")

        document = Document(
            page_content=text,
            metadata={
                "source": file.name,
                "source_path": file.relative_to(KNOWLEDGE_DIR).as_posix(),
                "file_type": "markdown",
            },
        )

        documents.append(document)

    # 2. Charger les fichiers PDF
    pdf_files = sorted(KNOWLEDGE_DIR.rglob("*.pdf"))

    for file in pdf_files:
        loader = PyPDFLoader(str(file))

        pages = loader.load()

        for page in pages:
            if not page.page_content.strip():
                continue

            page.metadata["source"] = file.name
            page.metadata["source_path"] = (
                file.relative_to(KNOWLEDGE_DIR).as_posix()
            )
            page.metadata["file_type"] = "pdf"

            documents.append(page)

    return documents

def split_documents(documents):

    # 4. Créer le découpeur
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    # 5. Découper tous les documents
    chunks = splitter.split_documents(
        documents
    )

    return chunks
