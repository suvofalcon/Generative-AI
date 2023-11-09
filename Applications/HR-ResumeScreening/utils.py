from pypdf import PdfReader
from langchain.schema import Document

# Extract information from the PDF file


def get_pdf_text(pdf_doc):

    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


# iterate over files in a list of user uploaded pdf files one by one
def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:

        chunks = get_pdf_text(filename)

        # adding item to our list and adding metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "id": filename.id, "type": filename.type,
                      "size": filename.size, "unique_id": unique_id}
        ))

    return docs
