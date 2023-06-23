import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders import PlaywrightURLLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from pprint import pprint

os.environ["OPENAI_API_KEY"] = ''

collection_name = 'collection1'
local_directory = 'local_dir'
persist_directory = os.path.join(os.getcwd(), local_directory)

# loader = UnstructuredURLLoader(urls=urls, continue_on_failure=False)
# loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])

loader = DirectoryLoader('./docs')
# loader = UnstructuredHTMLLoader("html/")
kb_data = loader.load()

text_splitter = TokenTextSplitter(chunk_size=4000, chunk_overlap=0)
kb_doc = text_splitter.split_documents(kb_data)
print(kb_doc)
#
embeddings = OpenAIEmbeddings()

kb_db = Chroma.from_documents(kb_doc, embeddings, collection_name=collection_name,
                              persist_directory=persist_directory)

kb_db.persist()

kb_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
    VectorStoreRetriever(vectorstore=kb_db),
    verbose=True,
    max_tokens_limit=4000,
    return_source_documents=False)

# kb_qa = ChatVectorDBChain.from_llm(
#     ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', max_tokens=4000),
#     # OpenAI(temperature=0, model_name='gpt-3.5-turbo'),
#     vectorstore=kb_db,
#     top_k_docs_for_context=5,
#     return_source_documents=False,
# )

chat_history = []
query_statement = "Ask a question here"

while query_statement != 'exit':
    query_statement = input('Enter your question here: > ')
    if query_statement != exit:
        result = kb_qa({"question": query_statement, "chat_history": chat_history})
        pprint(result)

os.environ["OPENAI_API_KEY"] = ''
