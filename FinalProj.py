#Manage Imports

#llamaindex stuff
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.schema import IndexNode
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import (
    Settings,
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
)
from llama_index.llms.groq import Groq
#Document Readers
import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

#Initalize LLM and Embedding Model

#Using Groq Temporarily due to low computer ram.
import os
from dotenv import load_dotenv
load_dotenv()
llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=(os.getenv("GROQ_API_KEY")))
#llm = Ollama(model="llama3.2:latest", temperature=0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#Settings the models to be used
Settings.llm = llm
Settings.embed_model = embed_model
#Settings for the chunk size and overlap for efficient embedding at VectorStoreIndex
Settings.chunk_size = 512
Settings.chunk_overlap = 30

debug = True


#Document Processing
file_name = input(" Enter the (enter) file name inside the document directory: ")
document = '~/Workspace/COS243/LocalLib/documents/' + file_name #adjusted accordingly

#Ebooks(Text)

if file_name.lower().endswith('.epub'):
    if debug:
        print("Loading Ebook through Epub Reader.")
    book = epub.read_epub(document)
    #Initialize list to store all the text
    all_text = []
    if debug and book.get_metadata('DC', 'title'):
        print(f"Loaded Ebook of Title : {book.get_metadata('DC', 'title')[0][0]}\n\n")
    elif debug: 
        print("Loading Unssuccessful\n\n")

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        #Parse it with soup
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        text = soup.get_text()

        all_text.append(text)
    
    #Creating Document Object so it can be embedded
    full_text = [Document(text=t) for t in all_text]
    if debug and full_text:
        print("Converted Document Object\n\n")
    elif debug: 
        print("Convertion Unssuccessful\n\n")

#More efficient document parsing modules can be added in the future

#General Document Loading
else:
    if debug:
        print('Loading Document Object through SimpleDirectoryReader.')
    full_text = SimpleDirectoryReader(input_files=[document]).load_data()
    if debug and full_text[0].metadata:
        print(f"Loaded Document of Title : {full_text[0].metadata['file_name']}\n\n")
    elif debug: 
        print("Loading Unssuccessful\n\n")

#Embedding the Document
vector_index = VectorStoreIndex.from_documents(full_text,show_progress=debug)
#for summary purpose
summary_index = SummaryIndex.from_documents(full_text,show_progress=debug)

#Define Query Engines
vector_query_engine = vector_index.as_query_engine()
summary_query_engine = summary_index.as_query_engine()

# define tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="vector_tool",
            description=(
                f"Useful for retrieving specific context from {file_name}"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=summary_query_engine,
        metadata=ToolMetadata(
            name="summary_tool",
            description=(
                f"Useful for summarization questions related to {file_name}"
            ),
        ),
    ),
]

#Build document specific agent
if debug:
    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm = llm,
        verbose=True,
    )
else:
    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm = llm,
        verbose=False,
    )

#Index nodes for future multi-doc module
objects = []

file_summary = (
    "Use this index if you need to lookup specific facts about"
    f" {file_name}."
)

node = IndexNode(
    text=file_summary, index_id=file_name, obj=agent
)
objects.append(node)
node_index = VectorStoreIndex(
    objects=objects,
)
#Force citation
query_engine = CitationQueryEngine.from_args(
    index=node_index,
    similarity_top_k = 2,
    citation_chunk_size=512,
    citation_chunk_overlap=20,
    show_progress=debug
)


response = query_engine.query("explainthe first three topics")

print("\n\n",response)