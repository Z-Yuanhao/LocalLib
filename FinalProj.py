import os
import gradio as gr
from dotenv import load_dotenv
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import docx2txt
from llama_index.core import (
    Settings,
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
#Groq used during dev because of low computer performance
#from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.schema import IndexNode

# Load environment variables
load_dotenv()

# Initialize global variables
vector_query_engine = None
query_engine = None
debug = True

# LLM and embedding model initialization
llm = Ollama(model="llama3.2:latest", temperature=0)
#llm = Groq(
#    model="llama3-groq-70b-8192-tool-use-preview",
#    api_key=os.getenv("GROQ_API_KEY")
#)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 30

def process_file(file_path):
    """
    Reads and processes an uploaded document.
    Supports EPUB, DOCX, and general documents.
    """
    if debug:
        print(f"Processing file: {file_path}")
    
    if file_path.lower().endswith('.epub'):
        # EPUB-specific processing
        if debug:
            print("Loading EPUB file through Epub Reader.")
        try:
            book = epub.read_epub(file_path)
            all_text = []
            
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                all_text.append(soup.get_text())
            
            full_text = [Document(text=t) for t in all_text]
            if debug:
                print("EPUB successfully processed.")
            return full_text, "EPUB successfully processed."
        
        except Exception as e:
            if debug:
                print(f"Error processing EPUB: {str(e)}")
            return None, f"Error processing EPUB: {str(e)}"
    
    elif file_path.lower().endswith('.docx'):
        # DOCX-specific processing using docx2txt
        if debug:
            print("Loading DOCX file through docx2txt.")
        try:
            all_text = docx2txt.process(file_path)
            full_text = [Document(text=all_text)]
            if debug:
                print("DOCX successfully processed.")
            return full_text, "DOCX successfully processed."
        
        except Exception as e:
            if debug:
                print(f"Error processing DOCX: {str(e)}")
            return None, f"Error processing DOCX: {str(e)}"
    
    else:
        # General document loading
        if debug:
            print("Loading Document Object through SimpleDirectoryReader.")
        try:
            full_text = SimpleDirectoryReader(input_files=[file_path]).load_data()
            if debug and full_text and hasattr(full_text[0], "metadata"):
                print(f"Loaded Document of Title: {full_text[0].metadata.get('file_name', 'Unknown Title')}\n\n")
            elif debug:
                print("Loading unsuccessful or metadata unavailable.\n\n")
            return full_text, "General document successfully processed."
        
        except Exception as e:
            if debug:
                print(f"Error processing document: {str(e)}")
            return None, f"Error processing document: {str(e)}"



def create_index(full_text, file_name):
    """
    Embeds the document content, initializes the query engine, and builds tools.
    """
    global vector_query_engine, query_engine
    
    try:
        if debug:
            print("Creating vector index...")
        
        # Create vector index
        vector_index = VectorStoreIndex.from_documents(full_text, show_progress=debug)
        vector_query_engine = vector_index.as_query_engine()

        # Create tools and agents
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=f"Useful for retrieving information from {file_name}.",
                ),
            ),
        ]
        
        agent = ReActAgent.from_tools(
            query_engine_tools,
            llm=llm,
            verbose=debug,
        )
        
        # Create citation query engine
        objects = []
        file_summary = f"Use this index if you need to lookup specific facts about {file_name}."
        node = IndexNode(text=file_summary, index_id=file_name, obj=agent)
        objects.append(node)
        node_index = VectorStoreIndex(objects=objects)
        
        query_engine = CitationQueryEngine.from_args(
            index=node_index,
            similarity_top_k=4,
            citation_chunk_size=512,
            citation_chunk_overlap=20,
        )
        
        if debug:
            print("Index and query engine successfully created.")
        return "Index and query engine successfully created."
    
    except Exception as e:
        if debug:
            print(f"Error creating index: {str(e)}")
        return f"Error creating index: {str(e)}"



with gr.Blocks() as interface:
    gr.Markdown("# Document Query Interface")

    # Export conversation history
    def export_history():
        global history
        export_content = "\n\n".join([f"You: {q}\nAI: {r}" for q, r in history])
        return export_content

    export_button = gr.Button("Export Conversation History")

    # File upload and processing
    file_input = gr.File(label="Upload Document")
    file_status = gr.Textbox(label="File Processing Status", interactive=False)

    # Conversation history in chat format
    conversation_history = gr.Chatbot(label="Conversation History")

    # Query interface
    query_input = gr.Textbox(
        label="Enter your query",
        placeholder="Type your question and press Enter",
    )
    query_button = gr.Button("Submit Query")

    # Pre-made prompt submission
    premade_prompts = gr.Dropdown(
        choices=["Summarize the document.", "What happened in the first two chapters?", "Explain the title."],
        label="Select a pre-made prompt",
        interactive=True,
    )
    premade_button = gr.Button("Submit Pre-made Prompt")

    # Global state for history
    history = []  # Chat history

    # Workflow for file processing
    def workflow(file):
        full_text, status = process_file(file.name)
        if full_text:
            index_status = create_index(full_text, file.name)
            return status + "\n" + index_status
        return status

    # Query handling with chat-like history update and clearing input
    def handle_query_with_chat_history(query):
        global history

        if query_engine is None:
            response = "No document is loaded. Please upload and process a file first."
        else:
            try:
                # Pass history as part of context for the LLM
                conversation_context = "\n".join(
                    [f"You: {q}\nAI: {r}" for q, r in history]
                )
                full_prompt = f"{conversation_context}\nYou: {query}\nAI:"
                response = str(query_engine.query(full_prompt))
            except Exception as e:
                response = f"Error during query execution: {str(e)}"
        
        # Update conversation history
        history.append((query, response))
        return history, ""  # Return updated history and clear query input

    # Pre-made prompt submission handler
    def handle_premade_prompt(prompt):
        if prompt:
            return handle_query_with_chat_history(prompt)
        return history, ""
  
    def export_history_with_title():
        global history
        export_content = "\n\n".join([f"You: {q}\nAI: {r}" for q, r in history])
        file_path = "conversation_history.txt"  # Set your custom file title here
        with open(file_path, "w") as f:
            f.write(export_content)
        return file_path

    # Gradio actions
    file_input.change(workflow, inputs=[file_input], outputs=[file_status])
    query_input.submit(handle_query_with_chat_history, inputs=[query_input], outputs=[conversation_history, query_input])
    query_button.click(handle_query_with_chat_history, inputs=[query_input], outputs=[conversation_history, query_input])
    premade_button.click(handle_premade_prompt, inputs=[premade_prompts], outputs=[conversation_history, query_input])
    export_button.click(export_history_with_title, inputs=[], outputs=[gr.File(label="Download Conversation History")])

interface.launch(share=True)
