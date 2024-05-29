import streamlit as st

from tempfile import NamedTemporaryFile
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import streamlit.components.v1 as components


# increase width of streamlit sidebar for wide screen
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            max-width: 1800px !important;
        }
        div[data-testid="stMarkdownContainer"] {
            word-wrap: break-word;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# PDF process and embedding
def process_pdf(pdf_path):
    """For a given pdf file, split it into multiple chunks, 
    then use OpenAIEmbedding to create embedding for each chunk, then store them in a vector db

    Args:
        pdf_path (str): file path to pdf file
    """
    openai_embeddings = OpenAIEmbeddings()
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    print(f"total number of pages: {len(pages)}")

    # create a in-memory vector database.
    vectordb = Chroma.from_documents(documents=pages,
                                 embedding=openai_embeddings,
                                 collection_metadata= {"hnsw:search_ef" : 100, "hnsw:space": "cosine"}) # chroma by default use l2 distance, but we want to use cosine distance
    print("vector db created")
    
    st.session_state.vectordb = vectordb


### LLM part
## Create a Q&A chain to address user's question
qa_chat_prompt_template = """You are a helpful AI assistant. Based on context and question below, please generate helpful answer. Please ONLY USE information from context, if you can't find answer from context, just say "I don't know"
Context: {context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_chat_prompt_template)

def ask_question(question):
    """Create a Langchain RetrievalQA chain to address user's question, use gpt 3.5 here.

    Args:
        question (str): User's question

    Returns:
        _type_: return a dict: {'result': answer, 'source_documents': source_documents}
    """
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
        retriever=st.session_state.vectordb.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True)  # return source documents for citation.
    
    return qa_chain({"query":question})

def reset():
    st.session_state.vectordb = None
    st.session_state.messages = []
    st.session_state.tmp_pdf_path = None


def display_pdf(page_nr=None):
    """
    call back function when user click one of reference page: show the coooresponding page in sidebar.
    """
    if page_nr:
        page_url = f"http://localhost:8900/{st.session_state.pdf_filename}#page={page_nr+1}" 
    else:
        page_url = f"http://localhost:8900/{st.session_state.pdf_filename}" 
    
    highlighted_html = f"""
                <iframe src="{page_url}" width="1050" height="1800" type="application/pdf"></iframe>
                <script>
                    // align pdf iframe width to slidebar width
                    var sidebarWith = document.querySelectorAll('[data-testid="stSidebar"]')[0].offsetWidth
                    document.getElementsByTagName("iframe")[0].offsetWidth = sidebarWith
                </script>
                """ 
    with st.sidebar:
        st.markdown(body = highlighted_html, unsafe_allow_html=True)
    

def button_clicked(page_nr):
    display_pdf(page_nr)
        

    
def print_answer(answer, chat_index):
    st.markdown(answer['result'])
    with st.expander(f"Source from document"):
        columns = st.columns(2)
        for index, doc in enumerate(answer['source_documents']):
            with columns[index % 2]:
                import time
                timestr = time.strftime("%Y%m%d-%H%M%S")
                st.button(str(doc.page_content)[0:40] + "...", 
                    on_click=button_clicked, 
                    args=[doc.metadata['page']],
                    key=f'{chat_index}_{timestr}_{index}') 
    
def main():
    
    st.header("PDF chatbot", divider='rainbow')
    st.caption(":rocket::rocket::rocket:  Power by Meno Data! Feeling interested, [contact us](mailto:info@menodata.com) :rocket::rocket::rocket: ")

    # Upload button for pdf file and process pdf file.
    upload_file = st.file_uploader("Upload your word document", type="pdf", on_change=reset)
    # st.set_option('deprecation.showfileUploaderEncoding', False)

    if upload_file is None:
        st.text("Please upload an word document")
    else:
        if 'vectordb' not in st.session_state or not st.session_state.vectordb:
            with NamedTemporaryFile(dir='./static', delete=False, suffix=".pdf") as tmp:
                tmp.write(upload_file.getvalue())
                st.session_state.pdf_filename = tmp.name.split('/')[-1]
                with st.spinner('Processing document...'):
                    process_pdf(tmp.name)
                    st.success('Document processed!')
                    display_pdf()
                
        
    # chat window for user questions.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        
    # Display chat messages from history on app rerun
    for chat_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == 'assistant':
                print_answer(message["content"], int((chat_index+1)/2))
            else:
                st.markdown(message["content"])
            
    
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🤖 is thinking ......"):
            answer = ask_question(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                print_answer(answer, int(len(st.session_state.messages)/2))
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                

if __name__ == "__main__":
    main()
