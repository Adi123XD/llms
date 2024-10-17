import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print(os.getenv("GOOGLE_API_KEY"))

st.set_page_config("Chat PDF")
# Custom CSS for chat style with right/left alignment
st.markdown(
    """
    <style>
    .chat-box {
        display: flex;
        align-items: center;
        margin-left: 10px;
        margin-right: 10px;
    }
    .chat-message {
        margin-left: 10px;
        margin-right: 10px;
        padding: 10px;
        border-radius: 10px;
        color: white;
    }
    .user-message {
        justify-content: flex-end;  /* Align user messages to the right */
        margin-left: auto; /* Push the user message to the right */
    }
    .bot-message {
        justify-content: flex-start;  /* Align bot messages to the left */
        width: 100%;  /* Bot message occupies full width */
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
    
def get_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss-index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. Do not provide incorrect answers.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# Displaying messages with right/left alignment
def display_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-box user-message"><span class="chat-message">{message["content"]}</span></div>', unsafe_allow_html=True)
        else:
            st.image("luffy.png", width=50) 
            st.markdown(f'<div class="chat-box bot-message"><span class="chat-message">{message["content"]}</span></div>', unsafe_allow_html=True)
        #     user_col1, user_col2 = st.columns([8, 1])
        #     with user_col1:
        #         st.markdown("")  # Empty space for right alignment
        #     with user_col2:
        #         # st.image("human.png", width=50)  # Avatar for user
        #         st.markdown(f'<div class="chat-box user-message"><span class="chat-message">{message["content"]}</span></div>', unsafe_allow_html=True)
        # else:
        #     luffy_col1, luffy_col2 = st.columns([1, 8])
        #     with luffy_col1:
        #         st.image("luffy.png", width=50)  # Avatar for bot
        #     with luffy_col2:
        #         st.markdown(f'<div class="chat-box bot-message"><span class="chat-message">{message["content"]}</span></div>', unsafe_allow_html=True)

def main():
    st.header("Chat with PDF using Gemini💁")


    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    # Chat interface
    user_question = st.chat_input("Ask a Question from the PDF files")
    if user_question:
        with st.spinner("Processing your question..."):
            # Process user question and get response
            response = user_input(user_question)

            # Display user's input
            st.session_state.messages.append({"role": "user", "content": user_question})
            # Display bot's response
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display the entire conversation
        display_chat()

if __name__ == "__main__":
    main()
