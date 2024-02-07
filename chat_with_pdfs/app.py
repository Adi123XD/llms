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

def get_pdf(pdf_docs):
    text =""
    for pdf in pdf_docs:
        # read the pdf pages
        pdf_reader = PdfReader(pdf)
        # the pdf is read in many pages get all the text from those pages
        for page in pdf_reader.pages:
            text=text+page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # loading the free google genai embeddings 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # creating a vectore store for our embeddings 
    vector_store= FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss-index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,, don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # loading the gemini pro model using langchain_google_genai's function ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model ="gemini-pro",temperature=0.3)
    # create a prompt out of the prompt template
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain= load_qa_chain(model , chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss-index",embeddings)
    docs = new_db.similarity_search(user_question)
    chain= get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply:",response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    user_question = st.text_input("Ask a Question from the PDF files")
    if (user_question):
        user_input(user_question)
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
if __name__=="__main__":
    main()




