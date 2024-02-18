# import the dependencies 
import streamlit as st
import re
import requests
import os
import google.generativeai as genai 
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

# load the model 
print(os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# code for youtube_transcriber
def extract_video_id(url):
    pattern = r"(?<=v=)[a-zA-Z0-9_-]+(?=&|\|?|$)"
    match = re.search(pattern , url)
    if match :
        print(match.group(0))
        return match.group(0)


def extract_transcript_details(video_id):
    try:
        # video_id = url.split("=")[1][:-2]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # this will be in the form of a list 
        transcript_text =""
        for i in transcript:
            transcript_text+= " "+i["text"]
        return transcript_text
    except Exception as e:
        error_message = str(e)
        if "No transcripts were found for any of the requested language codes" in error_message:
            st.error("Transcripts are not available in English for this video.")
        else:
            st.error(f"An error occurred: {error_message}")
def generate_gemini_content(transcript_text):
    model = genai.GenerativeModel("gemini-pro")
    prompt =f'''You are a youtube video summarizer. You will be taking the transcript text and summarizing the 
entire video and providing the important summary in points within 250  to 300 words. 
The transcript text will be appended here : {transcript_text}'''
    response = model.generate_content(prompt)
    return response.text


# code for chatbot 
def get_response(question):
    response = chat.send_message(question, stream=True)
    return response
def get_response_vext(question):
    headers={
    'Content-Type': 'application/json',
    'Apikey': f'Api-Key {os.getenv("VEXT_API_KEY")}'
    }
    data = {
        'payload':question
    }
    url ='https://payload.vextapp.com/hook/97QBZ0M9T7/catch/$(adi123)'
    response = requests.post(url ,json=data , headers=headers )
    return response

# code for chat_with_pdfs
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
    model2 = ChatGoogleGenerativeAI(model ="gemini-pro",temperature=0.3)
    # create a prompt out of the prompt template
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain= load_qa_chain(model2 , chain_type="stuff", prompt=prompt)
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


st.set_page_config(page_title="BetterAI")
st.header("AI for you !!")
choice = st.selectbox(
    label="How can I help you ??",
    options=("ChatBot","Youtube_Transcriber","Chat_with_PDFs")
)
if choice =="Youtube_Transcriber":
    st.subheader("Youtube video Notes generator")
    link = st.text_input(label="" , placeholder="Enter the video link ...")
    if link:
        video_id = extract_video_id(link)
        st.image(f'http://img.youtube.com/vi/{video_id}/0.jpg',use_column_width=True)
    if st.button("Get Notes"):
        transcript_text = extract_transcript_details(video_id)
        if transcript_text:
            summary = generate_gemini_content(transcript_text)
            st.markdown("## Detailed Notes : ")
            st.write(summary)
if choice =="ChatBot":
    st.subheader("Aur Batao ...")
    # initialise the chat history session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"]=[]
    input = st.text_input(label= "", placeholder="bolo bolo...")
    if input :
        response = get_response(input)
        # Add the user query and response into the session state
        st.session_state["chat_history"].append(("You", input))
        # st.subheader("ChatADI")
        for chunk in response:
            st.write(chunk.text)
        st.session_state['chat_history'].append(("Aur Batao ",response.text))
    st.subheader("Chat History ")
    for role,text in st.session_state['chat_history']:
        st.write(f"{role}:{text}")
if choice=="Chat_with_PDFs":
    st.subheader("Chat_with_PDFs")
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