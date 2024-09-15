import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain



# Load environment variables
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print(os.getenv("GOOGLE_API_KEY"))


# Function to load Gemini pro vision model and get responses for image input
def get_gemini_response_image(input, image):
    model = genai.GenerativeModel("gemini-pro-vision")
    if input:
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

# Function to load Gemini pro model for conversational chat
def get_gemini_chat_response(question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

# Function to extract transcript details from YouTube video
prompt= "Youtube video summarizer, taking the transcript text and summarize the entire video and providing the important summary in points within 250 words,please provide the summary of the text given here """
# def extract_transcript_details(youtube_video_url, language_code='en'):
#     try:
#         video_id = youtube_video_url.split("=")[1]
#         transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])

#         transcript = ""
#         for i in transcript_text:
#             transcript += "" + i["text"]
#         return transcript
#     except Exception as e:
#         st.error("Error: Could not receive transcript.")
#         st.error(str(e))
#         return None

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

# Function to generate summary based on prompt from Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to read PDF and extract text
def get_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
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
    # new_db = FAISS.load_local("faiss-index",embeddings,allow_dangerous_deserialization=True)
    new_db = FAISS.load_local("faiss-index",embeddings)
    docs = new_db.similarity_search(user_question)
    chain= get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply:",response["output_text"])

#functions to lead environment variables 
model = genai.GenerativeModel('gemini-pro-vision')
def get_gemini_response(input,image,prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        #Read the file into bytes, by converting it into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type, #Get the mime type of the up
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    

# Main function
def main():
    st.set_page_config(page_title="Integrated Streamlit App")
    st.sidebar.title("Menu")

    selected_option = st.sidebar.selectbox("Select Project", ["Image to Text", "ConvoLog", "PDF Chat", "Invoice Parsing", "Transcriber"])

  
    if selected_option == "Image to Text":
        st.title("Gemini Application")
        input_text = st.text_input("Input Prompt:", key="input_image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        image = ""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        submit = st.button("Tell me about the image")
        if submit or input_text:
            response = get_gemini_response_image(input_text, image)
            st.subheader("The Response is")
            st.write(response)

    elif selected_option == "ConvoLog":
        st.title("Gemini LLM Application")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        input_text = st.text_input("Input:", key="input_convo")
        submit = st.button("Ask the question")
        if submit or input_text:
            response = get_gemini_chat_response(input_text)
            st.session_state['chat_history'].append(("You",input_text))
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))
        st.subheader("The chat history is")
        if 'chat_history' in st.session_state:
         for role, text in st.session_state['chat_history']:
            st.write(f"{role}:{text}")

    elif selected_option == "PDF Chat":
        st.title("Chat with PDF using Gemini")
        user_question = st.text_input("Ask a Question from the PDF files")
        if user_question:
            user_input(user_question)
        with st.sidebar:
            st.title("Menu")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    elif selected_option == "Invoice Parsing":
        st.title("Multilanguage Invoice Extractor")
        input = st.text_input("Input Prompt:", key="input_invoice")
        uploaded_file = st.file_uploader("Choose the image of Invoice...", type=["jpg", "jpeg", "png"])
        image = ""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        submit = st.button("Tell me about the invoice")
        input_prompt = "You are an expert in understanding invoices. We will upload an image as an invoice and you will have to answer any questions based on the uploaded invoice."
        if submit:
            if submit:
              image_data = input_image_details(uploaded_file)
            response = get_gemini_response(input, image_data,input_prompt)

            st.subheader("The Response is")
            st.write(response)

    elif selected_option == "Transcriber":
        st.title("Youtube transcript to detailed notes converter")
        prompt= "Youtube video summarizer, taking the transcript text and summarize the entire video and providing the important summary in points within 250 words,please provide the summary of the text given here """
        youtube_link = st.text_input("Enter youtube video link")
        if youtube_link:
        #     video_id = youtube_link.split("=")[1]
        #     st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
        # if st.button("Get Detailed Notes"):
        #     transcript_text = extract_transcript_details(youtube_link)
        #     if transcript_text:
        #         summary = generate_gemini_content(transcript_text, prompt)
        #         st.markdown("Detailed Notes")
        #         st.write(summary)
            video_id = extract_video_id(youtube_link)
            st.image(f'http://img.youtube.com/vi/{video_id}/0.jpg',use_column_width=True)
        if st.button("Get Notes"):
            transcript_text = extract_transcript_details(video_id)
            if transcript_text:
                summary = generate_gemini_content(transcript_text,prompt)
                st.markdown("## Detailed Notes : ")
                st.write(summary)

    
    

if __name__ == "__main__":
    main()