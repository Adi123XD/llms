from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os 
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# load the gemini model 
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])
def get_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Set up the streamlit app 
st.set_page_config(page_title="ChatADI")
st.header("Chat with ChatADI")


# initialise the chat history session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"]=[]

input = st.text_input(label= "", placeholder="Message ChatADI...")
if input :
    response = get_response(input)
    # Add the user query and response into the session state
    st.session_state["chat_history"].append(("You", input))
    st.subheader("ChatADI")
    for chunk in response:
        st.write(chunk.text)
    st.session_state['chat_history'].append(("ChatADI",response.text))
st.subheader("Chat History ")
for role,text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")
