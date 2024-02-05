import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from llama model 
def getllamareponse(input_text,no_words,blog_style):
    print("hello world")
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={
                            'max_new_tokens':256,
                            'temperature':0.01
                        })
    template ='''
        Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words
        '''
    prompts = PromptTemplate(input_variables=['blog_style','input_text','no_words'],template=template)
    # Generate a response from the llama model 
    response = llm(prompts.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response



st.set_page_config(page_title="Generate Blogs",
                   initial_sidebar_state="collapsed",
                   layout="centered",
                   page_icon="🤖")
st.header("Generate Blogs 🤖")
input_text = st.text_input("Enter the Blog topic")

# creating more colomns for additional 2 fields
col1,col2 = st.columns([5,5])
with col1:
    no_words = st.text_input("Number of words")

with col2:
    blog_style = st.selectbox("Writing blog for ",("Researchers","Data Scientitsts" , "Common People"),index=0)
submit = st.button("Generate")

if submit:
    st.write(getllamareponse(input_text,no_words,blog_style))
