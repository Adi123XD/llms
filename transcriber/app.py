import streamlit as st
import re
import google.generativeai as genai 
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
load_dotenv()
import os 
print(os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title="Youtube transcriber")
# prompt ='''You are a youtube video summarizer. You will be taking the transcript text and summarizing the 
# entire video and providing the important summary in points within 250  to 300 words. 
# The transcript text will be appended here : '''


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
        pass
    except Exception as e:
        raise e
def generate_gemini_content(transcript_text):
    model = genai.GenerativeModel("gemini-pro")
    prompt =f'''You are a youtube video summarizer. You will be taking the transcript text and summarizing the 
entire video and providing the important summary in points within 250  to 300 words. 
The transcript text will be appended here : {transcript_text}'''
    response = model.generate_content(prompt)
    return response.text
st.header("Youtube video Notes generator")
link = st.text_input(label="" , placeholder="Enter the video link ...")
if link:
    # vid = link.split("=")[1][:-2]
    # index_of_ampersand = vid.find("&")
    # video_id= vid[0:index_of_ampersand]
    video_id = extract_video_id(link)
    st.image(f'http://img.youtube.com/vi/{video_id}/0.jpg',use_column_width=True)
if st.button("Get Notes"):
    transcript_text = extract_transcript_details(video_id)
    if transcript_text:
        summary = generate_gemini_content(transcript_text)
        st.markdown("## Detailed Notes : ")
        st.write(summary)
