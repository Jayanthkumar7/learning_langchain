from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt
from transformers.utils import logging
logging.set_verbosity_error()

load_dotenv()
if "processing" not in st.session_state:
    st.session_state.processing  = False
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

st.header("JK Model")

paper_selected = st.selectbox("Select the Rearch paper", ["Attention is all you need","BERT:pre Training of deep Bidirectinal Transformers","GPT 3: Language models are few shot learners","Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("select explanation style",["beginer Firendly","Technical","code-oriented","Mathematical"])

length_input = st.selectbox("select explanation length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long(detailed explanation )"])

template = load_prompt('template.json')

template.save("template.json")

prompt = template.invoke({
    'paper_selected':paper_selected,
    "style_input":style_input,
    "length_input":length_input
})

summarize_clicked = st.button("Summarize",disabled=st.session_state.processing)
if summarize_clicked:
    st.session_state.processing = True
    
    try :
        with st.spinner("generating summary !!"):
            result = model.invoke(prompt)
        st.write(result.content)
        
        
    finally:
        st.session_state.processing = False
