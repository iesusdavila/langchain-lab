import requests
import streamlit as st

def get_summary_response(input_text):
    response=requests.post("http://localhost:8000/summary/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

def get_llamacpp_response(input_text):
    response=requests.post("http://localhost:8000/llmcpp/invoke",
    json={'input':input_text})

    return response.json()['output']

st.title('Langchain Demo With LlamaCpp')
input_text=st.text_input("Escribe el tema que quieres hacer un resumen")
input_text_llamacpp=st.text_input("Escribe el tema que quieres consultar con LlamaCpp")

if input_text:
    st.write(get_summary_response(input_text))

if input_text_llamacpp:
    st.write(get_llamacpp_response(input_text_llamacpp))