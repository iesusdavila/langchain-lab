import requests
import streamlit as st

def get_consult_response(input_text):
    response=requests.post("http://localhost:8000/ros2_consultant/invoke",
    json={'input':input_text})

    print(response.json())
    return response.json()['output']['answer']

def get_llamacpp_response(input_text):
    response=requests.post("http://localhost:8000/llm/invoke",
    json={'input':input_text})

    print(response.json())
    return response.json()['output']

st.title('Asistente Virtual ROS2 Humble')
input_text=st.text_input("Escribe tu pregunta sobre ROS2 Humble aquí")

if input_text:
    st.write(get_consult_response(input_text))