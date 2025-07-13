import time
import requests
import streamlit as st
from datetime import datetime

def get_consult_response(input_text):
    try:
        response = requests.post("http://localhost:8000/ros2_consultant/invoke",
                               json={'input': input_text})
        response.raise_for_status()
        return response.json()['output']['answer']
    except Exception as e:
        return f"Error al conectar con ROS2 Consultant: {str(e)}"

def get_quiz_response(input_text):
    try:
        response = requests.post("http://localhost:8000/ros2_quiz/invoke",
                               json={'input': input_text})
        response.raise_for_status()
        return response.json()['output']['answer'], response.json()['output']['context']
    except Exception as e:
        return f"Error al conectar con ROS2 Quiz: {str(e)}"

def get_llm_response(input_text):
    try:
        response = requests.post("http://localhost:8000/llm/invoke",
                               json={'input': input_text})
        response.raise_for_status()
        return response.json()['output']["content"]
    except Exception as e:
        return f"Error al conectar con LLM: {str(e)}"

def display_message(message, is_user=True):
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)

def main():
    st.set_page_config(
        page_title="ROS2 Humble Virtual Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []
    if "ros2_messages" not in st.session_state:
        st.session_state.ros2_messages = []
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = "llm"

    with st.sidebar:
        chat_type = st.selectbox(
            "Tipo de Chat:",
            ["Chat with ROS2 Consultant", "ROS2 Quiz", "Chat with LLM"],
            key="chat_selector"
        )
        
        st.divider()

        st.header("Options")
        
        if st.button("Clear LLM Chat"):
            st.session_state.llm_messages = []
            st.rerun()
        
        if st.button("Clear ROS2 Chat"):
            st.session_state.ros2_messages = []
            st.rerun()
        
        if st.button("Clear Both Chats"):
            st.session_state.llm_messages = []
            st.session_state.ros2_messages = []
            st.rerun()
        
        st.divider()
        
        st.subheader("Statistics")
        st.metric("LLM Messages", len(st.session_state.get("llm_messages", [])))
        st.metric("ROS2 Messages", len(st.session_state.get("ros2_messages", [])))
        
        st.divider()

        st.subheader("Chat View")
        
        if chat_type == "Chat with LLM" and st.session_state.ros2_messages:
            st.subheader("ROS2 Chat View")
            with st.expander("See latest ROS2 messages"):
                for msg in st.session_state.ros2_messages[-3:]:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    st.text(f"{role_icon} {msg['content'][:50]}...")
        
        elif not chat_type == "Chat with LLM" and st.session_state.llm_messages:
            st.subheader("llm Chat View")
            with st.expander("See latest LLM messages"):
                for msg in st.session_state.llm_messages[-3:]:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    st.text(f"{role_icon} {msg['content'][:50]}...")
        
    if chat_type == "Chat with LLM":
        st.title("Chat with LLM")
        st.session_state.active_chat = "llm"
        
        llm_container = st.container()
        with llm_container:
            for message in st.session_state.llm_messages:
                display_message(message["content"], message["role"] == "user")
        
        llm_input = st.chat_input("Write your message to your assistant here...")
        
        if llm_input:
            st.session_state.llm_messages.append({
                "role": "user", 
                "content": llm_input,
                "timestamp": datetime.now()
            })
            
            with st.spinner("Generating response..."):
                llm_response = get_llm_response(llm_input)
            
            st.session_state.llm_messages.append({
                "role": "assistant", 
                "content": llm_response,
                "timestamp": datetime.now()
            })
            
            st.rerun()
    
    elif chat_type == "Chat with ROS2 Consultant":
        st.title("Chat with ROS2 Consultant")
        st.session_state.active_chat = "ros2"
        
        ros2_container = st.container()
        with ros2_container:
            for message in st.session_state.ros2_messages:
                display_message(message["content"], message["role"] == "user")
        
        ros2_input = st.chat_input("Write your question about ROS2 Humble here...")
        
        if ros2_input:
            st.session_state.ros2_messages.append({
                "role": "user", 
                "content": ros2_input,
                "timestamp": datetime.now()
            })
            
            with st.spinner("Consulting ROS2 Humble..."):
                ros2_response = get_consult_response(ros2_input)
            
            st.session_state.ros2_messages.append({
                "role": "assistant", 
                "content": ros2_response,
                "timestamp": datetime.now()
            })
            
            st.rerun()
    
    elif chat_type == "ROS2 Quiz":
        st.title("ROS2 Quiz")
        st.session_state.active_chat = "quiz"

        input_text=st.text_input("Write the topic you want to make a Quiz")

        if input_text:
            with st.spinner("Searching the documentation..."):
                start=time.process_time()
                response, context = get_quiz_response(input_text)
                st.write(response)
                response_time = time.process_time()-start
        
            st.info(f"Response time: {response_time:.2f} seconds")

            with st.expander("Reference documents used"):
                for i, doc in enumerate(context):
                    st.info(f"**Document {i+1}: {doc['metadata']['title']}**")
                    st.info(doc["page_content"])
                    st.write(f"*Source: {doc['metadata']['source']}*")
                    
if __name__ == "__main__":
    main()