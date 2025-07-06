import streamlit as st
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
import time
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "langchain-lab")

st.set_page_config(
    page_title="Real-Time LLM Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("Real-Time LLM Chat")

@st.cache_resource
def initialize_chat_model():
    return ChatLlamaCpp(
        model_path="models/models--MaziyarPanahi--Llama-3.2-1B-Instruct-GGUF/snapshots/b64ae94264258a3d7516a34a8c54928d32b19869/Llama-3.2-1B-Instruct.Q4_K_M.gguf",
        n_ctx=2048, 
        n_gpu_layers=10,  
        n_threads=8,
        temperature=0.7,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "system", 
        "content": "Eres un asistente útil y amigable. Responde de manera concisa y clara a las preguntas del usuario."
    })

if "chat_model" not in st.session_state:
    st.session_state.chat_model = initialize_chat_model()

col1, col2 = st.columns([1, 4])

with col2:
    st.header("💬 IA Chat")
    
    chat_container = st.container(height=500)
    with chat_container:
        for i, message in enumerate(st.session_state.messages[1:]):
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                        <div style="background-color: #0084ff; color: white; padding: 8px 12px; 
                                    border-radius: 18px; max-width: 70%; word-wrap: break-word;">
                            {message["content"]}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif message["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                        <div style="background-color: #f1f3f4; color: #333; padding: 8px 12px; 
                                    border-radius: 18px; max-width: 70%; word-wrap: break-word;">
                            🤖 {message["content"]}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    
    st.divider()
    
    st.markdown("### ✍️ Write your message:")
    if prompt := st.chat_input("Write your message here...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            langchain_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            try:
                start_time = time.time()
                response = st.session_state.chat_model.invoke(langchain_messages)
                end_time = time.time()
                
                if hasattr(response, 'content'):
                    response_content = response.content
                else:
                    response_content = str(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
                st.session_state.last_response_time = end_time - start_time
        
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al generar respuesta: {str(e)}")

with col1:
    st.header("Chat Control")
    
    with st.expander("Model Monitoring", expanded=True):
        try:
            if hasattr(st.session_state.chat_model, 'client'):
                client = st.session_state.chat_model.client
                
                n_ctx = client.n_ctx()
                st.metric("Total Contexto", f"{n_ctx:,} tokens")
                                
                total_tokens = client.n_tokens
                st.metric("Used Tokens", f"{total_tokens:,}")
                
                if n_ctx > 0:
                    usage_percentage = (total_tokens / n_ctx) * 100
                    st.metric("Used Context", f"{usage_percentage:.1f}%")
                    
                    st.progress(min(usage_percentage / 100, 1.0))
                    
                    if usage_percentage > 90:
                        st.error("🚨 Critical Context - Consider cleaning up the chat")
                    elif usage_percentage > 80:
                        st.warning("⚠️ Context almost full")
                
            else:
                st.info("Don't have a client available for monitoring")
        except Exception as e:
            st.error(f"Error in monitoring: {str(e)}")
    
    with st.expander("📈 Chat Statistics", expanded=True):
        user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        assistant_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
        
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("👤 Your messages", user_messages)
        with col_stats2:
            st.metric("🤖 AI Answers", assistant_messages)
        
        if hasattr(st.session_state, 'last_response_time'):
            st.metric("⏱️ Last Answer", f"{st.session_state.last_response_time:.2f}s")
    
    with st.expander("⚙️ Model Configuration"):
        st.info("**Model:** Llama-3.2-1B")
        st.info(f"**Temperature:** {st.session_state.chat_model.temperature}")
        st.info(f"**Max Tokens:** {st.session_state.chat_model.max_tokens}")
    
    st.subheader("🛠️ Controlls")
    
    if st.button("🗑️ Clean Chat", type="primary", use_container_width=True):
        st.session_state.messages = [{
            "role": "system", 
            "content": "Eres un asistente útil y amigable. Responde de manera concisa y clara a las preguntas del usuario."
        }]
        st.success("Cleaned chat successfully!")
        st.rerun()
    
    if st.button("💾 Exporter Chat", type="secondary", use_container_width=True):
        if len(st.session_state.messages) > 1:
            chat_history = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in st.session_state.messages[1:]
            ])
            st.download_button(
                label="📥 Download conversation",
                data=chat_history,
                file_name=f"chat_export_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Don't have messages to export yet. Start chatting!")