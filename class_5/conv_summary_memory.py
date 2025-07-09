from langchain.memory import ConversationSummaryMemory
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains import ConversationChain

llm = LlamaCpp(
    # model_path="models/Mistral-7B-Instruct-v0.3.IQ4_XS.gguf",
    model_path="models/models--MaziyarPanahi--Llama-3.2-1B-Instruct-GGUF/snapshots/b64ae94264258a3d7516a34a8c54928d32b19869/Llama-3.2-1B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    temperature=0.0,
    verbose=True,
    n_gpu_layers=10,  
    n_threads=8
)

memory = ConversationSummaryMemory(llm=llm)

memory.save_context(
    {"input": "Hola, mi nombre es Iesus"},
    {"output": "Hola Iesus, ¿qué tal? Soy un modelo de IA llamado Coco."}
)
# Tambien podemos usar memory.chat_memory.add_user_message y memory.chat_memory.add_ai_message
#memory.chat_memory.add_user_message("Hola, mi nombre es Iesus")
#memory.chat_memory.add_ai_message("Hola Iesus, ¿qué tal? Soy un modelo de IA llamado Coco.")
memory.save_context(
    {"input": "Estoy investigando los diferentes tipos de memoria conversacional."},
    {"output": "Eso es interesante, ¿cuáles son algunos ejemplos?"}
)
memory.save_context(
    {"input": "He estado viendo ConversationBufferMemory y ConversationBufferWindowMemory."},
    {"output": "Eso es interesante, ¿cuál es la diferencia?"}
)
memory.save_context(
    {"input": "La memoria buffer solo almacena toda la conversación, ¿verdad?"},
    {"output": "Eso tiene sentido, ¿y qué hay de ConversationBufferWindowMemory?"}
)
memory.save_context(
    {"input": "La memoria buffer window almacena los últimos k mensajes, descartando el resto."},
    {"output": "¡Muy interesante!"}
)

memory.load_memory_variables({})

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

response = chain.invoke({"input": "Recuerdame cual es mi nombre por favor"})

print("="*20)
print(response['response'])