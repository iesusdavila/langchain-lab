import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain_community.llms.llamacpp import LlamaCpp

# Config LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""

st.title("Conocimiento de futbol")
input_text = st.text_input("Ingresa el nombre de un jugador de futbol")

response_schemas = [
    ResponseSchema(
        name="nombre",
        description="Tema de la pregunta del usuario",
        type="string"
    ),
    ResponseSchema(
        name="data",
        description="Informacion del jugador",
        type="string"
    ),
    ResponseSchema(
        name="edad",
        description="Edad del jugador",
        type="integer"
    ),
    ResponseSchema(
        name="hechos",
        description="Hechos interesantes en el año de nacimiento del jugador",
        type="List[string]"
    ),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template=(
        "Eres un experto en futbol y conoces a todos los jugadores de la historia. "
        "Tu tarea es responder a las preguntas de los usuarios sobre jugadores de futbol. "
        "Responde con un JSON que contenga exactamente las claves: nombre, data, edad y hechos (todo en minúsculas).\n"
        "{format_instructions}\n\n"
        "Nombre del jugador: {nombre}\n"
        "Responde a la pregunta del usuario con la informacion del jugador, su edad y hechos interesantes en el año de nacimiento del jugador.\n"
        "Asegurate de que la respuesta use el campo 'hechos' en minúsculas.\n"
        "No incluyas ningun texto adicional, solo el JSON con la informacion solicitada.\n"
    ),
    input_variables=["nombre"],
    partial_variables={"format_instructions": format_instructions}
)

llm = LlamaCpp(
    model_path="models/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8/meta-llama-3.1-8b-instruct-q4_0.gguf",        
    n_ctx=2048,
    verbose=True,
    n_gpu_layers=20,
    n_threads=8
)

if input_text:
    chain = prompt | llm | output_parser
    result = chain.invoke({"nombre": input_text})
    
    st.write(result)
