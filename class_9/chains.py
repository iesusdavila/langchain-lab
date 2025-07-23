from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_groq import ChatGroq
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROP_API_KEY']

llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

# Actor Agent Prompt 
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Eres un investigador experto en IA.

            1. {first_instruction}
            2. Reflexiona y critica tu respuesta. Sé riguroso para maximizar la mejora.
            3. Después de la reflexión, **enumera de 1 a 3 consultas de búsqueda por separado** para investigar mejoras. 
            No las incluyas en la reflexión.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Responda la pregunta del usuario anterior utilizando el formato requerido."),
    ]
)

first_responder_prompt_template = actor_prompt_template.partial(
    time=lambda: datetime.datetime.now().isoformat(),
    first_instruction="Ofrece una respuesta de maximo 250 palabras"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion') 

validator = PydanticToolsParser(tools=[AnswerQuestion])

# Revisor section
revise_instructions = """Revisa tu respuesta anterior con la nueva información.
- Debes usar la crítica anterior para añadir información importante a tu respuesta.
- Debes incluir citas numéricas en tu respuesta revisada para asegurar que pueda verificarse.
- Agrega una sección de "Referencias" al final de tu respuesta (no se considera el límite de palabras). En el formato:
- [1] https://example.com
- [2] https://example.com
- Debes usar la crítica anterior para eliminar información superflua de tu respuesta y asegurarte de que no supere las 250 palabras.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# response = first_responder_chain.invoke({
#     "messages": [HumanMessage("AI Agents taking over content creation")]
# })s

# print(response)
