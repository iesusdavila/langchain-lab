from langchain_groq import ChatGroq
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

groq_api_key=os.environ['GROP_API_KEY']

llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

system_prompt = SystemMessagePromptTemplate.from_template(
    "Eres un asistente que ayuda a buscar información sobre celebridades. "
)

# First prompt to get basic information about the celebrity
first_user_prompt = HumanMessagePromptTemplate.from_template(
    """Dime algo sobre la celebridad {name}.""",
    input_variables=["name"]
)

first_prompt = ChatPromptTemplate.from_messages([system_prompt,first_user_prompt])

chain_one = (
    {"name": lambda x: x['name']}
    | first_prompt
    | llm 
    | {"person": lambda x: x.content,}
)

celebrity_info_msg = chain_one.invoke({"name": "Cristiano Ronaldo"})
print(celebrity_info_msg)

# Second prompt to ask for the date of birth of the celebrity
second_user_prompt = HumanMessagePromptTemplate.from_template(
    """Dime en que fecha nació {person}""",
    input_variables=["person"]
)

second_prompt = ChatPromptTemplate.from_messages([system_prompt,second_user_prompt])

chain_two = (
    {"person": lambda x: x['person']}
    | second_prompt
    | llm 
    | {"dob": lambda x: x.content,}
)

celebrity_dob_msg = chain_two.invoke({"person": celebrity_info_msg['person']})
print(celebrity_dob_msg)

# Third prompt to ask for major events around the date of birth
third_user_prompt = HumanMessagePromptTemplate.from_template(
    """Menciona 5 eventos importantes que ocurrieron alrededor de la fecha de nacimiento {dob}""",
    input_variables=["dob"]
)

third_prompt = ChatPromptTemplate.from_messages([system_prompt,third_user_prompt])

chain_three = (
    {"dob": lambda x: x['dob']}
    | third_prompt
    | llm 
    | {"description": lambda x: x.content,}
)

celebrity_events_msg = chain_three.invoke({"dob": celebrity_dob_msg['dob']})
print(celebrity_events_msg)