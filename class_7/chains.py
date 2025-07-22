from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROP_API_KEY']

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm