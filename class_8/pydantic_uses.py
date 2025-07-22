from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from typing import Optional
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

groq_api_key=os.environ['GROP_API_KEY']

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

class Country(BaseModel):
    """Information about a country"""

    name: str = Field(description="name of the country")
    language: str = Field(description="language of the country")
    capital: str = Field(description="Capital of the country")
 
structured_llm = llm.with_structured_output(Country)

response = structured_llm.invoke("Tell me about France")
print(response)

# ====================================================================
# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


structured_llm = llm.with_structured_output(Joke)

response = structured_llm.invoke("Tell me a joke about cats")
print(response)

