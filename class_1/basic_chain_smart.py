import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_experimental.smart_llm.base import SmartLLMChain

st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic you want")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)

llm = LlamaCpp(
    model_path="models/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8/meta-llama-3.1-8b-instruct-q4_0.gguf",        
    n_ctx=2048,
    temperature=0.8,
    verbose=True,
    n_threads=8,
    n_gpu_layers=20
)

chain1 = SmartLLMChain(
    llm=llm, 
    prompt=first_input_prompt,
    output_key='person',
    verbose=True
)

chain2 = SmartLLMChain(
    llm=llm,
    prompt=second_input_prompt,
    output_key='dob',
    verbose=True
)

chain3 = SmartLLMChain(
    llm=llm,
    prompt=third_input_prompt,
    output_key='description',
    verbose=True
)

parent_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

if input_text:
    result = parent_chain({'name': input_text})
    st.write(result)
