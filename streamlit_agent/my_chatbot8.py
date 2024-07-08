from langchain.agents import AgentExecutor, create_openai_functions_agent
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from pathlib import Path
from langchain.agents.agent_types import AgentType


st.set_page_config(page_title="Combined Chatbot with Memory and SQL DB", page_icon="🦜")
st.title("🦜 Combined Chatbot with Memory and SQL DB")


if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets["openai_api_key"]
else:
    st.info("Enter an OpenAI API Key in the secrets.toml file to continue")
    st.stop()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOpenAI(api_key=openai_api_key)


# Setup agent for SQL
llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)


@st.cache_resource(ttl="2h")
def configure_db():
    # Make the DB connection read-only to reduce risk of injection attacks
    db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
    creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))


db = configure_db()
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True, return_intermediate_steps=True)


def sql_query_chat(message):
    response = agent_executor.invoke({"question": message})
    return response


chat_history = []

with st.tabs[1]:
    for entry in chat_history:
        response = sql_query_chain.invoke({"question": entry[0]})
        st.write(response)
