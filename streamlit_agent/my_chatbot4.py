# it makes a bit more conversation
# kinda slow
import streamlit as st
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import AgentExecutor
from sqlalchemy import create_engine
import sqlite3
import pandas as pd
import re
import ast
from operator import itemgetter
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chains import create_sql_query_chain

# Streamlit app setup
st.set_page_config(page_title="Combined Chatbot with Memory and SQL DB", page_icon="🦜")
st.title("🦜 Combined Chatbot with Memory and SQL DB")

# Load OpenAI API Key from secrets
if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets["openai_api_key"]
else:
    st.info("Enter an OpenAI API Key in the secrets.toml file to continue")
    st.stop()

# Initialize message history
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")


# Function to truncate messages to fit within token limit
def truncate_messages(messages, max_tokens):
    total_tokens = 0
    truncated_messages = []
    for message in reversed(messages):
        message_tokens = len(message["content"].split())
        if total_tokens + message_tokens > max_tokens:
            break
        truncated_messages.append(message)
        total_tokens += message_tokens
    return list(reversed(truncated_messages))


# Set up SQL agent with OpenAI
llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)


@st.cache_resource(ttl="2h")
def configure_db():
    db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
    creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))


db = configure_db()
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


# Function to query the database and return results as list
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


# Retrieve proper nouns to handle high-cardinality columns
artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")

from langchain_community.vectorstores import FAISS

# Ensure the OpenAIEmbeddings gets the correct API key
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_db = FAISS.from_texts(artists + albums, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

# Define the SQL query chain
query_chain = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=query_chain).assign(result=itemgetter("query") | execute_query)
    | answer_prompt
    | llm
    | StrOutputParser()
)

# System message for the agent
SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)

agent_executor = AgentExecutor(agent=llm, tools=tools, messages_modifier=system_message)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    msgs.add_user_message(user_query)  # Save user query in history
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = chain.invoke({"question": user_query})

        # Convert query result to a DataFrame if it's SQL
        if "SQL query" in response:
            query_result = db.run(response)
            if query_result is not None and len(query_result) > 0:
                df = pd.DataFrame(
                    query_result, columns=[desc[0] for desc in query_result.cursor.description]
                )
                st.write("### Query Result")
                st.dataframe(df)
                response_text = df.to_string()
            else:
                response_text = "No results found."
        else:
            response_text = response

        # Update message history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        msgs.add_ai_message(response_text)  # Save AI response in history
        st.write(response_text)

# Display message history for debugging
with view_messages:
    view_messages.json(st.session_state.langchain_messages)
