# automatic explanation

import streamlit as st
import time
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import tiktoken  # Import tiktoken for accurate token counting

st.set_page_config(page_title="Custom LLM with SQL DB")
st.title("Custom LLM with SQL DB 📈")

# Get an OpenAI API Key from secrets.toml
if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets["openai_api_key"]
else:
    st.info("Enter an OpenAI API Key in the secrets.toml file to continue")
    st.stop()

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")


# Function to count tokens using tiktoken
def count_tokens(messages):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_tokens = sum([len(encoding.encode(msg.content)) for msg in messages])
    return total_tokens


# Truncate conversation history to fit within token limit
def truncate_messages(messages, max_tokens):
    total_tokens = 0
    truncated_messages = []
    for message in reversed(messages):
        message_tokens = count_tokens([message])
        if total_tokens + message_tokens > max_tokens:
            break
        truncated_messages.append(message)
        total_tokens += message_tokens
    return list(reversed(truncated_messages))


# Set up the LangChain for memory
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOpenAI(api_key=openai_api_key)
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Setup agent for SQL
llm = OpenAI(api_key=openai_api_key, temperature=0, streaming=True)


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
    agent_executor_kwargs={"return_intermediate_steps": True},
)

# Check if 'messages' exists in the session state or if the 'Clear message history' button is pressed
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display all messages from the session state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user query from the chat input
user_query = st.chat_input(placeholder="Ask me anything from the database!")

if user_query:
    # Append user query to the session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Save user query in history
    msgs.add_user_message(user_query)

    # Calculate current tokens and truncate message history if necessary
    max_total_tokens = 4096  # Maximum tokens allowed by the model
    reserved_tokens = 500  # Reserve tokens for the current query and response
    prompt_tokens = count_tokens(
        msgs.messages + [type("msg", (object,), {"content": user_query})()]
    )
    max_history_tokens = max_total_tokens - reserved_tokens

    # Ensure prompt_tokens does not exceed max_total_tokens
    if prompt_tokens > max_total_tokens:
        max_history_tokens = max_total_tokens - reserved_tokens
        truncated_history = truncate_messages(msgs.messages, max_history_tokens)
    else:
        truncated_history = msgs.messages

    with st.spinner(text="Analyzing the database..."):
        # Get the assistant's response using your agent
        response = agent.invoke(user_query, history=truncated_history)

    # Extract response content
    response_content = response["output"]

    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)

    # Save AI response in history
    msgs.add_ai_message(response_content)

    # Function to prettify intermediate steps
    def prettify_intermediate_steps(steps):
        prettified_steps = []
        for i, step in enumerate(steps):
            action, result = step
            tool = action.tool
            tool_input = action.tool_input
            log = action.log

            log_before_action = log.split("Action")[0].strip()
            prettified_action = f"**Step {i + 1}:**\n\n{log_before_action}\n\nAction: {tool}\n\nAction Input: {tool_input}\n\n"
            prettified_steps.append(f"{prettified_action}\n\nResult: {result}")
        return "\n\n".join(prettified_steps)

    # Prettify intermediate steps
    inter_steps = response["intermediate_steps"]
    prettified_inter_steps = prettify_intermediate_steps(inter_steps)

    # Display intermediate steps in an expander
    with st.container(height=300, border=True):
        st.markdown("See explanation:")
        st.write(prettified_inter_steps)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    view_messages.json(st.session_state.langchain_messages)
