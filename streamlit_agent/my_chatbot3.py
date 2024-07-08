# working chatbot with steps shown

import streamlit as st
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

st.set_page_config(page_title="Combined Chatbot with Memory and SQL DB", page_icon="🦜")
st.title("🦜 Combined Chatbot with Memory and SQL DB")

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


# Truncate conversation history to fit within token limit
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

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    msgs.add_user_message(user_query)  # Save user query in history
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        msgs.add_ai_message(response)  # Save AI response in history
        st.write(response)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    view_messages.json(st.session_state.langchain_messages)


# Code without callbacks

import streamlit as st
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

st.set_page_config(page_title="Combined Chatbot with Memory and SQL DB", page_icon="🦜")
st.title("🦜 Combined Chatbot with Memory and SQL DB")

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


# Truncate conversation history to fit within token limit
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

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    msgs.add_user_message(user_query)  # Save user query in history
    with st.chat_message("assistant"):
        response = agent.run(user_query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        msgs.add_ai_message(response)  # Save AI response in history
        st.write(response)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    view_messages.json(st.session_state.langchain_messages)

    # code without callbacks and with invoke function

    import streamlit as st
    from pathlib import Path
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_openai import ChatOpenAI, OpenAI
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_community.utilities import SQLDatabase
    from langchain.agents.agent_types import AgentType
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from sqlalchemy import create_engine
    import sqlite3
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory

    st.set_page_config(page_title="Combined Chatbot with Memory and SQL DB", page_icon="🦜")
    st.title("🦜 Combined Chatbot with Memory and SQL DB")

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

    # Truncate conversation history to fit within token limit
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

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask me anything!")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        msgs.add_user_message(user_query)  # Save user query in history
        with st.chat_message("assistant"):
            response = agent.invoke({"input": user_query})
            st.session_state.messages.append({"role": "assistant", "content": response})
            msgs.add_ai_message(response)  # Save AI response in history
            st.write(response)

    # Draw the messages at the end, so newly generated ones show up immediately
    with view_messages:
        view_messages.json(st.session_state.langchain_messages)
