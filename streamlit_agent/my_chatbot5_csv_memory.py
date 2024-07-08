import streamlit as st
import pandas as pd
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import PandasDataframeTool


st.set_page_config(page_title="Custom LLM with CSV Data")
st.title("Custom LLM with CSV Data 📈")

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

# Set up the LangChain for memory
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Specify the model name gpt-3.5-turbo in ChatOpenAI
chain = prompt | ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Setup agent for CSV
# Specify the model name gpt-3.5-turbo in ChatOpenAI
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0, streaming=True)

# Initialize memory
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=500, return_messages=True)


@st.cache_resource(ttl="2h")
def load_csv():
    csv_file = Path(__file__).parent / "customer_service_dataset.csv"
    df = pd.read_csv(csv_file)
    return df


df = load_csv()

# Initialize the agent with memory
agent = initialize_agent(
    tools=[pandas_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# Check if 'messages' exists in the session state or if the 'Clear message history' button is pressed
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display all messages from the session state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user query from the chat input
user_query = st.chat_input(placeholder="Ask me anything from the dataset!")

if user_query:
    # Append user query to the session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Save user query in history
    msgs.add_user_message(user_query)

    with st.spinner(text="Analyzing the dataset..."):
        # Get the assistant's response using your agent
        response = agent.invoke(user_query)

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
    inter_steps = response.get("intermediate_steps", [])
    if inter_steps:
        prettified_inter_steps = prettify_intermediate_steps(inter_steps)

        # Display intermediate steps in an expander
        with st.expander("See explanation"):
            st.write(prettified_inter_steps)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    view_messages.json(st.session_state.langchain_messages)
