import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import tiktoken

st.set_page_config(page_title="Custom LLM with Excel DB")
st.title("Custom LLM with Excel DB 📈")

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


# Detailed instructions for the AI on how to interact with the SQL database
prefix = f"""
You are an agent designed to interact with a SQL database.
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
Then you should query the schema of the most relevant tables.

When reacting react to basic conversation:
- Respond to greetings such as "Hello" or "Hi".
- Answer basic questions like "How are you?" with a friendly tone.
- Remind the user that you are an SQL agent and that you can help them interact with the database.
- Present the database schema to the user.
- Invite the user to ask questions based on the schema.
Example of a basic interaction:

User: Hello OR Hi
SQL Agent: Hi there! How can I help you today? Remember, I'm an SQL agent trained to interact with our database. Feel free to ask me anything about it.

User: How are you?
SQL Agent: I'm doing great, thank you! How can I assist you with the database today? Here's the schema for your reference:
- **Orders**: ("Row ID," "Order ID," "Order Date," "Ship Date," "Sales," "Profit," and more)
- **People**: ("Regional Manager," "Region")
- **Returns**: ("Returned," "Order ID")
Feel free to ask any question you have about the data!

When generating SQL queries for the SQLite database, ensure the following:
- Use double quotes or square brackets for column names that contain spaces.
- Do not enclose column names in single quotes.
- Use functions like strftime correctly by applying them directly to column names.
For example:
To query the total sales and profit for the year 2021 where the category is 'Office Supplies', use the following format:
SELECT SUM(Sales) AS Total_Sales, SUM(Profit) AS Total_Profit FROM Orders WHERE Category = 'Office Supplies' AND strftime('%Y', "Order Date") = '2021';

When generating responses, please use the word "dollars" instead of the dollar sign "$". For example, if the total sales amount is 183,939.98, the response should be "183,939.98 dollars" instead of "$183,939.98$".

When Formatting Responses:
- Use a clear , user friendly way to visualize the response.
- If the user asks for a summary or a count, provide a single answer.
- If the user asks for a list of items (e.g., "List all categories" or "Show all orders from 2021"), format the response as a list.
- Remember use the word "dollars" instead of the dollar sign "$".
- Clearly format financial figures on separate lines for readability using new lines.
- If the output is too large, display only the first 5 entries by default and inform the user.
- If the question does not seem related to the database, just return "The query does not relate to the database" as the answer.


For example:
User: What are the total sales and profit for the "Office Supplies" category in 2021?
SQL Agent: The total sales for the "Office Supplies" category in 2021 is 183,939.98 dollars.
The total profit for the "Office Supplies" category in 2021 is 35,061.23 dollars.

Another example:
User: Find the orders from the "West" region along with the name of the regional manager.
SQL Agent: Here are the first 5 orders from the "West" region along with the name of the regional manager:

Order ID: CA-2021-138688
Product Name: Self-Adhesive Address Labels for Typewriters by Universal
Sales: 14.62 dollars
Profit: 6.87 dollars
Regional Manager: Sadie Pawthorne

Order ID: CA-2019-115812
Product Name: Eldon Expressions Wood and Plastic Desk Accessories, Cherry Wood
Sales: 48.86 dollars
Profit: 14.17 dollars
Regional Manager: Sadie Pawthorne

Order ID: CA-2019-115812
Product Name: Newell 322
Sales: 7.28 dollars
Profit: 1.97 dollars
Regional Manager: Sadie Pawthorne

Order ID: CA-2019-115812
Product Name: Mitel 5320 IP Phone VoIP phone
Sales: 907.15 dollars
Profit: 90.72 dollars
Regional Manager: Sadie Pawthorne

Order ID: CA-2019-115812
Product Name: DXL Angle-View Binders with Locking Rings by Samsill
Sales: 18.50 dollars
Profit: 5.78 dollars
Regional Manager: Sadie Pawthorne

If you need more entries, please specify the number of entries you want to retrieve.

Additionally, provide clear and concise natural language explanations in the intermediate_steps for each step you take and the reasons behind those actions. This will help non-programmers understand your thought process and how you arrived at the final answer.
"""

# Additional instructions for the AI on the next steps after receiving the input question
suffix = """I should look at the tables in the database to see what I can query. Then I should query the schema of the most relevant tables.
For each step I take, I should explain in simple, natural language why I am taking that step and how it helps in answering the question.
"""

# Creating the prompt structure with system, human, and AI messages, and incorporating prefix and suffix
messages = [
    SystemMessagePromptTemplate.from_template(prefix),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessagePromptTemplate.from_template(suffix),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate.from_messages(messages)

# Specify the model name gpt-3.5-turbo in ChatOpenAI
chain = prompt | ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="history",
)

# Setup agent for SQL
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0, streaming=True)


# Function to read Excel file into SQLite file-based database
@st.cache_resource(ttl="2h")
def excel_to_sqlite(file_path):
    db_path = "database.db"

    # Create a writable SQLite database connection first
    con = sqlite3.connect(db_path)
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        # Ensure 'Order Date' is in the correct datetime format
        if "Order Date" in df.columns:
            df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        df.to_sql(sheet_name, con, index=False, if_exists="replace")  # Load each sheet into SQLite
    con.close()

    # Change the database to read-only mode
    con_read_only = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
    )  # Create a read-only SQLite database connection
    return SQLDatabase(create_engine("sqlite:///database.db", creator=lambda: con_read_only))


# Read the Excel file from the project folder
file_path = Path("streamlit_agent/(US)Sample-Superstore.xlsx")
db = excel_to_sqlite(file_path)  # Load the Excel file into the SQLite file-based database

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="openai-tools",
    prompt=prompt,
    agent_executor_kwargs={"return_intermediate_steps": True},
)


def clear_message_history():
    st.session_state.pop("messages", None)
    msgs.clear()


# Check if 'messages' exists in the session state or if the 'Clear chat history' button is pressed
if "messages" not in st.session_state or st.sidebar.button(
    "Clear chat history", on_click=clear_message_history
):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display all messages from the session state
for msg in st.session_state.messages:
    if "expander" in msg:
        with st.expander(msg["expander"]):
            st.write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Get user query from the chat input
user_query = st.chat_input(placeholder="Ask me anything from the database!")

# Action descriptions for prettifying intermediate steps
action_descriptions = {
    "sql_db_list_tables": "I have to check the list of available tables in the database.",
    "sql_db_schema": "I have to look at the schema of the '{}' table to understand its structure and the available columns.",
    "sql_db_query": "Now I have to execute a query to get the required data from the '{}' table.",
}


def prettify_intermediate_steps(steps):
    prettified_steps = []
    num_steps = len(steps)
    for i, step in enumerate(steps, 1):
        action, result = step
        tool = action.tool
        tool_input = action.tool_input
        log = action.log

        # Determine the description based on the action type
        if tool in action_descriptions:
            if tool == "sql_db_schema" or tool == "sql_db_query":
                # Use the table name in the description
                table_name = (
                    tool_input.get("table_names")
                    if tool == "sql_db_schema"
                    else tool_input.get("query").split("FROM")[1].split()[0]
                )
                if num_steps > 1:
                    description = (
                        f"**Step {i}:** \n\n **{action_descriptions[tool].format(table_name)}**"
                    )
                else:
                    description = f"**{action_descriptions[tool].format(table_name)}**"
                if tool == "sql_db_query":
                    query_description = (
                        f"**Now the following SQL query has been run:** `{tool_input.get('query')}`"
                    )
            else:
                if num_steps > 1:
                    description = f"**Step {i}:** \n\n **{action_descriptions[tool]}**"
                else:
                    description = f"**{action_descriptions[tool]}**"
        else:
            if num_steps > 1:
                description = f"**Step {i}: ** \n\n **Performed an action.**"
            else:
                description = "**Performed an action.**"

        prettified_steps.append(
            f"{description}\n\n{query_description if tool == 'sql_db_query' else ''}\n\n**The result of this action returns:** {result}"
        )
    return "\n\n".join(prettified_steps)


if user_query:
    # Append user query to the session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Save user query in history
    msgs.add_user_message(user_query)

    # Calculate current tokens and truncate message history if necessary
    max_total_tokens = 4096  # Maximum tokens allowed by the model
    reserved_tokens = 1000  # Reserve tokens for the current query and response
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
        try:
            response = agent.invoke({"input": user_query, "history": truncated_history})
        except Exception as e:
            st.error(f"Error: {e}")
            response = {"output": str(e), "intermediate_steps": []}

    # Extract response content
    response_content = response["output"]

    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)

    # Save AI response in history
    msgs.add_ai_message(response_content)

    # Prettify intermediate steps
    inter_steps = response["intermediate_steps"]

    if inter_steps:
        prettified_inter_steps = prettify_intermediate_steps(inter_steps)

        # Save the whole expander content in the session state
        expander_content = f"**See explanation**\n{prettified_inter_steps}"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "expander": "**See explanation**",
                "content": prettified_inter_steps,
            }
        )

        # Display intermediate steps in an expander
        with st.expander("**See explanation**"):
            st.write(prettified_inter_steps)
    else:
        # Display message if no intermediate steps are provided
        no_explanation_message = (
            "**The agent does not provide any further explanations for its response.**"
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "expander": "**See explanation**",
                "content": no_explanation_message,
            }
        )
        with st.expander("**See explanation**"):
            st.write(no_explanation_message)


# Draw the messages at the end, so newly generated ones show up immediately
# with view_messages:
#    view_messages.json(st.session_state.langchain_messages)
