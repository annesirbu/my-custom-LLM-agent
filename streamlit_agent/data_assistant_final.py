import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import sqlite3
import psycopg2
from psycopg2 import sql  # For dynamic SQL queries
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

# from langchain.agents.agent_types import AgentType
import tiktoken
import subprocess
import sys
import uuid
import openai
from contextlib import contextmanager


# Ensure necessary packages are installed
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import openpyxl
except ImportError:
    install_package("openpyxl")

# Streamlit app setup
st.set_page_config(page_title="Data Assistant")
st.title("Data Assistant 📈")

# Load environment variables from the .env file
load_dotenv()

# Use environment variables for OpenAI API key and PostgreSQL URL
openai_api_key = os.getenv("OPENAI_API_KEY")
database_url = os.getenv("DATABASE_URL")

# Ensure the OpenAI API key is set
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
openai.api_key = openai_api_key


# Context manager for PostgreSQL connection
@contextmanager
def get_pg_connection():
    try:
        conn = psycopg2.connect(database_url)
        yield conn.cursor()
    except Exception as e:
        st.error(f"Failed to connect to the database: {e}")
    finally:
        conn.commit()
        conn.close()


# Function to check if table exists
def table_exists(table_name):
    with get_pg_connection() as pg_cursor:
        pg_cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
            );
        """,
            (table_name,),
        )
        return pg_cursor.fetchone()[0]  # Returns True if the table exists


# Function to create the table if it doesn't exist
def create_table_if_not_exist():
    table_name = "interactions_experiment"  # Replace with your actual table name

    if not table_exists(table_name):
        with get_pg_connection() as pg_cursor:
            pg_cursor.execute(
                f"""
                CREATE TABLE {table_name} (
                    session_id TEXT,
                    id SERIAL PRIMARY KEY,
                    participant_id TEXT,
                    treatment TEXT,
                    user_query TEXT,
                    assistant_response TEXT,
                    intermediate_steps TEXT,
                    simplified_intermediate_steps TEXT,
                    user_query_sent_time TIMESTAMP,
                    response_displayed_time TIMESTAMP,
                    explanation_button_displayed_time TIMESTAMP,
                    explanation_clicked_time TIMESTAMP,
                    explanation_clicked BOOLEAN DEFAULT FALSE,
                    explanation_displayed_time TIMESTAMP
                );
            """
            )


# Function to check and add missing columns, also add composite primary key
def add_columns_if_not_exist():
    table_name = "interactions_experiment"

    with get_pg_connection() as pg_cursor:
        # Function to check if a column exists
        def column_exists(table_name, column_name):
            query = """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name=%s AND column_name=%s
            );
            """
            pg_cursor.execute(query, (table_name, column_name))
            return pg_cursor.fetchone()[0]  # Returns True if the column exists, False otherwise

        # Add missing columns if they don't exist
        if not column_exists(table_name, "participant_id"):
            pg_cursor.execute(f"""ALTER TABLE {table_name} ADD COLUMN participant_id TEXT""")

        if not column_exists(table_name, "treatment"):
            pg_cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN treatment TEXT")

        if not column_exists(table_name, "explanation_displayed_time"):
            pg_cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN explanation_displayed_time TIMESTAMP"
            )

        if not column_exists(table_name, "user_query_sent_time"):
            pg_cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN user_query_sent_time TIMESTAMP")

        if not column_exists(table_name, "response_displayed_time"):
            pg_cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN response_displayed_time TIMESTAMP"
            )

        if not column_exists(table_name, "explanation_button_displayed_time"):
            pg_cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN explanation_button_displayed_time TIMESTAMP"
            )

        if not column_exists(table_name, "explanation_clicked_time"):
            pg_cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN explanation_clicked_time TIMESTAMP"
            )

        if not column_exists(table_name, "explanation_clicked"):
            pg_cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN explanation_clicked BOOLEAN DEFAULT FALSE"
            )

        # Check if a composite primary key exists on session_id and id
        pg_cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM information_schema.table_constraints
            WHERE table_name = '{table_name}'
            AND constraint_type = 'PRIMARY KEY';
        """
        )
        primary_key_exists = pg_cursor.fetchone()[0]

        # If the primary key exists and is not composite, drop and recreate it
        if primary_key_exists:
            pg_cursor.execute(
                f"""
                ALTER TABLE {table_name}
                DROP CONSTRAINT IF EXISTS {table_name}_pkey,
                ADD PRIMARY KEY (session_id, id);
            """
            )


# Call the function to alter the table and add columns if they don't exist
create_table_if_not_exist()
add_columns_if_not_exist()


# Function to save interaction into PostgreSQL
def save_interaction(
    session_id,
    question_id,
    participant_id,
    treatment,
    user_query,
    assistant_response,
    intermediate_steps,
    simplified_intermediate_steps,
    user_query_sent_time,
    response_displayed_time,
    explanation_button_displayed_time,
    explanation_clicked_time,
    explanation_clicked,
    explanation_displayed_time,
):
    intermediate_steps_str = str(intermediate_steps)  # Convert to string
    simplified_intermediate_steps_str = str(simplified_intermediate_steps)  # Convert to string
    user_query_sent_time_str = (
        user_query_sent_time.strftime("%Y-%m-%d %H:%M:%S") if user_query_sent_time else None
    )
    response_displayed_time_str = (
        response_displayed_time.strftime("%Y-%m-%d %H:%M:%S") if response_displayed_time else None
    )
    explanation_button_displayed_time_str = (
        explanation_button_displayed_time.strftime("%Y-%m-%d %H:%M:%S")
        if explanation_button_displayed_time
        else None
    )
    explanation_clicked_time_str = (
        explanation_clicked_time.strftime("%Y-%m-%d %H:%M:%S") if explanation_clicked_time else None
    )
    explanation_displayed_time_str = (
        explanation_displayed_time.strftime("%Y-%m-%d %H:%M:%S")
        if explanation_displayed_time
        else None
    )

    with get_pg_connection() as pg_cursor:
        pg_cursor.execute(
            """
            INSERT INTO interactions_experiment
            (session_id, id, participant_id, treatment, user_query, assistant_response, intermediate_steps,
            simplified_intermediate_steps, user_query_sent_time, response_displayed_time,
            explanation_button_displayed_time, explanation_clicked_time, explanation_clicked, explanation_displayed_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                question_id,
                participant_id,
                treatment,
                user_query,
                assistant_response,
                intermediate_steps_str,
                simplified_intermediate_steps_str,
                user_query_sent_time_str,
                response_displayed_time_str,
                explanation_button_displayed_time_str,
                explanation_clicked_time_str,
                explanation_clicked,
                explanation_displayed_time_str,
            ),
        )


# Function to update explanation clicked
def update_explanation_clicked(session_id, question_id):
    with get_pg_connection() as pg_cursor:
        pg_cursor.execute(
            """
            UPDATE interactions_experiment
            SET explanation_clicked = TRUE, explanation_clicked_time = CURRENT_TIMESTAMP, explanation_displayed_time = CURRENT_TIMESTAMP
            WHERE session_id = %s AND id = %s
            """,
            (session_id, question_id),
        )


# Assign a unique session ID if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Initialize question counter for the session
if "question_counter" not in st.session_state:
    st.session_state["question_counter"] = 0

session_id = st.session_state["session_id"]

# Capture the participant_id and treatment from the URL using st.query_params
query_params = st.query_params
participant_id = query_params.get(
    "participant", None
)  # Retrieve SAVEDID (participant_id) from the URL
treatment = query_params.get("treatment", None)  # Retrieve treatment value (1 or 2) from the URL

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message(
        "Hello! I am the new AI-powered data assistant designed by Superstore. How may I help you?"
    )

view_messages = st.expander("View the message contents in session state")


# Function to count tokens using tiktoken
def count_tokens(messages):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
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
prefix = """
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

When reacting to basic conversation:
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
- Provide paragraph texts that are easy to understand for humans as your response
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

# Specify the model name gpt-4o-mini in ChatOpenAI
chain = prompt | ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="history",
)

# Setup agent for SQL
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0, streaming=True)


# Function to read Excel file into SQLite file-based database
@st.cache_resource(ttl="2h")
def excel_to_sqlite(file_path):
    db_path = "database.db"

    # Create a writable SQLite database connection first
    con = sqlite3.connect(db_path)
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

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
    st.session_state["question_counter"] = 0
    msgs.clear()


# Check if 'messages' exists in the session state or if the 'Clear chat history' button is pressed
if "messages" not in st.session_state or st.sidebar.button(
    "Clear chat history", on_click=clear_message_history
):
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am the new AI-powered data assistant designed by Superstore. How may I help you?",
        }
    ]


# Function to handle explanation click event
def handle_explanation_click(interaction_id):
    update_explanation_clicked(session_id, interaction_id)
    st.session_state[f"expander_{interaction_id}"] = True


# Display all messages from the session state

if treatment == 2:
    # if treatment is None:
    for msg in st.session_state.get("messages", []):
        if "expander" in msg:
            interaction_id = msg.get("interaction_id")
            if st.session_state.get(f"expander_{interaction_id}", False):
                with st.expander(msg["expander"], expanded=True):
                    st.write(msg["content"])
            else:
                if st.button(
                    "See explanation",
                    key=f"button_{interaction_id}",
                    on_click=lambda id=interaction_id: handle_explanation_click(id),
                ):
                    st.session_state[f"expander_{interaction_id}"] = True
                    with st.expander(msg["expander"], expanded=True):
                        st.write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])
else:
    for msg in st.session_state.get("messages", []):
        if "expander" in msg:
            with st.expander(msg.get("expander", "See explanation"), expanded=False):
                st.write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])


# Function to execute SQL queries every time
def execute_sql_query(query):
    with sqlite3.connect("database.db") as con:
        cur = con.cursor()
        cur.execute(query)
        results = cur.fetchall()
    return results


# Function to process user query
def process_user_query(query):
    # This function would typically be more complex, integrating logic to form SQL queries based on user input
    sql_query = f"SELECT * FROM interactions_experiment WHERE user_query LIKE '%{query}%'"
    results = execute_sql_query(sql_query)
    return results


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

        prettified_steps.append(
            f"{description}\n\n{query_description if tool == 'sql_db_query' else ''}\n\n**The result of this action returns:** {result}"
        )
    return "\n\n".join(prettified_steps)


# Ensure the OpenAI client is correctly instantiated
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def explain_intermediate_steps(intermediate_steps):
    prompt = (
        "Explain the following steps in simple, natural language for a non-technical user. Start your answer directly with the response and do not interact with the prompt. Do not start with sentences like: Sure! Here are the steps explained in simple language:"
        "Use the first person when explaining like for example: I checked the databases available. Before every step, write the step and its corresponding number."
        "Also, only if the following steps contain SQL query, please provide it in a code block format that users can try out themselves. Do not provide any SQL statements starting with CREATE TABLE:\n\n"
        f"{intermediate_steps}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You explain the steps in simple, natural language for a non-technical user. Start your answer directly with the response and do not interact with the prompt. Do not start with sentences like: Sure! Here are the steps explained in simple language:"
                "Use the first person when explaining like for example: I checked the databases available. Before every step, write the step and its corresponding number."
                "Also, only if the following steps contain SQL query, please provide it in a code block format that users can try. Do not display any SQL statements starting with CREATE TABLE, write only the ones starting with SELECT",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.1,  # Set the temperature to a low value for precise responses
    )
    message = response.choices[0].message.content.strip()
    return message


# Get user query from the chat input
user_query = st.chat_input(placeholder="Ask me anything from the database!")

if user_query:
    # Record timestamp for when the user query is sent
    user_query_sent_time = pd.Timestamp.now()

    # Increment the question counter for the session
    st.session_state["question_counter"] += 1
    question_id = st.session_state["question_counter"]

    # Append user query to the session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Save user query in history
    msgs.add_user_message(user_query)

    # Calculate current tokens and truncate message history if necessary
    max_total_tokens = 14600  # Maximum tokens allowed by the model
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

    # Record timestamp for when the response is displayed
    response_displayed_time = pd.Timestamp.now()

    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.chat_message("assistant").write(response_content)

    # Save AI response in history
    msgs.add_ai_message(response_content)

    # Prettify intermediate steps
    inter_steps = response["intermediate_steps"]
    prettified_inter_steps = prettify_intermediate_steps(inter_steps)

    # Add a spinner for generating explanation
    with st.spinner(text="Generating explanation..."):
        simplified_intermediate_steps = explain_intermediate_steps(prettified_inter_steps)

    explanation_button_displayed_time = None
    explanation_clicked = None
    explanation_clicked_time = None
    explanation_displayed_time = None

    if treatment == 2:
        # if treatment is None:
        explanation_button_displayed_time = pd.Timestamp.now()
        if inter_steps:
            # Update session state with simplified steps
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "expander": "See explanation",
                    "content": simplified_intermediate_steps,
                    "interaction_id": question_id,
                }
            )

            # Display intermediate steps with a button to see the explanation
            if st.button(
                "See explanation",
                key=f"button_{question_id}",
                on_click=lambda id=question_id: handle_explanation_click(id),
            ):
                st.session_state[f"expander_{question_id}"] = True
                with st.expander("See explanation", expanded=True):
                    st.write(simplified_intermediate_steps)
        else:
            # Display message if no intermediate steps are provided
            no_explanation_message = (
                "**The agent does not provide any further explanation for its response.**"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "expander": "See explanation",
                    "content": no_explanation_message,
                    "interaction_id": question_id,
                }
            )
            if st.button(
                "See explanation",
                key=f"button_{question_id}",
                on_click=lambda id=question_id: handle_explanation_click(id),
            ):
                st.session_state[f"expander_{question_id}"] = True
                with st.expander("See explanation", expanded=True):
                    st.write(no_explanation_message)
    else:
        if inter_steps:
            # Update session state with simplified steps
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "expander": "See explanation",
                    "content": simplified_intermediate_steps,
                    "interaction_id": question_id,
                }
            )

            explanation_displayed_time = pd.Timestamp.now()

            # Display intermediate steps with expander
            with st.expander("See explanation", expanded=True):
                st.write(simplified_intermediate_steps)
        else:
            # Display message if no intermediate steps are provided
            no_explanation_message = (
                "**The agent does not provide any further explanation for its response.**"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "expander": "See explanation",
                    "content": no_explanation_message,
                    "interaction_id": question_id,
                }
            )

            explanation_displayed_time = pd.Timestamp.now()

            with st.expander("See explanation", expanded=True):
                st.write(no_explanation_message)

    # Save interaction to the database
    save_interaction(
        session_id,
        question_id,
        participant_id,
        treatment,
        user_query,
        response_content,
        prettified_inter_steps,
        simplified_intermediate_steps if inter_steps else no_explanation_message,
        user_query_sent_time,
        response_displayed_time,
        explanation_button_displayed_time,
        explanation_clicked_time,
        explanation_clicked,
        explanation_displayed_time,
    )
