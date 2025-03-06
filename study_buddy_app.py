'''
import streamlit as st
from StudyBuddy import study_agent
st.title("StudyBuddy AI")
user_input=st.text_input("Where to start, where to break and let go... Where to change, where to look and to grow...")
if user_input:
    ai_response=study_agent(user_input)
    st.markdown(ai_response)
'''
import streamlit as st
from textwrap import dedent
from agno.agent import Agent, AgentMemory
from agno.memory.classifier import MemoryClassifier
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.memory.manager import MemoryManager
from agno.memory.summarizer import MemorySummarizer
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.youtube import YouTubeTools
from agno.knowledge import AgentKnowledge
from agno.vectordb.lancedb import LanceDb
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.search import SearchType
from agno.tools.crawl4ai import Crawl4aiTools
from agno.tools.website import WebsiteTools
from agno.tools.wikipedia import WikipediaTools
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from rich import print
st.set_page_config(page_title="StudyBuddy", page_icon="📚", layout="wide")
st.title("📚 StudyBuddy - Your Personal Learning Companion! 🎓")

agent_storage = SqliteAgentStorage(table_name="study_sessions", db_file="tmp/agents.db")
memory_db = SqliteMemoryDb(
    table_name="study_memory",
    db_file="tmp/agent_memory.db",
)
vector_db=LanceDb(
            uri="/tmp/lancedb",
            table_name="study_documents",
            embedder=GeminiEmbedder(dimensions=1536)
)
knowledge_base = PDFUrlKnowledgeBase(
    vector_db=vector_db,
    search_type=SearchType.keyword
    
)

# Comment out after first run
#knowledge_base.load(recreate=True)

st.title("StuddyBuddy AI")

# User ID input
user_id = st.text_input("Enter your user ID", value="default_user")

# Session selection
new_session = st.checkbox("Start a new study session?")

agent_storage = SqliteAgentStorage(table_name="agent_sessions", db_file="tmp/agents.db")
###############################
# if not new_session:
#   existing_sessions = agent_storage.get_all_session_ids(user_id)
#    if existing_sessions:
#        session_options = ["Most recent"] + existing_sessions
#        selected_session = st.selectbox("Choose a session to continue", options=session_options)
#        if selected_session == "Most recent":
#            session_id = existing_sessions[0]
#        else:
#            session_id = selected_session
#    else:
#        st.warning("No existing sessions found. Starting a new session.")
#        session_id = None
#else:
#    session_id = None
####################################
groqmod=Groq(id="llama-3.3-70b-versatile")
goomod=Gemini(id="gemini-2.0-flash")
ollmod=Ollama(id="llama3.2")

def create_study_buddy(user_id, session_id=None):
    return Agent(
        name="StudyBuddy",
        user_id=user_id,
        session_id=session_id,
        model=goomod,
        memory=AgentMemory(
            db=memory_db,
            create_user_memories=True,
            update_user_memories_after_run=True,
            classifier=MemoryClassifier(model=goomod),
            summarizer=MemorySummarizer(model=goomod),
            manager=MemoryManager(model=goomod, db=memory_db, user_id=user_id),
        ),
        storage=agent_storage,
        knowledge=knowledge_base,
        tools=[DuckDuckGoTools(), YouTubeTools(), Crawl4aiTools(max_length=None), WebsiteTools(), WikipediaTools(), PDFReader(),WebsiteReader(),CSVReader(),TextReader()],
        description=dedent("""\
        You are StudyBuddy, an expert educational mentor with deep expertise in personalized learning! 📚

        Your mission is to be an engaging, adaptive learning companion that helps users achieve their
        educational goals through personalized guidance, interactive learning, and comprehensive resource curation.
        """),# Your description here,
        instructions=dedent("""\
        Follow these steps for an optimal learning experience:

        1. Initial Assessment
        - Learn about the user's background, goals, and interests
        - Assess current knowledge level
        - Identify preferred learning styles

        2. Learning Path Creation
        - Design customized study plans, use DuckDuckGo to find resources
        - Set clear milestones and objectives
        - Adapt to user's pace and schedule
        - Use the material given in the knowledge base

        3. Content Delivery
        - Break down complex topics into digestible chunks
        - Use relevant analogies and examples
        - Connect concepts to user's interests
        - Provide multi-format resources (text, video, interactive)
        - Use the material given in the knowledge base

        4. Resource Curation
        - Find relevant learning materials using DuckDuckGo
        - Recommend quality educational content
        - Share community learning opportunities
        - Suggest practical exercises
        - Use the material given in the knowledge base
        - Use urls with pdf links if provided by the user

        5. Be a friend
        - Provide emotional support if the user feels down
        - Interact with them like how a close friend or homie would


        Your teaching style:
        - Be encouraging and supportive
        - Use emojis for engagement (📚 ✨ 🎯)
        - Incorporate interactive elements
        - Provide clear explanations
        - Use memory to personalize interactions
        - Adapt to learning preferences
        - Include progress celebrations
        - Offer study technique tips

        Remember to:
        - Keep sessions focused and structured
        - Provide regular encouragement
        - Celebrate learning milestones
        - Address learning obstacles
        - Maintain learning continuity\
        """),
        additional_context=dedent(f"""\
        - User ID: {user_id}
        - Session Type: {"New Session" if session_id is None else "Continuing Session"}
        - Available Tools: Web Search, YouTube Resources
        - Memory System: Active
        """),
        add_history_to_messages=True,
        num_history_responses=3,
        show_tool_calls=True,
        markdown=True,
        debug_mode=True
    )

if not new_session:
    existing_sessions = agent_storage.get_all_session_ids(user_id)
    if existing_sessions:
        session_options = ["Most recent"] + existing_sessions
        selected_session = st.selectbox("Choose a session to continue", options=session_options)
        if selected_session == "Most recent":
            session_id = existing_sessions[0]
        else:
            session_id = selected_session
    else:
        st.warning("No existing sessions found. Starting a new session.")
        session_id = None
else:
    session_id = None

study_buddy = create_study_buddy(user_id, session_id)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Where to start, Where to break and let go?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in study_buddy.run(prompt, stream=True):
            full_response += response.content
            message_placeholder.markdown(full_response + "▌")
        #message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
