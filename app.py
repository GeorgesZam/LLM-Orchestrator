import streamlit as st
import ollama
import networkx as nx
from pyvis.network import Network
import pandas as pd
import json
import time
from streamlit_chat import message
import os
import re
import requests
import psutil
import platform
import subprocess
from datetime import datetime

# Streamlit page configuration
st.set_page_config(page_title="LLM Orchestrator", layout="wide")

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
    }
    .model-card {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        background: none;  /* Remove gray background */
    }
    .response-container {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)


def get_gpu_memory():
    """Get available GPU memory in GB"""
    try:
        if platform.system() == "Darwin":  # macOS
            cmd = "system_profiler SPDisplaysDataType"
            output = subprocess.check_output(cmd.split()).decode()
            if "Metal" in output:
                return 4  # Estimation for Metal
            return 0
        elif platform.system() == "Windows":
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
            try:
                output = subprocess.check_output(cmd.split())
                return int(output.decode().strip()) / 1024  # Convert to GB
            except:
                return 0
        elif platform.system() == "Linux":
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
                return int(output.decode().strip()) / 1024
            except:
                return 0
        return 0
    except:
        return 0


# Machine configuration
if 'MACHINE_CONFIG' not in st.session_state:
    st.session_state.MACHINE_CONFIG = {
        'max_model_size': 14,  # Maximum size in billions of parameters
        'ram_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1),  # Total RAM in GB
        'gpu_memory_gb': get_gpu_memory()  # GPU memory in GB
    }


# Configuration interface
def show_machine_config():
    st.sidebar.markdown("### 🖥️ Machine Configuration")
    st.session_state.MACHINE_CONFIG['max_model_size'] = st.sidebar.number_input(
        "Max model size (B parameters)",
        min_value=1,
        max_value=70,
        value=st.session_state.MACHINE_CONFIG['max_model_size']
    )

    # Display detected resources
    st.sidebar.info(f"""
        💾 RAM: {st.session_state.MACHINE_CONFIG['ram_gb']} GB
        🎮 GPU: {st.session_state.MACHINE_CONFIG['gpu_memory_gb']} GB

        Recommendations:
        - 8GB RAM min. per 7B model
        - 16GB RAM min. per 13B+ model
        - GPU recommended for large models
    """)


def initialize_session_state():
    """Initialize session state variables"""
    if 'OLLAMA_HOST' not in st.session_state:
        st.session_state.OLLAMA_HOST = "http://localhost:11434"
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'chat_counter' not in st.session_state:
        st.session_state.chat_counter = 0


def get_available_models():
    """Get list of available models on the machine"""
    try:
        response = st.session_state.client.list()
        models = {}

        for model in response.models:
            if hasattr(model, 'model'):
                name = str(model.model)
                base_name = name.split(':')[0]
                models[base_name] = name

        return models
    except Exception as e:
        st.error(f"Error retrieving models: {str(e)}")
        return {}


def get_manager_prompt(question, max_time, available_models):
    """Create prompt for LLM manager"""
    return f"""You are an expert in LLM model selection. Quickly choose the most appropriate model.

Question: "{question}"

Available models:
{json.dumps(list(available_models.values()), indent=2)}

Simple rules:
- Calculations, greetings: small model
- Knowledge questions: medium model
- Creative questions, suggestions: large model
- Complex analysis: large model

JSON format only:
{{
    "selected_models": ["model_name"],
    "strategy": "Short reason"
}}"""


def get_manager_decision(question, max_time):
    """Get quick decision from manager"""
    available_models = get_available_models()

    if not available_models:
        st.error("No models available")
        return None

    manager_model = list(available_models.values())[0]

    try:
        response = st.session_state.client.chat(
            model=manager_model,
            messages=[{
                'role': 'user',
                'content': get_manager_prompt(question, max_time, available_models)
            }],
            stream=False,
            options={'temperature': 0}
        )

        content = response['message']['content'].strip()
        if content.startswith("```"):
            content = re.sub(r'^```.*\n|```$', '', content)

        decision = json.loads(content)
        selected_model = decision['selected_models'][0]

        if selected_model in available_models.values():
            return decision
        else:
            fallback_model = list(available_models.values())[0]
            return {
                'selected_models': [fallback_model],
                'strategy': f"Default model"
            }

    except Exception as e:
        st.error(f"Error: {str(e)}")
        fallback_model = list(available_models.values())[0]
        return {
            'selected_models': [fallback_model],
            'strategy': "Default model (error)"
        }


def analyze_question_complexity(question):
    """Analyze question complexity"""
    simple_patterns = [
        r'^\d+[\s+\-*/]\d+',  # Simple math
        r'^(hi|hello|hey)\b',  # Greetings
        r'^what(\s+is)?\s+\d+[\s+\-*/]\d+',  # Math questions
    ]

    for pattern in simple_patterns:
        if re.match(pattern, question.lower()):
            return "SMALL"

    return None


def get_appropriate_model(available_models, question):
    """Select most appropriate model based on question"""
    complexity = analyze_question_complexity(question)
    if complexity == "SMALL":
        return list(available_models.values())[0]
    else:
        return list(available_models.values())[0]


def get_response(model, question, chat_id):
    """Get response with chat context"""
    try:
        response_container = st.empty()
        response_container.markdown(f"*🤖 Response from {model}...*")

        full_response = ""

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant. Maintain context of the conversation.'}
        ]

        chat = st.session_state.chats[chat_id]
        for msg in chat['messages']:
            messages.append(msg)

        messages.append({'role': 'user', 'content': question})

        stream_response = st.session_state.client.chat(
            model=model,
            messages=messages,
            stream=True
        )

        for chunk in stream_response:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                response_container.markdown(f"*🤖 {model}:*\n{full_response}▌")

        response_container.markdown(f"*🤖 {model}:*\n{full_response}")

        chat['messages'].append({
            'role': 'user',
            'content': question,
            'model': model
        })
        chat['messages'].append({
            'role': 'assistant',
            'content': full_response,
            'model': model
        })
        chat['model'] = model

        save_chats()

        return {'status': 'success', 'content': full_response}

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {'status': 'error', 'content': str(e)}


def create_new_chat():
    """Create a new chat"""
    chat_id = str(st.session_state.chat_counter)
    st.session_state.chats[chat_id] = {
        'title': f"New Chat {chat_id}",
        'messages': [],
        'model': None,
        'created_at': datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_counter += 1
    save_chats()


def save_chats():
    """Save chats to session state"""
    # In a real app, save to database
    pass


def load_chats():
    """Load chats from session state"""
    # In a real app, load from database
    pass


def setup_ollama_connection():
    """Setup Ollama server connection"""
    st.sidebar.markdown("### Server Configuration")

    server_type = st.sidebar.radio(
        "Server Location:",
        ["Local", "Remote"],
        help="Choose where your Ollama server is running"
    )

    if server_type == "Local":
        host = st.sidebar.text_input(
            "Local Server URL",
            value="http://localhost:11434",
            help="Usually http://localhost:11434"
        )
    else:
        host = st.sidebar.text_input(
            "Remote Server URL",
            value="http://",
            help="Example: http://your-server:11434"
        )

    if st.sidebar.button("Connect to Server"):
        try:
            st.session_state.OLLAMA_HOST = host
            st.session_state.client = ollama.Client(host=host)
            st.sidebar.success("Connected to Ollama server!")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {str(e)}")
            st.session_state.client = None


def show_chat_sidebar():
    """Show sidebar with chat list"""
    st.sidebar.markdown("### Your Chats")

    if st.sidebar.button("New Chat"):
        create_new_chat()
        st.rerun()

    st.sidebar.markdown("---")

    for chat_id, chat in sorted(st.session_state.chats.items(), key=lambda x: x[1]['created_at'], reverse=True):
        col1, col2 = st.sidebar.columns([4, 1])

        title = chat['title']
        if chat['messages']:
            first_msg = chat['messages'][0]['content']
            title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg

        if col1.button(f"📝 {title}", key=f"chat_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.rerun()


def show_current_chat():
    """Show current chat"""
    if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            latest_chat_id = max(st.session_state.chats.keys())
            st.session_state.current_chat_id = latest_chat_id
        else:
            create_new_chat()
            st.rerun()
            return

    chat = st.session_state.chats[st.session_state.current_chat_id]

    if chat.get('model'):
        st.markdown(f"*🤖 Current model: {chat['model']}*")

    for msg in chat['messages']:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            model_name = msg.get('model', 'Unknown')
            st.markdown(f"**Assistant ({model_name}):** {msg['content']}")


def main():
    """Main entry point"""
    st.markdown("<h1 class='main-header'>LLM Orchestrator</h1>", unsafe_allow_html=True)

    initialize_session_state()
    setup_ollama_connection()
    load_chats()

    show_chat_sidebar()

    if not st.session_state.client:
        st.warning("Please configure and connect to an Ollama server first")
        return

    show_current_chat()

    if st.session_state.current_chat_id:
        question = st.text_area("Enter your message:", height=100)

        col1, col2 = st.columns([3, 1])
        with col1:
            max_time = st.slider(
                "Response Intelligence Level",
                min_value=30,
                max_value=180,
                value=60,
                help="Higher value = more thoughtful responses"
            )
        with col2:
            st.markdown(f"""
            Intelligence:
            - Level 1 (30)
            - Level 2 (60)
            - Level 3 (120)
            - Level 4 (180)
            """)

        if st.button("Send"):
            if question:
                with st.spinner("Analyzing..."):
                    decision = get_manager_decision(question, max_time)

                    if decision:
                        st.markdown("### Strategy:")
                        st.markdown(decision['strategy'])

                        for model_name in decision['selected_models']:
                            response = get_response(
                                model_name,
                                question,
                                st.session_state.current_chat_id
                            )


if __name__ == "__main__":
    main()
